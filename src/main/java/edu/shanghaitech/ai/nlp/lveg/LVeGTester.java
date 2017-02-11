package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.PCFGLA.TreeAnnotations;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.eval.EnglishPennTreebankParseEvaluator;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.lveg.impl.MaxRuleParser;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeGTester extends LearnerConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2210108238288350384L;
	
	protected static EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> scorer;
	protected static StateTreeList trainTrees;
	protected static StateTreeList testTrees;
	protected static StateTreeList devTrees;
	
	protected static LVeGGrammar grammar;
	protected static LVeGLexicon lexicon;
	
	protected static MaxRuleParser<?, ?> mrParser;
	protected static ThreadPool mparser;

	protected static String treeFile;
	protected static Options opts;
	
	
	public static void main(String[] args) throws Exception {
		String fparams = args[0];
		try {
			args = readFile(fparams, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		opts = (Options) optionParser.parse(args, true);
		// configurations
		initialize(opts, true); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = loadData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG tester] " + (endTime - startTime) / 1000.0 + "\n");
	}

	
	private static void train(Map<String, StateTreeList> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);
		testTrees = trees.get(ID_TEST);
		devTrees = trees.get(ID_DEV);
		
		treeFile = sublogroot + opts.imgprefix;
		
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		/* to ease the parameters tuning */
		GaussianMixture.config(opts.expzero, opts.maxmw, opts.ncomponent, opts.nwratio, random, mogPool);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, gaussPool);
		Optimizer.config(opts.choice, random, opts.maxsample, opts.bsize, opts.minmw); // FIXME no errors, just alert you...
		
		// load grammar
		logger.trace("--->Loading grammars from \'" + subdatadir + opts.inGrammar + "\'...\n");
		GrammarFile gfile = (GrammarFile) GrammarFile.load(subdatadir + opts.inGrammar);
		grammar = gfile.getGrammar();
		lexicon = gfile.getLexicon();
		
		/*
		logger.trace(grammar);
		logger.trace(lexicon);
		System.exit(0);
		*/
		
		lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		lexicon.labelTrees(testTrees); // save the search time cost by finding a specific tag-word
		lexicon.labelTrees(devTrees); // pair in in Lexicon.score(...)
		
		scorer = new EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>(
				new HashSet<String>(Arrays.asList(new String[] { "ROOT", "PSEUDO" })), 
				new HashSet<String>(Arrays.asList(new String[] { "''", "``", ".", ":", "," })));
		mrParser = new MaxRuleParser<Tree<State>, Tree<String>>(grammar, lexicon, opts.maxLenParsing, opts.reuse, opts.ef1prune);
		mparser = new ThreadPool(mrParser, opts.nttest);
		
		logger.info("\n---F1 CONFIG---\n[parallel: batch-" + opts.pbatch + ", grad-" + 
				opts.pgrad + ", eval-" + opts.peval + ", test-" + opts.pf1 + "]\n\n");
		
		f1entry(testTrees, numberer, false);
		if (opts.ef1ontrain) { 
			scorer.reset();
			f1entry(trainTrees, numberer, true);
		}
		if (opts.ef1ondev) {
			scorer.reset();
			f1entry(devTrees, numberer, false);
		}
		// kill threads
		grammar.shutdown();
		lexicon.shutdown();
		mparser.shutdown();
	}
	
	
	public static void f1entry(StateTreeList trees, Numberer numberer, boolean istrain) {
		if (opts.pf1) {
			parallelFscore(opts, mparser, trees, numberer, istrain);
		} else {
			serialFscore(opts, mrParser, trees, numberer, istrain);
		}
	}
	

	public static void parallelFscore(Options opts, ThreadPool mparser, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		Tree<State> goldTree = null;
		Tree<String> parsedTree = null;
		int nUnparsable = 0, cnt = 0, idx = 0;
		List<Tree<State>> trees = filterTrees(opts, stateTreeList, numberer, istrain);
		for (Tree<State> tree : trees) {
			mparser.execute(tree);
			while (mparser.hasNext()) {
				goldTree = trees.get(idx);
				parsedTree = (Tree<String>) mparser.getNext();
				if (!saveTree(goldTree, parsedTree, numberer, idx)) {
					nUnparsable++;
				}
				/*
				logger.trace(idx + " --- " + strTree2stateTree(goldTree, numberer) + "\n");
				logger.trace(idx + " --- " + parsedTree + "\n");
				*/
				idx++;
			}
			// logger.trace("submit -------- " + (cnt++) + "\n");
		}
		while (!mparser.isDone()) {
			while (mparser.hasNext()) {
				goldTree = trees.get(idx);
				parsedTree = (Tree<String>) mparser.getNext();
				if (!saveTree(goldTree, parsedTree, numberer, idx)) {
					nUnparsable++;
				}
				/*
				logger.trace(idx + " --- " + strTree2stateTree(goldTree, numberer) + "\n");
				logger.trace(idx + " --- " + parsedTree + "\n");
				*/
				idx++;
			}
		}
		mparser.reset();
		logger.trace("\n[max rule parser: " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + trees.size() + ") samples]\n");
		logger.trace(scorer.display() + "\n\n");
	}
	
	
	public static void serialFscore(Options opts, MaxRuleParser<?, ?> mrParser, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		int nUnparsable = 0, idx = 0;
		List<Tree<State>> trees = filterTrees(opts, stateTreeList, numberer, istrain);
		for (Tree<State> tree : trees) {
			Tree<String> parsedTree = mrParser.parse(tree);
			if (!saveTree(tree, parsedTree, numberer, idx)) {
				nUnparsable++;
			}
			idx++; // index the State tree
		}
		logger.trace("\n[max rule parser: " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + trees.size() + ") samples]\n");
		logger.trace(scorer.display() + "\n\n");
	}
	
	
	public static List<Tree<State>> filterTrees(Options opts, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		int cnt = 0;
		List<Tree<State>> trees = new ArrayList<Tree<State>>();
		int maxlen = istrain ? /*1*/opts.eonlylen : (opts.eonextradev ? opts.eonlylen + 5 : opts.eonlylen);
		for (Tree<State> tree : stateTreeList) {
			if (opts.eonlylen > 0) {
				if (tree.getYield().size() > maxlen) { continue; }
			}
			if (istrain && opts.eratio > 0) {
				if (random.nextDouble() > opts.eratio) { continue; }
			}
			if (opts.efirstk > 0) {
				if (++cnt > opts.efirstk) { break; } // DEBUG
			}
			trees.add(tree);
			/*
			Tree<String> strTree = strTree2stateTree(Tree<State> tree, Numberer numberer)
			logger.trace((cnt - 1) + "\t" + strTree + "\n");
			*/
			// logger.trace((cnt - 1) + "\t" + MethodUtil.debugTree(tree, false, (short) -1, numberer, true) + "\n");
		}
		return trees;
	}
	
	
	public static boolean saveTree(Tree<State> tree, Tree<String> parsedTree, Numberer numberer, int idx) {
		try {
			Tree<String> goldTree = null;
			if (opts.ef1imwrite) {
				String treename = treeFile + "_gd_" + idx;
				goldTree = StateTreeList.stateTreeToStringTree(tree, numberer);
				MethodUtil.saveTree2image(null, treename, goldTree, numberer);
				goldTree = TreeAnnotations.unAnnotateTree(goldTree, false);
				MethodUtil.saveTree2image(null, treename + "_ua", goldTree, numberer);
				
				treename = treeFile + "_te_" + idx;
				MethodUtil.saveTree2image(null, treename, parsedTree, numberer);
				parsedTree = TreeAnnotations.unAnnotateTree(parsedTree, false);
				MethodUtil.saveTree2image(null, treename + "_ua", parsedTree, numberer);
			} else {
				goldTree = StateTreeList.stateTreeToStringTree(tree, numberer);
				goldTree = TreeAnnotations.unAnnotateTree(goldTree, false);
				parsedTree = TreeAnnotations.unAnnotateTree(parsedTree, false);
			}
			scorer.evaluate(parsedTree, goldTree);
			logger.trace(idx + "\tgold  : " + goldTree + "\n");
			logger.trace(idx + "\tparsed: " + parsedTree + "\n");
			return true;
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}
	
}
