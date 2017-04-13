package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import edu.berkeley.nlp.PCFGLA.TreeAnnotations;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.eval.EnglishPennTreebankParseEvaluator;
import edu.shanghaitech.ai.nlp.lveg.impl.PCFGParser;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeGPCFG extends LearnerConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 8232031232691463175L;
	
	protected static EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> scorer;
	protected static StateTreeList trainTrees;
	protected static StateTreeList testTrees;
	protected static StateTreeList devTrees;
	
	protected static LVeGGrammar grammar;
	protected static LVeGLexicon lexicon;
	
	protected static PCFGParser<?, ?> pcfgParser;
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
		/*
		loadToyTrees(opts);
		System.exit(0);
		*/
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
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, 
				opts.nwratio, opts.riserate, opts.rtratio, opts.hardcut, random, mogPool);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, gaussPool);
		Optimizer.config(opts.choice, random, opts.maxsample, opts.bsize, opts.minmw, opts.sampling); // FIXME no errors, just alert you...
		
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
		pcfgParser = new PCFGParser<Tree<State>, Tree<String>>(grammar, lexicon, opts.maxLenParsing, 
				opts.ntcyker, opts.pcyker, opts.ef1prune, false);
		mparser = new ThreadPool(pcfgParser, opts.nttest);
		
		logger.info("\n---F1 CONFIG---\n[parallel: batch-" + opts.pbatch + ", grad-" + 
				opts.pgrad + ", eval-" + opts.peval + ", test-" + opts.pf1 + "]\n\n");
		
		sorter = new PriorityQueue<Tree<State>>(opts.bsize + 5, wcomparator);
		
		StringBuffer sb = new StringBuffer();
		sb.append("[test ]" + f1entry(testTrees, numberer, false) + "\n");
		if (opts.ef1ontrain) { 
			scorer.reset();
			sb.append("[train]" + f1entry(trainTrees, numberer, true) + "\n");
		}
		if (opts.ef1ondev) {
			scorer.reset();
			sb.append("[dev  ]" + f1entry(devTrees, numberer, false) + "\n");
		}
		logger.info("[summary]\n" + sb.toString() + "\n");
		// kill threads
		grammar.shutdown();
		lexicon.shutdown();
		mparser.shutdown();
	}
	
	
	public static String f1entry(StateTreeList trees, Numberer numberer, boolean istrain) {
		if (opts.pf1) {
			return parallelFscore(opts, mparser, trees, numberer, istrain);
		} else {
			return serialFscore(opts, pcfgParser, trees, numberer, istrain);
		}
	}
	

	public static String parallelFscore(Options opts, ThreadPool mparser, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		Tree<State> goldTree = null;
		Tree<String> parsedTree = null;
		int nUnparsable = 0, cnt = 0, idx = 0;
		List<Tree<State>> trees = new ArrayList<Tree<State>>(stateTreeList.size());
		filterTrees(opts, stateTreeList, trees, numberer, istrain);
		
		for (Tree<State> tree : trees) {
			mparser.execute(tree);
			while (mparser.hasNext()) {
				goldTree = trees.get(idx);
				parsedTree = (Tree<String>) mparser.getNext();
				if (!saveTree(goldTree, parsedTree, numberer, idx)) {
					nUnparsable++;
				}
				idx++;
			}
		}
		while (!mparser.isDone()) {
			while (mparser.hasNext()) {
				goldTree = trees.get(idx);
				parsedTree = (Tree<String>) mparser.getNext();
				if (!saveTree(goldTree, parsedTree, numberer, idx)) {
					nUnparsable++;
				}
				idx++;
			}
		}
		mparser.reset();
		String summary = scorer.display();
		logger.trace("\n[max rule parser: " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + trees.size() + ") samples]\n");
		logger.trace(summary + "\n\n");
		return summary;
	}
	
	
	public static String serialFscore(Options opts, PCFGParser<?, ?> mrParser, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {

		int nUnparsable = 0, idx = 0;
		
		List<Tree<State>> trees = new ArrayList<Tree<State>>(stateTreeList.size());
		filterTrees(opts, stateTreeList, trees, numberer, istrain);

		for (Tree<State> tree : trees) {
			Tree<String> parsedTree = mrParser.parse(tree);
			if (!saveTree(tree, parsedTree, numberer, idx)) {
				nUnparsable++;
			}
			idx++; // index the State tree
		}
		String summary = scorer.display();
		logger.trace("\n[max rule parser: " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + trees.size() + ") samples]\n");
		logger.trace(summary + "\n\n");
		return summary;
	}
	
	
	public static boolean saveTree(Tree<State> tree, Tree<String> parsedTree, Numberer numberer, int idx) {
		try {
			Tree<String> goldTree = null;
			if (opts.ef1imwrite) {
				String treename = treeFile + "_gd_" + idx;
				goldTree = StateTreeList.stateTreeToStringTree(tree, numberer);
				FunUtil.saveTree2image(null, treename, goldTree, numberer);
				goldTree = TreeAnnotations.unAnnotateTree(goldTree, false);
				FunUtil.saveTree2image(null, treename + "_ua", goldTree, numberer);
				
				treename = treeFile + "_te_" + idx;
				FunUtil.saveTree2image(null, treename, parsedTree, numberer);
				parsedTree = TreeAnnotations.unAnnotateTree(parsedTree, false);
				FunUtil.saveTree2image(null, treename + "_ua", parsedTree, numberer);
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
