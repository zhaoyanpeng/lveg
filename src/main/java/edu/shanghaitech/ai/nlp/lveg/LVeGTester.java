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
import edu.shanghaitech.ai.nlp.eval.EnglishPennTreebankParseEvaluator;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.lveg.impl.MaxRuleParser;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;
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
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, 
				opts.nwratio, opts.riserate, opts.rtratio, opts.hardcut, random, mogPool);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, gaussPool);
		
//		debugGrammars();
		
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
		mrParser = new MaxRuleParser<Tree<State>, Tree<String>>(grammar, lexicon, opts.maxslen, 
				opts.ntcyker, opts.pcyker, opts.ef1prune, opts.usemasks);
		mparser = new ThreadPool(mrParser, opts.nttest);
		
		logger.info("\n---F1 CONFIG---\n[parallel: batch-" + opts.pbatch + ", grad-" + 
				opts.pgrad + ", eval-" + opts.peval + ", test-" + opts.pf1 + "]\n\n");
		sorter = new PriorityQueue<>(3000, wcomparator);
		
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
	
	protected static void debugGrammars() {
		for (int i = 0; i < 7; i++) {
			String data = subdatadir + "lveg_" + i + ".gr";
			logger.trace("--->Loading grammars from \'" + data  + "\'\n\n");
			GrammarFile gfile = (GrammarFile) GrammarFile.load(data);
			grammar = gfile.getGrammar();
			lexicon = gfile.getLexicon();
			
			GrammarRule brule = grammar.getBRule((short) 5, (short) 64, (short) 9);
			logger.debug(brule + "\tW=" + brule.getWeight() + "\n");
			brule = grammar.getURule((short) 0, (short) 5, RuleType.RHSPACE);
			logger.debug(brule + "\tW=" + brule.getWeight() + "\n");
			brule = grammar.getBRule((short) 6, (short) 5, (short) 10);
			logger.debug(brule + "\tW=" + brule.getWeight() + "\n");
			brule = grammar.getBRule((short) 6, (short) 5, (short) 3);
			logger.debug(brule + "\tW=" + brule.getWeight() + "\n");
			brule = grammar.getBRule((short) 5, (short) 5, (short) 5);
			logger.debug(brule + "\tW=" + brule.getWeight() + "\n");
			brule = grammar.getBRule((short) 5, (short) 5, (short) 3);
			logger.debug(brule + "\tW=" + brule.getWeight() + "\n");
			brule = grammar.getURule((short) 5, (short) 18, RuleType.LRURULE);
			logger.debug(brule + "\tW=" + brule.getWeight() + "\n");
		}
		System.exit(0);
	}
	
	public static String f1entry(StateTreeList trees, Numberer numberer, boolean istrain) {
		if (opts.pf1) {
			return parallelFscore(opts, mparser, trees, numberer, istrain);
		} else {
			return serialFscore(opts, mrParser, trees, numberer, istrain);
		}
	}
	

	public static String parallelFscore(Options opts, ThreadPool mparser, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		Tree<State> goldTree = null;
		Tree<String> parsedTree = null;
		int nUnparsable = 0, cnt = 0, idx = 0;
		List<Tree<State>> trees = new ArrayList<>(stateTreeList.size());
		filterTrees(opts, stateTreeList, trees, numberer, istrain);
		
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
		String summary = scorer.display();
		logger.trace("\n[max rule parser: " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + trees.size() + ") samples]\n");
		logger.trace(summary + "\n\n");
		return summary;
	}
	
	
	public static String serialFscore(Options opts, MaxRuleParser<?, ?> mrParser, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
//		String str = "(ROOT (NP^g (@NP^g (NN EDUCATION) (NNS ADS)) (: :)))";
//		String str = "(ROOT (SINV^g (@SINV^g (@SINV^g (VP^g (ADVP^g (JJS Hardest)) (NN hit)) (VP^g (VBP are))) (SBAR^g (WHNP^g (WP what)) (S^g (NP^g (PRP he)) (VP^g (VBZ calls) (S^g (NP^g (NP^g (@NP^g (@NP^g (`` ``) (JJ secondary)) ('' '')) (NNS sites)) (SBAR^g (WHNP^g (WDT that)) (S^g (ADVP^g (RB primarily)) (VP^g (VBP serve) (NP^g (NN neighborhood) (NNS residents))))))))))) (. .)))";
//		Tree<String> strtree = (new Trees.PennTreeReader(new StringReader(str))).next();	
//		Tree<State> statetree = StateTreeList.stringTreeToStateTree(strtree, numberer);
//		mrParser.parse(statetree);
		
		int nUnparsable = 0, idx = 0;
		
		List<Tree<State>> trees = new ArrayList<>(stateTreeList.size());
		filterTrees(opts, stateTreeList, trees, numberer, istrain);
		
//		List<Tree<State>> trees = getToyTrees(numberer);
		
		/*
		int cnt = 0;
		for (Tree<State> tree : trees) {
			Tree<String> goldTree = StateTreeList.stateTreeToStringTree(tree, numberer);
			goldTree = 		TreeAnnotations.unAnnotateTree(goldTree, false);
			logger.trace("\t" + goldTree + "\n");
			logger.trace(++cnt + "\t" + FunUtil.debugTree(tree, false, (short) 2, numberer, true) + "\n");
			if (cnt >= 348) {
				System.exit(0);
			}
		}
		*/
		for (Tree<State> tree : trees) {
			/*
			if (tree.getYield().size() > 3) {
				continue;
			}
			logger.trace(FunUtil.debugTree(tree, false, (short) 2, numberer, true) + "\n");
			*/
//			logger.trace(FunUtil.debugTree(tree, false, (short) 2, numberer, true) + "\n");
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
