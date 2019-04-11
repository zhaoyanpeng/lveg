package edu.shanghaitech.ai.nlp.lvet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import edu.shanghaitech.ai.nlp.data.ObjectFileManager;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.Constraint;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.TaggerFile;
import edu.shanghaitech.ai.nlp.eval.EnglishPennTreebankTagEvaluator;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lvet.impl.MaxRuleTagger;
import edu.shanghaitech.ai.nlp.lvet.impl.TagTPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeTTester extends LVeTConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2210108238288350384L;
	
	protected static EnglishPennTreebankTagEvaluator.LabeledConstituentEval<String> scorer;
	protected static List<List<TaggedWord>> trainTrees;
	protected static List<List<TaggedWord>> testTrees;
	protected static List<List<TaggedWord>> devTrees;
	
	protected static TagTPair ttpairs;
	protected static TagWPair twpairs;
	
	protected static MaxRuleTagger<?, ?> mrTagger;
	protected static ThreadPool mtagger;

	protected static String treeFile;
	protected static Options opts;
	
	protected static boolean doTest = false;
	
	public static void main(String[] args) throws Exception {
		String fparams = args[0];
		try {
			if (args.length > 1 && "test".equals(args[1].trim())) {
				doTest = true;
			}
			args = FunUtil.readFile(fparams, StandardCharsets.UTF_8).split(",");
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
		Map<String, List<List<TaggedWord>>> trees = loadData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG tester] " + (endTime - startTime) / 1000.0 + "\n");
	}

	
	private static void train(Map<String, List<List<TaggedWord>>> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);
		testTrees = trees.get(ID_TEST);
		devTrees = trees.get(ID_DEV);
		
		treeFile = sublogroot + opts.imgprefix;
		
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		Pair.LEADING_IDX = numberer.number(Pair.LEADING);
		Pair.ENDING_IDX = numberer.number(Pair.ENDING);
		
		/* to ease the parameters tuning */
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, 
				opts.nwratio, opts.riserate, opts.rtratio, opts.hardcut, random, null);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, null);
		
//		debugGrammars();
		
		// load grammar
		logger.trace("--->Loading grammars from \'" + subdatadir + opts.inGrammar + "\'...\n");
		TaggerFile gfile = (TaggerFile) TaggerFile.load(subdatadir + opts.inGrammar);
		ttpairs = gfile.getGrammar();
		twpairs = gfile.getLexicon();
		
		
//		logger.trace(ttpairs);
//		logger.trace(twpairs);
//		System.exit(0);
		
		
		twpairs.labelSequences(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		twpairs.labelSequences(testTrees); // save the search time cost by finding a specific tag-word
		twpairs.labelSequences(devTrees); // pair in in Lexicon.score(...)
		
		
		Set<String>[][][] allmasks = null, masks = null;
		if (opts.consfile != null && !opts.consfile.equals("")) { // not safe, since load() may returns null
			Object cons = ObjectFileManager.ObjectFile.load(opts.datadir + opts.consfile);
			if (cons != null) {
				allmasks = ((Constraint) cons).getConstraints(); 
			}
		}
		
		List<List<TaggedWord>> tstTrees = new ArrayList<List<TaggedWord>>();
		List<Integer> idxes = new ArrayList<Integer>();
		int idx = 0, cnt = 0;
		
		List<List<TaggedWord>> goTrees = devTrees;
		if (doTest) {
			goTrees = testTrees;
		}
		for (List<TaggedWord> tree : goTrees) {
			if (tree.size() <= opts.eonlylen) {
				tstTrees.add(tree);
				idxes.add(idx);
			}
			idx += 1;
		}
		
		if (allmasks != null) {
			masks = new HashSet[idxes.size()][][];
			for (int i = 0; i < idxes.size(); i++) {
				masks[i] = allmasks[idxes.get(i)];
			}
			System.out.println("--only eval " + masks.length + " sentences");
		}
		
		
		scorer = new EnglishPennTreebankTagEvaluator.LabeledConstituentEval<String>(
				new HashSet<String>(Arrays.asList(new String[] { })),
				new HashSet<String>(Arrays.asList(new String[] { "''", "``", ".", ":", "," })));
		mrTagger = new MaxRuleTagger<List<TaggedWord>, List<String>>(ttpairs, twpairs, opts.maxslen, opts.ef1prune);
		mtagger = new ThreadPool(mrTagger, opts.nttest);
		
		logger.info("\n---F1 CONFIG---\n[parallel: batch-" + opts.pbatch + ", grad-" + 
				opts.pgrad + ", eval-" + opts.peval + ", test-" + opts.pf1 + "]\n\n");
		sorter = new PriorityQueue<>(3000, wcomparator);
		
		StringBuffer sb = new StringBuffer();
		
		sb.append("[test ]" + f1entry(tstTrees, numberer, false) + "\n");
		
//		sb.append("[test ]" + f1entry(testTrees, numberer, false) + "\n");
//		if (opts.ef1ontrain) { 
//			scorer.reset();
//			sb.append("[train]" + f1entry(trainTrees, numberer, true) + "\n");
//		}
//		if (opts.ef1ondev) {
//			scorer.reset();
//			sb.append("[dev  ]" + f1entry(devTrees, numberer, false) + "\n");
//		}
		logger.info("[summary]\n" + sb.toString() + "\n");
		// kill threads
		ttpairs.shutdown();
		twpairs.shutdown();
		mtagger.shutdown();
	}
	
	protected static void debugGrammars() {
		for (int i = 0; i < 7; i++) {
			String data = subdatadir + "lveg_" + i + ".gr";
			logger.trace("--->Loading grammars from \'" + data  + "\'\n\n");
		}
		System.exit(0);
	}
	
	public static String f1entry(List<List<TaggedWord>> trees, Numberer numberer, boolean istrain) {
		if (opts.pf1) {
			return parallelFscore(opts, mtagger, trees, numberer, istrain);
		} else {
			return serialFscore(opts, mrTagger, trees, numberer, istrain);
		}
	}
	

	public static String parallelFscore(Options opts, ThreadPool mparser, List<List<TaggedWord>> trees, Numberer numberer, boolean istrain) {
		List<TaggedWord> goldTree = null;
		List<String> parsedTree = null;
		int nUnparsable = 0, cnt = 0, idx = 0;
//		List<Tree<State>> trees = new ArrayList<>(stateTreeList.size());
//		filterTrees(opts, stateTreeList, trees, numberer, istrain);
		
		for (List<TaggedWord> tree : trees) {
			mparser.execute(tree);
			while (mparser.hasNext()) {
				goldTree = trees.get(idx);
				parsedTree = (List<String>) mparser.getNext();
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
				parsedTree = (List<String>) mparser.getNext();
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
		logger.trace("\n[max rule parser: " + nUnparsable + " unparsable sample(s) of " + trees.size() + "(" + trees.size() + ") samples]\n");
		logger.trace(summary + "\n\n");
		return summary;
	}
	
	
	public static String serialFscore(Options opts, MaxRuleTagger<?, ?> mrParser, List<List<TaggedWord>> trees, Numberer numberer, boolean istrain) {
//		String str = "(ROOT (NP^g (@NP^g (NN EDUCATION) (NNS ADS)) (: :)))";
//		String str = "(ROOT (SINV^g (@SINV^g (@SINV^g (VP^g (ADVP^g (JJS Hardest)) (NN hit)) (VP^g (VBP are))) (SBAR^g (WHNP^g (WP what)) (S^g (NP^g (PRP he)) (VP^g (VBZ calls) (S^g (NP^g (NP^g (@NP^g (@NP^g (`` ``) (JJ secondary)) ('' '')) (NNS sites)) (SBAR^g (WHNP^g (WDT that)) (S^g (ADVP^g (RB primarily)) (VP^g (VBP serve) (NP^g (NN neighborhood) (NNS residents))))))))))) (. .)))";
//		Tree<String> strtree = (new Trees.PennTreeReader(new StringReader(str))).next();	
//		Tree<State> statetree = StateTreeList.stringTreeToStateTree(strtree, numberer);
//		mrParser.parse(statetree);
		
		int nUnparsable = 0, idx = 0;
		
//		List<Tree<State>> trees = new ArrayList<>(stateTreeList.size());
//		filterTrees(opts, stateTreeList, trees, numberer, istrain);
		
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
		for (List<TaggedWord> tree : trees) {
			/*
			if (tree.getYield().size() > 3) {
				continue;
			}
			logger.trace(Debugger.debugTree(tree, false, (short) 2, numberer, true) + "\n");
			*/
//			logger.trace(Debugger.debugTree(tree, false, (short) 2, numberer, true) + "\n");
			List<String> parsedTree = mrParser.parse(tree, idx);
			if (!saveTree(tree, parsedTree, numberer, idx)) {
				nUnparsable++;
			}
			idx++; // index the State tree
		}
		String summary = scorer.display();
		logger.trace("\n[max rule parser: " + nUnparsable + " unparsable sample(s) of " + trees.size() + "(" + trees.size() + ") samples]\n");
		logger.trace(summary + "\n\n");
		return summary;
	}
	
	
	public static boolean saveTree(List<TaggedWord> tree, List<String> parsedTree, Numberer numberer, int idx) {
		try {
			List<String> goldTree = new ArrayList<String>(tree.size() + 5);
			if (opts.ef1imwrite) {
				// pass
			} else {
				for (TaggedWord word : tree) {
					goldTree.add((String)numberer.object(word.getTagIdx()));
				}
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
