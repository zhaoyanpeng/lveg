package edu.shanghaitech.ai.nlp.lvet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lvet.impl.TagTPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeTTrainer extends LVeTConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3889795373391559958L;
	protected static List<List<TaggedWord>> trainTrees;
	protected static List<List<TaggedWord>> testTrees;
	protected static List<List<TaggedWord>> devTrees;
	
	protected static Optimizer toptimizer;
	protected static Optimizer woptimizer;
	
	protected static TagTPair ttpairs;
	protected static TagWPair twpairs;
	
	protected static Options opts;
	
	
	public static void main(String[] args) throws Exception {
		// main thread
		Thread main = Thread.currentThread();
		main.setName(ThreadPool.MAIN_THREAD);
		// parsing parameters
		String fparams = args[0];
		try {
			args = FunUtil.readFile(fparams, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		opts = (Options) optionParser.parse(args, true);
		// configurations
		initialize(opts, false); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		
		Numberer wrapper = new Numberer();
		Map<String, List<List<TaggedWord>>> sequences = loadData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(sequences, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeT learner] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	private static void train(Map<String, List<List<TaggedWord>>> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);
		testTrees = trees.get(ID_TEST);
		devTrees = trees.get(ID_DEV);
		
//		int iepoch = 0;
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		ttpairs = new TagTPair(numberer, -1);
		twpairs = new TagWPair(numberer, -1);
		
		/* to ease the parameters tuning */
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, 
				opts.nwratio, opts.riserate, opts.rtratio, opts.hardcut, random, null);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, null);
		Optimizer.config(opts.choice, random, opts.maxsample, opts.bsize, opts.minmw, opts.sampling); // FIXME no errors, just alert you...
		
		boolean away = false;
		if (opts.loadGrammar && opts.inGrammar != null && away) {
//			logger.trace("--->Loading grammars from \'" + subdatadir + opts.inGrammar + "\'...\n");
//			GrammarFile gfile = (GrammarFile) GrammarFile.load(subdatadir + opts.inGrammar);
//			grammar = gfile.getGrammar();
//			lexicon = gfile.getLexicon();
//			goptimizer = grammar.getOptimizer();
//			loptimizer = grammar.getOptimizer();
//			
//			String regx = ".*?(\\d+)";
//			Pattern pat = Pattern.compile(regx);
//			Matcher matcher = pat.matcher(opts.inGrammar);
//			if (matcher.find()) { // training continued starting from the input grammar
//				iepoch = Integer.valueOf(matcher.group(1).trim()) + 1;
//			}
		} else {
			toptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			woptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			ttpairs.setOptimizer(toptimizer);
			twpairs.setOptimizer(woptimizer);
			logger.trace("--->Tallying trees...\n");
			for (List<TaggedWord> tree : trainTrees) {
				ttpairs.tallyTaggedSample(tree);
				twpairs.tallyTaggedSample(tree);
				
			}
			logger.trace("\n--->Going through the training set is over...");
			
			ttpairs.postInitialize();
			twpairs.postInitialize();
			logger.trace("post-initializing is over.\n");
			
			// reset the rule weight
//			if (opts.resetw || opts.usemasks) {
//				logger.trace("--->Reset rule weights according to treebank grammars...\n");
//				resetRuleWeight(grammar, lexicon, numberer, opts.mwfactor, opts);
//			}
			
			ttpairs.initializeOptimizer();
			twpairs.initializeOptimizer();
			logger.trace("\n--->Initializing optimizer is over...\n");
		}
		
		logger.trace(ttpairs);
		logger.trace(twpairs);
//		System.exit(0);
		
		twpairs.labelSequences(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		twpairs.labelSequences(testTrees); // save the search time cost by finding a specific tag-word
		twpairs.labelSequences(devTrees); // pair in in Lexicon.score(...)
		
		//
		logger.error("stub\n");
	}
}
