package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.PCFGLA.TreeAnnotations;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.lveg.impl.LVeGParser;
import edu.shanghaitech.ai.nlp.lveg.impl.MaxRuleParser;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.impl.Valuator;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.GradientChecker;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeGToy extends LearnerConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 8827498326298180622L;
	
	protected static StateTreeList trainTrees;
	
	protected static Optimizer goptimizer;
	protected static Optimizer loptimizer;
	
	protected static LVeGGrammar grammar;
	protected static LVeGLexicon lexicon;
	
	protected static Valuator<?, ?> valuator;
	protected static LVeGParser<?, ?> lvegParser;
	protected static MaxRuleParser<?, ?> mrParser;
	
	protected static ThreadPool mvaluator;
	protected static ThreadPool trainer;
	
	protected static Tree<State> globalTree;
	protected static String treeFile;
	
	protected static Options opts;
	protected static int ntree = 3;
	
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
		initialize(opts, false); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		
//		showPPTrees(opts);
//		System.exit(0);
		
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = 
				/*makeAugmentedData(wrapper, opts);*/
				/*loadPPTrees(wrapper, opts);*/
				/*makeData(wrapper, opts); */
				makeComplexData(wrapper, opts);
		
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG learner] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	private static void train(Map<String, StateTreeList> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);

		treeFile = sublogroot + opts.imgprefix;
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		grammar = new SimpleLVeGGrammar(numberer, -1);
		lexicon = new SimpleLVeGLexicon(numberer, -1);
		
		/* to ease the parameters tuning */
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, 
				opts.nwratio, opts.riserate, opts.rtratio, opts.hardcut, random, mogPool);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, gaussPool);
		Optimizer.config(opts.choice, random, opts.maxsample, opts.bsize, opts.minmw, opts.sampling); // FIXME no errors, just alert you...
				
		if (opts.loadGrammar && opts.inGrammar != null) {
			logger.trace("--->Loading grammars from \'" + opts.datadir + opts.inGrammar + "\'...\n");
			GrammarFile gfile = (GrammarFile) GrammarFile.load(subdatadir + opts.inGrammar);
			grammar = gfile.getGrammar();
			lexicon = gfile.getLexicon();
			goptimizer = grammar.getOptimizer();
			loptimizer = grammar.getOptimizer();
		} else {
			goptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			loptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			grammar.setOptimizer(goptimizer);
			lexicon.setOptimizer(loptimizer);
			logger.trace("--->Tallying trees...\n");
			for (Tree<State> tree : trainTrees) {
				lexicon.tallyStateTree(tree);
				grammar.tallyStateTree(tree);
				
			}
			logger.trace("\n--->Going through the training set is over...");
			
			grammar.postInitialize();
			lexicon.postInitialize();
			logger.trace("post-initializing is over.\n");
			
//			resetPPrule(grammar, lexicon);
//			resetInterule(grammar, lexicon);
			
			grammar.initializeOptimizer();
			lexicon.initializeOptimizer();
			logger.trace("\n--->Initializing optimizer is over...\n");
		}
		
//		customize(grammar, lexicon); // reset grammar rules
		
//		logger.trace(grammar);
//		logger.trace(lexicon);
//		System.exit(0);
		
		
		// DEBUG print initial grammars
		logger.info("\n\n----------PRINT INITIAL GRAMMARS----------\n\n");
		printGrammars();
		
		
		lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		
		lvegParser = new LVeGParser<Tree<State>, List<Double>>(grammar, lexicon, opts.maxslen, 
				opts.ntcyker, opts.pcyker, opts.iosprune, opts.usemasks);
		mrParser = new MaxRuleParser<Tree<State>, Tree<String>>(grammar, lexicon, opts.maxslen, 
				opts.ntcyker, opts.pcyker, opts.ef1prune, opts.usemasks);
		valuator = new Valuator<Tree<State>, Double>(grammar, lexicon, opts.maxslen, 
				opts.ntcyker, opts.pcyker, opts.ellprune, opts.usemasks);
		mvaluator = new ThreadPool(valuator, opts.nteval);
		trainer = new ThreadPool(lvegParser, opts.ntbatch);
		double ll = Double.NEGATIVE_INFINITY;
		
		// initial likelihood of the training set
		logger.trace("\n-------ll of the training data initially is... ");
		long beginTime = System.currentTimeMillis();
//		ll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
		ll = serialLL(opts, valuator, trainTrees, numberer, true);
		long endTime = System.currentTimeMillis();
		logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		
		logger.info("\n---SGD CONFIG---\n[parallel: batch-" + opts.pbatch + ", grad-" + 
				opts.pgrad + ", eval-" + opts.peval + "] " + Params.toString(false) + "\n");
		
		if (opts.pbatch) {
			parallelInBatch(numberer, ll);
		} else {
			serialInBatch(numberer, ll);
		}
		
		
		// DEBUG print final grammars
		logger.info("\n\n----------PRINT FINAL GRAMMARS----------\n\n");
		printGrammars();
		
		logger.trace(grammar);
		logger.trace(lexicon);
		
		// kill threads
		grammar.shutdown();
		lexicon.shutdown();
		trainer.shutdown();
		mvaluator.shutdown();
	}
	
	protected static void debugrad(boolean debug) {
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getURuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			goptimizer.debug(entry.getKey(), debug);
		}
		uRuleMap = lexicon.getURuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			loptimizer.debug(entry.getKey(), debug);
		}
		/*
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : bRuleMap.entrySet()) {
			goptimizer.debug(entry.getKey(), debug);
		}
		*/
		/*goptimizer.debug(null, debug);*/
	}
	
	
	protected static void printGrammars() {
		GrammarRule rule = null;
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getURuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			rule = entry.getKey();
			logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.getWeight() + "\n----------\n");
		}
		uRuleMap = lexicon.getURuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			rule = entry.getKey();
			logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.getWeight() + "\n----------\n");
		}
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : bRuleMap.entrySet()) {
			rule = entry.getKey();
			logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.getWeight() + "\n----------\n");
		}
	}
	
	
	protected static void jointrainer(short nfailed) {
		while (!trainer.isDone()) {
			while (trainer.hasNext()) {
				List<Double> score = (List<Double>) trainer.getNext();
				if (Double.isInfinite(score.get(0)) || Double.isInfinite(score.get(1))) {
					nfailed++;
				} else {
					logger.trace("\n~~~score: " + FunUtil.double2str(score, precision, -1, false, true) + "\n");
				}
			}
		}
	}
	
	
	public static void parallelInBatch(Numberer numberer, double prell) throws Exception {
		List<Double> scoresOfST = new ArrayList<>(3);
		List<Double> trllist = new ArrayList<>();
		List<Double> dellist = new ArrayList<>();
		int cnt = 0;
		do {			
			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
			boolean exit = false;
			short isample = 0, idx = 0, nfailed = 0;
			long beginTime, endTime, startTime = System.currentTimeMillis();
			long batchstart = System.currentTimeMillis(), batchend;
			for (Tree<State> tree : trainTrees) {
				if (opts.eonlylen > 0) {
					if (tree.getYield().size() > opts.eonlylen) { continue; }
				}
				trainer.execute(tree);
				while (trainer.hasNext()) { // not really block the main thread
					List<Double> score = (List<Double>) trainer.getNext();
					if (Double.isInfinite(score.get(0)) || Double.isInfinite(score.get(1))) {
						nfailed++;
					} else {
						logger.trace("\n~~~score: " + FunUtil.double2str(score, precision, -1, false, true) + "\n");
					}
				}
				
				isample++;
				if (++idx % opts.bsize == 0) {
					jointrainer(nfailed); // block the main thread until get all the feedbacks of submitted tasks
					trainer.reset();  // after the whole batch
					batchend = System.currentTimeMillis();
					
					if (opts.dgradnbatch > 0 && ((isample % (opts.bsize * opts.dgradnbatch)) == 0)) { debugrad(true); }
					
					// apply gradient descent
					logger.trace("+++Apply gradient descent for the batch " + (isample / opts.bsize) + "... ");
					beginTime = System.currentTimeMillis();
					
					grammar.applyGradientDescent(scoresOfST);
					lexicon.applyGradientDescent(scoresOfST);
					
					endTime = System.currentTimeMillis();
					logger.trace((endTime - beginTime) / 1000.0 + "... batch time: " + (batchend - batchstart) / 1000.0 + ", nfailed: " + nfailed + "\n");
					idx = 0;
					
					if ((isample % (opts.bsize * opts.nbatch)) == 0) {
						exit = peep(isample, cnt, numberer, trllist, dellist, false);
						if (exit) { break; }
					}
					nfailed = 0;
					batchstart = System.currentTimeMillis();
				}
			}
			jointrainer(nfailed); // if the last batch has not been joined
			trainer.reset();
			if (exit) { // 
				logger.info("\n---exiting since the log likelihood are not increasing any more.\n");
				break;
			} else {
				exit = ends(idx, isample, cnt, startTime, scoresOfST, numberer, trllist, dellist);
				if (exit) { 
					logger.info("\n---exiting since the log likelihood are not increasing any more.\n");
					break; 
				}
			}
		} while(++cnt < opts.nepoch);
		
		finals(trllist, dellist, numberer, true); // better finalize it in a specialized test class
	}
	
	
	public static void serialInBatch(Numberer numberer, double prell) throws Exception {
		List<Double> scoresOfST = null;
		List<Double> trllist = new ArrayList<>();
		List<Double> dellist = new ArrayList<>();
		int cnt = 0;
		int a = 200;
		do {			
//			if (cnt % a == 0) {
//				logger.info("\n---check similarity");
//				checkSimilarity(grammar);
//			}
			
			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
			double length = 0;
			boolean exit = false;
			short isample = 0, idx = 0;
			long beginTime, endTime, startTime = System.currentTimeMillis();
			long batchstart = System.currentTimeMillis(), batchend;
			for (Tree<State> tree : trainTrees) {
				length = tree.getYield().size(); 
				if (opts.eonlylen > 0) {
					if (length > opts.eonlylen) { continue; }
				}
				
				logger.trace("---Sample " + isample + "...\t");
				beginTime = System.currentTimeMillis();
				
				scoresOfST = lvegParser.evalRuleCounts(tree, (short) 0);
				scoresOfST.add(length);
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\t");
				
				if (Double.isInfinite(scoresOfST.get(0)) || Double.isInfinite(scoresOfST.get(1))) {
					logger.trace("scores: " + FunUtil.double2str(scoresOfST, precision, -1, false, true) + "\n");
					logger.trace("--------- " + StateTreeList.stateTreeToStringTree(tree, numberer) + "\n");
					continue;
				}
				logger.trace("scores: " + FunUtil.double2str(scoresOfST, precision, -1, false, true) + "\teval gradients... ");
				beginTime = System.currentTimeMillis();
				
				grammar.evalGradients(scoresOfST);
				lexicon.evalGradients(scoresOfST);
				scoresOfST.clear();
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\n");
				
				isample++;
				if (++idx % opts.bsize == 0) {
					batchend = System.currentTimeMillis();
					
					if (opts.dgradnbatch > 0 && ((isample % (opts.bsize * opts.dgradnbatch)) == 0)) { 
						debugrad(true); 
						GradientChecker.gradcheck(grammar, lexicon, lvegParser, valuator, tree, opts.maxsample);
					}
					
					// apply gradient descent
					logger.trace("+++Apply gradient descent for the batch " + (isample / opts.bsize) + "... ");
					beginTime = System.currentTimeMillis();
					
					grammar.applyGradientDescent(scoresOfST);
					lexicon.applyGradientDescent(scoresOfST);
					
					endTime = System.currentTimeMillis();
					logger.trace((endTime - beginTime) / 1000.0 + "... batch time: " + (batchend - batchstart) / 1000.0 + "\n");
					idx = 0;
					
					if ((isample % (opts.bsize * opts.nbatch)) == 0) {
						exit = peep(isample, cnt, numberer, trllist, dellist, false);
						if (exit) { break; }
					}
					batchstart = System.currentTimeMillis();
				}
			}
			if (exit) { // 
				logger.info("\n---[I] exiting since the log likelihood are not increasing any more.\n");
				break;
			} else {
				exit = ends(idx, isample, cnt, startTime, scoresOfST, numberer, trllist, dellist);
				if (exit) { 
					logger.info("\n[O]---exiting since the log likelihood are not increasing any more.\n");
					break; 
				}
			}
			
//			if (cnt > 500 && cnt % 200 == 0) {
//			if (cnt < 600 && cnt % a == 0) {
//			if (cnt > 0 && cnt % a == 0) { // does not work for n = 3
//			if (cnt % a == 0) { // work for n == 3
//			if (cnt % a == 0) {
//				logger.info("\n---check similarity");
//				checkSimilarity(grammar);
//			}
			
		} while(++cnt < opts.nepoch);
		
		finals(trllist, dellist, numberer, true); // better finalize it in a specialized test class
	}
	
	
	
	
	public static void finals(List<Double> trllist, List<Double> dellist, Numberer numberer, boolean exit) {
		logger.trace("Convergence Path [train]: " + trllist + "\nMAX: " + Collections.max(trllist) + "\n");
		logger.trace("\n----------training is over after " + trllist.size() + " batches----------\n");
		
		if (opts.saveGrammar) {
			logger.info("\n-------saving the final grammar file...");
			GrammarFile gfile = new GrammarFile(grammar, lexicon);
			String filename = subdatadir + opts.outGrammar + "_final.gr";
			if (gfile.save(filename)) {
				logger.info("to \'" + filename + "\' successfully.\n");
			} else {
				logger.info("to \'" + filename + "\' unsuccessfully.\n");
			}
		}
	}
	
	
	
	
	public static boolean ends(int idx, int isample, int cnt, long startTime, List<Double> scoresOfST, 
			Numberer numberer, List<Double> trllist, List<Double> dellist) throws Exception {
		// if not a multiple of batchsize
		long beginTime, endTime;
		if (idx != 0) {
			logger.trace("+++Apply gradient descent for the last batch " + (isample / opts.bsize + 1) + "... ");
			beginTime = System.currentTimeMillis();
			grammar.applyGradientDescent(scoresOfST);
			lexicon.applyGradientDescent(scoresOfST);
			endTime = System.currentTimeMillis();
			logger.trace((endTime - beginTime) / 1000.0 + "\n");
			scoresOfST.clear();
		}
		
		// a coarse summary
		endTime = System.currentTimeMillis();
		logger.trace("===Average time each sample cost is " + (endTime - startTime) / (1000.0 * isample) + "\n");
		
		// likelihood of the data set
		logger.trace("\n----------log-likelihood in epoch is under evaluation----------\n");
		/*boolean exit = peep(isample, cnt, numberer, trllist, dellist, true);*/
		boolean exit = false;
		logger.trace("\n------------------------evaluation over------------------------\n");
		
		// we shall clear the inside and outside score in each state 
		// of the parse tree after the training on a sample 
		trainTrees.reset();
		trainTrees.shuffle(rnd4shuffle);
		logger.trace("\n-------epoch " + String.format("%1$3s", cnt) + " ends  -------\n");
		return exit;
	}
	
	
	
	
	public static boolean peep(int isample, int cnt, Numberer numberer, List<Double> trllist, List<Double> dellist, boolean ends) throws Exception {
		double trll = 1e-8;
		long beginTime, endTime;
		int ibatch = (isample / opts.bsize);
		// likelihood of the training set
		if (opts.eontrain) {
			logger.trace("\n-------ll of the training data after " + ibatch + " batches in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			if (opts.peval) {
				trll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
			} else {
				trll = serialLL(opts, valuator, trainTrees, numberer, true);
			}
			endTime = System.currentTimeMillis();
			logger.trace("------->" + trll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			trainTrees.reset();
			trllist.add(trll);
			
			if (trll > bestscore) {
				bestscore = trll;
				cntdrop = 0;
			} else {
				cntdrop++;
			}
		}
		return false;
	}
	
	
	/**
	 * We have to evaluate the grammar on only a fraction of training data because the evaluation is quite time-consumed. But it is the 
	 * evaluation on the whole validation dataset or the whole test dataset that can tells whether your trained model is good or bad.
	 */
	public static double parallelLL(Options opts, ThreadPool valuator, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		double ll = 0, sumll = 0;
		int nUnparsable = 0, cnt = 0;
		int maxlen = istrain ? opts.eonlylen : (opts.eonextradev ? opts.eonlylen + 5 : opts.eonlylen);
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
			valuator.execute(tree);
			while (valuator.hasNext()) {
				ll = (double) valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll)) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
		}
		while (!valuator.isDone()) {
			while (valuator.hasNext()) {
				ll = (double) valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll) || ll > 0) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
		}
		valuator.reset();
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + cnt + ") samples]\n");
		return sumll;
	}
	
	
	public static double serialLL(Options opts, Valuator<?, ?> valuator, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		double ll = 0, sumll = 0;
		int nUnparsable = 0, cnt = 0;
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
			ll = valuator.probability(tree);
			if (Double.isInfinite(ll) || Double.isNaN(ll) || ll > 0) {
				nUnparsable++;
			} else {
				sumll += ll;
			}
		}
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + cnt + ") samples]\n");
		return sumll;
	}
	
	
	protected static Map<String, StateTreeList> makeComplexData(Numberer wraper, Options opts) {
		StateTreeList trainTrees;
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, StateTreeList> trees = new HashMap<>(3, 1);
		List<Tree<String>> strTrees = new ArrayList<>();
		
		String str1 = "(ROOT (S (A w_0) (E (B w_1) (C w_2))))";
		String str2 = "(ROOT (S (A w_0) (F (B w_1) (C w_2))))";
		String str3 = "(ROOT (S (A w_0) (G (E (B w_1) (C w_2)))))";
		String str4 = "(ROOT (S (A w_0) (F (E (D (B w_1) (C w_2))))))";
		String str5 = "(ROOT (L (A w_0) (M (B w_1) (C w_2))))";
		String str6 = "(ROOT (L (K (A w_0) (M (B w_1) (C w_2)))))";
		String str7 = "(ROOT (K (J (A w_0) (M (B w_1) (C w_2)))))";
		String str8 = "(ROOT (J (A w_0) (M (B w_1) (C w_2))))";
		strTrees.add((new Trees.PennTreeReader(new StringReader(str1))).next());
		strTrees.add((new Trees.PennTreeReader(new StringReader(str2))).next());
		strTrees.add((new Trees.PennTreeReader(new StringReader(str3))).next());
		strTrees.add((new Trees.PennTreeReader(new StringReader(str4))).next());
		strTrees.add((new Trees.PennTreeReader(new StringReader(str5))).next());
		strTrees.add((new Trees.PennTreeReader(new StringReader(str6))).next());
		strTrees.add((new Trees.PennTreeReader(new StringReader(str7))).next());
		strTrees.add((new Trees.PennTreeReader(new StringReader(str8))).next());
		
		trainTrees = stringTreeToStateTree(strTrees, numberer, opts, false);
		trees.put(ID_TRAIN, trainTrees);
		return trees;
	}
	
	protected static Map<String, StateTreeList> makeData(Numberer wraper, Options opts) {
		StateTreeList trainTrees;
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, StateTreeList> trees = new HashMap<>(3, 1);
		List<Tree<String>> strTrees = new ArrayList<>();
		for (int i = 0; i < ntree; i++) {
			String string = "(ROOT (A_" + i + " (B X_" + i + ")))";
			strTrees.add((new Trees.PennTreeReader(new StringReader(string))).next());
		}
		trainTrees = stringTreeToStateTree(strTrees, numberer, opts, false);
		trees.put(ID_TRAIN, trainTrees);
		return trees;
	}
	
	
	protected static Map<String, StateTreeList> makeAugmentedData(Numberer wraper, Options opts) {
		StateTreeList trainTrees;
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, StateTreeList> trees = new HashMap<>(3, 1);
		List<Tree<String>> strTrees = new ArrayList<>();
		for (int i = 0; i < ntree; i++) {
//			String string = "(ROOT (A_" + i + " (B X_" + i + ")))";
			String string = "(ROOT (A_" + i + " (B (C X_" + i + "))))";
			strTrees.add((new Trees.PennTreeReader(new StringReader(string))).next());
		}
		trainTrees = stringTreeToStateTree(strTrees, numberer, opts, false);
		trees.put(ID_TRAIN, trainTrees);
		return trees;
	}
	
	
	protected static void resetInterule(LVeGGrammar grammar, LVeGLexicon lexicon) {
		GrammarRule rule = grammar.getURule((short) 2, 3, RuleType.LRURULE);
		rule.addWeightComponent(rule.type, (short) (ntree - 1), (short) -1);
	}
	
	protected static void checkSimilarity(LVeGGrammar grammar) {
		GrammarRule rule = grammar.getURule((short) 2, 3, RuleType.LRURULE);
		int ncomp = rule.weight.ncomponent();
		GaussianDistribution gdi, gdj;
		Component compi, compj;
		double pivot = 1e-5, range = 4;
		for (int i = 0; i < ncomp; i++) {
			compi = rule.weight.getComponent((short) i);
			
//			if (Math.exp(compi.getWeight()) < 1e-2) {
//				compi.setWeight(Math.log(1e-2));
//				logger.info("---reset the small mixing weight");
//			}
//			if (Math.exp(compi.getWeight()) > 1e1) {
//				compi.setWeight(Math.log(2));
//				logger.info("---reset the small mixing weight");
//			}
			
//			if (Math.exp(compi.getWeight()) < 1e-2 ||
//					Math.exp(compi.getWeight()) > 1e1) {
//				compi.setWeight(0);
//				gdi = compi.squeeze(GrammarRule.Unit.P);
//				resetGD(gdi, range);
//				gdi = compi.squeeze(GrammarRule.Unit.UC);
//				resetGD(gdi, range);
//				logger.info("---reset the small mixing weight");
//				continue;
//			}
			/*
			for (int j = i + 1; j < ncomp; j++) {
				compj = rule.weight.getComponent((short) j);
				gdi = compi.squeeze(GrammarRule.Unit.P);
				gdj = compj.squeeze(GrammarRule.Unit.P);
				double value = eulerDistance(gdi, gdj);
				if (value < pivot) {
					resetGD(gdj, range);
					logger.info("\n---reset P parameters");
				}
				
				gdi = compi.squeeze(GrammarRule.Unit.UC);
				gdj = compj.squeeze(GrammarRule.Unit.UC);
				value = eulerDistance(gdi, gdj);
				if (value < pivot) {
					resetGD(gdj, range);
					logger.info("\n---reset UC parameters");
				}
			}
			*/
			
			for (int j = i + 1; j < ncomp; j++) {
				
				compj = rule.weight.getComponent((short) j);
				gdi = compi.squeeze(RuleUnit.P);
				gdj = compj.squeeze(RuleUnit.P);
				
				double value = integral(gdi, gdj);
				if (Math.exp(value) > pivot) {
					resetGD(gdj, range);
					logger.info("\n---reset P parameters");
				}
				
				gdi = compi.squeeze(RuleUnit.UC);
				gdj = compj.squeeze(RuleUnit.UC);
				
				value = integral(gdi, gdj);
				if (Math.exp(value) > pivot) {
					resetGD(gdj, range);
					logger.info("\n---reset UC parameters");
				}
			}
			
		}
	}
	
	
	protected static double eulerDistance(GaussianDistribution gdi, GaussianDistribution gdj) {
		int dim = gdi.getDim();
		double value = 0, vtmp = 0, epsilon = 0;
		List<Double> mus0 = gdi.getMus(), vars0 = gdi.getVars();
		List<Double> mus1 = gdj.getMus(), vars1 = gdj.getVars();
		assert(gdi.getDim() == gdj.getDim());
		
		for (int i = 0; i < dim; i++) {
			value += Math.pow(mus0.get(i) - mus1.get(i), 2);
		}
		value = Math.sqrt(value);
		return value;
	}
	
	
	protected static void resetGD(GaussianDistribution gd, double range) {
		List<Double> mus = gd.getMus(), vars = gd.getVars();
		int dim = gd.getDim();
		mus.clear();
		vars.clear();
		for (int i = 0; i < dim; i++) {
			double rndn = (random.nextDouble() - 0.5) * range;
			mus.add(rndn);
			vars.add(0.0);
		}
	}
	
	
	protected static double integral(GaussianDistribution gdi, GaussianDistribution gdj) {
		int dim = gdi.getDim();
		double value = 0, vtmp = 0, epsilon = 0;
		List<Double> mus0 = gdi.getMus(), vars0 = gdi.getVars();
		List<Double> mus1 = gdj.getMus(), vars1 = gdj.getVars();
		assert(gdi.getDim() == gdj.getDim());
		
		for (int i = 0; i < dim; i++) {
			double mu0 = mus0.get(i), mu1 = mus1.get(i);
			double vr0 = Math.exp(vars0.get(i) * 2), vr1 = Math.exp(vars1.get(i) * 2);
			vtmp = vr0 + vr1 + epsilon;
			// int NN dx, // in logarithmic form
			double shared0 = -0.5 * Math.log(vtmp) - Math.pow(mu0 - mu1, 2) / (2 * vtmp);
			// complete integral
			value += shared0; 
		}
		value += Math.log(2 * Math.PI) * (-dim / 2.0); // normalizer in Gaussian is implicitly cached
		return value;
	}
	
	
	public static Map<String, StateTreeList> loadPPTrees(Numberer wraper, Options opts) {
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, StateTreeList> trees = new HashMap<>(3, 1);
		List<Tree<String>> strTrees = loadStringTree(opts.datadir + "wsj_toy_tree_ppa", opts);
		StateTreeList trainTrees = stringTreeToStateTree(strTrees, numberer, opts, false);
		trees.put(ID_TRAIN, trainTrees);
		return trees;
	}
	
	
	public static void showPPTrees(Options opts) throws Exception {
		List<Tree<String>> trees = loadStringTree(opts.datadir + "wsj_toy_tree_ppa", opts);
		int idx = 0;
		String name, prefix = sublogroot + opts.imgprefix + "_gd";
		for (Tree<String> tree : trees) {
			name = prefix + "_" + idx;
			System.out.println(idx + "\t" + tree);
			FunUtil.saveTree2image(null, name, tree, null);
			Tree<String> gold = TreeAnnotations.unAnnotateTree(tree, false);
			name += "_ud";
			System.out.println(idx + "\t" + gold);
			FunUtil.saveTree2image(null, name, gold, null);
			idx++;
		}
	}
	
	protected static Map<String, StateTreeList> ppAttachment(Numberer wraper, Options opts) {
		StateTreeList trainTrees;
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, StateTreeList> trees = new HashMap<>(3, 1);
		List<Tree<String>> strTrees = new ArrayList<>();
		for (int i = 0; i < ntree; i++) {
			String string = "(ROOT (A_" + i + " (B X_" + i + ")))";
			strTrees.add((new Trees.PennTreeReader(new StringReader(string))).next());
		}
		trainTrees = stringTreeToStateTree(strTrees, numberer, opts, false);
		trees.put(ID_TRAIN, trainTrees);
		return trees;
	}
	
	
	protected static void resetPPrule(LVeGGrammar grammar, LVeGLexicon lexicon) {
		GrammarRule rule = grammar.getBRule((short) 8, (short) 9, (short) 3);
		rule.addWeightComponent(rule.type, (short) 1, (short) -1);
		rule = grammar.getURule((short) 3, 7, RuleType.LRURULE);
		rule.addWeightComponent(rule.type, (short) 1, (short) -1);
	}
	
	protected static void customize(LVeGGrammar grammar, LVeGLexicon lexicon) {
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		
		Component comp01 = ur01.getWeight().getComponent((short) 0);
		Component comp03 = ur03.getWeight().getComponent((short) 0);
		Component comp12 = ur12.getWeight().getComponent((short) 0);
		Component comp32 = ur32.getWeight().getComponent((short) 0);
		Component comp20 = ur20.getWeight().getComponent((short) 0);
		Component comp21 = ur21.getWeight().getComponent((short) 0);
		
		GaussianDistribution gd01 = comp01.squeeze(RuleUnit.C);
		GaussianDistribution gd03 = comp03.squeeze(RuleUnit.C);
		GaussianDistribution gd12p = comp12.squeeze(RuleUnit.P);
		GaussianDistribution gd12c = comp12.squeeze(RuleUnit.UC);
		GaussianDistribution gd32p = comp32.squeeze(RuleUnit.P);
		GaussianDistribution gd32c = comp32.squeeze(RuleUnit.UC);
		GaussianDistribution gd20 = comp20.squeeze(RuleUnit.P);
		GaussianDistribution gd21 = comp21.squeeze(RuleUnit.P);
		
		double mua = 1.0, mub = -1.0;
		gd01.getMus().set(0, mua);
		gd03.getMus().set(0, mub);
		gd12p.getMus().set(0, mua);
		gd12c.getMus().set(0, mua);
		gd32p.getMus().set(0, mub);
		gd32c.getMus().set(0, mub);
		gd20.getMus().set(0, mua);
		gd21.getMus().set(0, mub);
		
		double std = 3;
		double vara = Math.log(std), varb = Math.log(std);
		gd01.getVars().set(0, vara);
		gd03.getVars().set(0, varb);
		gd12p.getVars().set(0, vara);
		gd12c.getVars().set(0, vara);
		gd32p.getVars().set(0, varb);
		gd32c.getVars().set(0, varb);
		gd20.getVars().set(0, vara);
		gd21.getVars().set(0, varb);
		
		Map<GrammarRule, GrammarRule> urmap = grammar.getURuleMap();
		urmap.get(ur01).setWeight(ur01.getWeight());
		urmap.get(ur03).setWeight(ur03.getWeight());
		urmap.get(ur12).setWeight(ur12.getWeight());
		urmap.get(ur32).setWeight(ur32.getWeight());
		urmap = lexicon.getURuleMap();
		urmap.get(ur20).setWeight(ur20.getWeight());
		urmap.get(ur21).setWeight(ur21.getWeight());
	}
}
