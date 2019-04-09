package edu.shanghaitech.ai.nlp.lvet;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.shanghaitech.ai.nlp.data.ObjectFileManager.TaggerFile;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lvet.impl.LVeTTagger;
import edu.shanghaitech.ai.nlp.lvet.impl.MaxRuleTagger;
import edu.shanghaitech.ai.nlp.lvet.impl.TagTPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.impl.Valuator;
import edu.shanghaitech.ai.nlp.lvet.io.CoNLLFileReader;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.GradientCheckerT;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeTToy extends LVeTConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6746808071430099232L;
	
	protected static List<List<TaggedWord>> trainTrees;
	
	protected static Optimizer toptimizer;
	protected static Optimizer woptimizer;
	
	protected static TagTPair ttpairs;
	protected static TagWPair twpairs;
	
	protected static Valuator<?, ?> valuator;
	protected static LVeTTagger<?, ?> lvetTagger;
	protected static MaxRuleTagger<?, ?> mrTagger;
	
	protected static ThreadPool mvaluator;
	protected static ThreadPool trainer;
	
	protected static Options opts;
	protected static int ntree = 3;
	
	public static void main(String[] args) throws Exception {
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
		
//		System.exit(0);
		
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, List<List<TaggedWord>>> trees = 
				/* loadSequences(wrapper, opts); */
				complexSequences(wrapper, opts);
				/*simpleSequences(wrapper, opts);*/
		
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG learner] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	private static void train(Map<String, List<List<TaggedWord>>> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);

		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		ttpairs = new TagTPair(numberer, -1);
		twpairs = new TagWPair(numberer, -1);
		
		/* to ease the parameters tuning */
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, 
				opts.nwratio, opts.riserate, opts.rtratio, opts.hardcut, random, null);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, null);
		Optimizer.config(opts.choice, random, opts.maxsample, opts.bsize, opts.minmw, opts.sampling); // FIXME no errors, just alert you...
				
		if (opts.loadGrammar && opts.inGrammar != null) {
			logger.trace("--->Loading grammars from \'" + opts.datadir + opts.inGrammar + "\'...\n");
			TaggerFile gfile = (TaggerFile) TaggerFile.load(subdatadir + opts.inGrammar);
			ttpairs = gfile.getGrammar();
			twpairs = gfile.getLexicon();
			toptimizer = ttpairs.getOptimizer();
			woptimizer = twpairs.getOptimizer();
		} else {
			toptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			woptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			ttpairs.setOptimizer(toptimizer);
			twpairs.setOptimizer(woptimizer);
			logger.trace("--->Tallying trees...\n");
			for (List<TaggedWord> tree : trainTrees) {
				ttpairs.tallyTaggedWords(tree);
				twpairs.tallyTaggedWords(tree);
				
			}
			logger.trace("\n--->Going through the training set is over...");
			
			ttpairs.postInitialize();
			twpairs.postInitialize();
			logger.trace("post-initializing is over.\n");
			
//			resetPPrule(grammar, lexicon);
//			resetInterule(grammar, lexicon);
			
			ttpairs.initializeOptimizer();
			twpairs.initializeOptimizer();
			logger.trace("\n--->Initializing optimizer is over...\n");
		}
		
//		customize(grammar, lexicon); // reset grammar rules
		
		logger.trace(ttpairs);
		logger.trace(twpairs);
		
		
		// DEBUG print initial grammars
		logger.info("\n\n----------PRINT INITIAL GRAMMARS----------\n\n");
		printGrammars();
		
		
		twpairs.labelSequences(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		
		lvetTagger = new LVeTTagger<List<TaggedWord>, List<Double>>(ttpairs, twpairs, opts.maxslen, opts.iosprune);
		
		mrTagger = new MaxRuleTagger<List<TaggedWord>, List<String>>(ttpairs, twpairs, opts.maxslen, opts.ef1prune);
		valuator = new Valuator<List<TaggedWord>, Double>(ttpairs, twpairs, opts.maxslen, opts.ellprune);
		mvaluator = new ThreadPool(valuator, opts.nteval);
		trainer = new ThreadPool(lvetTagger, opts.ntbatch);
		double ll = Double.NEGATIVE_INFINITY;
		
		// initial likelihood of the training set
		logger.trace("\n-------ll of the training data initially is... ");
		long beginTime = System.currentTimeMillis();
//		ll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
//		ll = serialLL(opts, valuator, trainTrees, numberer, true);
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
		
		logger.trace(ttpairs);
		logger.trace(twpairs);
		
		// kill threads
		ttpairs.shutdown();
		twpairs.shutdown();
		trainer.shutdown();
		mvaluator.shutdown();
	}
	
	protected static void debugrad(boolean debug) {
		Map<GrammarRule, GrammarRule> uRuleMap = ttpairs.getEdgeMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			toptimizer.debug(entry.getKey(), debug);
		}
		uRuleMap = twpairs.getEdgeMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			woptimizer.debug(entry.getKey(), debug);
		}
		/*goptimizer.debug(null, debug);*/
	}
	
	
	protected static void printGrammars() {
		GrammarRule rule = null;
		Map<GrammarRule, GrammarRule> uRuleMap = ttpairs.getEdgeMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			rule = entry.getKey();
			logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.getWeight() + "\n----------\n");
		}
		uRuleMap = twpairs.getEdgeMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
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
			for (List<TaggedWord> tree : trainTrees) {
				if (opts.eonlylen > 0) {
					if (tree.size() > opts.eonlylen) { continue; }
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
					
					ttpairs.applyGradientDescent(scoresOfST);
					twpairs.applyGradientDescent(scoresOfST);
					
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
			for (List<TaggedWord> tree : trainTrees) {
				length = tree.size(); 
				if (opts.eonlylen > 0) {
					if (length > opts.eonlylen) { continue; }
				}
				
				logger.trace("---Sample " + isample + "...\t");
				beginTime = System.currentTimeMillis();
				
				scoresOfST = lvetTagger.evalEdgeCounts(tree, (short) 0);
				scoresOfST.add(length);
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\t");
				
				if (Double.isInfinite(scoresOfST.get(0)) || Double.isInfinite(scoresOfST.get(1))) {
					logger.trace("scores: " + FunUtil.double2str(scoresOfST, precision, -1, false, true) + "\n");
					//logger.trace("--------- " + StateTreeList.stateTreeToStringTree(tree, numberer) + "\n");
					continue;
				}
				logger.trace("scores: " + FunUtil.double2str(scoresOfST, precision, -1, false, true) + "\teval gradients... ");
				beginTime = System.currentTimeMillis();
				
				ttpairs.evalGradients(scoresOfST);
				twpairs.evalGradients(scoresOfST);
				scoresOfST.clear();
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\n");
				
				isample++;
				if (++idx % opts.bsize == 0) {
					batchend = System.currentTimeMillis();
					
					if (opts.dgradnbatch > 0 && ((isample % (opts.bsize * opts.dgradnbatch)) == 0)) { 
						debugrad(true); 
						GradientCheckerT.gradcheck(ttpairs, twpairs, lvetTagger, valuator, tree, opts.maxsample);
					}
					
					// apply gradient descent
					logger.trace("+++Apply gradient descent for the batch " + (isample / opts.bsize) + "... ");
					beginTime = System.currentTimeMillis();
					
					ttpairs.applyGradientDescent(scoresOfST);
					twpairs.applyGradientDescent(scoresOfST);
					
					endTime = System.currentTimeMillis();
					logger.trace((endTime - beginTime) / 1000.0 + "... batch time: " + (batchend - batchstart) / 1000.0 + "\n");
					idx = 0;
					
					if ((isample % (opts.bsize * opts.nbatch)) == 0) {
						exit = peep(isample, cnt, numberer, trllist, dellist, false);
						if (exit) { break; }
					}
					batchstart = System.currentTimeMillis();
				}
				break;
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
			TaggerFile gfile = new TaggerFile(ttpairs, twpairs);
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
			ttpairs.applyGradientDescent(scoresOfST);
			twpairs.applyGradientDescent(scoresOfST);
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
		// of the parse tree after the training on a sample; NOT necessarily.
		Collections.shuffle(trainTrees);
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
	public static double parallelLL(Options opts, ThreadPool valuator, List<List<TaggedWord>> stateTreeList, Numberer numberer, boolean istrain) {
		double ll = 0, sumll = 0;
		int nUnparsable = 0, cnt = 0;
		int maxlen = istrain ? opts.eonlylen : (opts.eonextradev ? opts.eonlylen + 5 : opts.eonlylen);
		for (List<TaggedWord> tree : stateTreeList) {
			if (opts.eonlylen > 0) {
				if (tree.size() > maxlen) { continue; }
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
	
	public static double serialLL(Options opts, Valuator<?, ?> valuator, List<List<TaggedWord>> stateTreeList, Numberer numberer, boolean istrain) {
		double ll = 0, sumll = 0;
		int nUnparsable = 0, cnt = 0;
		int maxlen = istrain ? /*1*/opts.eonlylen : (opts.eonextradev ? opts.eonlylen + 5 : opts.eonlylen);
		for (List<TaggedWord> tree : stateTreeList) {
			if (opts.eonlylen > 0) {
				if (tree.size() > maxlen) { continue; }
			}
			if (istrain && opts.eratio > 0) {
				if (random.nextDouble() > opts.eratio) { continue; }
			}
			if (opts.efirstk > 0) {
				if (++cnt > opts.efirstk) { break; } // DEBUG
			}
			
			ll = valuator.probability(tree);
			System.out.println("~~>" + ll);
			
//			ll = valuator.probability(tree, true);
//			System.out.println("-->" + ll);
			
			if (Double.isInfinite(ll) || Double.isNaN(ll) || ll > 0) {
				nUnparsable++;
			} else {
				sumll += ll;
			}
		}
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + cnt + ") samples]\n");
		return sumll;
	}
	
	public static Map<String, List<List<TaggedWord>>> loadSequences(Numberer wraper, Options opts) throws Exception {
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		numberer.number(Pair.LEADING); // starting of the sequence
		numberer.number(Pair.ENDING); // ending of the sequence
		
		Map<String, List<List<TaggedWord>>> trees = new HashMap<>(3, 1);
		List<List<TaggedWord>> sequences = CoNLLFileReader.read(opts.datadir + "wsj.21.toy.dep");
		labelTags(sequences, numberer, opts, false);
		trees.put(ID_TRAIN, sequences);
		debugNumbererTag(numberer, opts); // DEBUG
		return trees;
	}
	
	public static Map<String, List<List<TaggedWord>>> simpleSequences(Numberer wraper, Options opts) throws Exception {
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		numberer.number(Pair.LEADING); // starting of the sequence
		numberer.number(Pair.ENDING); // ending of the sequence
		
//		String strings = "A\tW_1\t\nB\tW_2\t\nC\tW_3\t\n\n" +
//						 "A\tW_1\t\nD\tW_2\t\nC\tW_3\t\n\n";
		String strings = "A\tW_1\t\nB\tW_2\t\n\n" +
						 "A\tW_1\t\nC\tW_2\t\n\n";
		BufferedReader bstring = new BufferedReader(new StringReader(strings));
		
		Map<String, List<List<TaggedWord>>> trees = new HashMap<>(3, 1);
		CoNLLFileReader.config(0, 1);
		List<List<TaggedWord>> sequences = CoNLLFileReader.read(bstring);
		labelTags(sequences, numberer, opts, false);
		trees.put(ID_TRAIN, sequences);
		debugNumbererTag(numberer, opts); // DEBUG
		return trees;
	}
	
	public static Map<String, List<List<TaggedWord>>> complexSequences(Numberer wraper, Options opts) throws Exception {
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		numberer.number(Pair.LEADING); // starting of the sequence
		numberer.number(Pair.ENDING); // ending of the sequence
		
		String strings = "A\tW_1\t\nB\tW_2\t\nC\tW_3\t\n\n" +
						 "A\tW_1\t\nC\tW_2\t\nD\tW_1\t\n\n" + 
						 "B\tW_3\t\nD\tW_2\t\nC\tW_1\t\n\n" + 
						 "B\tW_3\t\nD\tW_2\t\nB\tW_1\t\n\n";
		BufferedReader bstring = new BufferedReader(new StringReader(strings));
		
		Map<String, List<List<TaggedWord>>> trees = new HashMap<>(3, 1);
		CoNLLFileReader.config(0, 1);
		List<List<TaggedWord>> sequences = CoNLLFileReader.read(bstring);
		labelTags(sequences, numberer, opts, false);
		trees.put(ID_TRAIN, sequences);
		debugNumbererTag(numberer, opts); // DEBUG
		return trees;
	}
	
	
	
}
