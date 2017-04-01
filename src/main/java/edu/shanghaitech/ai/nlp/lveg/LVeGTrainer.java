package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import edu.berkeley.nlp.PCFGLA.TreeAnnotations;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.lveg.impl.LVeGParser;
import edu.shanghaitech.ai.nlp.lveg.impl.MaxRuleParser;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.impl.Valuator;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * There is only one Grammar instance shared by trainer, lvegParser, maxRuleParser, and valuator.
 * 
 * @author Yanpeng Zhao
 *
 */
public class LVeGTrainer extends LearnerConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1249878080098056557L;
	
	protected static List<Tree<State>> ftrainTrees;
	
	protected static StateTreeList trainTrees;
	protected static StateTreeList testTrees;
	protected static StateTreeList devTrees;
	
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
	
	protected static double bestTrainLL = 0.0;
	protected static double bestDevLL = 0.0;
	protected static int nbatch = 0;
	
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
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = loadData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG learner] " + (endTime - startTime) / 1000.0 + "\n");
		logger.trace("[summary] " + opts.runtag + "\t" + nbatch + "\t" + bestDevLL + "\t" + bestTrainLL + "\t" + (endTime - startTime) / 1000.0 + "\n");
	}
	
	
	private static void train(Map<String, StateTreeList> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);
		testTrees = trees.get(ID_TEST);
		devTrees = trees.get(ID_DEV);
		
		treeFile = sublogroot + opts.imgprefix;
		
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		grammar = new SimpleLVeGGrammar(numberer, -1, opts.useref, refSubTypes);
		lexicon = new SimpleLVeGLexicon(numberer, -1, opts.useref, refSubTypes);
		
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
			/*
			System.out.println("mog : " + mogPool.getNumActive());
			System.out.println("gd  : " + gaussPool.getNumActive());
			*/
			logger.trace("\n--->Going through the training set is over...");
			grammar.postInitialize();
			lexicon.postInitialize();
			logger.trace("post-initializing is over.\n");
		}
		// reset the rule weight
		if (opts.resetw || opts.usemasks) {
			logger.trace("--->Reset rule weights according to treebank grammars...\n");
			resetRuleWeight(grammar, lexicon, numberer, opts.mwfactor);
		}
		/*
		logger.trace(grammar);
		logger.trace(lexicon);
		System.exit(0);
		*/
		lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		lexicon.labelTrees(testTrees); // save the search time cost by finding a specific tag-word
		lexicon.labelTrees(devTrees); // pair in in Lexicon.score(...)
		
		lvegParser = new LVeGParser<Tree<State>, List<Double>>(grammar, lexicon, opts.maxLenParsing, 
				opts.ntcyker, opts.pcyker, opts.reuse, opts.iosprune, opts.usemasks, opts.cntprune);
		mrParser = new MaxRuleParser<Tree<State>, Tree<String>>(grammar, lexicon, opts.maxLenParsing, 
				opts.ntcyker, opts.pcyker, opts.reuse, opts.ef1prune, false);
		valuator = new Valuator<Tree<State>, Double>(grammar, lexicon, opts.maxLenParsing, 
				opts.ntcyker, opts.pcyker, opts.reuse, opts.ellprune, false);
		mvaluator = new ThreadPool(valuator, opts.nteval);
		trainer = new ThreadPool(lvegParser, opts.ntbatch);
		double ll = Double.NEGATIVE_INFINITY;
		
		// initial likelihood of the training set
		logger.trace("\n-------ll of the training data initially is... ");
		long beginTime = System.currentTimeMillis();
		if (opts.eontrain) {
			ll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
			// ll = serialLL(opts, valuator, trainTrees, numberer, true);
		}
		long endTime = System.currentTimeMillis();
		logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		
		// set a global tree for debugging
		for (Tree<State> tree : testTrees) {
			if (tree.getYield().size() == opts.eonlylen) {
				globalTree = tree.shallowClone();
				break;
			}
		}
		/* State tree to String tree */
		if (opts.ellimwrite) {
			String treename = treeFile + "_gd";
			Tree<String> stringTree = StateTreeList.stateTreeToStringTree(globalTree, numberer);
			FunUtil.saveTree2image(null, treename, stringTree, numberer);
			stringTree = TreeAnnotations.unAnnotateTree(stringTree, false);
			FunUtil.saveTree2image(null, treename + "_ua", stringTree, numberer);
			
			Tree<String> parseTree = mrParser.parse(globalTree);
			FunUtil.saveTree2image(null, treeFile + "_ini", parseTree, numberer);
			stringTree = TreeAnnotations.unAnnotateTree(stringTree, false);
			FunUtil.saveTree2image(null, treeFile + "_ini_ua", parseTree, numberer);
		}
		
		logger.info("\n---SGD CONFIG---\n[parallel: batch-" + opts.pbatch + ", grad-" + 
				opts.pgrad + ", eval-" + opts.peval + "] " + Params.toString(false) + "\n");
		
		ftrainTrees = new ArrayList<Tree<State>>(trainTrees.size());
		for (Tree<State> tree : trainTrees) {
			if (opts.eonlylen > 0) {
				if (tree.getYield().size() > opts.eonlylen) { continue; }
			}
			ftrainTrees.add(tree);
		}
		sorter = new PriorityQueue<Tree<State>>(opts.bsize + 5, wcomparator);
		
		if (opts.pbatch) {
			parallelInBatch(numberer);
		} else {
			serialInBatch(numberer);
		}
		// kill threads
		grammar.shutdown();
		lexicon.shutdown();
		trainer.shutdown();
		mvaluator.shutdown();
	}
	
	
	protected static void debugrad(boolean debug) {
		goptimizer.debug(null, debug);
	}
	
	
	protected static void jointrainer(short nfailed) {
		while (!trainer.isDone()) {
			while (trainer.hasNext()) {
				List<Double> score = (List<Double>) trainer.getNext();
				if (score == null) {
					nfailed++;
				} else {
					logger.trace("\n~~~score: " + FunUtil.double2str(score, precision, -1, false, true) + "\n");
				}
			}
		}
	}
	
	
	protected static int getBatch(int ibegin, List<Tree<State>> batch) {
		if (batch != null) { batch.clear(); } 
		int iend = ibegin + opts.bsize, nsample = ftrainTrees.size(), diff;
		iend = (diff = (iend - nsample)) > 0 ? iend = nsample : iend;
		for (int i = ibegin; i < iend; i++) {
			batch.add(ftrainTrees.get(i));
		}
		if (diff > 0) {
			Collections.shuffle(ftrainTrees, rnd4shuffle);
			for (int i = 0; i < diff; i++) {
				batch.add(ftrainTrees.get(i));
			}
			iend = diff;
		}
		// sort the samples by descending sentence length
		
		sorter.clear();
		sorter.addAll(batch);
		batch.clear();
		while (!sorter.isEmpty()) {
			batch.add(sorter.poll());
		}
		
		/*
		for (Tree<State> tree : batch) {
			System.out.println(tree.getYield().size() + "\t" + ibegin + "\t" + diff + "\t" + nsample);
		}
		*/
		return iend;
	}
	
	
	public static void parallelInBatch(Numberer numberer) throws Exception {
		long bTime, eTime, epochBTime, batchBTime, batchETime;
		int iprebeg, ibegin = 0, nfailed, iepoch = 0, ibatch = 0;
		List<Double> trllist = new ArrayList<Double>();
		List<Double> dellist = new ArrayList<Double>();
		List<Double> scoresOfST = new ArrayList<Double>(3);
		List<Tree<State>> batch = new ArrayList<Tree<State>>(opts.bsize + 5);
		do {
			logger.trace("\n\n-------epoch " + iepoch + " begins-------\n\n");
			boolean exit = false;
			epochBTime = System.currentTimeMillis();
			while (true) {
				ibatch++;
				if ((iepoch > opts.epochskipk - 1) && (opts.nbatch == 0 || ((ibatch % opts.nbatch) == 0))) {
					exit = opts.nbatch == 0 ? true : peep(ibatch, iepoch, numberer, trllist, dellist, false);
					if (exit) { break; }
				}
				nfailed = 0;
				iprebeg = ibegin;
				ibegin = getBatch(ibegin, batch);
				batchBTime = System.currentTimeMillis();
				for (Tree<State> tree : batch) {
					trainer.execute(tree);
					while (trainer.hasNext()) {
						List<Double> score = (List<Double>) trainer.getNext();
						if (score == null) {
							nfailed++;
						} else {
							logger.trace("\n~~~score: " + FunUtil.double2str(score, precision, -1, false, true) + "\n");
						}
					}
				}
				jointrainer((short) nfailed); // block the main thread until get all the feedbacks of submitted tasks
				batchETime = System.currentTimeMillis();
				
				// apply gradient descent
				logger.trace("+++Apply gradient descent for the batch " + ibatch + "... ");
				bTime = System.currentTimeMillis();
				
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(true); }
				grammar.applyGradientDescent(scoresOfST);
				lexicon.applyGradientDescent(scoresOfST);
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(false); }
				
				eTime = System.currentTimeMillis();
				logger.trace((eTime - bTime) / 1000.0 + "... batch time: " + (batchETime - batchBTime) / 1000.0 + "\n");
				
				if (ibegin < iprebeg) {
					break; // epoch ends
				}
			}
			if (exit) { // 
				logger.info("\n---[*] exiting since the log likelihood are not increasing any more.\n");
				break;
			} else {
				endepoch(epochBTime, ibatch, iepoch, numberer, trllist, dellist);
			}
		} while (++iepoch < opts.nepoch);
		finals(trllist, dellist, numberer, ibatch, iepoch); // better finalize it in a specialized test class
	}
	
	
	public static void serialInBatch(Numberer numberer) throws Exception {
		long bTime, eTime, epochBTime, batchBTime, batchETime;
		int iprebeg, ibegin = 0, iepoch = 0, ibatch = 0, length, isample = 0;
		List<Double> trllist = new ArrayList<Double>();
		List<Double> dellist = new ArrayList<Double>();
		List<Double> scoresOfST = new ArrayList<Double>(3);
		List<Tree<State>> batch = new ArrayList<Tree<State>>(opts.bsize + 5);
		do {
			logger.trace("\n\n-------epoch " + iepoch + " begins-------\n\n");
			boolean exit = false;
			epochBTime = System.currentTimeMillis();
			while (true) {
				ibatch++;
				if ((iepoch > opts.epochskipk - 1) && (opts.nbatch == 0 || ((ibatch % opts.nbatch) == 0))) {
					exit = opts.nbatch == 0 ? true : peep(ibatch, iepoch, numberer, trllist, dellist, false);
					if (exit) { break; }
				}
				isample = 0;
				iprebeg = ibegin;
				ibegin = getBatch(ibegin, batch);
				batchBTime = System.currentTimeMillis();
				for (Tree<State> tree : batch) {
					isample++;
					length = tree.getYield().size(); 
					logger.trace("---Sample " + isample + "...\t");
					bTime = System.currentTimeMillis();
					
					double scoreT = lvegParser.evalRuleCountWithTree(tree, (short) 0);
					double scoreS = lvegParser.evalRuleCount(tree, (short) 0);
					
					eTime = System.currentTimeMillis();
					logger.trace((eTime - bTime) / 1000.0 + "\t");
					
					scoresOfST.add(scoreT);
					scoresOfST.add(scoreS);
					scoresOfST.add((double) length);
					
					logger.trace("scores: " + FunUtil.double2str(scoresOfST, precision, -1, false, true) + "\teval gradients... ");
					bTime = System.currentTimeMillis();
					
					grammar.evalGradients(scoresOfST);
					lexicon.evalGradients(scoresOfST);
					scoresOfST.clear();
					
					eTime = System.currentTimeMillis();
					logger.trace((eTime - bTime) / 1000.0 + "\n");
				}
				batchETime = System.currentTimeMillis();
				
				// apply gradient descent
				logger.trace("+++Apply gradient descent for the batch " + ibatch + "... ");
				bTime = System.currentTimeMillis();
				
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(true); }
				grammar.applyGradientDescent(scoresOfST);
				lexicon.applyGradientDescent(scoresOfST);
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(false); }
				
				eTime = System.currentTimeMillis();
				logger.trace((eTime - bTime) / 1000.0 + "... batch time: " + (batchETime - batchBTime) / 1000.0 + "\n");
				if (ibegin < iprebeg) {
					break; // epoch ends
				}
			}
			if (exit) { // 
				logger.info("\n---[*] exiting since the log likelihood are not increasing any more.\n");
				break;
			} else {
				endepoch(epochBTime, ibatch, iepoch, numberer, trllist, dellist);
			}
		} while (++iepoch < opts.nepoch);
		finals(trllist, dellist, numberer, ibatch, iepoch); // better finalize it in a specialized test class
	}
	
	
	public static boolean endepoch(double epochBTime, int ibatch, int iepoch,
			Numberer numberer, List<Double> trllist, List<Double> dellist) throws Exception {
		// coarse summary
		long epochETime = System.currentTimeMillis();
		logger.trace("===Average time each sample consumed is " + (epochETime - epochBTime) / (1000.0 * (ibatch * opts.bsize)) + "\n");
		
		// likelihood of the data set
		logger.trace("\n----------log-likelihood in epoch is under evaluation----------\n");
		boolean exit = peep(ibatch, iepoch, numberer, trllist, dellist, true);
		logger.trace("\n------------------------evaluation over------------------------\n");
		
		// shall we clear the inside and outside score in each state of the parse tree after the training on a sample?
		// trainTrees.reset(); // CHECK
		logger.trace("\n-------epoch " + String.format("%1$3s", iepoch) + " ends  -------\n");
		return exit;
	}
	
	
	public static void finals(List<Double> trllist, List<Double> dellist, Numberer numberer, int ibatch, int iepoch) {
		if (!trllist.isEmpty()) { bestTrainLL = Collections.max(trllist); }
		if (!dellist.isEmpty()) { bestDevLL = Collections.max(dellist); }
		nbatch = ibatch;
		logger.trace("Convergence Path [train]: " + trllist + "\tMAX: " + bestTrainLL + "\n");
		logger.trace("Convergence Path [ dev ]: " + dellist + "\tMAX: " + bestDevLL + "\n");
		
		logger.trace("\n----------training is over after " + iepoch + " epoches " + ibatch + " batches----------\n");
		
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
	
	
	public static boolean peep(int ibatch, int iepoch, Numberer numberer, List<Double> trllist, 
			List<Double> dellist, boolean ends) throws Exception {
		long beginTime, endTime;
		double trll = 1e-8, dell = 1e-8;
		// likelihood of the training set
		if (!ends && opts.eontrain) {
			logger.trace("\n-------ll of the training data after " + ibatch + " batches in epoch " + iepoch + " is... ");
			beginTime = System.currentTimeMillis();
			if (opts.peval) {
				trll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
			} else {
				trll = serialLL(opts, valuator, trainTrees, numberer, true);
			}
			endTime = System.currentTimeMillis();
			logger.trace("------->" + trll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			// trainTrees.reset();
			trllist.add(trll);
		}
		
		// check dropping count according to the log likelihood on the development set
		boolean exit = false, save = false;
		if (!ends && opts.eondev && (ibatch % opts.enbatchdev) == 0) {
			logger.trace("\n-------ll of the dev data after " + ibatch + " batches in epoch " + iepoch + " is... ");
			beginTime = System.currentTimeMillis();
			if (opts.peval) {
				dell = parallelLL(opts, mvaluator, devTrees, numberer, false);
			} else {
				dell = serialLL(opts, valuator, devTrees, numberer, false);
			}
			endTime = System.currentTimeMillis();
			logger.trace("------->" + dell + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			// devTrees.reset();
			dellist.add(dell);
			if (dell > bestscore) {
				bestscore = dell;
				cntdrop = 0;
				save = true;
			} else {
				cntdrop++;
				exit = (cntdrop >= opts.nAllowedDrop);
			}
		}
		
		// visualize the parse tree
		if (opts.ellimwrite) {
			String treename = ends ? treeFile + "_" + iepoch : treeFile + "_" + iepoch + "_" + ibatch;
			Tree<String> parseTree = mrParser.parse(globalTree);
			FunUtil.saveTree2image(null, treename, parseTree, numberer);
			parseTree = TreeAnnotations.unAnnotateTree(parseTree, false);
			FunUtil.saveTree2image(null, treename + "_ua", parseTree, numberer);
		}
		
		// save the intermediate grammars
		GrammarFile gfile = new GrammarFile(grammar, lexicon);
		if (opts.saveGrammar && save) { // always save the best grammar to the same file
			String filename = subdatadir + opts.outGrammar + "_best.gr";
			if (gfile.save(filename)) { 
				logger.info("\n------->the best [dev] grammar [iepoch = " + iepoch + ", ibatch = " + ibatch + "]\n");
			}
		}
		// save the grammar at the end of each epoch or after every # of batches
		save = (((ibatch % opts.nbatchSave) == 0) && opts.outGrammar != null);
		if (opts.saveGrammar && (ends || save)) {
			logger.info("\n-------saving grammar file...");
			String filename = (ends ? subdatadir + opts.outGrammar + "_" + iepoch + ".gr" 
					: subdatadir + opts.outGrammar + "_" + iepoch + "_" + ibatch + ".gr");
			if (gfile.save(filename)) {
				logger.info("to \'" + filename + "\' successfully.\n");
			} else {
				logger.info("to \'" + filename + "\' unsuccessfully.\n");
			}
		}
		// save the best grammar according to the log likelihood on training set
		if (trll < 0 && trll > besttrain) {
			besttrain = trll;
			String filename = subdatadir + opts.outGrammar + "_best_train.gr";
			if (gfile.save(filename)) { 
				logger.info("\n------->the best [train] grammar [iepoch = " + iepoch + ", ibatch = " + ibatch + "]\n");
			}
		}
		return exit;
	}
	
	
	/**
	 * We have to evaluate the grammar on only a fraction of training data because the evaluation is quite time-consumed. But it is the 
	 * evaluation on the whole validation dataset or the whole test dataset that can tells whether your trained model is good or bad.
	 */
	public static double parallelLL(Options opts, ThreadPool valuator, StateTreeList stateTreeList, Numberer numberer, boolean istrain) {
		double ll = 0, sumll = 0;
		int nUnparsable = 0, cnt = 0;
		List<Tree<State>> trees = new ArrayList<Tree<State>>(stateTreeList.size());
		filterTrees(opts, stateTreeList, trees, numberer, istrain);
		for (Tree<State> tree : trees) {
//			Tree<String> stringTree = StateTreeList.stateTreeToStringTree(tree, numberer);
//			logger.trace("\n" + cnt + "\t" + stringTree);
			valuator.execute(tree);
			while (valuator.hasNext()) {
				ll = (double) valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll)) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
			cnt++;
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
		List<Tree<State>> trees = new ArrayList<Tree<State>>(stateTreeList.size());
		filterTrees(opts, stateTreeList, trees, numberer, istrain);
		for (Tree<State> tree : trees) {
//			Tree<String> stringTree = StateTreeList.stateTreeToStringTree(tree, numberer);
//			logger.trace("\n" + cnt + "\t" + stringTree + "\n");
			ll = valuator.probability(tree);
//			logger.trace("\n" + cnt + "\t" + ll + "\n");
//			System.exit(0);
			if (Double.isInfinite(ll) || Double.isNaN(ll) || ll > 0) {
				nUnparsable++;
			} else {
				sumll += ll;
			}
			cnt++;
		}
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + cnt + ") samples]\n");
		return sumll;
	}
	
}
