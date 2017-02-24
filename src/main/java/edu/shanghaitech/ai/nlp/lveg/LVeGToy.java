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
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeGToy extends LearnerConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 8827498326298180622L;
	
	protected static StateTreeList trainTrees;
//	protected static StateTreeList testTrees;
//	protected static StateTreeList devTrees;
	
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
	protected static int ntree = 50;
	
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
		Map<String, StateTreeList> trees = makeData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG learner] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	private static void train(Map<String, StateTreeList> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);
		/*
		testTrees = trees.get(ID_TEST);
		devTrees = trees.get(ID_DEV);
		*/
		treeFile = sublogroot + opts.imgprefix;
		
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		grammar = new SimpleLVeGGrammar(numberer, -1);
		lexicon = new SimpleLVeGLexicon(numberer, -1);
		
		/* to ease the parameters tuning */
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, opts.nwratio, random, mogPool);
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
			System.out.println("mog : " + mogPool.getNumActive());
			System.out.println("gd  : " + gaussPool.getNumActive());
			
			logger.trace("\n--->Going through the training set is over...");
			grammar.postInitialize();
			lexicon.postInitialize();
			logger.trace("post-initializing is over.\n");
		}
		
//		customize(grammar, lexicon); // reset grammar rules
		
		/*
		logger.trace(grammar);
		logger.trace(lexicon);
		System.exit(0);
		*/
		lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		/*
		lexicon.labelTrees(testTrees); // save the search time cost by finding a specific tag-word
		lexicon.labelTrees(devTrees); // pair in in Lexicon.score(...)
		*/
		lvegParser = new LVeGParser<Tree<State>, List<Double>>(grammar, lexicon, opts.maxLenParsing, opts.reuse, opts.iosprune, opts.cntprune);
		mrParser = new MaxRuleParser<Tree<State>, Tree<String>>(grammar, lexicon, opts.maxLenParsing, opts.reuse, opts.ef1prune);
		valuator = new Valuator<Tree<State>, Double>(grammar, lexicon, opts.maxLenParsing, opts.reuse, opts.ellprune);
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
		/*
		// set a global tree for debugging
		for (Tree<State> tree : testTrees) {
			if (tree.getYield().size() == opts.eonlylen) {
				globalTree = tree.shallowClone();
				break;
			}
		}
		*/
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
		
		if (opts.pbatch) {
			parallelInBatch(numberer, ll);
		} else {
			serialInBatch(numberer, ll);
		}
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
		
		/*goptimizer.debug(null, debug);*/
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
	
	
	public static void parallelInBatch(Numberer numberer, double prell) throws Exception {
		List<Double> scoresOfST = new ArrayList<Double>(3);
		List<Double> trllist = new ArrayList<Double>();
		List<Double> dellist = new ArrayList<Double>();
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
					if (score == null) {
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
		List<Double> scoresOfST = new ArrayList<Double>(3);
		List<Double> trllist = new ArrayList<Double>();
		List<Double> dellist = new ArrayList<Double>();
		int cnt = 0;
		do {			
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
				
				// if (isample < 3) { isample++; continue; } // DEBUG test grammar loading
				
				logger.trace("---Sample " + isample + "...\t");
				beginTime = System.currentTimeMillis();
				
				double scoreT = lvegParser.evalRuleCountWithTree(tree, (short) 0);
				double scoreS = lvegParser.evalRuleCount(tree, (short) 0);
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\t");
				
				scoresOfST.add(scoreT);
				scoresOfST.add(scoreS);
				scoresOfST.add(length);
				
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
//						debugrad(true); 
						FunUtil.gradcheck(grammar, lexicon, lvegParser, valuator, tree, opts.maxsample);
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
		} while(++cnt < opts.nepoch);
		
		finals(trllist, dellist, numberer, true); // better finalize it in a specialized test class
	}
	
	
	
	
	public static void finals(List<Double> trllist, List<Double> dellist, Numberer numberer, boolean exit) {
		long beginTime, endTime;
		
		logger.trace("Convergence Path [train]: " + trllist + "\tMAX: " + Collections.max(trllist) + "\n");
		/*
		logger.trace("Convergence Path [ dev ]: " + dellist + "\tMAX: " + Collections.max(dellist) + "\n");
		*/
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
		
		// since the final grammar may not be the best grammar, the evaluations below are not 
		// so helpful for analyzing results.
		if (exit) { return; }
		/*
		logger.trace("------->evaluating on the training dataset...");
		beginTime = System.currentTimeMillis();
		double ll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
		endTime = System.currentTimeMillis();
		logger.trace("ll is "+ ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		trainTrees.reset();
		
		logger.trace("------->evaluating on the valuation dataset...");
		beginTime = System.currentTimeMillis();
		ll = parallelLL(opts, mvaluator, devTrees, numberer, false);
		endTime = System.currentTimeMillis();
		logger.trace("ll is "+ ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		devTrees.reset();
		
		logger.trace("------->evaluating on the test dataset...");
		beginTime = System.currentTimeMillis();
		ll = parallelLL(opts, mvaluator, testTrees, numberer, false);
		endTime = System.currentTimeMillis();
		logger.trace("ll is "+ ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		testTrees.reset();
		*/
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
		long beginTime, endTime;
		double trll = 1e-8, dell = 1e-8;
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
		}
		/*
		// check dropping count according to the log likelihood on the development set
		boolean exit = false, save = false;
		if (opts.eondev && (ibatch % opts.enbatchdev) == 0) {
			logger.trace("\n-------ll of the dev data after " + ibatch + " batches in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			if (opts.peval) {
				dell = parallelLL(opts, mvaluator, devTrees, numberer, false);
			} else {
				dell = serialLL(opts, valuator, devTrees, numberer, false);
			}
			endTime = System.currentTimeMillis();
			logger.trace("------->" + dell + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			devTrees.reset();
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
			String treename = ends ? treeFile + "_" + cnt : treeFile + "_" + cnt + "_" + ibatch;
			Tree<String> parseTree = mrParser.parse(globalTree);
			FunUtil.saveTree2image(null, treename, parseTree, numberer);
			parseTree = TreeAnnotations.unAnnotateTree(parseTree, false);
			FunUtil.saveTree2image(null, treename + "_ua", parseTree, numberer);
		}
		
		// save the intermediate grammars
		GrammarFile gfile = new GrammarFile(grammar, lexicon);
		if (opts.saveGrammar && save) { // always save the best grammar to the same file
			String filename = subdatadir + opts.outGrammar + "_best" + ".gr";
			if (gfile.save(filename)) { 
				logger.info("\n------->the best grammar [cnt = " + cnt + ", ibatch = " + ibatch + "]\n");
			}
		}
		// save the grammar at the end of each epoch or after every # of batches
		save = (((ibatch % opts.nbatchSave) == 0) && opts.outGrammar != null);
		if (opts.saveGrammar && (ends || save)) {
			logger.info("\n-------saving grammar file...");
			String filename = (ends ? subdatadir + opts.outGrammar + "_" + cnt + ".gr" 
					: subdatadir + opts.outGrammar + "_" + cnt + "_" + ibatch + ".gr");
			if (gfile.save(filename)) {
				logger.info("to \'" + filename + "\' successfully.\n");
			} else {
				logger.info("to \'" + filename + "\' unsuccessfully.\n");
			}
		}
		return exit;
		*/
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
		}
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + "(" + cnt + ") samples]\n");
		return sumll;
	}
	
	public static Map<String, StateTreeList> makeData(Numberer wraper, Options opts) {
		StateTreeList trainTrees;
		Numberer numberer = wraper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, StateTreeList> trees = new HashMap<String, StateTreeList>(3, 1);
		List<Tree<String>> strTrees = new ArrayList<Tree<String>>();
		for (int i = 0; i < ntree; i++) {
			String string = "(ROOT (A_" + i + " (B X_" + i + ")))";
			strTrees.add((new Trees.PennTreeReader(new StringReader(string))).next());
		}
//		int idx = 0;
//		for (int i = 0; i < 1; i++) {
//			String string = "(ROOT (A_" + idx + " (B X_" + idx + ")))";
//			strTrees.add((new Trees.PennTreeReader(new StringReader(string))).next());
//		}
		trainTrees = stringTreeToStateTree(strTrees, numberer, opts, false);
		trees.put(ID_TRAIN, trainTrees);
		return trees;
	}
	
	
	private static void customize(LVeGGrammar grammar, LVeGLexicon lexicon) {
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, GrammarRule.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, GrammarRule.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, GrammarRule.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, GrammarRule.LHSPACE, true);	
		
		Component comp01 = ur01.getWeight().getComponent((short) 0);
		Component comp03 = ur03.getWeight().getComponent((short) 0);
		Component comp12 = ur12.getWeight().getComponent((short) 0);
		Component comp32 = ur32.getWeight().getComponent((short) 0);
		Component comp20 = ur20.getWeight().getComponent((short) 0);
		Component comp21 = ur21.getWeight().getComponent((short) 0);
		
		GaussianDistribution gd01 = comp01.squeeze(GrammarRule.Unit.C);
		GaussianDistribution gd03 = comp03.squeeze(GrammarRule.Unit.C);
		GaussianDistribution gd12p = comp12.squeeze(GrammarRule.Unit.P);
		GaussianDistribution gd12c = comp12.squeeze(GrammarRule.Unit.UC);
		GaussianDistribution gd32p = comp32.squeeze(GrammarRule.Unit.P);
		GaussianDistribution gd32c = comp32.squeeze(GrammarRule.Unit.UC);
		GaussianDistribution gd20 = comp20.squeeze(GrammarRule.Unit.P);
		GaussianDistribution gd21 = comp21.squeeze(GrammarRule.Unit.P);
		
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
