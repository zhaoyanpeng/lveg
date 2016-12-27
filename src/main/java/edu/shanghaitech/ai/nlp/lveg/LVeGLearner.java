package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.lveg.impl.LVeGParser;
import edu.shanghaitech.ai.nlp.lveg.impl.MaxRuleParser;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexiconOld;
import edu.shanghaitech.ai.nlp.lveg.impl.Valuator;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGLearner extends LearnerConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1249878080098056557L;
	
	protected static StateTreeList trainTrees;
	protected static StateTreeList testTrees;
	protected static StateTreeList devTrees;
	
	protected static LVeGGrammar grammar;
	protected static LVeGLexicon lexicon;
	
	protected static LVeGParser<?, ?> lvegParser;
	protected static MaxRuleParser<?, ?> mrParser;
	
	protected static ThreadPool mvaluator;
	protected static ThreadPool trainer;
	
	protected static Tree<State> globalTree;
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
		initialize(opts); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = loadData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG learner] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	
	private static void train(Map<String, StateTreeList> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);
		testTrees = trees.get(ID_TEST);
		devTrees = trees.get(ID_DEV);
		
		double ll = Double.NEGATIVE_INFINITY;
		String filename = opts.imagePrefix + "_gd";
		treeFile = opts.imagePrefix + "_tr";
		
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		grammar = new SimpleLVeGGrammar(numberer, -1);
		lexicon = new SimpleLVeGLexiconOld(numberer, -1);
		Valuator<?, ?> valuator = new Valuator<Tree<State>, Double>(grammar, lexicon, true);
		mvaluator = new ThreadPool(valuator, opts.nthreadeval);
				
		if (opts.loadGrammar && opts.inGrammar != null) {
			logger.trace("--->Loading grammars from \'" + opts.datadir + opts.inGrammar + "\'...\n");
			GrammarFile gfile = (GrammarFile) GrammarFile.load(opts.datadir + opts.inGrammar);
			grammar = gfile.getGrammar();
			lexicon = gfile.getLexicon();
			lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
			lexicon.labelTrees(testTrees); // no need to label the data if we load it from the object file
			lexicon.labelTrees(devTrees); // no need to ... if ...
			Optimizer.config(random, opts.maxsample, opts.batchsize); // FIXME no errors, just alert you...
			valuator = new Valuator<Tree<State>, Double>(grammar, lexicon, true);			
			mvaluator = new ThreadPool(valuator, opts.nthreadeval);
			
			for (Tree<State> tree : trainTrees) {
				if (tree.getYield().size() == 6) {
					globalTree = tree.shallowClone();
					break;
				} // a global tree
			}
		} else {
			Optimizer goptimizer = new ParallelOptimizer(LVeGLearner.random, opts.maxsample, 
					opts.batchsize, opts.ntheadgrad, opts.parallelgrad, opts.parallelmode, opts.pverbose);
			Optimizer loptimizer = new ParallelOptimizer(LVeGLearner.random, opts.maxsample, 
					opts.batchsize, opts.ntheadgrad, opts.parallelgrad, opts.parallelmode, opts.pverbose);
			grammar.setOptimizer(goptimizer);
			lexicon.setOptimizer(loptimizer);
			
			for (Tree<State> tree : trainTrees) {
				lexicon.tallyStateTree(tree);
				grammar.tallyStateTree(tree);
				if (tree.getYield().size() == 6) {
					globalTree = tree.shallowClone();
				} // a global tree
			}
			logger.trace("\n--->Going through the training set is over...");
			grammar.postInitialize();
			
//			lexicon.postInitialize();
			lexicon.postInitialize(trainTrees);
			
			logger.trace("post-initializing is over.\n");
			lexicon.labelTrees(trainTrees);
			lexicon.labelTrees(testTrees);
			lexicon.labelTrees(devTrees);
		}
		/*
		// initial likelihood of the training set
		logger.trace("\n-------ll of the training data initially is... ");
		long beginTime = System.currentTimeMillis();
		ll = calculateLL(grammar, mvaluator, trainTrees);
		long endTime = System.currentTimeMillis();
		logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");	
		*/
		lvegParser = new LVeGParser<Tree<State>, List<Double>>(grammar, lexicon, true);
		mrParser = new MaxRuleParser<Tree<State>, Tree<String>>(grammar, lexicon, true);
		trainer = new ThreadPool(lvegParser, opts.nthreadbatch);
		
		MethodUtil.saveTree2image(globalTree, filename, null, numberer);
		Tree<String> parseTree = mrParser.parse(globalTree);
		MethodUtil.saveTree2image(null, treeFile + "_ini", parseTree, numberer);
		
		logger.info("\n---SGD CONFIG---\n[parallel: batch-" + opts.parallelbatch + ", grad-" + opts.parallelgrad +"] " + Params.toString(false) + "\n");
		
		if (opts.parallelbatch) {
			parallelInBatch(numberer, ll);
		} else {
			serialInBatch(numberer, ll);
		}
	}
	
	
	public static void parallelInBatch(Numberer numberer, double prell) throws Exception {
		List<Double> scoresOfST = new ArrayList<Double>(2);
		List<Double> trllist = new ArrayList<Double>();
		List<Double> dellist = new ArrayList<Double>();
		int cnt = 0;
		do {			
			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
			short isample = 0, idx = 0, nfailed = 0;
			long beginTime, endTime, startTime = System.currentTimeMillis();
			long batchstart = System.currentTimeMillis(), batchend;
			for (Tree<State> tree : trainTrees) {
				if (opts.onlyLength > 0) {
					if (tree.getYield().size() > opts.onlyLength) { continue; }
				}
				trainer.execute(tree);
				while (trainer.hasNext()) {
					List<Double> score = (List<Double>) trainer.getNext();
					if (score == null) {
						nfailed++;
					} else {
						logger.trace("\n~~~score: " + MethodUtil.double2str(score, precision, -1, false, true) + "\n");
					}
				}
				
				isample++;
				if (++idx % opts.batchsize == 0) {
					while (!trainer.isDone()) {
						while (trainer.hasNext()) {
							List<Double> score = (List<Double>) trainer.getNext();
							if (score == null) {
								nfailed++;
							} else {
								logger.trace("\n~~~score: " + MethodUtil.double2str(score, precision, -1, false, true) + "\n");
							}
						}
					}
					trainer.reset();
					batchend = System.currentTimeMillis();
					
					// apply gradient descent
					logger.trace("+++Apply gradient descent for the batch " + (isample / opts.batchsize) + "... ");
					beginTime = System.currentTimeMillis();
					
					grammar.applyGradientDescent(scoresOfST);
					lexicon.applyGradientDescent(scoresOfST);
					
					endTime = System.currentTimeMillis();
					logger.trace((endTime - beginTime) / 1000.0 + "... batch time: " + (batchend - batchstart) / 1000.0 + ", nfailed: " + nfailed + "\n");
					idx = 0;
					
					if ((isample % (opts.batchsize * opts.nbatch)) == 0) {
						peep(isample, cnt, numberer, trllist, dellist);
						prell = trllist.get(trllist.size() - 1);
					}
					nfailed = 0;
					batchstart = System.currentTimeMillis();
				}
			}
			ends(idx, isample, cnt, startTime, scoresOfST, numberer, trllist, dellist);
			
		// relative error could be negative
		} while(++cnt < opts.nepoch);
		
		finals(trllist, dellist);
	}
	
	public static void serialInBatch(Numberer numberer, double prell) throws Exception {
		List<Double> scoresOfST = new ArrayList<Double>(2);
		List<Double> trllist = new ArrayList<Double>();
		List<Double> dellist = new ArrayList<Double>();
		int cnt = 0;
		do {			
			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
			short isample = 0, idx = 0;
			long beginTime, endTime, startTime = System.currentTimeMillis();
			long batchstart = System.currentTimeMillis(), batchend;
			for (Tree<State> tree : trainTrees) {
				if (opts.onlyLength > 0) {
					if (tree.getYield().size() > opts.onlyLength) { continue; }
				}
				
				logger.trace("---Sample " + isample + "...\t");
				beginTime = System.currentTimeMillis();
				
				double scoreT = lvegParser.evalRuleCountWithTree(tree, (short) 0);
				double scoreS = lvegParser.evalRuleCount(tree, (short) 0);
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\t");
				
				scoresOfST.add(scoreT);
				scoresOfST.add(scoreS);
				
				logger.trace("scores: " + MethodUtil.double2str(scoresOfST, precision, -1, false, true) + "\teval gradients... ");
				beginTime = System.currentTimeMillis();
				
				grammar.evalGradients(scoresOfST);
				lexicon.evalGradients(scoresOfST);
				scoresOfST.clear();
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\n");
				
				isample++;
				if (++idx % opts.batchsize == 0) {
					batchend = System.currentTimeMillis();
					
					// apply gradient descent
					logger.trace("+++Apply gradient descent for the batch " + (isample / opts.batchsize) + "... ");
					beginTime = System.currentTimeMillis();
					
					grammar.applyGradientDescent(scoresOfST);
					lexicon.applyGradientDescent(scoresOfST);
					
					endTime = System.currentTimeMillis();
					logger.trace((endTime - beginTime) / 1000.0 + "... batch time: " + (batchend - batchstart) / 1000.0 + "\n");
					idx = 0;
					
					if ((isample % (opts.batchsize * opts.nbatch)) == 0) {
						peep(isample, cnt, numberer, trllist, dellist);
						prell = trllist.get(trllist.size() - 1);
					}
					batchstart = System.currentTimeMillis();
				}
			}
			ends(idx, isample, cnt, startTime, scoresOfST, numberer, trllist, dellist);
			
		// relative error could be negative
		} while(++cnt < opts.nepoch);
		
		finals(trllist, dellist);
	}
	
	
	public static void finals(List<Double> trllist, List<Double> dellist) {
		long beginTime, endTime;
		logger.trace("Convergence Path [train]: " + trllist + "\n");
		logger.trace("Convergence Path [ dev ]: " + dellist + "\n");
		
		logger.trace("\n----------training is over----------\n");
		
		logger.info("\n-------saving grammar file...");
		GrammarFile gfile = new GrammarFile(grammar, lexicon);
		String filename = opts.datadir + opts.outGrammar + "_final.gr";
		if (gfile.save(filename)) {
			logger.info("to \'" + filename + "\' successfully.\n");
		} else {
			logger.info("to \'" + filename + "\' unsuccessfully.\n");
		}
		
		logger.trace("------->evaluating on the training dataset...");
		beginTime = System.currentTimeMillis();
		double ll = calculateLL(opts, mvaluator, trainTrees, true);
		endTime = System.currentTimeMillis();
		logger.trace("ll is "+ ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		trainTrees.reset();
		
		logger.trace("------->evaluating on the valuation dataset...");
		beginTime = System.currentTimeMillis();
		ll = calculateLL(opts, mvaluator, devTrees, false);
		endTime = System.currentTimeMillis();
		logger.trace("ll is "+ ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		devTrees.reset();
		
		logger.trace("------->evaluating on the test dataset...");
		beginTime = System.currentTimeMillis();
		ll = calculateLL(opts, mvaluator, testTrees, false);
		endTime = System.currentTimeMillis();
		logger.trace("ll is "+ ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		testTrees.reset();
	}
	
	
	public static void ends(int idx, int isample, int cnt, long startTime, List<Double> scoresOfST, 
			Numberer numberer, List<Double> trllist, List<Double> dellist) throws Exception {
		// if not a multiple of batchsize
		long beginTime, endTime;
		if (idx != 0) {
			logger.trace("+++Apply gradient descent for the last batch " + (isample / opts.batchsize) + "... ");
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
		peep(isample, cnt, numberer, trllist, dellist);
		logger.trace("\n----------          log-likelihood in epoch          ----------\n");
		
		// we shall clear the inside and outside score in each state 
		// of the parse tree after the training on a sample 
		trainTrees.reset();
		trainTrees.shuffle(random);
		
		logger.trace("\n----------           epoch " + cnt + " ends          ----------\n");
	}
	
	
	public static void peep(int isample, int cnt, Numberer numberer, List<Double> trllist, List<Double> dellist) throws Exception {
		long beginTime, endTime;
		double trll = 0, dell = 0;
		// save the intermediate grammars
		if (opts.saveGrammar && opts.outGrammar != null) {
			logger.info("\n-------saving grammar file...");
			GrammarFile gfile = new GrammarFile(grammar, lexicon);
			String filename = opts.datadir + opts.outGrammar + "_" + (isample / opts.batchsize) + "_" + cnt + ".gr";
			if (gfile.save(filename)) {
				logger.info("to \'" + filename + "\' successfully.");
			} else {
				logger.info("to \'" + filename + "\' unsuccessfully.");
			}
		}
		// likelihood of the training set
		if (opts.evalontrain) {
			logger.trace("\n-------ll of the training data after " + (isample / opts.batchsize) + " batches in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			trll = calculateLL(opts, mvaluator, trainTrees, true);
			endTime = System.currentTimeMillis();
			logger.trace("------->" + trll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			trainTrees.reset();
		}
		
		if (opts.evalondev) {
			logger.trace("\n-------ll of the dev data after " + (isample / opts.batchsize) + " batches in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			dell = calculateLL(opts, mvaluator, devTrees, false);
			endTime = System.currentTimeMillis();
			logger.trace("------->" + dell + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			devTrees.reset();
		}
		// visualize the parse tree
		Tree<String> parseTree = mrParser.parse(globalTree);
		String filename = treeFile + "_" + cnt + "_" + (isample / (opts.batchsize * opts.nbatch));
		MethodUtil.saveTree2image(null, filename, parseTree, numberer);
		// store the log score
		trllist.add(trll);
		dellist.add(dell);
	}
	
	
	/**
	 * We have to evaluate the grammar on only a fraction of training data because the evaluation is quite time-consumed. But it is the 
	 * evaluation on the whole validation dataset or the whole test dataset that can tells whether your trained model is good or bad.
	 */
	public static double calculateLL(Options opts, ThreadPool valuator, StateTreeList stateTreeList, boolean istrain) {
		int nUnparsable = 0, cnt = 0;
		double ll = 0, sumll = 0;
		for (Tree<State> tree : stateTreeList) {
			if (opts.onlyLength > 0) {
				if (tree.getYield().size() > opts.onlyLength) { continue; }
			}
			if (opts.firstk > 0) {
				if (++cnt > opts.firstk) { break; } // DEBUG
			}
			if (istrain && opts.evalfraction > 0) {
				if (random.nextDouble() > opts.evalfraction) { continue; }
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
		cnt = 0;
		while (!valuator.isDone()) {
			while (valuator.hasNext()) {
				ll = (double) valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll)) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
		}
		valuator.reset();
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + " training samples]\n");
		return sumll;
	}
	
}