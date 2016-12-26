package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.lveg.ObjectFileManager.GrammarFile;
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

	public static void main(String[] args) throws Exception {
		String fparams = "param.ini";
		try {
			args = readFile(fparams, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		Options opts = (Options) optionParser.parse(args, true);
		// configurations
		initialize(opts); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = loadData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(trees, wrapper, opts);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeG learner] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	
	private static void train(Map<String, StateTreeList> trees, Numberer wrapper, Options opts) throws Exception {
		StateTreeList trainTrees = trees.get(ID_TRAIN);
		StateTreeList testTrees = trees.get(ID_TEST);
		StateTreeList devTrees = trees.get(ID_DEV);
		
		double ll = Double.NEGATIVE_INFINITY;
		Tree<State> globalTree = null;
		String filename = opts.imagePrefix + "_gd";
		String treeFile = opts.imagePrefix + "_tr";
		
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		LVeGGrammar grammar = new LVeGGrammar(null, numberer, -1);
		LVeGLexicon lexicon = new SimpleLVeGLexicon();
		Valuator<?, ?> valuator = new Valuator<Tree<State>, Double>(grammar, lexicon, true);
		ThreadPool mvaluator = new ThreadPool(valuator, opts.nthreadeval);
				
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
					globalTree = tree.copy();
					break;
				} // a global tree
			}
		} else {
			Optimizer goptimizer = new ParallelOptimizer(LVeGLearner.random, opts.maxsample, 
					opts.batchsize, opts.ntheadgrad, opts.parallelgrad, opts.parallelmode, true/*opts.pverbose*/);
			Optimizer loptimizer = new ParallelOptimizer(LVeGLearner.random, opts.maxsample, 
					opts.batchsize, opts.ntheadgrad, opts.parallelgrad, opts.parallelmode, true/*opts.pverbose*/);
			grammar.setOptimizer(goptimizer);
			lexicon.setOptimizer(loptimizer);
			
			for (Tree<State> tree : trainTrees) {
				lexicon.tallyStateTree(tree);
				grammar.tallyStateTree(tree);
				if (tree.getYield().size() == 6) {
					globalTree = tree.copy();
				} // a global tree
			}
			logger.trace("\n--->Going through the training set is over...");
			grammar.postInitialize(0.0);
			lexicon.postInitialize(trainTrees, numberer.size());
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
		LVeGParser<?, ?> lvegParser = new LVeGParser<Tree<State>, List<Double>>(grammar, lexicon, true);
		MaxRuleParser<?, ?> mrParser = new MaxRuleParser<Tree<State>, Tree<String>>(grammar, lexicon, true);
		ThreadPool trainer = new ThreadPool(lvegParser, opts.nthreadbatch);
		
		MethodUtil.saveTree2image(globalTree, filename, null, numberer);
		Tree<String> parseTree = mrParser.parse(globalTree);
		MethodUtil.saveTree2image(null, treeFile + "_ini", parseTree, numberer);
		
		logger.info("\n---SGD CONFIG---\n[parallel: batch-" + opts.parallelbatch + ", grad-" + opts.parallelgrad +"] " + Params.toString(false) + "\n");
		
		if (opts.parallelbatch) {
			parallelInBatch(opts, grammar, lexicon, mvaluator, trainer, numberer, mrParser, trees, globalTree, treeFile, ll);
		} else {
			serialInBatch(opts, grammar, lexicon, mvaluator, lvegParser, numberer, mrParser, trees, globalTree, treeFile, ll);
		}
		
		/*
		if (relativError < opts.relativerror) {
			System.out.println("Maximum-relative-error reaching.");
		}
		*/
	}
	
	
	public static void parallelInBatch(Options opts, LVeGGrammar grammar, LVeGLexicon lexicon, ThreadPool mvaluator, ThreadPool trainer, Numberer numberer,
			MaxRuleParser<?, ?> mrParser, Map<String, StateTreeList> trees, Tree<State> globalTree, String treeFile, double prell) throws Exception {
		List<Double> scoresOfST = new ArrayList<Double>(2);
		List<Double> likelihood = new ArrayList<Double>();
		StateTreeList trainTrees = trees.get(ID_TRAIN);
		StateTreeList testTrees = trees.get(ID_TEST);
		StateTreeList devTrees = trees.get(ID_DEV);
		int cnt = 0;
		double ll;
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
						if (opts.saveGrammar && opts.outGrammar != null) {
							GrammarFile gfile = new GrammarFile(grammar, lexicon);
							String gfilename = opts.datadir + opts.outGrammar + "_" + (isample / opts.batchsize) + "_" + cnt + ".gr";
							if (gfile.save(gfilename)) {
								logger.info("\n-------save grammar file to \'" + gfilename + "\'");
							} else {
								logger.error("\n-------failed to save grammar file to \'" + gfilename + "\'");
							}
						}
						// likelihood of the training set
						logger.trace("\n-------ll of the training data after " + (isample / opts.batchsize) + " batches in epoch " + cnt + " is... ");
						beginTime = System.currentTimeMillis();
						ll = calculateLL(opts, grammar, mvaluator, trainTrees);
						endTime = System.currentTimeMillis();
						logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
						trainTrees.reset();
						// visualize the parse tree
						Tree<String> parseTree = mrParser.parse(globalTree);
						String filename = treeFile + "_" + cnt + "_" + (isample / (opts.batchsize * opts.nbatch));
						MethodUtil.saveTree2image(null, filename, parseTree, numberer);
						// store the log score
						likelihood.add(ll);
					}
					nfailed = 0;
					batchstart = System.currentTimeMillis();
				}
			}
			
			// if not a multiple of batchsize
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
			
			// likelihood of the training set
			logger.trace("-------ll of the training data in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			ll = calculateLL(opts, grammar, mvaluator, trainTrees);
			endTime = System.currentTimeMillis();
			likelihood.add(ll);
			logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			
			// we shall clear the inside and outside score in each state 
			// of the parse tree after the training on a sample 
			trainTrees.reset();
			trainTrees.shuffle(random);
			
			logger.trace("-------epoch " + cnt + " ends-------\n");
			
		// relative error could be negative
		} while(++cnt <= 8) /*while (cnt > 1 && Math.abs(relativError) < opts.relativerror && droppingiter < opts.droppintiter)*/;
		logger.trace("Convergence Path: " + likelihood + "\n");
	}
	
	public static void serialInBatch(Options opts, LVeGGrammar grammar, LVeGLexicon lexicon, ThreadPool mvaluator, LVeGParser<?, ?> lvegParser, Numberer numberer,
			MaxRuleParser<?, ?> mrParser, Map<String, StateTreeList> trees, Tree<State> globalTree, String treeFile, double prell) throws Exception {
		List<Double> scoresOfST = new ArrayList<Double>(2);
		List<Double> likelihood = new ArrayList<Double>();
		StateTreeList trainTrees = trees.get(ID_TRAIN);
		StateTreeList testTrees = trees.get(ID_TEST);
		StateTreeList devTrees = trees.get(ID_DEV);
		int cnt = 0;
		double ll;
		
		do {			
			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
			short isample = 0, idx = 0;
			long beginTime, endTime, startTime = System.currentTimeMillis();
			long batchstart = System.currentTimeMillis(), batchend;
			for (Tree<State> tree : trainTrees) {
				if (opts.onlyLength > 0) {
					if (tree.getYield().size() > opts.onlyLength) { continue; }
				}
				
//				if (isample < batchsize) { isample++; continue; }
				
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
						if (opts.saveGrammar && opts.outGrammar != null) {
							GrammarFile gfile = new GrammarFile(grammar, lexicon);
							String gfilename = opts.datadir + opts.outGrammar + "_" + (isample / opts.batchsize) + "_" + cnt + ".gr";
							if (gfile.save(gfilename)) {
								logger.info("\n-------save grammar file to \'" + gfilename + "\'");
							} else {
								logger.error("\n-------failed to save grammar file to \'" + gfilename + "\'");
							}
						}
						// likelihood of the training set
						logger.trace("\n-------ll of the training data after " + (isample / opts.batchsize) + " batches in epoch " + cnt + " is... ");
						beginTime = System.currentTimeMillis();
						ll = calculateLL(opts, grammar, mvaluator, trainTrees);
						endTime = System.currentTimeMillis();
						logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
						trainTrees.reset();
						// visualize the parse tree
						Tree<String> parseTree = mrParser.parse(globalTree);
						String filename = treeFile + "_" + cnt + "_" + (isample / (opts.batchsize * opts.nbatch));
						MethodUtil.saveTree2image(null, filename, parseTree, numberer);
						// store the log score
						likelihood.add(ll);
					}
					batchstart = System.currentTimeMillis();
				}
			}
			
			// if not a multiple of batchsize
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
			
			// likelihood of the training set
			logger.trace("-------ll of the training data in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			ll = calculateLL(opts, grammar, mvaluator, trainTrees);
			endTime = System.currentTimeMillis();
			likelihood.add(ll);
			logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			
			// we shall clear the inside and outside score in each state 
			// of the parse tree after the training on a sample 
			trainTrees.reset();
			trainTrees.shuffle(random);
			
			logger.trace("-------epoch " + cnt + " ends-------\n");
			
		// relative error could be negative
		} while(++cnt <= 8) /*while (cnt > 1 && Math.abs(relativError) < opts.relativerror && droppingiter < opts.droppintiter)*/;
		
		logger.trace("Convergence Path: " + likelihood + "\n");
	}
	
	
	public static double calculateLL(Options opts, LVeGGrammar grammar, ThreadPool valuator, StateTreeList stateTreeList) {
		int nUnparsable = 0, cnt = 0;
		double ll = 0, sumll = 0;
		for (Tree<State> tree : stateTreeList) {
			if (opts.onlyLength > 0) {
				if (tree.getYield().size() > opts.onlyLength) { continue; }
			}
			if (++cnt > 200) { break; } // DEBUG
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
