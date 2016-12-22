package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.PCFGLA.Option;
import edu.berkeley.nlp.PCFGLA.OptionParser;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGLearner extends Recorder {
	
	public static short dim        =  5;
	public static short maxrandom  =  1;
	public static short batchsize  = 50;
	public static short ncomponent =  5;
	public static short maxlength  = 30;
	
	public static int randomseed = 0;
	public static int precision  = 3;
	
	public static Random random;
	
	public final static String KEY_TAG_SET = "tags";
	public final static String TOKEN_UNKNOWN = "UNK";
	
	private final static boolean LVG = true;
	private final static String ID_TRAINING = "training";
	private final static String ID_VALIDATION = "validation";
	
	
	public static class Options {
		@Option(name = "-out", required = true, usage = "Output file for grammar (Required)")
		public String outFile;
		
		@Option(name = "-in", usage = "Input file for grammar (Optional: unimplemented)")
		public String inFile;

		@Option(name = "-pathToCorpus", usage = "Path to corpus (Default: null)")
		public String pathToCorpus = null;
		
		@Option(name = "-randomSeed", usage = "Seed for random number generator (Default: 111)")
		public int randomSeed = 111;
		
		@Option(name = "-treebank", usage = "Language: WSJ, CHINESE, SINGLEFILE (Default: SINGLEFILE)")
		public TreeBankType treebank = TreeBankType.SINGLEFILE;
//		public TreeBankType treebank = TreeBankType.WSJ;
		
		
		@Option(name = "-skipSection", usage = "Skip a particular section of the WSJ training corpus (Needed for training Mark Johnsons reranker (Default: -1)")
		public int skipSection = -1;
		
		@Option(name = "-skipBilingual", usage = "Skip the bilingual portion of the Chinese treebank (Needed for training the bilingual reranker (Default: false)")
		public boolean skipBilingual = false;
		
		@Option(name = "-keepFunctionLabel", usage = "Retain predicted function labels. Model must have been trained with function labels (Default: false)")
		public boolean keepFunctionLabel = false;
		
		@Option(name = "-simpleLexicon", usage = "Use the simple generative lexicon (Default: true)")
		public boolean simpleLexicon = true;
		
		@Option(name = "-featurizedLexicon", usage = "Use the featurized lexicon (default: false)")
		public boolean featurizedLexicon = false;
		
		@Option(name = "-trainingFraction", usage = "The fraction of the training corpus to keep (Default: 1.0)")
		public double trainingFraction = 1.0;
		
		@Option(name = "-trainOnDevSet", usage = "Include the development set into the training set (Default: false)")
		public boolean trainOnDevSet = false;
		
		@Option(name = "-maxSentenceLength", usage = "Maximum sentence length (Default: <= 10000)")
		public int maxSentenceLength = 10000;
		
		@Option(name = "-binarization", usage = "Left/Right binarization (Default: RIGHT)")
		public Binarization binarization = Binarization.RIGHT;
		
		@Option(name = "-horizontalMarkovization", usage = "Horizontal markovization (Default: 0)")
		public int horizontalMarkovization = 0;
		
		@Option(name = "-verticalMarkovization", usage = "Vertical markovization (Default: 1")
		public int verticalMarkovization = 1;
		
		@Option(name = "-lowercase", usage = "Lowercase All Words in the Treebank (Default: false)")
		public boolean lowercase = false;
		
		@Option(name = "-rareThreshold", usage = "Rare word threshold (Default: 20)")
		public int rareThreshold = 20;
		
		@Option(name = "-rarerThreshold", usage = "Rarer word threshold (Default: 10)")
		public int rarerThreshold = 10;
		
		@Option(name = "-filterThreshold", usage = "Filter rules with probability below this threshold (Default: 1.0e-30)")
		public double filterThreshold = 1.0e-30;
		
		@Option(name = "-verbose", usage = "Verbose/Quiet (Default: false)")
		public boolean verbose = false;
		
		@Option(name = "-ncomponent", usage = "Number of gaussian components (Default: 10)")
		public short ncomponent = 2;
		
		@Option(name = "-batchsize", usage = "Sample times in one gradient update step (Default: 500)")
		public short batchsize = 3;
		
		@Option(name = "-maxramdom", usage = "Maximum random value (Default: 10)")
		public short maxrandom = 1;
		
		@Option(name = "-maxiter", usage = "Maximum iteration time (Default: 1000)")
		public int maxiter = 1000;
		
		@Option(name = "-droppingiter", usage = "Maximum consecutive dropping time (Default: 6)")
		public int droppintiter = 6;
		
		@Option(name = "-relativerror", usage = "Desirable maximum relative difference between the neighboring iterations (1e-6)")
		public double relativerror = 1e-6;
		
		@Option(name = "-lr", usage = "Learning rate (Default: 1.0)")
		public double lr = 1.0;
		
		@Option(name = "-dim", usage = "Dimension of the latent vector (Default: 5)")
		public short dim = 2;
		
		@Option(name = "-logType", usage = "Console (false) or file (true) (Default: true)")
		public boolean logType = false;
		
		@Option(name = "-logFile", usage = "Log file name (Default: log/log)")
		public String logFile = "log/log";
	}
	
	
	public static class Params {
		public static double lr = 0.02;
		public static boolean reg = true;
		public static boolean clip = true;
		public static double absmax = 5.0;
		public static double wdecay = 0.02;
		public static double momentum = 0.9;
		// L1 (true) or L2 (false)
		public static boolean l1 = true;
		
		public static String toString(boolean placeholder) {
			return "Params [lr = " + lr + ", reg = " + ", clip = " + clip + ", absmax = " + 
					absmax + ", wdecay = " + wdecay + ", momentum = " + momentum + ", l1 = " + l1 + "]";
		}
	}
	

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		OptionParser optionParser = new OptionParser(Options.class);
		Options opts = (Options) optionParser.parse(args, true);
		System.out.println("Calling with " + optionParser.getPassedInOptions());
		
		initializeLearner(opts); // 
		
		Map<String, List<Tree<String>>> stringTrees = loadData(opts);
		
		// Numberer numbererTag = getNumbererTag(data, opts);
		Numberer numbererTag = Numberer.getGlobalNumberer(KEY_TAG_SET);
		
		Map<String, StateTreeList> stateTrees = stringTreeToStateTree(stringTrees, numbererTag, opts);
		
		long startTime = System.currentTimeMillis();
		train(stateTrees, numbererTag, opts);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	
	@SuppressWarnings("unused")
	private static void train(Map<String, StateTreeList> stateTrees, Numberer numbererTag, Options opts) throws Exception {
		StateTreeList trainTrees = new StateTreeList(stateTrees.get(ID_TRAINING));
		StateTreeList validationTrees = new StateTreeList(stateTrees.get(ID_VALIDATION));
		
/*		// DEBUG
		MethodUtil.debugShuffle(trainTrees);
		System.exit(1);
*/		
/*		// DEBUG
		String imageName = "log/atree_unary_rule_chain_";
		MethodUtil.lenUnaryRuleChain(trainTrees, (short) 2, imageName);
		System.exit(0);
*/		
		Tree<State> globalTree = null;
		String oldFilename = "log/groundtruth";
		String filename, newFilename = "log/maxrulerets_chart";
		
		batchsize = 6;
		short maxsample = 3;
		short ntheadgrad = 3, nthreadeval = 6, nthreadbatch = 3; // eval gradients, calculate ll, parallelize in the batch
		boolean parallelgrad = true, parallelbatch = true;
		boolean useOldGram = false, saveNewGram = false;
		int cnt = 0, droppingiter = 0, maxLength = 7, nbatch = 1;
		
		LVeGGrammar grammar = new LVeGGrammar(null, -1);
		LVeGLexicon lexicon = new SimpleLVeGLexicon();
		Valuator<?> valuator = new Valuator<Double>(grammar, lexicon, true);
		MultiThreadedParser mvaluator = new MultiThreadedParser(valuator, nthreadeval);
				
		if (useOldGram && opts.inFile != null) {
			GrammarFile gfile = GrammarFile.load(opts.inFile);
			grammar = gfile.getGrammar();
			lexicon = gfile.getLexicon();
			lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
			Optimizer.config(random, maxsample, batchsize); // FIXME no errors, just alert you...
			valuator = new Valuator<Double>(grammar, lexicon, true);			
			mvaluator = new MultiThreadedParser(valuator, nthreadeval);
			
			for (Tree<State> tree : trainTrees) {
				if (tree.getYield().size() == 6) {
					globalTree = tree.copy();
					break;
				} // a global tree
			}
			
			// likelihood of the training set
			logger.trace("-------ll of the training data initially is... ");
			long beginTime = System.currentTimeMillis();
			double ll = calculateLL(grammar, mvaluator, trainTrees, maxLength);
			long endTime = System.currentTimeMillis();
			logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");	
		} else {
			Optimizer goptimizer = new ParallelOptimizer(LVeGLearner.random, maxsample, batchsize, ntheadgrad);
			Optimizer loptimizer = new ParallelOptimizer(LVeGLearner.random, maxsample, batchsize, ntheadgrad);
			grammar.setOptimizer(goptimizer);
			lexicon.setOptimizer(loptimizer);
			
			for (Tree<State> tree : trainTrees) {
				lexicon.tallyStateTree(tree);
				grammar.tallyStateTree(tree);
				if (tree.getYield().size() == 6) {
					globalTree = tree.copy();
				} // a global tree
			}
			logger.trace("Going through the training set is over.\n");
			
			grammar.postInitialize(0.0);
			lexicon.postInitialize(trainTrees, numbererTag.size());
			logger.trace("Post-initializing is over.\n");
		}
		LVeGParser<?> lvegParser = new LVeGParser<List<Double>>(grammar, lexicon, true);
		MaxRuleParser<?> mrParser = new MaxRuleParser<Tree<String>>(grammar, lexicon, true);
		
		MultiThreadedParser trainer = new MultiThreadedParser(lvegParser, nthreadbatch);
		
		MethodUtil.saveTree2image(globalTree, oldFilename, null);
		Tree<String> parseTree = mrParser.parse(globalTree);
		MethodUtil.saveTree2image(null, newFilename + "_ini", parseTree);
		
//		logger.debug(grammar);
//		logger.debug(lexicon);

		/*// check if there is any circle in the unary grammar rules
		// TODO move this self-checking procedure to the class Grammar
		logger.debug("---Circle Detection.\n");
		if (MethodUtil.checkUnaryRuleCircle(grammar, lexicon, true)) {
			logger.error("Circle (WithC) was found in the unary grammar rules.");
			return;
		}
		*/
		
		double prell = 0.0;
		double relativError = 0, ll, maxll = prell;
		List<Double> scoresOfST = new ArrayList<Double>(2);
		List<Double> likelihood = new ArrayList<Double>();
		
		logger.info("\n---SGD CONFIG---\n[parallel: " + parallelgrad + "] " + Params.toString(false) + "\n");
		if (parallelgrad) { newFilename = "log/maxrulerets_parallel"; }
		
		if (parallelbatch) {
		
		do {			
			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
			short isample = 0, idx = 0, nfailed = 0;
			long beginTime, endTime, startTime = System.currentTimeMillis();
			long batchstart = System.currentTimeMillis(), batchend;
			for (Tree<State> tree : trainTrees) {

				if (tree.getYield().size() > maxLength) { continue; }
				
				trainer.parse(tree);
				
				while (trainer.hasNext()) {
					List<Double> score = (List<Double>) trainer.getNext();
					if (score == null) {
						nfailed++;
					} else {
						logger.trace("\n~~~score: " + MethodUtil.double2str(score, precision, -1, false, true) + "\n");
					}
				}
				
				
//				logger.trace("---Sample " + isample + "...\t");
//				beginTime = System.currentTimeMillis();
//				
//				double scoreT = lvegParser.evalRuleCountWithTree(tree, (short) 0);
//				double scoreS = lvegParser.evalRuleCount(tree, (short) 0);
//				
//				endTime = System.currentTimeMillis();
//				logger.trace( + (endTime - beginTime) / 1000.0 + "\t");
//				
//				scoresOfST.add(scoreT);
//				scoresOfST.add(scoreS);
//				
//				logger.trace("scores: " + MethodUtil.double2str(scoresOfST, precision, -1, false, true) + "\teval gradients... ");
//				beginTime = System.currentTimeMillis();
//				
//				grammar.evalGradients(scoresOfST, parallelgrad);
//				lexicon.evalGradients(scoresOfST, parallelgrad);
//				scoresOfST.clear();
//				
//				endTime = System.currentTimeMillis();
//				logger.trace( + (endTime - beginTime) / 1000.0 + "\n");
				
				
				isample++;
				if (++idx % batchsize == 0) {
					
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
					logger.trace("+++Apply gradient descent for the batch " + (isample / batchsize) + "... ");
					beginTime = System.currentTimeMillis();
					
					grammar.applyGradientDescent(scoresOfST);
					lexicon.applyGradientDescent(scoresOfST);
					
					endTime = System.currentTimeMillis();
					logger.trace((endTime - beginTime) / 1000.0 + "... batch time: " + (batchend - batchstart) / 1000.0 + ", nfailed: " + nfailed + "\n");
					idx = 0;
					
					if ((isample % (batchsize * nbatch)) == 0) {
						// likelihood of the training set
						logger.trace("\n-------ll of the training data after " + (isample / batchsize) + " batches in epoch " + cnt + " is... ");
						beginTime = System.currentTimeMillis();
						ll = calculateLL(grammar, mvaluator, trainTrees, maxLength);
						endTime = System.currentTimeMillis();
						logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
						trainTrees.reset();
						// visualize the parse tree
						parseTree = mrParser.parse(globalTree);
						filename = newFilename + "_" + cnt + "_" + (isample / (batchsize * nbatch));
						MethodUtil.saveTree2image(null, filename, parseTree);
						// store the log score
						likelihood.add(ll);
					}
					
					nfailed = 0;
					batchstart = System.currentTimeMillis();
				}
			}
			
			// if not a multiple of batchsize
			logger.trace("+++Apply gradient descent for the last batch " + (isample / batchsize) + "... ");
			beginTime = System.currentTimeMillis();
			grammar.applyGradientDescent(scoresOfST);
			lexicon.applyGradientDescent(scoresOfST);
			endTime = System.currentTimeMillis();
			logger.trace((endTime - beginTime) / 1000.0 + "\n");
			scoresOfST.clear();
			
			// a coarse summary
			endTime = System.currentTimeMillis();
			logger.trace("===Average time each sample cost is " + (endTime - startTime) / (1000.0 * isample) + "\n");
			
			// likelihood of the training set
			logger.trace("-------ll of the training data in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			ll = calculateLL(grammar, mvaluator, trainTrees, maxLength);
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
		
		} else {
		
		do {			
			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
			short isample = 0, idx = 0;
			long beginTime, endTime, startTime = System.currentTimeMillis();
			long batchstart = System.currentTimeMillis(), batchend;
			for (Tree<State> tree : trainTrees) {
				if (tree.getYield().size() > maxLength) { continue; }
				
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
				
				grammar.evalGradients(scoresOfST, parallelgrad);
				lexicon.evalGradients(scoresOfST, parallelgrad);
				scoresOfST.clear();
				
				endTime = System.currentTimeMillis();
				logger.trace( + (endTime - beginTime) / 1000.0 + "\n");
				
				isample++;
				if (++idx % batchsize == 0) {
					batchend = System.currentTimeMillis();
					
					// apply gradient descent
					logger.trace("+++Apply gradient descent for the batch " + (isample / batchsize) + "... ");
					beginTime = System.currentTimeMillis();
					
					grammar.applyGradientDescent(scoresOfST);
					lexicon.applyGradientDescent(scoresOfST);
					
					endTime = System.currentTimeMillis();
					logger.trace((endTime - beginTime) / 1000.0 + "... batch time: " + (batchend - batchstart) / 1000.0 + "\n");
					idx = 0;
					
					if ((isample % (batchsize * nbatch)) == 0) {
						
						
						if (saveNewGram && opts.outFile != null) {
							GrammarFile gfile = new GrammarFile(grammar, lexicon);
							String gfilename = opts.outFile + "_" + (isample / batchsize) + "_" + cnt + ".gr";
							if (gfile.save(gfilename)) {
								logger.info("\n-------save grammar file to \'" + gfile + "\'");
							} else {
								logger.info("\n-------failed to save grammar file to \'" + gfile + "\'");
							}
						}
						
						
						// likelihood of the training set
						logger.trace("\n-------ll of the training data after " + (isample / batchsize) + " batches in epoch " + cnt + " is... ");
						beginTime = System.currentTimeMillis();
						ll = calculateLL(grammar, mvaluator, trainTrees, maxLength);
						endTime = System.currentTimeMillis();
						logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
						trainTrees.reset();
						// visualize the parse tree
						parseTree = mrParser.parse(globalTree);
						filename = newFilename + "_" + cnt + "_" + (isample / (batchsize * nbatch));
						MethodUtil.saveTree2image(null, filename, parseTree);
						// store the log score
						likelihood.add(ll);
					}
		
					batchstart = System.currentTimeMillis();
				}
			}
			
			// if not a multiple of batchsize
			logger.trace("+++Apply gradient descent for the last batch " + (isample / batchsize) + "... ");
			beginTime = System.currentTimeMillis();
			grammar.applyGradientDescent(scoresOfST);
			lexicon.applyGradientDescent(scoresOfST);
			endTime = System.currentTimeMillis();
			logger.trace((endTime - beginTime) / 1000.0 + "\n");
			scoresOfST.clear();
			
			// a coarse summary
			endTime = System.currentTimeMillis();
			logger.trace("===Average time each sample cost is " + (endTime - startTime) / (1000.0 * isample) + "\n");
			
			// likelihood of the training set
			logger.trace("-------ll of the training data in epoch " + cnt + " is... ");
			beginTime = System.currentTimeMillis();
			ll = calculateLL(grammar, mvaluator, trainTrees, maxLength);
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
		
		
		}
		
		
//		do {			
//			logger.trace("\n\n-------epoch " + cnt + " begins-------\n\n");
//			short isample = 0, idx = 0;
//			long beginTime, endTime, startTime = System.currentTimeMillis();
//			for (Tree<State> tree : trainTrees) {
//				
//				if (tree.getYield().size() > maxLength) { continue; }
//				
//				logger.trace("---Sample " + isample + "\tis being processed... ");
//				beginTime = System.currentTimeMillis();
//				
//				double scoreT = lvegParser.evalRuleCountWithTree(tree, (short) idx);
//				double scoreS = lvegParser.evalRuleCount(tree, (short) idx);
//				
//				endTime = System.currentTimeMillis();
//				logger.trace( + (endTime - beginTime) / 1000.0 + "\n");
//				
//				scoresOfST.add(scoreT);
//				scoresOfST.add(scoreS);
//				
//				isample++;
//				if (++idx % batchsize == 0) {
//					// apply gradient descent
//					logger.trace("+++Apply gradient descent for the batch " + (isample / batchsize) + "... ");
//					beginTime = System.currentTimeMillis();
//					
//					grammar.applyGradientDescent(scoresOfST);
//					lexicon.applyGradientDescent(scoresOfST);
//					
//					endTime = System.currentTimeMillis();
//					logger.trace((endTime - beginTime) / 1000.0 + "\n");
//					scoresOfST.clear();
//					idx = 0;
//					
//					if ((isample % (batchsize * nbatch)) == 0) {
//						// likelihood of the training set
//						logger.trace("\n-------ll of the training data after " + (isample / batchsize) + " batches in epoch " + cnt + " is... ");
//						ll = calculateLL(grammar, lexicon, trainTrees, maxLength);
//						logger.trace("------->" + ll + "\n");
//						trainTrees.reset();
//						// visualize the parse tree
//						parseTree = mrParser.parse(globalTree);
//						filename = newFilename + "_" + cnt + "_" + (isample / (batchsize * nbatch));
//						MethodUtil.saveTree2image(null, filename, parseTree);
//						// store the log score
//						likelihood.add(ll);
//					}
//				}
//			}
//			
//			// if not a multiple of batchsize
//			logger.trace("+++Apply gradient descent for the last batch " + (isample / batchsize) + "... ");
//			beginTime = System.currentTimeMillis();
//			grammar.applyGradientDescent(scoresOfST);
//			lexicon.applyGradientDescent(scoresOfST);
//			endTime = System.currentTimeMillis();
//			logger.trace((endTime - beginTime) / 1000.0 + "\n");
//			scoresOfST.clear();
//			
//			// a coarse summary
//			endTime = System.currentTimeMillis();
//			logger.trace("===Average time each sample cost is " + (endTime - startTime) / (1000.0 * isample) + "\n");
//			
//			// likelihood of the training set
//			logger.trace("-------ll of the training data in epoch " + cnt + " is... ");
//			ll = calculateLL(grammar, lexicon, trainTrees, maxLength);
//			likelihood.add(ll);
//			logger.trace(ll + "\n");
//			
//			// we shall clear the inside and outside score in each state 
//			// of the parse tree after the training on a sample 
//			trainTrees.reset();
//			trainTrees.shuffle(random);
//			
//			// CHECK shuffle the training data, does this work?
//			Collections.shuffle(Arrays.asList(trainTrees.toArray()), random);
//			
//			logger.trace("-------epoch " + cnt + " ends-------\n");
//			
//		// relative error could be negative
//		} while(++cnt <= 8) /*while (cnt > 1 && Math.abs(relativError) < opts.relativerror && droppingiter < opts.droppintiter)*/;
		
		
		logger.trace("Convergence Path: " + likelihood + "\n");
		/*
		if (relativError < opts.relativerror) {
			System.out.println("Maximum-relative-error reaching.");
		}
		*/
	}
	
	
	public static double calculateLL(LVeGGrammar grammar, MultiThreadedParser valuator, StateTreeList stateTreeList, int maxLength) {
		int nUnparsable = 0, cnt = 0;
		double ll = 0, sumll = 0;
		for (Tree<State> tree : stateTreeList) {
			if (tree.getYield().size() > maxLength) { continue; }
			if (++cnt > 200) { break; } // DEBUG
			valuator.parse(tree);
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
	
	
/*	
	public static double calculateLL(LVeGGrammar grammar, LVeGLexicon lexicon, LVeGParser parser, StateTreeList stateTreeList, int maxLength) {
		int nUnparsable = 0, cnt = 0;
		double ll = 0, sumll = 0;
		MultiThreadedValuator valuator = new MultiThreadedValuator(parser, 6);
		for (Tree<State> tree : stateTreeList) {
			if (tree.getYield().size() > maxLength) { continue; }
			if (++cnt > 200) { break; } // DEBUG
			valuator.parse(tree);
			while (valuator.hasNext()) {
				ll = valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll)) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
		}
		valuator.shutdown();
		cnt = 0;
		while (!valuator.isDone()) {
			while (valuator.hasNext()) {
				ll = valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll)) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
		}
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + " training samples]\n");
		return sumll;
	}
*/	
	
	
//	public static double calculateLL(LVeGGrammar grammar, LVeGLexicon lexicon, Valuator parser, StateTreeList stateTreeList, int maxLength) {
//		int nUnparsable = 0, cnt = 0;
//		double ll = 0, sumll = 0;
//		for (Tree<State> tree : stateTreeList) {
//			if (tree.getYield().size() > maxLength) { continue; }
//			if (++cnt > 200) { break; }
//			ll = parser.probability(tree);
//			if (Double.isInfinite(ll) || Double.isNaN(ll)) {
//				nUnparsable++;
//			} else {
//				// logger.trace("\n===> sub-ll: " + ll + ", id: " + (cnt - 1) + "\n"); // DEBUG
//				sumll += ll;
//			}
//		}
//		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + stateTreeList.size() + " training samples]\n");
//		return sumll;
//	}
	
	
	private static void initializeLearner(Options opts) {
		if (opts.outFile == null) {
			throw new IllegalArgumentException("Output file is required.");
		} else {
			if (opts.logType) {
				logger = logUtil.getFileLogger(opts.logFile);
			} else {
				logger = logUtil.getConsoleLogger();
			}
			System.out.println("Grammar file will be saved to " + opts.outFile + ".");
		}
		dim = opts.dim;
		maxrandom = opts.maxrandom;
		batchsize = opts.batchsize;
		ncomponent = opts.ncomponent;
		random = new Random(opts.randomSeed);
		System.out.println("Random number generator seeded at " + opts.randomSeed + ".");
	}
	
	
	private static Map<String, List<Tree<String>>> loadData(Options opts) {
		System.out.println("Loading trees from " + opts.pathToCorpus + " and using languange " + opts.treebank);
		
		boolean onlyTest = false, manualAnnotation = false;
		Map<String, List<Tree<String>>> datasets = new HashMap<String, List<Tree<String>>>();
		
		Corpus corpus = new Corpus(
				opts.pathToCorpus, opts.treebank, opts.trainingFraction,
				onlyTest,
				opts.skipSection, opts.skipBilingual, opts.keepFunctionLabel);
		List<Tree<String>> trainTrees = Corpus.binarizeAndFilterTrees(
				corpus.getTrainTrees(), opts.verticalMarkovization, opts.horizontalMarkovization, 
				opts.maxSentenceLength, opts.binarization, manualAnnotation, opts.verbose);
		List<Tree<String>> validationTrees = Corpus.binarizeAndFilterTrees(
				corpus.getValidationTrees(), opts.verticalMarkovization, opts.horizontalMarkovization, 
				opts.maxSentenceLength, opts.binarization, manualAnnotation, opts.verbose);
		
		if (opts.trainOnDevSet) {
			System.out.println("Adding development set to the training data.");
			trainTrees.addAll(validationTrees);
		}
		if (opts.lowercase) {
			System.out.println("Lowercasing the treebank.");
			Corpus.lowercaseWords(trainTrees);
			Corpus.lowercaseWords(validationTrees);
		}
		logger.trace("There are " + trainTrees.size() + " trees in the training set.\n");
		datasets.put(ID_TRAINING, trainTrees);
		datasets.put(ID_VALIDATION, validationTrees);
		
		return datasets;
	}
	
	
	private static Map<String, StateTreeList> stringTreeToStateTree(
			Map<String, List<Tree<String>>> stringTrees, Numberer numbererTag, Options opts) {
		Map<String, StateTreeList> datasets = new HashMap<String, StateTreeList>();
		
		StateTreeList trainStateTrees = new StateTreeList(stringTrees.get(ID_TRAINING), numbererTag);
		StateTreeList validationStateTrees = new StateTreeList(stringTrees.get(ID_VALIDATION), numbererTag);
		
		debugNumbererTag(numbererTag, opts);
		
		datasets.put(ID_TRAINING, trainStateTrees);
		datasets.put(ID_VALIDATION, validationStateTrees);
		
		if (opts.simpleLexicon) {
			System.out.println("Replacing words appearing less than 5 times with their signature.");
			LVeGCorpus.replaceRareWords(trainStateTrees, new SimpleLVeGLexicon(), opts.rareThreshold);
		}
		
		return datasets;
	}
	
	
	@SuppressWarnings("unused")
	private static Numberer getNumbererTag(Map<String, List<Tree<String>>> data, Options opts) {
		/* Initialize tagNumberer via indexing tags */
		Numberer numbererTag = Numberer.getGlobalNumberer(KEY_TAG_SET);
		StateTreeList.initializeNumbererTag(data.get(ID_TRAINING), numbererTag);
		StateTreeList.initializeNumbererTag(data.get(ID_VALIDATION), numbererTag);
		
		debugNumbererTag(numbererTag, opts);
		
		return numbererTag;
	}
	
	
	private static void debugNumbererTag(Numberer numbererTag, Options opts) {
		if (opts.verbose || true) {
			for (int i = 0; i < numbererTag.size(); i++) {
//				logger.trace("Tag " + i + "\t" +  (String) numbererTag.object(i) + "\n"); // DEBUG
			}
		}
		logger.debug("There are " + numbererTag.size() + " observed tags.\n");
	}

}
