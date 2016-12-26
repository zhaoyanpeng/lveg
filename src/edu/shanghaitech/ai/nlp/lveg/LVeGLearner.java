package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
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
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;
import edu.shanghaitech.ai.nlp.lveg.ObjectFileManager.GrammarFile;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGLearner extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1249878080098056557L;
	public final static String KEY_TAG_SET = "tags";
	public final static String TOKEN_UNKNOWN = "UNK";
	private final static String ID_TRAIN = "train";
	private final static String ID_TEST = "test";
	private final static String ID_DEV = "dev";
	
	public static short dim        =  5;
	public static short maxrandom  =  1;
	public static short ncomponent =  5;
	public static short maxlength  = 30;
	
	public static int randomseed = 0;
	public static int precision  = 3;
	
	public static Random random;
	private static Options opts;
	
	public static class Params {
		public static double lr;
		public static boolean l1;
		public static boolean reg;
		public static boolean clip;
		public static double absmax;
		public static double wdecay;
		public static double momentum;
		
		public static void config(Options opts) {
			lr = opts.lr;
			l1 = opts.l1;
			reg = opts.reg;
			clip = opts.clip;
			absmax = opts.absmax;
			wdecay = opts.wdecay;
			momentum = opts.momentum;
		}
		
		public static String toString(boolean placeholder) {
			return "Params [lr = " + lr + ", reg = " + reg + ", clip = " + clip + ", absmax = " + 
					absmax + ", wdecay = " + wdecay + ", momentum = " + momentum + ", l1 = " + l1 + "]";
		}
	}
	
	public static class Options {
		/* corpus section begins */
		@Option(name = "-datadir", required = true, usage = "absolute path pointing to the data directory (default: null)")
		public String datadir = null;
		@Option(name = "-train", required = true, usage = "path to the training data (default: null)")
		public String train = null;
		@Option(name = "-test", usage = "path to the test data (default: null")
		public String test = null;
		@Option(name = "-dev", usage = "path to the development data (default: null)")
		public String dev = null;
		@Option(name = "-inCorpus", usage = "input object file of the training, development, and test data (default: null" )
		public String inCorpus = null;
		@Option(name = "-outCorpus", usage = "output object file of the training, development, and test data (default: null" )
		public String outCorpus = null;
		@Option(name = "-saveCorpus", usage = "save corpus (true) or not (false) (default: false)")
		public boolean saveCorpus = false;
		@Option(name = "-loadCorpus", usage = "load corpus (true) or not (false) (default: false)")
		public boolean loadCorpus = false;
		/* corpus section ends */
		
		/* optimization-parameter section begins*/
		@Option(name = "-lr", usage = "learning rate (default: 1.0)")
		public double lr = 0.02;
		@Option(name = "-reg", usage = "using regularization (true) or not (false) (default: true)")
		public boolean reg = true;
		@Option(name = "-clip", usage = "clip the gradients (true) or not (false) (default: true)")
		public boolean clip = true;
		@Option(name = "-absmax", usage = "threshold for clipping gradients (default: 5.0)")
		public double absmax = 5.0;
		@Option(name = "-wdecay", usage = "weight decay rate (default: 0.02)")
		public double wdecay = 0.02;
		@Option(name = "-momentum", usage = "momentum used in the gradient update (default: 0.9)")
		public double momentum = 0.9;
		@Option(name = "-l1", usage = "using l1 regularization (true) or l2 regularization (false) (default: true)")
		public boolean l1 = true;
		/* optimization-parameter section ends*/
		
		/* grammar-data section begins */
		@Option(name = "-inGrammar", usage = "input object file of the grammar (default: null)")
		public String inGrammar = null;
		@Option(name = "-outGrammar", usage = "output object file of the grammar (default: null)")
		public String outGrammar = null;
		@Option(name = "-saveGrammar", usage = "save grammar (true) or not (false) (default: false)")
		public boolean saveGrammar = false;
		@Option(name = "-loadGrammar", usage = "load grammar (true) or not (false) (default: false)")
		public boolean loadGrammar = false;
		/* grammar-data section ends */
		
		/* parallel configurations section begins */
		@Option(name = "-nthreadgrad", usage = "# of threads for computing gradients (default: half number of the available cores)")
		public short ntheadgrad = 1;
		@Option(name = "-nthreadeval", usage = "# of threads for evaluating the grammar (default: half number of the available cores)")
		public short nthreadeval = 1;
		@Option(name = "-nthreadbatch", usage = "# of threads for minibatch training (default: half number of the available cores)")
		public short nthreadbatch = 1;
		@Option(name = "-parallelgrad", usage = "parallize the gradient computation (true) or not (false) (default: true)")
		public boolean parallelgrad = true;
		@Option(name = "-parallelbatch", usage = "parallize the minibatch training (true) or not (false) (default: false)")
		public boolean parallelbatch = true;
		/* parallel-configurations section ends */
		
		/* training configurations section begins */
		@Option(name = "-ncomponent", usage = "# of gaussian components (default: 2)")
		public short ncomponent = 2;
		@Option(name = "-dim", usage = "dimension of the gaussian (default: 2)")
		public short dim = 2;
		@Option(name = "-maxramdom", usage = "maximum random value in initializing MoGul parameters (Default: 1)")
		public short maxrandom = 1;
		@Option(name = "-batchsize", usage = "# of the samples in a batch (default: 128)")
		public short batchsize = 128;
		@Option(name = "-maxsample", usage = "sampling times when approximating gradients (default: 3)")
		public short maxsample = 3;
		@Option(name = "-evalfraction", usage = "fraction of the training data on which the grammar evaluation is conducted (default: 0.2)")
		public double evalfraction = 0.2;
		@Option(name = "-nepoch", usage = "# of epoches (default: 10)")
		public int nepoch = 10;
		@Option(name = "-relativediff", usage = "maximum relative difference between the neighboring iterations (default: 1e-6)")
		public double relativerror = 1e-6;
		@Option(name = "-onlyLength", usage = "train on only the sentences of length less than or equal to the specific length, no such constraints if it is negative (default: 50)")
		public int onlyLength = 50;
		/* training-configurations section ends */
		
		/* logger configurations section begins */
		@Option(name = "-logType", usage = "console (false) or file (true) (default: true)")
		public boolean logType = false;
		@Option(name = "-logFile", usage = "log file name (default: log/log)")
		public String logFile = "log/log";
		@Option(name = "-nbatch", usage = "# of batchs after which the grammar evaluation is conducted (default: 1)")
		public short nbatch = 1;
		/* logger-configurations section ends */
		
		
		
		/* file prefix section begins */
		@Option(name = "-imagePrefix", usage = "prefix of the image of the parse tree obtained from max rule parser (default: maxrule")
		public String imagePrefix = "log/maxrule";
		/* file prefix section ends */
		
		
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
		
		@Option(name = "-maxiter", usage = "Maximum iteration time (Default: 1000)")
		public int maxiter = 1000;
		
		@Option(name = "-droppingiter", usage = "Maximum consecutive dropping time (Default: 6)")
		public int droppintiter = 6;
	}

	public static void main(String[] args) throws Exception {
		// configurations
		initializeLearner("param.ini");
		// loading data
		Numberer numbererTag = Numberer.getGlobalNumberer(KEY_TAG_SET);
		Map<String, StateTreeList> trees = loadData(numbererTag);
		// training
		long startTime = System.currentTimeMillis();
		train(trees, numbererTag);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	
	@SuppressWarnings("unused")
	private static void train(Map<String, StateTreeList> trees, Numberer numbererTag) throws Exception {
		StateTreeList trainTrees = trees.get(ID_TRAIN);
		StateTreeList testTrees = trees.get(ID_TEST);
		StateTreeList devTrees = trees.get(ID_DEV);
		
		double ll = Double.NEGATIVE_INFINITY;
		Tree<State> globalTree = null;
		String filename = opts.imagePrefix + "_gd";
		String treeFile = opts.imagePrefix + "_tr";
		
		
		LVeGGrammar grammar = new LVeGGrammar(null, -1);
		LVeGLexicon lexicon = new SimpleLVeGLexicon();
		Valuator<?> valuator = new Valuator<Double>(grammar, lexicon, true);
		MultiThreadedParser mvaluator = new MultiThreadedParser(valuator, opts.nthreadeval);
				
		if (opts.loadGrammar && opts.inGrammar != null) {
			GrammarFile gfile = GrammarFile.load(opts.datadir + opts.inGrammar);
			grammar = gfile.getGrammar();
			lexicon = gfile.getLexicon();
			lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
			Optimizer.config(random, opts.maxsample, opts.batchsize); // FIXME no errors, just alert you...
			valuator = new Valuator<Double>(grammar, lexicon, true);			
			mvaluator = new MultiThreadedParser(valuator, opts.nthreadeval);
			
			for (Tree<State> tree : trainTrees) {
				if (tree.getYield().size() == 6) {
					globalTree = tree.copy();
					break;
				} // a global tree
			}
		} else {
			Optimizer goptimizer = new ParallelOptimizer(LVeGLearner.random, opts.maxsample, opts.batchsize, opts.ntheadgrad, opts.parallelgrad);
			Optimizer loptimizer = new ParallelOptimizer(LVeGLearner.random, opts.maxsample, opts.batchsize, opts.ntheadgrad, opts.parallelgrad);
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
			lexicon.postInitialize(trainTrees, numbererTag.size());
			logger.trace("post-initializing is over.\n");
		}
		/*
		// initial likelihood of the training set
		logger.trace("\n-------ll of the training data initially is... ");
		long beginTime = System.currentTimeMillis();
		ll = calculateLL(grammar, mvaluator, trainTrees);
		long endTime = System.currentTimeMillis();
		logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");	
		*/
		LVeGParser<?> lvegParser = new LVeGParser<List<Double>>(grammar, lexicon, true);
		MaxRuleParser<?> mrParser = new MaxRuleParser<Tree<String>>(grammar, lexicon, true);
		MultiThreadedParser trainer = new MultiThreadedParser(lvegParser, opts.nthreadbatch);
		
		MethodUtil.saveTree2image(globalTree, filename, null);
		Tree<String> parseTree = mrParser.parse(globalTree);
		MethodUtil.saveTree2image(null, treeFile + "_ini", parseTree);
		
		logger.info("\n---SGD CONFIG---\n[parallel: batch-" + opts.parallelbatch + ", grad-" + opts.parallelgrad +"] " + Params.toString(false) + "\n");
		
		if (opts.parallelbatch) {
			parallelInBatch(grammar, lexicon, mvaluator, trainer, mrParser, trees, globalTree, treeFile, ll);
		} else {
			serialInBatch(grammar, lexicon, mvaluator, lvegParser, mrParser, trees, globalTree, treeFile, ll);
		}
		
		/*
		if (relativError < opts.relativerror) {
			System.out.println("Maximum-relative-error reaching.");
		}
		*/
	}
	
	
	public static void parallelInBatch(LVeGGrammar grammar, LVeGLexicon lexicon, MultiThreadedParser mvaluator, MultiThreadedParser trainer, 
			MaxRuleParser<?> mrParser, Map<String, StateTreeList> trees, Tree<State> globalTree, String treeFile, double prell) throws Exception {
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
				trainer.parse(tree);
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
						// likelihood of the training set
						logger.trace("\n-------ll of the training data after " + (isample / opts.batchsize) + " batches in epoch " + cnt + " is... ");
						beginTime = System.currentTimeMillis();
						ll = calculateLL(grammar, mvaluator, trainTrees);
						endTime = System.currentTimeMillis();
						logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
						trainTrees.reset();
						// visualize the parse tree
						Tree<String> parseTree = mrParser.parse(globalTree);
						String filename = treeFile + "_" + cnt + "_" + (isample / (opts.batchsize * opts.nbatch));
						MethodUtil.saveTree2image(null, filename, parseTree);
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
			ll = calculateLL(grammar, mvaluator, trainTrees);
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
	
	public static void serialInBatch(LVeGGrammar grammar, LVeGLexicon lexicon, MultiThreadedParser mvaluator, LVeGParser<?> lvegParser, 
			MaxRuleParser<?> mrParser, Map<String, StateTreeList> trees, Tree<State> globalTree, String treeFile, double prell) throws Exception {
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
						ll = calculateLL(grammar, mvaluator, trainTrees);
						endTime = System.currentTimeMillis();
						logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
						trainTrees.reset();
						// visualize the parse tree
						Tree<String> parseTree = mrParser.parse(globalTree);
						String filename = treeFile + "_" + cnt + "_" + (isample / (opts.batchsize * opts.nbatch));
						MethodUtil.saveTree2image(null, filename, parseTree);
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
			ll = calculateLL(grammar, mvaluator, trainTrees);
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
	
	
	public static double calculateLL(LVeGGrammar grammar, MultiThreadedParser valuator, StateTreeList stateTreeList) {
		int nUnparsable = 0, cnt = 0;
		double ll = 0, sumll = 0;
		for (Tree<State> tree : stateTreeList) {
			if (opts.onlyLength > 0) {
				if (tree.getYield().size() > opts.onlyLength) { continue; }
			}
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
	
	
	private static void initializeLearner(String paramfile) {
		String[] args = null;
		try {
			args = MethodUtil.readFile(paramfile, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		opts = (Options) optionParser.parse(args, true);
		
		if (opts.outGrammar == null) {
			throw new IllegalArgumentException("Output file is required.");
		}
		if (opts.logType) {
			logger = logUtil.getFileLogger(opts.logFile);
		} else {
			logger = logUtil.getConsoleLogger();
		}
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		logger.info("Random number generator seeded at " + opts.randomSeed + ".\n");
		
		dim = opts.dim;
		maxrandom = opts.maxrandom;
		ncomponent = opts.ncomponent;
		random = new Random(opts.randomSeed);
		Params.config(opts);
	}
	
	
	protected static  Map<String, StateTreeList> loadData(Numberer numbererTag) {
		Map<String, StateTreeList> trees = new HashMap<String, StateTreeList>(3, 1);
		List<Tree<String>> trainString = loadStringTree(opts.datadir + opts.train, numbererTag);
		List<Tree<String>> testString = loadStringTree(opts.datadir + opts.test, numbererTag);
		List<Tree<String>> devString = loadStringTree(opts.datadir + opts.dev, numbererTag);
		if (opts.trainOnDevSet) {
			logger.info("Adding development set to the training data.\n");
			trainString.addAll(devString);
		}
		StateTreeList trainTrees = stringTreeToStateTree(trainString, numbererTag);
		StateTreeList testTrees = stringTreeToStateTree(testString, numbererTag);
		StateTreeList devTrees = stringTreeToStateTree(devString, numbererTag);
		trees.put(ID_TRAIN, trainTrees);
		trees.put(ID_TEST, testTrees);
		trees.put(ID_DEV, devTrees);
		return trees;
	}
	
	
	protected static List<Tree<String>> loadStringTree(String path, Numberer numbererTag) {
		logger.trace("--->Loading trees from " + path + " and using languange " + opts.treebank + "\n");
		boolean onlyTest = false, manualAnnotation = false;
		Corpus corpus = new Corpus(
				path, opts.treebank, opts.trainingFraction, onlyTest,
				opts.skipSection, opts.skipBilingual, opts.keepFunctionLabel);
		List<Tree<String>> data = Corpus.binarizeAndFilterTrees(
				corpus.getTrainTrees(), opts.verticalMarkovization, opts.horizontalMarkovization, 
				opts.maxSentenceLength, opts.binarization, manualAnnotation, opts.verbose);
		return data;
	}
	
	
	protected static StateTreeList stringTreeToStateTree(List<Tree<String>> stringTrees, Numberer numbererTag) {
		if (opts.lowercase) {
			System.out.println("Lowercasing the treebank.");
			Corpus.lowercaseWords(stringTrees);
		}
		logger.trace("--->There are " + stringTrees.size() + " trees in the corpus.\n");
		StateTreeList stateTreeList = new StateTreeList(stringTrees, numbererTag);
		MethodUtil.debugNumbererTag(numbererTag, opts); // DEBUG
		if (opts.simpleLexicon) {
			System.out.println("Replacing words appearing less than 5 times with their signature.");
			LVeGCorpus.replaceRareWords(stateTreeList, new SimpleLVeGLexicon(), opts.rareThreshold);
		}
		return stateTreeList;
	}
	
}
