package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
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
	/**
	 * 
	 */
	private static final long serialVersionUID = -4975261706356545370L;
	public final static String KEY_TAG_SET = "tags";
	public final static String TOKEN_UNKNOWN = "UNK";
	public final static String ID_TRAINING = "training";
	public final static String ID_VALIDATION = "validation";
	
	public static short dim        =  5;
	public static short maxrandom  =  1;
	public static short batchsize  = 50;
	public static short ncomponent =  5;
	public static short maxlength  = 30;
	
	public static int randomseed = 0;
	public static int precision  = 3;
	
	public static Random random;
	
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
			return "Params [lr = " + lr + ", reg = " + ", clip = " + clip + ", absmax = " + 
					absmax + ", wdecay = " + wdecay + ", momentum = " + momentum + ", l1 = " + l1 + "]";
		}
	}
	
	public static class Options {
		/* corpus section begins */
		@Option(name = "-test", usage = "path to the test data (default: null")
		public String test = null;
		@Option(name = "-dev", usage = "path to the development data (default: null)")
		public String dev = null;
		@Option(name = "-train", usage = "path to the training data (default: null)")
		public String train = null;
		@Option(name = "-inCorpus", usage = "input object file of the training, development, and test data (default: null" )
		public String inCorpus = null;
		@Option(name = "-outCorpus", usage = "output object file of the training, development, and test data (default: null" )
		public String outCorpus = null;
		@Option(name = "-saveCorpus", usage = "save corpus (true) or not (false) (default: false)")
		public boolean saveCorpus = false;
		@Option(name = "-loadCorpus", usage = "load corpus (true) or not (false) (default: false)")
		public boolean loadCorpus = false;
		/* corpus section ends */
		
		/* grammar-data section begins */
		@Option(name = "-in", usage = "input object file of the grammar (default: null)")
		public String inGrammar = null;
		@Option(name = "-out", usage = "output object file of the grammar (default: null)")
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
		@Option(name = "-maxsample", usage = "sampling times when approximate gradients (default: 3)")
		public short maxsample = 3;
		@Option(name = "-evalfraction", usage = "fraction of the training data on which the grammar evaluation is conducted (default: 0.2)")
		public double evalfraction = 0.2;
		@Option(name = "-nepoch", usage = "# of epoches (default: 10)")
		public int nepoch = 10;
		@Option(name = "-relativediff", usage = "maximum relative difference between the neighboring iterations (default: 1e-6)")
		public double relativerror = 1e-6;
		@Option(name = "-onlyLength", usage = "train on only the sentences of length less than or equal to the specific length (default: 50)")
		public int onlyLength = 50;
		/* training-configurations section ends */
		
		/* optimization-parameter section begins*/
		@Option(name = "-lr", usage = "learning rate (default: 1.0)")
		public double lr = 1.0;
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
		public String imagePrefix = "maxrule";
		/* file prefix section ends */
		
		
		@Option(name = "-seed", usage = "seed for random number generator (default: 111)")
		public int seed = 111;
		
		@Option(name = "-maxLength", usage = "maximum sentence length (default: <= 10000)")
		public int maxLength = 10000;
		
		@Option(name = "-treebank", usage = "language: WSJ, CHINESE, SINGLEFILE (default: SINGLEFILE)")
		public TreeBankType treebank = TreeBankType.SINGLEFILE;

		
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
	}
	

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		OptionParser optionParser = new OptionParser(Options.class);
		Options opts = (Options) optionParser.parse(args, true);
		System.out.println("Calling with " + optionParser.getPassedInOptions());
		
		initializeLearner(opts); // 
		
		Map<String, List<Tree<String>>> stringTrees = loadData(opts);
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
		
		Tree<State> globalTree = null;
		String oldFilename = "log/" + opts.imagePrefix + "_gd";
		String filename, newFilename = "log/" + opts.imagePrefix + "_tr";
		
		batchsize = 60;
		short maxsample = 3;
		short ntheadgrad = 6, nthreadeval = 6, nthreadbatch = 6; // eval gradients, calculate ll, parallelize in the batch
		boolean parallelgrad = true, parallelbatch = false;
		boolean useOldGram = false, saveNewGram = false;
		int cnt = 0, droppingiter = 0, maxLength = 7, nbatch = 1;
		
		LVeGGrammar grammar = new LVeGGrammar(null, -1);
		LVeGLexicon lexicon = new SimpleLVeGLexicon();
		Valuator<?> valuator = new Valuator<Double>(grammar, lexicon, true);
		MultiThreadedParser mvaluator = new MultiThreadedParser(valuator, nthreadeval);
				
		if (useOldGram && opts.inGrammar != null) {
			GrammarFile gfile = GrammarFile.load(opts.inGrammar);
			grammar = gfile.getGrammar();
			lexicon = gfile.getLexicon();
			lexicon.labelTrees(trainTrees); // FIXME no errors, just alert you to pay attention to it 
			Optimizer.config(random, maxsample, batchsize); // FIXME no errors, just alert you...
			valuator = new Valuator<Double>(grammar, lexicon, true);			
			mvaluator = new MultiThreadedParser(valuator, nthreadeval);
			
			for (Tree<State> tree : trainTrees) {
				if (tree.getYield().size() == 6) {
					globalTree = tree.shallowClone();
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
						
						
						if (saveNewGram && opts.outGrammar != null) {
							GrammarFile gfile = new GrammarFile(grammar, lexicon);
							String gfilename = opts.outGrammar + "_" + (isample / batchsize) + "_" + cnt + ".gr";
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
	
	
	private static void initializeLearner(Options opts) {
		if (opts.outGrammar == null) {
			throw new IllegalArgumentException("Output file is required.");
		} else {
			if (opts.logType) {
				logger = logUtil.getFileLogger(opts.logFile);
			} else {
				logger = logUtil.getConsoleLogger();
			}
			System.out.println("Grammar file will be saved to " + opts.outGrammar + ".");
		}
		dim = opts.dim;
		maxrandom = opts.maxrandom;
		batchsize = opts.batchsize;
		randomseed = opts.seed;
		ncomponent = opts.ncomponent;
		random = new Random(opts.seed);
		System.out.println("Random number generator seeded at " + opts.seed + ".");
		Params.config(opts);
	}
	
	
	private static Map<String, List<Tree<String>>> loadData(Options opts) {
		System.out.println("Loading trees from " + opts.train + " and using languange " + opts.treebank);
		
		boolean onlyTest = false, manualAnnotation = false;
		Map<String, List<Tree<String>>> datasets = new HashMap<String, List<Tree<String>>>();
		
		Corpus corpus = new Corpus(
				opts.train, opts.treebank, opts.trainingFraction,
				onlyTest,
				opts.skipSection, opts.skipBilingual, opts.keepFunctionLabel);
		List<Tree<String>> trainTrees = Corpus.binarizeAndFilterTrees(
				corpus.getTrainTrees(), opts.verticalMarkovization, opts.horizontalMarkovization, 
				opts.maxLength, opts.binarization, manualAnnotation, opts.verbose);
		List<Tree<String>> validationTrees = Corpus.binarizeAndFilterTrees(
				corpus.getValidationTrees(), opts.verticalMarkovization, opts.horizontalMarkovization, 
				opts.maxLength, opts.binarization, manualAnnotation, opts.verbose);
		
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
		
		MethodUtil.debugNumbererTag(numbererTag, opts);
		
		datasets.put(ID_TRAINING, trainStateTrees);
		datasets.put(ID_VALIDATION, validationStateTrees);
		
		if (opts.simpleLexicon) {
			System.out.println("Replacing words appearing less than 5 times with their signature.");
			LVeGCorpus.replaceRareWords(trainStateTrees, new SimpleLVeGLexicon(), opts.rareThreshold);
		}
		
		return datasets;
	}

}
