package edu.shanghaitech.ai.nlp.lveg;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;

import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.PCFGLA.Option;
import edu.berkeley.nlp.PCFGLA.OptionParser;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGLearner extends Recorder {
	
	public static short dim        =  2;
	public static short maxrandom  =  1;
	public static short batchsize  = 50;
	public static short ncomponent =  2;
	public static short maxlength  = 30;
	
	public static int randomseed = 0;
	public static int precision  = 3;
	
	private static Random random;
	public static Logger logger = null;
	
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
		public short batchsize = 500;
		
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
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		OptionParser optionParser = new OptionParser(Options.class);
		Options opts = (Options) optionParser.parse(args, true);
		System.out.println("Calling with " + optionParser.getPassedInOptions());
		
		initializeLearner(opts); // 
		
		Map<String, List<Tree<String>>> stringTrees = loadData(opts);
		
		// Numberer numbererTag = getNumbererTag(data, opts);
		Numberer numbererTag = Numberer.getGlobalNumberer(KEY_TAG_SET);
		
		Map<String, StateTreeList> stateTrees = stringTreeToStateTree(stringTrees, numbererTag, opts);
		
		train(stateTrees, numbererTag, opts);
	}
	
	
	private static void train(Map<String, StateTreeList> stateTrees, Numberer numbererTag, Options opts) {
		StateTreeList trainTrees = new StateTreeList(stateTrees.get(ID_TRAINING));
		StateTreeList validationTrees = new StateTreeList(stateTrees.get(ID_VALIDATION));
		
		// MethodUtil.isChildrenSizeZero(trainTrees);
		
		// MethodUtil.isParentEqualToChild(trainTrees);
		LVeGGrammar grammar = new LVeGGrammar(null, opts.filterThreshold, -1);
		LVeGLexicon lexicon = new SimpleLVeGLexicon();
			
		for (Tree<State> tree : trainTrees) {
			lexicon.tallyStateTree(tree);
			grammar.tallyStateTree(tree);
		}
		
		grammar.postInitialize(0.0);
		lexicon.postInitialize(trainTrees, numbererTag.size());
		
		short idp = 5, idc = 1;
//		for (Tree<State> tree : trainTrees) {
//			if (MethodUtil.containsRule(tree, idp, idc)) {
//				System.out.println(tree);
//			}
//		}

		
		logger.debug(grammar);
		logger.debug(lexicon);
		
//		System.out.println(grammar);
//		System.out.println(lexicon);
		
//		if (grammar.containsRule(rule, true)) {
//			System.out.println("oops");
//		}
//		System.exit(0);
		
		// check if there is any circle in the unary grammar rules
		// TODO move this self-checking procedure to the class Grammar
		logger.debug("---Circle Detection.\n");
		if (MethodUtil.checkUnaryRuleCircle(grammar, lexicon, true)) {
			logger.error("Circle (WithC) was found in the unary grammar rules.");
			return;
		}
		
		/*// DEBUG Note necessary, the circle is reversible 
		if (MethodUtil.checkUnaryRuleCircle(grammar, lexicon, false)) {
			logger.error("Circle (WithP) was found in the unary grammar rules.");
			System.exit(0);
		}
		*/
		
		LVeGGrammar maxGrammar = null, preGrammar = null;
		LVeGLexicon maxLexicon = null, preLexicon = null;
		
		int cnt = 0, droppingiter = 0;
//		double prell = calculateLL(grammar, lexicon, validationTrees);	
		double prell = 0.0;
		double relativError = 0, ll, maxll = prell;
		do {
			cnt++;
			
			LVeGParser parser = new LVeGParser(grammar, lexicon);
			short isample = 0;
			long startTime = System.currentTimeMillis();
			for (Tree<State> tree : trainTrees) {
//				if (tree.getYield().size() != 5) { continue; }
				
				/*// DEBUG
				System.out.println(tree.getTerminalYield());
				System.out.println(tree.getYield());
				*/
				
//				parser.doInsideOutside(tree);
				
				/*// DEBUG 
				parser.doInsideOutsideWithTree(tree);
				MethodUtil.debugTree(tree, false, (short) 2);
				trainTrees.resetScore(tree);
				MethodUtil.debugTree(tree, false, (short) 2);
				*/
				
				if (tree.getYield().size() > 10) { continue; }
				
				parser.evalRuleCountWithTree(tree);
				parser.evalRuleCount(tree);
				
				isample++;
				LVeGLearner.logger.trace("Sample " + isample + "...");
				if (isample >= 1) {
					break;
				}
			}
			long endTime = System.currentTimeMillis();
			
			System.out.println("Average time each sample cost is " + (endTime - startTime) / (1000.0 * isample));
			
			MethodUtil.debugCount(grammar, lexicon, null, null); // DEBUG
			
			// apply gradient descent
			grammar.applyGradientDescent(random, opts.lr);
			lexicon.applyGradientDescent(random, opts.lr);
			
			/*
			ll = calculateLL(grammar, lexicon, validationTrees);
			relativError = (ll - prell) / prell;
			
			if (ll > maxll) {
				maxll = ll;
				prell = ll;
				maxGrammar = grammar;
				maxLexicon = lexicon;
				droppingiter = 0;
			} else {
				droppingiter++;
				if (droppingiter >= opts.droppintiter) {
					System.out.println("Maximum-allowed-dropping-time reaching");
				}
			}
			
			if (cnt > opts.maxiter) {
				System.out.println("Maximum-iteration-time exceeding.");
				break;
			}
			*/
			
			// we shall clear the inside and outside score in each state 
			// of the parse tree after the training on a sample 
			trainTrees.resetScore();
			validationTrees.resetScore();
			
			LVeGLearner.logger.trace("Epoch " + cnt + "...");
			if (cnt >= 2) {
				break;
			}
		// relative error could be negative
		} while(cnt > 0) /*while (cnt > 1 && Math.abs(relativError) < opts.relativerror && droppingiter < opts.droppintiter)*/;
		
		
		/*
		if (relativError < opts.relativerror) {
			System.out.println("Maximum-relative-error reaching.");
		}
		*/
	}
	
	
	public static double calculateLL(LVeGGrammar grammar, LVeGLexicon lexicon, StateTreeList stateTreeList) {
		
		int nUnparsable = 0;
		double ll = 0, sumll = 0;
		LVeGParser parser = new LVeGParser(grammar, lexicon);
		for (Tree<State> tree : stateTreeList) {
			ll = parser.probability(tree);
			ll = Math.log(ll);
			if (Double.isInfinite(ll) || Double.isNaN(ll)) {
				nUnparsable++;
			} else {
				sumll += ll;
			}
		}
		logger.trace("There is (are) " + nUnparsable + " unparsable sample(s)");
		return sumll;
	}
	
	
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
		logger.debug("There are " + trainTrees.size() + " trees in the training set.");
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
				logger.trace("Tag " + i + "\t" +  (String) numbererTag.object(i)); // DEBUG
			}
		}
		logger.debug("There are " + numbererTag.size() + " observed tags.");
	}

}
