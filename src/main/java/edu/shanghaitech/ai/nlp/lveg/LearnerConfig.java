package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.ObjectFileManager.CorpusFile;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer.ParallelMode;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Option;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class LearnerConfig extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5745194882841709365L;
	protected final static String ID_TRAIN = "train";
	protected final static String ID_TEST = "test";
	protected final static String ID_DEV = "dev";
	
	public final static String KEY_TAG_SET = "tag";
	
	public static short dim;
	public static short ncomponent;
	public static short maxrandom;
	public static short maxlength;
	public static int randomseed;
	public static int precision;
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
		@Option(name = "-parallelmode", usage = "the mode of parallelizing gradient evaluation (default: THREAD_POOL")
		public ParallelMode parallelmode = ParallelMode.THREAD_POOL;
		@Option(name = "-pverbose", usage = "silent (false) the parallel optimizer or not (true) (default: true)")
		public boolean pverbose = true;
		/* parallel-configurations section ends */
		
		/* training configurations section begins */
		@Option(name = "-batchsize", usage = "# of the samples in a batch (default: 128)")
		public short batchsize = 128;
		@Option(name = "-maxsample", usage = "sampling times when approximating gradients (default: 3)")
		public short maxsample = 3;
		@Option(name = "-nepoch", usage = "# of epoches (default: 10)")
		public int nepoch = 10;
		@Option(name = "-relativediff", usage = "maximum relative difference between the neighboring iterations (default: 1e-6)")
		public double relativerror = 1e-6;
		@Option(name = "-ncomponent", usage = "# of gaussian components (default: 2)")
		public short ncomponent = 2;
		@Option(name = "-dim", usage = "dimension of the gaussian (default: 2)")
		public short dim = 2;
		@Option(name = "-maxramdom", usage = "maximum random value in initializing MoGul parameters (Default: 1)")
		public short maxrandom = 1;
		/* training-configurations section ends */
		
		/* evaluation section begins */
		@Option(name = "-evalfraction", usage = "fraction of the training data on which the grammar evaluation is conducted (default: 0.2)")
		public double evalfraction = 0.2;
		@Option(name = "-onlyLength", usage = "train or eval on only the sentences of length less than or equal to the specific length, no such constraints if it is negative (default: 50)")
		public int onlyLength = 50;
		@Option(name = "-firstk", usage = "evaluate the grammar on the only first k samples, no such constraints if it is negative (default: 200)")
		public int firstk = 200;
		@Option(name = "-evalondev", usage = "evaluate the grammar on the dev-data (true) or not (false) (default: false)")
		public boolean evalondev = true;
		@Option(name = "-evalontrain", usage = "evaluate the grammar on the train-data (true) or not (false) (default: true)")
		public boolean evalontrain = true;
		/* evaluation section ends */
		
		/* logger configurations section begins */
		@Option(name = "-logType", usage = "console (false) or file (true) (default: true)")
		public boolean logType = false;
		@Option(name = "-logFile", usage = "log file name (default: log/log)")
		public String logFile = "log/log";
		@Option(name = "-nbatch", usage = "# of batchs after which the grammar evaluation is conducted (default: 1)")
		public short nbatch = 1;
		@Option(name = "-precision", usage = "precision of the output decimals (default: 3)")
		public int precision = 3;
		@Option(name = "-randomSeed", usage = "Seed for random number generator (Default: 111)")
		public int randomSeed = 111;
		/* logger-configurations section ends */
		
		/* file prefix section begins */
		@Option(name = "-imagePrefix", usage = "prefix of the image of the parse tree obtained from max rule parser (default: maxrule")
		public String imagePrefix = "log/maxrule";
		/* file prefix section ends */
		
		
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
	}
	
	protected static void initialize(Options opts) {
		if (opts.outGrammar == null) {
			throw new IllegalArgumentException("Output file is required.");
		}
		if (opts.logType) {
			logger = logUtil.getFileLogger(opts.logFile);
		} else {
			logger = logUtil.getConsoleLogger();
		}
		
		logger.info("Random number generator seeded at " + opts.randomSeed + ".\n");
		
		dim = opts.dim;
		precision = opts.precision;
		maxrandom = opts.maxrandom;
		ncomponent = opts.ncomponent;
		randomseed = opts.randomSeed;
		random = new Random(randomseed);
		Params.config(opts);
	}
	
	protected static  Map<String, StateTreeList> loadData(Numberer wrapper, Options opts) {
		Numberer numberer = null;
		StateTreeList trainTrees, testTrees, devTrees;
		Map<String, StateTreeList> trees = new HashMap<String, StateTreeList>(3, 1);
		if (opts.loadCorpus && opts.inCorpus != null) {
			logger.trace("--->Loading corpus from \'" + opts.datadir + opts.inCorpus + "\'...\n");
			CorpusFile corpus = (CorpusFile) CorpusFile.load(opts.datadir + opts.inCorpus);
			numberer = corpus.getNumberer();
			trainTrees = corpus.getTrain();
			testTrees = corpus.getTest();
			devTrees = corpus.getDev();
			wrapper.put(KEY_TAG_SET, numberer);
		} else {
			numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
			List<Tree<String>> trainString = loadStringTree(opts.datadir + opts.train, opts);
			List<Tree<String>> testString = loadStringTree(opts.datadir + opts.test, opts);
			List<Tree<String>> devString = loadStringTree(opts.datadir + opts.dev, opts);
			if (opts.trainOnDevSet) {
				logger.info("Adding development set to the training data.\n");
				trainString.addAll(devString);
			}
			trainTrees = stringTreeToStateTree(trainString, numberer, opts);
			testTrees = stringTreeToStateTree(testString, numberer, opts);
			devTrees = stringTreeToStateTree(devString, numberer, opts);

		}
		trees.put(ID_TRAIN, trainTrees);
		trees.put(ID_TEST, testTrees);
		trees.put(ID_DEV, devTrees);
		
		if (opts.saveCorpus && opts.outCorpus != null) {
			logger.info("\n-------saving corpus file...");
			CorpusFile corpus = new CorpusFile(trainTrees, testTrees, devTrees, numberer);
			String filename = opts.datadir + opts.outCorpus;
			if (corpus.save(filename)) {
				logger.info("to \'" + filename + "\' successfully.");
			} else {
				logger.info("to \'" + filename + "\' unsuccessfully.");
			}
		}
		return trees;
	}
	
	protected static List<Tree<String>> loadStringTree(String path, Options opts) {
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
	
	protected static StateTreeList stringTreeToStateTree(List<Tree<String>> stringTrees, Numberer numberer, Options opts) {
		if (opts.lowercase) {
			System.out.println("Lowercasing the treebank.");
			Corpus.lowercaseWords(stringTrees);
		}
		logger.trace("--->There are " + stringTrees.size() + " trees in the corpus.\n");
		StateTreeList stateTreeList = new StateTreeList(stringTrees, numberer);
		debugNumbererTag(numberer, opts); // DEBUG
		if (opts.simpleLexicon) {
			System.out.println("Replacing words appearing less than 5 times with their signature.");
			LVeGCorpus.replaceRareWords(stateTreeList, new SimpleLVeGLexicon(), opts.rareThreshold);
		}
		return stateTreeList;
	}
	
	protected static void debugNumbererTag(Numberer numberer, Options opts) {
		if (opts.verbose || true) {
			for (int i = 0; i < numberer.size(); i++) {
//				logger.trace("Tag " + i + "\t" +  (String) numberer.object(i) + "\n"); // DEBUG
			}
		}
		logger.debug("There are " + numberer.size() + " observed tags.\n");
	}
	
	public static String readFile(String path, Charset encoding) throws IOException {
		  byte[] encoded = Files.readAllBytes(Paths.get(path));
		  return new String(encoded, encoding);
	}
	
}
