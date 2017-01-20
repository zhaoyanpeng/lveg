package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.CorpusFile;
import edu.shanghaitech.ai.nlp.lveg.impl.GaussFactory;
import edu.shanghaitech.ai.nlp.lveg.impl.MoGFactory;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.optimization.Optimizer.OptChoice;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer.ParallelMode;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.ObjectPool;
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
	
	protected static String subdatadir;
	protected static String sublogroot;
	
	public final static String KEY_TAG_SET = "tag";
	public static double minmw = 1e-6;
	public static double nratio = 0.5;
	
	public static short cntdrop = 0;
	public static double bestscore = Double.NEGATIVE_INFINITY;
	
	public static short dim = 2;
	public static short ncomponent = 2;
	public static short maxlength = 7;
	public static double maxrandom = 1;
	public static int randomseed = 0;
	public static int precision = 3;
	public static Random random = new Random(randomseed);
	public static Random rnd4shuffle = new Random(11);
	
	public static ObjectPool<Short, GaussianMixture> mogPool;
	public static ObjectPool<Short, GaussianDistribution> gaussPool;

	public static class Params {
		public static double lr;
		public static boolean l1;
		public static boolean reg;
		public static boolean clip;
		public static double absmax;
		public static double wdecay;
		public static double lambda;
		public static double lambda1;
		public static double lambda2;
		public static double epsilon;
		
		public static void config(Options opts) {
			lr = opts.lr;
			l1 = opts.l1;
			reg = opts.reg;
			clip = opts.clip;
			absmax = opts.absmax;
			wdecay = opts.wdecay;
			lambda = opts.lambda;
			lambda1 = opts.lambda1;
			lambda2 = opts.lambda2;
			epsilon = opts.epsilon;
		}
		
		public static String toString(boolean placeholder) {
			return "Params [lr = " + lr + ", reg = " + reg + ", clip = " + clip + ", absmax = " + absmax + 
					", wdecay = " + wdecay + ", lambda = " + lambda + ", l1 = " + l1 + ", lambda1 = " + 
					lambda1 + ", lambda2 = " + lambda2 + ", epsilon = " + epsilon + "]";
		}
	}
	
	public static class Options {
		/* corpus section begins */
		@Option(name = "-datadir", required = true, usage = "absolute path to the data directory (default: null)")
		public String datadir = null;
		@Option(name = "-train", required = true, usage = "name of the training data (default: null)")
		public String train = null;
		@Option(name = "-test", usage = "name of the test data (default: null)")
		public String test = null;
		@Option(name = "-dev", usage = "name of the dev data (default: null)")
		public String dev = null;
		@Option(name = "-inCorpus", usage = "input: object file of the training/development/test data, and the tag set (default: null)" )
		public String inCorpus = null;
		@Option(name = "-outCorpus", usage = "output: object file of the training/development/test data, and the tag set (default: null)" )
		public String outCorpus = null;
		@Option(name = "-saveCorpus", usage = "save corpus to the object file (true) or not (false) (default: false)")
		public boolean saveCorpus = false;
		@Option(name = "-loadCorpus", usage = "load corpus from the object file(true) or not (false) (default: false)")
		public boolean loadCorpus = false;
		/* corpus section ends */
		
		/* optimization-parameter section begins*/
		@Option(name = "-lr", usage = "learning rate (default: 1.0)")
		public double lr = 0.02;
		@Option(name = "-reg", usage = "using regularization (true) or not (false) (default: true)")
		public boolean reg = true;
		@Option(name = "-clip", usage = "clipping the gradients (true) or not (false) (default: true)")
		public boolean clip = true;
		@Option(name = "-absmax", usage = "threshold for clipping gradients (default: 5.0)")
		public double absmax = 5.0;
		@Option(name = "-wdecay", usage = "weight decay rate (default: 0.02)")
		public double wdecay = 0.02;
		@Option(name = "-l1", usage = "using l1 regularization (true) or l2 regularization (false) (default: true)")
		public boolean l1 = true;
		@Option(name = "-minmw", usage = "minimum mixing weight (default: 1e-6)")
		public double minmw = 1e-6;
		@Option(name = "-expzero", usage = "relative magnitude calculated as a / b = exp(log(a) - log(b)) (default: 1e-6)")
		public double expzero = 1e-6;
		@Option(name = "-epsilon", usage = "a small constant to avoid the division by zero (default: 1e-8)")
		public double epsilon = 1e-8;
		@Option(name = "-choice", usage = "optimization methods: NORMALIZED, SGD, ADAGRAD, RMSPROP, ADADELTA, ADAM (default: ADAM)")
		public OptChoice choice = OptChoice.ADAM;
		@Option(name = "-lambda", usage = "momentum used in the gradient update (default: 0.9)")
		public double lambda = 0.9;
		@Option(name = "-lambda1", usage = "1st. order momentum (default: 0.9)")
		public double lambda1 = 0.9;
		@Option(name = "-lambda2", usage = "2nd. order momentum (default: 0.9)")
		public double lambda2 = 0.9;
		/* optimization-parameter section ends*/
		
		/* grammar-data section begins */
		@Option(name = "-inGrammar", usage = "input: object file of the grammar and lexicon (default: null)")
		public String inGrammar = null;
		@Option(name = "-outGrammar", usage = "output: object file of the grammar (default: null)")
		public String outGrammar = null;
		@Option(name = "-saveGrammar", usage = "save grammar to the object file (true) or not (false) (default: false)")
		public boolean saveGrammar = false;
		@Option(name = "-loadGrammar", usage = "load grammar from the object file (true) or not (false) (default: false)")
		public boolean loadGrammar = false;
		@Option(name = "-nbatchSave", usage = "# of batches after which the grammar is saved (default: 20")
		public short nbatchSave = 20;
		/* grammar-data section ends */
		
		/* parallel configurations section begins */
		@Option(name = "-ntbatch", usage = "# of threads for minibatch training (default: 1)")
		public short ntbatch = 1;
		@Option(name = "-ntgrad", usage = "# of threads for gradient calculation (default: 1)")
		public short ntgrad = 1;
		@Option(name = "-nteval", usage = "# of threads for grammar evaluation (default: 1)")
		public short nteval = 1;
		@Option(name = "-pclose", usage = "close all parallel switches (default: false)")
		public boolean pclose = false;
		@Option(name = "-pbatch", usage = "parallizing training in the minibatch (true) or not (false) (default: false)")
		public boolean pbatch = true;
		@Option(name = "-peval", usage = "parallizing evaluation section (true) or not (false) (default: false)")
		public boolean peval = true;
		@Option(name = "-pgrad", usage = "parallizeing gradient calculation (true) or not (false) (default: true)")
		public boolean pgrad = true;
		@Option(name = "-pmode", usage = "parallel mode of gradient evaluation: INVOKE_ALL, COMPLETION_SERVICE, CUSTOMIZED_BLOCK, FORK_JOIN, THREAD_POOL (default: THREAD_POOL)")
		public ParallelMode pmode = ParallelMode.THREAD_POOL;
		@Option(name = "-pverbose", usage = "silent (false) the parallel optimizer or not (true) (default: true)")
		public boolean pverbose = true;
		/* parallel-configurations section ends */
		
		/* training configurations section begins */
		@Option(name = "-reuse", usage = "whether reuse the chart for CYK and inside-outside score (true) or not (false) (default: true)")
		public boolean reuse = true;
		@Option(name = "-prune", usage = "avoid adding trivial components if the mixing weights are equal to zero (default: false)")
		public boolean prune = false;
		@Option(name = "-bsize", usage = "# of the samples in a batch (default: 128)")
		public short bsize = 128;
		@Option(name = "-nepoch", usage = "# of epoches for training (default: 10)")
		public int nepoch = 10;
		@Option(name = "-maxsample", usage = "maximum sampling time when approximating gradients (default: 3)")
		public short maxsample = 3;
		@Option(name = "-maxLenParsing", usage = "which is used to initialize the size of the chart used in CYK and [inside, outside] score calculation (default: 120)")
		public short maxLenParsing = 120;
		@Option(name = "-nAllowedDrop", usage = "# of allowed iterations in which the validation likelihood drops (default: 6)")
		public short nAllowedDrop = 6;
		@Option(name = "-relativediff", usage = "maximum relative difference between the neighboring iterations (default: 1e-6)")
		public double relativerror = 1e-6;
		@Option(name = "-nratio", usage = "fraction of negative values when initializing MoG parameters (Default: 0.5)")
		public double nratio = 0.5;
		@Option(name = "-maxramdom", usage = "maximum random double (int) value of the exponent part of MoG parameters (Default: 1)")
		public double maxrandom = 1;
		@Option(name = "-maxmw", usage = "maximum random exponent when initializing mixing weights (default: 1)")
		public double maxmw = 1;
		@Option(name = "-nwratio", usage = "fraction of negative exponents when initializing mixing weights (default: 0.5)")
		public double nwratio = 0.5;
		@Option(name = "-maxmu", usage = "maximum random mean of the gaussian (default: 1)")
		public double maxmu = 1;
		@Option(name = "-nmratio", usage = "fraction of negative values when initializing the means of gaussians (default: 0.5)")
		public double nmratio = 0.5;
		@Option(name = "-maxvar", usage = "maximum random exponent when initializing the variances of gaussians (default: 1)")
		public double maxvar = 1;
		@Option(name = "-nvratio", usage = "fraction of negative exponents when initializing the variances of gaussians (default: 0.5)")
		public double nvratio = 0.5;
		@Option(name = "-ncomponent", usage = "# of gaussian components (default: 2)")
		public short ncomponent = 2;
		@Option(name = "-dim", usage = "dimension of the gaussian (default: 2)")
		public short dim = 2;
		/* training-configurations section ends */
		
		/* evaluation section begins */
		@Option(name = "-eratio", usage = "fraction of the training data on which the grammar evaluation is conducted, no such constraints if it is negative (default: 0.2)")
		public double eratio = 0.2;
		@Option(name = "-eonlylen", usage = "training or evaluating on only the sentences of length less than or equal to the specific length, no such constraints if it is negative (default: 50)")
		public short eonlylen = 50;
		@Option(name = "-efirstk", usage = "evaluating the grammar on the only first k samples, no such constraints if it is negative (default: 200)")
		public short efirstk = 200;
		@Option(name = "-eondev", usage = "evaluating the grammar on the dev data (true) or not (false) (default: false)")
		public boolean eondev = true;
		@Option(name = "-eontrain", usage = "evaluating the grammar on the train data (true) or not (false) (default: true)")
		public boolean eontrain = true;
		@Option (name = "-eonextradev", usage = "evaluating the grammar on the sentences of length less than or equal to [eonlylen + 5] (true) or not (false) (default: true)")
		public boolean eonextradev = true;
		@Option (name = "-eondevprune", usage = "applying pruning when evaluating the grammar (true) or not (false) (default: false)")
		public boolean eondevprune = false;
		@Option(name = "-enbatchdev", usage = "# of batches after which the grammar is evaluated on the development dataset (default: 5)")
		public short enbatchdev = 5;
		/* evaluation section ends */
		
		/* logger configurations section begins */
		@Option(name = "-runtag", usage = "the flag assigned to a run specially (default: lveg)")
		public String runtag = "lveg";
		@Option(name = "-logtype", usage = "console (false) or file and console (true) (default: true)")
		public boolean logtype = false;
		@Option(name = "-logroot", usage = "log file name (default: log/log)")
		public String logroot = "log/";
		@Option(name = "-nbatch", usage = "# of batchs after which the grammar evaluation is conducted (default: 1)")
		public short nbatch = 1;
		@Option(name = "-precision", usage = "precision of the output decimals (default: 3)")
		public int precision = 3;
		@Option(name = "-rndomseed", usage = "seed for random number generator (default: 0)")
		public int rndomseed = 111;
		/* logger-configurations section ends */
		
		/* file prefix section begins */
		@Option(name = "-imgprefix", usage = "prefix of the image of the parse tree obtained from max rule parser (default: maxrule)")
		public String imgprefix = "lveg";
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
		// make directories
		subdatadir = opts.datadir + "/" + opts.runtag + "/";
		sublogroot = opts.logroot + "/" + opts.runtag + "/";
		MethodUtil.mkdir(sublogroot);
		MethodUtil.mkdir(subdatadir);
		if (opts.logtype) {
			logger = logUtil.getBothLogger(opts.logroot + "/" + opts.runtag);
		} else {
			logger = logUtil.getConsoleLogger();
		}
		
		logger.info("Random number generator seeded at " + opts.rndomseed + ".\n");
		
		if (opts.pclose) {
			opts.pbatch = false;
			opts.peval = false;
			opts.pgrad = false;
		} // ease the parameter tuning
		
		dim = opts.dim;
		minmw = opts.minmw;
		nratio = opts.nratio;
		precision = opts.precision;
		maxrandom = opts.maxrandom;
		randomseed = opts.rndomseed;
		ncomponent = opts.ncomponent;
		random = new Random(randomseed);
		Params.config(opts);
		
		GenericKeyedObjectPoolConfig config = new GenericKeyedObjectPoolConfig();
		config.setMaxTotalPerKey(Integer.MAX_VALUE);
		config.setMaxTotal(Integer.MAX_VALUE);
		
		config.setBlockWhenExhausted(true); // by default
		config.setMaxWaitMillis(250);
//		config.setTestOnBorrow(true); // by default
//		config.setTestOnCreate(true);
//		config.setTestOnReturn(true);
		
		MoGFactory mfactory = new MoGFactory(opts.ncomponent, opts.maxmw, opts.nwratio, random);
		GaussFactory gfactory = new GaussFactory(opts.dim, opts.maxmu, opts.maxvar, opts.nmratio, opts.nvratio, random);
		mogPool = new ObjectPool<Short, GaussianMixture>(mfactory, config);
		gaussPool = new ObjectPool<Short, GaussianDistribution>(gfactory, config);
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
			trainTrees = stringTreeToStateTree(trainString, numberer, opts, true);
			testTrees = stringTreeToStateTree(testString, numberer, opts, false);
			devTrees = stringTreeToStateTree(devString, numberer, opts, false);

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
	
	protected static StateTreeList stringTreeToStateTree(List<Tree<String>> stringTrees, 
			Numberer numberer, Options opts, boolean replaceRareWords) {
		if (opts.lowercase) {
			logger.trace("Lowercasing the treebank.\n");
			Corpus.lowercaseWords(stringTrees);
		}
		logger.trace("--->There are " + stringTrees.size() + " trees in the corpus.\n");
		StateTreeList stateTreeList = new StateTreeList(stringTrees, numberer);
		debugNumbererTag(numberer, opts); // DEBUG
		if (replaceRareWords) {
			logger.trace("Replacing words appearing less than 20 times with their signature.\n");
			LVeGCorpus.replaceRareWords(stateTreeList, new SimpleLVeGLexicon(), opts.rareThreshold);
		}
		return stateTreeList;
	}
	
	protected static void debugNumbererTag(Numberer numberer, Options opts) {
		if (opts.verbose) {
			for (int i = 0; i < numberer.size(); i++) {
				logger.trace("Tag " + i + "\t" +  (String) numberer.object(i) + "\n");
			}
		}
		logger.trace("There are " + numberer.size() + " observed tags.\n");
	}
	
	public static String readFile(String path, Charset encoding) throws IOException {
		  byte[] encoded = Files.readAllBytes(Paths.get(path));
		  return new String(encoded, encoding);
	}
	
}
