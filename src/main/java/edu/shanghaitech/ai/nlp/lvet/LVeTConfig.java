package edu.shanghaitech.ai.nlp.lvet;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.shanghaitech.ai.nlp.data.LVeGCorpus;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.io.CoNLLFileReader;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.optimization.Optimizer.OptChoice;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer.ParallelMode;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Option;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class LVeTConfig extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -895542569116561260L;
	public final static String KEY_TAG_SET = "tag";
	
	public final static String ID_TRAIN = "train";
	public final static String ID_TEST = "test";
	public final static String ID_DEV = "dev";
	
	protected static String subdatadir;
	protected static String sublogroot;
	public static double minmw = 1e-6;
	
	public static short cntdrop = 0;
	public static double besttrain = Double.NEGATIVE_INFINITY;
	public static double bestscore = Double.NEGATIVE_INFINITY;
	
	public static short ncomp = 1;
	public static short ndim = 4;
	
	public static int tgBase = 3;
	public static double tgRatio = 0.3;
	public static double tgProb = 1e-10;
	public static boolean iomask = false;
	public static double squeezeexp = 0.35;
	
	public static short dim = 2;
	public static short ncomponent = 2;
	public static short maxlength = 7;
	public static double maxrandom = 1;
	public static int randomseed = 0;
	public static int precision = 3;
	public static Random random = new Random(randomseed);
	public static Random rnd4shuffle = new Random(11);
	
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
		public double lr = 0.01;
		@Option(name = "-reg", usage = "using regularization (true) or not (false) (default: false)")
		public boolean reg = false;
		@Option(name = "-clip", usage = "clipping the gradients (true) or not (false) (default: false)")
		public boolean clip = false;
		@Option(name = "-absmax", usage = "threshold for clipping gradients (default: 5.0)")
		public double absmax = 5.0;
		@Option(name = "-wdecay", usage = "weight decay rate (default: 0.02)")
		public double wdecay = 0.02;
		@Option(name = "-l1", usage = "using l1 regularization (true) or l2 regularization (false) (default: true)")
		public boolean l1 = true;
		@Option(name = "-minmw", usage = "minimum mixing weight (default: 1e-6)")
		public double minmw = 1e-15;
		@Option(name = "-epsilon", usage = "a small constant to avoid the division by zero (default: 1e-8)")
		public double epsilon = 1e-8;
		@Option(name = "-choice", usage = "optimization methods: NORMALIZED, SGD, ADAGRAD, RMSPROP, ADADELTA, ADAM (default: ADAM)")
		public OptChoice choice = OptChoice.ADAM;
		@Option(name = "-lambda", usage = "momentum used in the gradient update (default: 0.9)")
		public double lambda = 0.9;
		@Option(name = "-lambda1", usage = "1st. order momentum (default: 0.9)")
		public double lambda1 = 0.9;
		@Option(name = "-lambda2", usage = "2nd. order momentum (default: 0.9)")
		public double lambda2 = 0.999;
		/* optimization-parameter section ends*/
		
		/* grammar-data section begins */
		@Option(name = "-inGrammar", usage = "input: object file of the grammar and lexicon (default: null)")
		public String inGrammar = null;
		@Option(name = "-outGrammar", usage = "output: object file of the grammar (default: null)")
		public String outGrammar = null;
		@Option(name = "-saveGrammar", usage = "save grammar to the object file (true) or not (false) (default: true)")
		public boolean saveGrammar = true;
		@Option(name = "-loadGrammar", usage = "load grammar from the object file (true) or not (false) (default: false)")
		public boolean loadGrammar = false;
		@Option(name = "-nbatchSave", usage = "# of batches after which the grammar is saved (default: 20")
		public short nbatchSave = 20;
		/* grammar-data section ends */
		
		/* parallel configurations section begins */
		@Option(name = "-ntcyker", usage = "# of threads for computing in/out-side scores with binary rules (default: 1)")
		public short ntcyker = 1;
		@Option(name = "-ntbatch", usage = "# of threads for minibatch training (default: 1)")
		public short ntbatch = 1;
		@Option(name = "-ntgrad", usage = "# of threads for gradient calculation (default: 1)")
		public short ntgrad = 1;
		@Option(name = "-nteval", usage = "# of threads for grammar evaluation (default: 1)")
		public short nteval = 1;
		@Option(name = "-nttest", usage = "# of threads for f1 score calculation (default: 1)")
		public short nttest = 1;
		@Option(name = "-pclose", usage = "close all parallel switches (default: false)")
		public boolean pclose = false;
		@Option(name = "-pcyker", usage = "parallizing cyk algorithm (true) or not (false) (default: true)")
		public boolean pcyker = true;
		@Option(name = "-pbatch", usage = "parallizing training in the minibatch (true) or not (false) (default: true)")
		public boolean pbatch = true;
		@Option(name = "-peval", usage = "parallizing evaluation section (true) or not (false) (default: true)")
		public boolean peval = true;
		@Option(name = "-pgrad", usage = "parallizeing gradient calculation (true) or not (false) (default: true)")
		public boolean pgrad = true;
		@Option(name = "-pmode", usage = "parallel mode of gradient evaluation: INVOKE_ALL, COMPLETION_SERVICE, CUSTOMIZED_BLOCK, FORK_JOIN, THREAD_POOL (default: THREAD_POOL)")
		public ParallelMode pmode = ParallelMode.THREAD_POOL;
		@Option(name = "-pverbose", usage = "silent (false) the parallel optimizer or not (true) (default: true)")
		public boolean pverbose = true;
		/* parallel-configurations section ends */
		
		/* training configurations section begins */
		@Option(name = "-iosprune", usage = "when evaluating inside-outside score, avoid adding trivial components if the mixing weights are equal to zero (default: false)")
		public boolean iosprune = false;
		@Option(name = "-sampling", usage = "whether use sampling techniques (true) in evaluating gradients or not (false) (default: false)")
		public boolean sampling = false;
		@Option(name = "-riserate", usage = "# of components allowed to be increased for every 100 more components (default: 2.0)")
		public double riserate = 2.0;
		@Option(name = "-maxnbig", usage = "reserve max first n components of MoG (default: 100)")
		public short maxnbig = 100;
		@Option(name = "-rtratio", usage = "ratio of # of retained components to total number of components (default: 0.2)")
		public double rtratio = 0.2;
		@Option(name = "-hardcut", usage = "just take K first biggest component (true) or not (false) (default: true)")
		public boolean hardcut = true;
		@Option(name = "-expzero", usage = "relative magnitude calculated as a / b = exp(log(a) - log(b)) (default: 1e-6)")
		public double expzero = 1e-6;
		@Option(name = "-bsize", usage = "# of the samples in a batch (default: 128)")
		public short bsize = 128;
		@Option(name = "-nepoch", usage = "# of epoches for training (default: 10)")
		public int nepoch = 10;
		@Option(name = "-maxsample", usage = "maximum sampling time when approximating gradients (default: 3)")
		public int maxsample = 3;
		@Option(name = "-maxslen", usage = "which is used to initialize the size of the chart used in CYK and [inside, outside] score calculation (default: 120)")
		public short maxslen = 120;
		@Option(name = "-nAllowedDrop", usage = "# of allowed iterations in which the validation likelihood drops (default: 6)")
		public short nAllowedDrop = 6;
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
		@Option(name = "-resetw", usage = "reset the mixing weight according to the treebank grammars (default: false)")
		public boolean resetw = false;
		@Option(name = "-resetc", usage = "reset the # of component of the rule weight according to its frequency (default: false) ")
		public boolean resetc = false;
		@Option(name = "-mwfactor", usage = "multiply a factor when reseting the mixint weight to the treebank grammars (default: 1)")
		public double mwfactor = 1.0;
		@Option(name = "-usemasks", usage = "must close this option in MaxRuleParser, use masks from treebank grammars to prune the nonterminals (default: false)")
		public boolean usemasks = false;
		@Option(name = "-tgbase", usage = "minimum number of nonterminals (default: 3)")
		public int tgbase = 3;
		@Option(name = "-tgratio", usage = "tgbase + size * tgratio (default: 0.3)")
		public double tgratio = 0.3;
		@Option(name = "-tgprob", usage = "the tag is pruned if its posterior probability is below this bound (default: 1e-10)")
		public double tgprob = 1e-10;
		@Option(name = "-iomask", usage = "which type of mask we would like to use, false: IO mask, true: posterior mask (default: false)")
		public boolean iomask = false;
		@Option(name = "-sexp", usage = "squeeze ratio in pruning components of inside/outside scores (default: 0.35)")
		public double sexp = 0.35;
		@Option(name = "-pivota", usage = "initialize # of components of the rule weight by its frequency (default: 200)")
		public double pivota = 100;
		@Option(name = "-pivotb", usage = "initialize # of components of the rule weight by its frequency (default: 5000)")
		public double pivotb = 5000;
		@Option(name = "-resetl", usage = "whether reset lexicon rule weights (true) or not (false) (default: false)")
		public boolean resetl = false;
		@Option(name = "-resetp", usage = "whether reset mus of the rule weight (true) or not (false) (default: false)")
		public boolean resetp = false;
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
		@Option(name = "-eonextradev", usage = "evaluating the grammar on the sentences of length less than or equal to [eonlylen + 5] (true) or not (false) (default: true)")
		public boolean eonextradev = true;
		@Option(name = "-ellprune", usage = "applying pruning when evaluating (log-likelihood) the grammar (true) or not (false) (default: false)")
		public boolean ellprune = false;
		@Option(name = "-ellimwrite", usage = "write parse tree to image (true) or not (false) when evaluating ll (default: false)")
		public boolean ellimwrite = false;
		@Option(name = "-epochskipk", usage = "first k epoches in which evaluation is not performed for time saving (default: 3)")
		public short epochskipk = 3;
		@Option(name = "-enbatchdev", usage = "# of batches after which the grammar is evaluated on the development dataset (default: 5)")
		public short enbatchdev = 5;
		/* evaluation section ends */
		
		/* evaluate f1 score section begins */
		@Option(name = "-pf1", usage = "parallelize f1 score calculation (true) or not (false) (default: false)")
		public boolean pf1 = false;
		@Option(name = "-ef1tag", usage = "the flag assigned to a f1 evaluation (default: \"\")")
		public String ef1tag = "";
		@Option(name = "-ef1prune", usage = "applying pruning when evaluating (f1-score) the grammar (true) or not (false) (default: false)")
		public boolean ef1prune = false;
		@Option(name = "-ef1ondev", usage = "evaluating f1-score on the development dataset (true) or not (false) (default: false)")
		public boolean ef1ondev = false;
		@Option(name = "-ef1ontrain", usage = "evaluating f1-score on the training dataset (true) or not (false) (default: false)")
		public boolean ef1ontrain = false;
		@Option(name = "-ef1imwrite", usage = "write parse tree to image (true) or not (false) when evaluating f1 score (default: false)")
		public boolean ef1imwrite = false;
		/* evaluate f1 score section ends */
		
		/* logger configurations section begins */
		@Option(name = "-runtag", usage = "the flag assigned to a run specially (default: lveg)")
		public String runtag = "lveg";
		@Option(name = "-logtype", usage = "file (1) or console (< 0) or file and console (0) (default: -1)")
		public int logtype = -1;
		@Option(name = "-logroot", usage = "log file name (default: log/log)")
		public String logroot = "log/";
		@Option(name = "-nbatch", usage = "# of batches after which the grammar evaluation is conducted (default: 1)")
		public short nbatch = 1;
		@Option(name = "-precision", usage = "precision of the output decimals (default: 3)")
		public int precision = 3;
		@Option(name = "-rndomseed", usage = "seed for random number generator (default: 0)")
		public int rndomseed = 111;
		/* logger-configurations section ends */
		
		/* file prefix section begins */
		@Option(name = "-dgradnbatch", usage = "# of batches after which the gradients are recorded (default: 1)")
		public short dgradnbatch = 1;
		/* file prefix section ends */
		
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
//		public Binarization binarization = Binarization.LEFT;
		
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
	
	
	public static void initialize(Options opts, boolean test) {
		if (opts.outGrammar == null && !test) {
			throw new IllegalArgumentException("Output file is required.");
		}
		// make directories
		String logfile = null;
		FunUtil.mkdir(opts.datadir);
		subdatadir = opts.datadir + "/" + opts.runtag + "/";
		if (!test) {
			sublogroot = opts.logroot + "/" + opts.runtag + "/";
			logfile = opts.logroot + "/" + opts.runtag;
			FunUtil.mkdir(subdatadir);
			if (opts.ellimwrite) { FunUtil.mkdir(sublogroot); }
		} else {
			String suffix = opts.ef1tag.equals("") ? opts.runtag + "_f1" : opts.runtag + "_f1_" + opts.ef1tag;
			sublogroot = opts.logroot + "/" + suffix + "/";
			logfile = opts.logroot + "/" + suffix;
			if (opts.ef1imwrite) { FunUtil.mkdir(sublogroot); }
		}
		if (opts.logtype == 0) {
			Recorder.logger = logUtil.getBothLogger(logfile);
		} else if (opts.logtype == 1) {
			Recorder.logger = logUtil.getFileLogger(logfile);
		} else {
			Recorder.logger = logUtil.getConsoleLogger();
		}
		
		logger.info("Random number generator seeded at " + opts.rndomseed + ".\n");
		
		if (opts.pclose) {
			opts.pbatch = false;
			opts.peval = false;
			opts.pgrad = false;
		} // ease the parameter tuning
		
		ndim = opts.dim;
		minmw = opts.minmw;
		precision = opts.precision;
		maxrandom = opts.maxrandom;
		randomseed = opts.rndomseed;
		ncomp = opts.ncomponent;
		random = new Random(randomseed);
		tgBase = opts.tgbase;
		tgRatio = opts.tgratio;
		tgProb = Math.log(opts.tgprob); // in logarithmic form
		iomask = opts.iomask;
		squeezeexp = opts.sexp;
		Params.config(opts);
	}
	
	public static  Map<String, List<List<TaggedWord>>> loadData(Numberer wrapper, Options opts) throws Exception {
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, List<List<TaggedWord>>> data = new HashMap<String, List<List<TaggedWord>>>(3, 1);
		List<List<TaggedWord>> train = CoNLLFileReader.read(opts.datadir + opts.train);
		List<List<TaggedWord>> test = CoNLLFileReader.read(opts.datadir + opts.test);
		List<List<TaggedWord>> dev = CoNLLFileReader.read(opts.datadir + opts.dev);
		
		numberer.number(Pair.ENDING); // ending of the sequence
		numberer.number(Pair.LEADING); // starting of the sequence
		labelTags(train, numberer, opts, true);
		labelTags(test, numberer, opts, false);
		labelTags(dev, numberer, opts, false);
		debugNumbererTag(numberer, opts); // DEBUG
		
		data.put(ID_TRAIN, train);
		data.put(ID_TEST, test);
		data.put(ID_DEV, dev);
		return data;
	}
	
	public static void labelTags(List<List<TaggedWord>> sequences, Numberer numberer, Options opts, boolean replaceRareWords) {
		for (List<TaggedWord> sequence : sequences) {
			for (TaggedWord word : sequence) {
				int idx = numberer.number(word.tag());
				word.setTagIdx(idx);
			}
		}
		logger.trace("--->There are " + sequences.size() + " trees in the corpus.\n");
		if (replaceRareWords) {
			logger.trace("Replacing words appearing less than 20 times with their signature.\n");
			LVeGCorpus.replaceRareWords(sequences, new TagWPair(), opts.rareThreshold);
		}
	}
	
	protected static void debugNumbererTag(Numberer numberer, Options opts) {
		if (opts.verbose) {
			for (int i = 0; i < numberer.size(); i++) {
				logger.trace("Tag " + i + "\t" +  (String) numberer.object(i) + "\n");
			}
		}
		logger.trace("There are " + numberer.size() + " observed tags.\n");
	}
}
