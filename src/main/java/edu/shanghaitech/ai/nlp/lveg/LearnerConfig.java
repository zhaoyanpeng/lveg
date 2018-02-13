package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.PCFGLA.TreeAnnotations;
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.data.LVeGCorpus;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.CorpusFile;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.Optimizer.OptChoice;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer.ParallelMode;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.ObjectPool;
import edu.shanghaitech.ai.nlp.util.Option;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class LearnerConfig extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5745194882841709365L;
	public final static String ID_TRAIN = "train";
	public final static String ID_TEST = "test";
	public final static String ID_DEV = "dev";
	
	protected static String subdatadir;
	protected static String sublogroot;
	
	public final static String KEY_TAG_SET = "tag";
	public static double minmw = 1e-6;
	
	public static short cntdrop = 0;
	public static double besttrain = Double.NEGATIVE_INFINITY;
	public static double bestscore = Double.NEGATIVE_INFINITY;
	
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
	protected static PriorityQueue<Tree<State>> sorter;
	
	public static ObjectPool<Short, GaussianMixture> mogPool;
	public static ObjectPool<Short, GaussianDistribution> gaussPool;
	
	public static Map<Short, Short> refSubTypes = null;
	public static String reference = "ROOT=1 S^g=22 @S^g=29 PP^g=40 IN=34 NP^g=57 @NP^g=61 DT=21 NNP=53 CD=27 NN=59 ``=1 "
			+ "''=1 POS=2 PRN^g=5 @PRN^g=8 -LRB-=2 VBN=35 NNS=51 VP^g=47 @VP^g=47 VBP=19 ,=1 CC=7 -RRB-=2 VBD=28 ADVP^g=28 "
			+ "RB=41 TO=1 .=3 VBZ=18 NNPS=6 SBAR^g=18 PRP=3 PRP$=5 VB=31 ADJP^g=26 JJ=56 QP^g=13 @PP^g=11 MD=3 UCP^g=2 @UCP^g=3 "
			+ "VBG=26 @SBAR^g=5 WHNP^g=4 @ADVP^g=4 RBR=4 :=5 SINV^g=2 @SINV^g=9 WP=2 WDT=2 JJR=5 PDT=2 RBS=1 @QP^g=14 @ADJP^g=10 "
			+ "FRAG^g=2 NAC^g=2 @NAC^g=3 WHADVP^g=2 WRB=2 JJS=6 $=1 PRT^g=2 RP=1 NX^g=3 @FRAG^g=3 WHPP^g=1 FW=1 SQ^g=2 @SQ^g=2 @NX^g=3 "
			+ "SBARQ^g=1 @SBARQ^g=2 EX=2 CONJP^g=1 WHADJP^g=1 SYM=1 #=1 @CONJP^g=1 LS=1 @WHADJP^g=1 INTJ^g=1 UH=1 WP$=1 X^g=1 @WHNP^g=1 "
			+ "RRC^g=1 @WHADVP^g=1 LST^g=1 @X^g=1 @INTJ^g=1 PRT|ADVP^g=50 @PRT^g=1 @LST^g=1 @RRC^g=1";

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
		@Option(name = "-efraction", usage = "evaluate grammars on a fraction of samples")
		public double efraction = 1.0;
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
			logger = logUtil.getBothLogger(logfile);
		} else if (opts.logtype == 1) {
			logger = logUtil.getFileLogger(logfile);
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
		precision = opts.precision;
		maxrandom = opts.maxrandom;
		randomseed = opts.rndomseed;
		ncomponent = opts.ncomponent;
		random = new Random(randomseed);
		tgBase = opts.tgbase;
		tgRatio = opts.tgratio;
		tgProb = Math.log(opts.tgprob); // in logarithmic form
		iomask = opts.iomask;
		squeezeexp = opts.sexp;
		Params.config(opts);
	}
	
	public static  Map<String, StateTreeList> loadData(Numberer wrapper, Options opts) {
		Numberer numberer = null;
		StateTreeList trainTrees, testTrees, devTrees;
		Map<String, StateTreeList> trees = new HashMap<>(3, 1);
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
			/*
			logger.trace("\n----------train---\n");
			debugTreeType(trainString, trainTrees, numberer);
			logger.trace("\n----------test ---\n");
			debugTreeType(testString, testTrees, numberer);
			logger.trace("\n----------dev  ---\n");
			debugTreeType(devString, devTrees, numberer);
			System.exit(0);
			*/
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
	
	protected static void makeSubTypes(Numberer numberer) {
		int size = numberer.size();
		String entrySep = " ", kvSep = "=";
		refSubTypes = new HashMap<>(size, 1);
		String[] entries = reference.split(entrySep);
		for (String entry : entries) {
			if (entry.length() > 1 && entry.contains(kvSep)) {
				String[] keyValue = entry.split(kvSep);
				refSubTypes.put((short) numberer.number(keyValue[0]), Short.valueOf(keyValue[1]));
			}
		}
	}
	
	protected static List<Tree<String>> loadStringTree(String path, Options opts) {
		logger.trace("--->Loading trees from " + path + " and using languange " + opts.treebank + "\n");
		boolean onlyTest = false, manualAnnotation = false;
		Corpus corpus = new Corpus(
				path, opts.treebank, opts.trainingFraction, onlyTest,
				opts.skipSection, opts.skipBilingual, opts.keepFunctionLabel);
		/*
		List<Tree<String>> trees = corpus.getDevTestingTrees(); // the same as the Corpus.getTrainTrees()
		for (int i = 0; i < 10; i++) {
			logger.trace(i + "\tstr   tree: " + trees.get(i) + "\n");
		}
		*/
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
	
	public static void debugTreeType(List<Tree<String>> strTrees, StateTreeList stateTrees, Numberer numberer) {
		for (int i = 0; i < 10; i++) {
			logger.trace(i + "\tstr   tree: " + TreeAnnotations.unAnnotateTree(strTrees.get(i), false) + "\n");
			logger.trace(i + "\tstate tree: " + strTree2stateTree(stateTrees.get(i), numberer) + "\n");
		}
	}
	
	public static Tree<String> strTree2stateTree(Tree<State> tree, Numberer numberer) {
		Tree<String> strTree = StateTreeList.stateTreeToStringTree(tree, numberer);
		strTree = TreeAnnotations.unAnnotateTree(strTree, false);
		return strTree;
	}
	
	public static String readFile(String path, Charset encoding) throws IOException {
		  byte[] encoded = Files.readAllBytes(Paths.get(path));
		  return new String(encoded, encoding);
	}
	
	public static void filterTrees(Options opts, StateTreeList stateTreeList, List<Tree<State>> container, Numberer numberer, boolean istrain) {
		int cnt = 0;
		if (container != null) { container.clear(); }
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
			container.add(tree);
			/*
			Tree<String> strTree = strTree2stateTree(Tree<State> tree, Numberer numberer)
			logger.trace((cnt - 1) + "\t" + strTree + "\n");
			*/
			// logger.trace((cnt - 1) + "\t" + FunUtil.debugTree(tree, false, (short) -1, numberer, true) + "\n");
		}
		// sort the samples by descending sentence length
		sorter.clear();
		sorter.addAll(container);
		container.clear();
		while (!sorter.isEmpty()) {
			container.add(sorter.poll());
		}
	}
	
	protected static Comparator<Tree<State>> wcomparator = new Comparator<Tree<State>>() {
		@Override
		public int compare(Tree<State> o1, Tree<State> o2) {
			return o2.getYield().size() - o1.getYield().size();
		}
	};
	
	
	protected static ArrayList<Tree<State>> sampleTrees(List<Tree<State>> trees, Options opts) {
		ArrayList<Tree<State>> newList = new ArrayList<>();
		for (Tree<State> tree : trees) {
			List<State> sentence = tree.getYield();
			int sentenceLength = sentence.size();
			if (sentenceLength > opts.eonlylen)
				continue;
			newList.add(tree);			
		}
		
		int total = newList.size();
		int nneed = (int) Math.ceil(total * opts.efraction);
		Random rnd = new Random(11);
		Set<Integer> idxes = new LinkedHashSet<>();
		while (idxes.size() < nneed) {
			Integer next = rnd.nextInt(total);
			idxes.add(next);
		}
		
		ArrayList<Tree<State>> filterList = new ArrayList<>();
		for (Integer idx : idxes) {
			filterList.add(newList.get(idx));
		}
		
		logger.debug("\n\n" + "total: " + total + ", nneed: " + nneed + ", fract: " + opts.efraction + ", " + trees.size());
		return filterList;
	}
	
	
	protected static void resetRuleWeight(LVeGGrammar grammar, LVeGLexicon lexicon, Numberer numberer, double factor, Options opts) {
		int ntag = numberer.size(), nrule, count, ncomp;
		List<GrammarRule> gUruleWithP, gBruleWithP, lUruleWithP;
		double prob, rulecnt, logprob;
		int a = 0, b = 0, c = 0, d = 0, e = 0;
		GaussianMixture ruleW;
		boolean resetc;
		/*
		// probabilities of lexicon rules
		// since LHS tags of lexicon rules and CNF rules do not overlap
		// we do not need to specifically initialize the probabilities of lexicon rules
		for (int i = 0; i < ntag; i++) {
			count = 0;
			lUruleWithP = lexicon.getURuleWithP(i);
			for (GrammarRule rule : lUruleWithP) {
				count += rule.getWeight().getBias();
			}
			for (GrammarRule rule : lUruleWithP) {
				ruleW = rule.getWeight();
				prob = Math.log(ruleW.getBias() / count * factor);
				ruleW.setWeight(0, prob);
				ruleW.setProb(prob);
			}
			logger.debug(i + "\t: " + count + "\n");
		}
		*/
		// for nonterminal rules
		for (int i = 0; i < ntag; i++) {
			count = 0;
			gUruleWithP = grammar.getURuleWithP(i);
			gBruleWithP = grammar.getBRuleWithP(i);
			lUruleWithP = lexicon.getURuleWithP(i);
			nrule = gUruleWithP.size() + gBruleWithP.size() + lUruleWithP.size();
			List<GrammarRule> rules = new ArrayList<>(nrule + 5);
			rules.addAll(gUruleWithP);
			rules.addAll(gBruleWithP);
			rules.addAll(lUruleWithP);
			for (GrammarRule rule : rules) {
				count += rule.getWeight().getBias();
			}
			
			for (GrammarRule rule : rules) {
				rulecnt = rule.getWeight().getBias();
				prob = rulecnt / count;
				ruleW = rule.getWeight();
				ncomp = opts.ncomponent;
				
				if (!opts.resetl && rule.type == RuleType.LHSPACE) {
					d++;
					resetc = false;
				} else {
					resetc = opts.resetc;
				}
				
				if (resetc && rulecnt > opts.pivota) {
					RuleType type = rule.getType();
					int increment = 0;
					
					if (rulecnt < opts.pivotb) {
						ncomp += 1;
						b++;
						increment = type == RuleType.LHSPACE ? 2 : 3;
					} else {
						ncomp += 2;
						c++;
						increment = type == RuleType.LHSPACE ? 2 : 3;
					}
//					rule.addWeightComponent(type, increment, (short) -1);
					
					ruleW = rule.getWeight();
					ruleW.setBias(rulecnt);
				} else { a++; }
				logprob = Math.log(prob);
				ruleW.setProb(logprob);
				
				ncomp = ruleW.ncomponent();
				prob = prob * factor /*/ ncomp*/;
				logprob = Math.log(prob);
				for (int icomp = 0; icomp < ncomp; icomp++) {
					ruleW.setWeight(icomp, logprob);
				}
			}
//			logger.debug(i + "\t: " + count + "\n");
		}
		logger.debug("# of 1-comp: " + a + ", # of 2-comps: " + b + ", # of 3-comps: " + c +
				", skip # of lexicon rules: " + d +
				", # of larger than " + opts.pivota + " is " + e + "\n");
		
		if (opts.resetp) {
//			resetRuleWeightParams(grammar, lexicon, numberer, opts);
		}
	}

/*	
	protected static void resetRuleWeightParams(LVeGGrammar grammar, LVeGLexicon lexicon, Numberer numberer, Options opts) {
		int ntag = numberer.size(), nrule;
		List<GrammarRule> gUruleWithP, gBruleWithP, lUruleWithP;
		// filters
		Set<Short> lexiconTags = new HashSet<>(ntag);
		for (int i = 0; i < ntag; i++) {
			lUruleWithP = lexicon.getURuleWithP(i);
			for (GrammarRule rule : lUruleWithP) {
				lexiconTags.add(rule.lhs);
				break;
			}
		}
		logger.trace("---lexicon tags: # is " + lexiconTags.size() + "; " + lexiconTags + "\n");
		
		// params
		int ncomp = 3;
		List[][] tmus = new List[ntag][ncomp];
		for (short itag = 0; itag < ntag; itag++) {
			if (lexiconTags.contains(itag)) { 
				logger.trace("skip lexicon tag: " + itag + "\n");
				continue; 	
			}
			for (int icomp = 0; icomp < ncomp; icomp++) {
				tmus[itag][icomp] = new ArrayList<>(opts.dim);
				for (short idim = 0; idim < opts.dim; idim++) {
					double rndn = (random.nextDouble() - opts.nmratio) * opts.maxmu;
					tmus[itag][icomp].add(rndn);
				}
			}
		}
		
		// reset
		for (short itag = 0; itag < ntag; itag++) {
			if (lexiconTags.contains(itag)) { continue; }
			gUruleWithP = grammar.getURuleWithP(itag);
			gBruleWithP = grammar.getBRuleWithP(itag);
			nrule = gUruleWithP.size() + gBruleWithP.size();
			List<GrammarRule> rules = new ArrayList<>(nrule + 5);
			rules.addAll(gUruleWithP);
			rules.addAll(gBruleWithP);
			
			for (GrammarRule rule : rules) {
				resetRuleWeightParams(rule, tmus);
			}
		}
	}
*/
/*	
	private static void resetRuleWeightParams(GrammarRule rule, List[][] tmus) {
		GaussianMixture ruleW = rule.weight;
		int ncomp = ruleW.ncomponent();
		RuleType type = rule.getType();
		
		switch (type) {
		case RHSPACE: { // rules for the root since it does not have subtypes
			UnaryGrammarRule urule = (UnaryGrammarRule) rule;
			List[] newmus = tmus[urule.rhs];
			for (short i = 0; i < ncomp; i++) {
				Component comp = ruleW.getComponent(i);
				if (newmus[i] != null) {
					GaussianDistribution gd = comp.squeeze(RuleUnit.C);
					List<Double> oldmus = gd.getMus();
					assert(oldmus.size() == newmus[i].size());
					oldmus.clear();
					oldmus.addAll(newmus[i]);
				}
			}
			break;
		}
		case LHSPACE: { // rules in the preterminal layer (discarded)
			logger.error("\n---not supposed to be reached\n");
			break;
		}
		case LRURULE: { // general unary rules 
			UnaryGrammarRule urule = (UnaryGrammarRule) rule;
			List[] pnewmus = tmus[urule.lhs];
			List[] cnewmus = tmus[urule.rhs];
			for (short i = 0; i < ncomp; i++) {
				Component comp = ruleW.getComponent(i);
				GaussianDistribution gd = comp.squeeze(RuleUnit.P);
				List<Double> oldmus = gd.getMus();
				assert(oldmus.size() == pnewmus[i].size());
				oldmus.clear();
				oldmus.addAll(pnewmus[i]);
				
				if (cnewmus[i] != null) {
					gd = comp.squeeze(RuleUnit.UC);
					oldmus = gd.getMus();
					assert(oldmus.size() == cnewmus[i].size());
					oldmus.clear();
					oldmus.addAll(cnewmus[i]);
				}
			}
			break;
		}
		case LRBRULE: { // general binary rules
			BinaryGrammarRule brule = (BinaryGrammarRule) rule;
			List[] pnewmus = tmus[brule.lhs];
			List[] lnewmus = tmus[brule.lchild];
			List[] rnewmus = tmus[brule.rchild];
			for (short i = 0; i < ncomp; i++) {
				Component comp = ruleW.getComponent(i);
				GaussianDistribution gd = comp.squeeze(RuleUnit.P);
				List<Double> oldmus = gd.getMus();
				assert(oldmus.size() == pnewmus[i].size());
				oldmus.clear();
				oldmus.addAll(pnewmus[i]);
				
				if (lnewmus[i] != null) {
					gd = comp.squeeze(RuleUnit.LC);
					oldmus = gd.getMus();
					assert(oldmus.size() == lnewmus[i].size());
					oldmus.clear();
					oldmus.addAll(lnewmus[i]);
				}
				
				if (rnewmus[i] != null) {
					gd = comp.squeeze(RuleUnit.RC);
					oldmus = gd.getMus();
					assert(oldmus.size() == rnewmus[i].size());
					oldmus.clear();
					oldmus.addAll(rnewmus[i]);
				}
			}
			break;
		}
		default:
			throw new RuntimeException("Not consistent with any grammar rule type. Type: " + type);
		}
	}
*/	
}
