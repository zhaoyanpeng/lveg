package edu.shanghaitech.ai.nlp.lvet;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.shanghaitech.ai.nlp.data.LVeGCorpus;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
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
	protected static PriorityQueue<List<TaggedWord>> sorter;
	
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
		CoNLLFileReader.config(3, 1);
	}
	
	public static  Map<String, List<List<TaggedWord>>> loadData(Numberer wrapper, Options opts) throws Exception {
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		Map<String, List<List<TaggedWord>>> data = new HashMap<String, List<List<TaggedWord>>>(3, 1);
		List<List<TaggedWord>> train = CoNLLFileReader.read(opts.datadir + opts.train);
		List<List<TaggedWord>> test = CoNLLFileReader.read(opts.datadir + opts.test);
		List<List<TaggedWord>> dev = CoNLLFileReader.read(opts.datadir + opts.dev);
		
		numberer.number(Pair.LEADING); // starting of the sequence
		numberer.number(Pair.ENDING); // ending of the sequence
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
			logger.trace("Replacing words appearing less than " + opts.rareThreshold + " times with their signature.\n");
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
	
	protected static Comparator<List<TaggedWord>> wcomparator = new Comparator<List<TaggedWord>>() {
		@Override
		public int compare(List<TaggedWord> o1, List<TaggedWord> o2) {
			return o2.size() - o1.size();
		}
	};
}
