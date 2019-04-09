package edu.shanghaitech.ai.nlp.lvet;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.LVeGCorpus;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lvet.impl.TagTPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.io.CoNLLFileReader;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.optimization.Optimizer.OptChoice;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer.ParallelMode;
import edu.shanghaitech.ai.nlp.syntax.State;
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
	
	protected static void resetRuleWeight(TagTPair grammar, TagWPair lexicon, Numberer numberer, double factor, Options opts) {
		int ntag = numberer.size(), count, ncomp;
		List<GrammarRule> gUruleWithP, lUruleWithP;
		double prob, rulecnt, logprob;
		int a = 0, b = 0, c = 0, d = 0, e = 0;
		GaussianMixture ruleW;
		boolean resetc;
		
		// probabilities of lexicon rules
		// since LHS tags of lexicon rules and CNF rules do not overlap
		// we do not need to specifically initialize the probabilities of lexicon rules
		for (int i = 0; i < ntag; i++) {
			count = 0;
			lUruleWithP = lexicon.getEdgeWithP(i);
			for (GrammarRule rule : lUruleWithP) {
				count += rule.getWeight().getBias();
			}
			short increment = 1;
			for (GrammarRule rule : lUruleWithP) {
				rulecnt = rule.getWeight().getBias();
				prob = rulecnt / count;
				ruleW = rule.getWeight();
				
				if (!opts.resetl) {
					d++;
					resetc = false;
				} else {
					resetc = opts.resetc;
				}
				
				if (resetc && rulecnt > opts.pivota) {
					c++;
//					rule.addWeightComponent(rule.type, increment, (short) -1);
					ruleW = rule.getWeight();
					ruleW.setBias(rulecnt);
				} else {
					a++;
				}
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
		
		// for nonterminal rules
		for (int i = 0; i < ntag; i++) {
			count = 0;
			gUruleWithP = grammar.getEdgeWithP(i);
			for (GrammarRule rule : gUruleWithP) {
				count += rule.getWeight().getBias();
			}
			
			short increment = 3;
			for (GrammarRule rule : gUruleWithP) {
				rulecnt = rule.getWeight().getBias();
				prob = rulecnt / count;
				ruleW = rule.getWeight();
				
				if (opts.resetc && rulecnt > opts.pivota) {
					b++;
//					rule.addWeightComponent(rule.type, increment, (short) -1);
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
		logger.debug("# of 1-comp: " + a + ", # of 2-comps: " + c + ", # of 4-comps: " + b +
				", skip # of lexicon rules: " + d +
				", # of larger than " + opts.pivota + " is " + e + "\n");
		
		if (opts.resetp) {
//			resetRuleWeightParams(grammar, lexicon, numberer, opts);
		}
	}
	
	public static void filterTrees(Options opts, List<List<TaggedWord>> stateTreeList, List<List<TaggedWord>> container, Numberer numberer, boolean istrain) {
		int cnt = 0;
		if (container != null) { container.clear(); }
		int maxlen = istrain ? /*1*/opts.eonlylen : (opts.eonextradev ? opts.eonlylen + 5 : opts.eonlylen);
		for (List<TaggedWord> tree : stateTreeList) {
			if (opts.eonlylen > 0) {
				if (tree.size() > maxlen) { continue; }
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
}
