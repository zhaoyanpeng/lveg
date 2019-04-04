package edu.shanghaitech.ai.nlp.lvet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.shanghaitech.ai.nlp.data.ObjectFileManager.TaggerFile;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lvet.impl.LVeTTagger;
import edu.shanghaitech.ai.nlp.lvet.impl.MaxRuleTagger;
import edu.shanghaitech.ai.nlp.lvet.impl.TagTPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.impl.Valuator;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.optimization.ParallelOptimizer;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class LVeTTrainer extends LVeTConfig {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3889795373391559958L;
	
	protected static List<List<TaggedWord>> ftrainTrees;
	
	protected static List<List<TaggedWord>> trainTrees;
	protected static List<List<TaggedWord>> testTrees;
	protected static List<List<TaggedWord>> devTrees;
	
	protected static Optimizer toptimizer;
	protected static Optimizer woptimizer;
	
	protected static TagTPair ttpairs;
	protected static TagWPair twpairs;
	
	protected static Valuator<?, ?> valuator;
	protected static LVeTTagger<?, ?> lvetTagger;
	protected static MaxRuleTagger<?, ?> mrTagger;
	
	protected static ThreadPool mvaluator;
	protected static ThreadPool trainer;
	
	protected static Options opts;
	
	protected static double bestTrainLL = 0.0;
	protected static double bestDevLL = 0.0;
	protected static int nbatch = 0;
	
	
	public static void main(String[] args) throws Exception {
		// main thread
		Thread main = Thread.currentThread();
		main.setName(ThreadPool.MAIN_THREAD);
		// parsing parameters
		String fparams = args[0];
		try {
			args = FunUtil.readFile(fparams, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		opts = (Options) optionParser.parse(args, true);
		// configurations
		initialize(opts, false); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		
		Numberer wrapper = new Numberer();
		Map<String, List<List<TaggedWord>>> sequences = loadData(wrapper, opts);
		// training
		long startTime = System.currentTimeMillis();
		train(sequences, wrapper);
		long endTime = System.currentTimeMillis();
		logger.trace("[total time consumed by LVeT learner] " + (endTime - startTime) / 1000.0 + "\n");
	}
	
	private static void train(Map<String, List<List<TaggedWord>>> trees, Numberer wrapper) throws Exception {
		trainTrees = trees.get(ID_TRAIN);
		testTrees = trees.get(ID_TEST);
		devTrees = trees.get(ID_DEV);
		
		int iepoch = 0;
		Numberer numberer = wrapper.getGlobalNumberer(KEY_TAG_SET);
		
		ttpairs = new TagTPair(numberer, -1);
		twpairs = new TagWPair(numberer, -1);
		
		/* to ease the parameters tuning */
		GaussianMixture.config(opts.maxnbig, opts.expzero, opts.maxmw, opts.ncomponent, 
				opts.nwratio, opts.riserate, opts.rtratio, opts.hardcut, random, null);
		GaussianDistribution.config(opts.maxmu, opts.maxvar, opts.dim, opts.nmratio, opts.nvratio, random, null);
		Optimizer.config(opts.choice, random, opts.maxsample, opts.bsize, opts.minmw, opts.sampling); // FIXME no errors, just alert you...
		
		boolean away = false;
		if (opts.loadGrammar && opts.inGrammar != null && away) {
			logger.trace("--->Loading grammars from \'" + subdatadir + opts.inGrammar + "\'...\n");
			TaggerFile gfile = (TaggerFile) TaggerFile.load(subdatadir + opts.inGrammar);
			ttpairs = gfile.getGrammar();
			twpairs = gfile.getLexicon();
			toptimizer = ttpairs.getOptimizer();
			woptimizer = twpairs.getOptimizer();
			
			String regx = ".*?(\\d+)";
			Pattern pat = Pattern.compile(regx);
			Matcher matcher = pat.matcher(opts.inGrammar);
			if (matcher.find()) { // training continued starting from the input grammar
				iepoch = Integer.valueOf(matcher.group(1).trim()) + 1;
			}
			/*
			logger.trace(grammar);
			logger.trace(lexicon);
			*/
		} else {
			toptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			woptimizer = new ParallelOptimizer(opts.ntgrad, opts.pgrad, opts.pmode, opts.pverbose);
			ttpairs.setOptimizer(toptimizer);
			twpairs.setOptimizer(woptimizer);
			logger.trace("--->Tallying trees...\n");
			for (List<TaggedWord> tree : trainTrees) {
				ttpairs.tallyTaggedWords(tree);
				twpairs.tallyTaggedWords(tree);
				
			}
			logger.trace("\n--->Going through the training set is over...");
			
			ttpairs.postInitialize();
			twpairs.postInitialize();
			logger.trace("post-initializing is over.\n");
			
			// reset the rule weight
//			if (opts.resetw || opts.usemasks) {
//				logger.trace("--->Reset rule weights according to treebank grammars...\n");
//				resetRuleWeight(grammar, lexicon, numberer, opts.mwfactor, opts);
//			}
			
			ttpairs.initializeOptimizer();
			twpairs.initializeOptimizer();
			logger.trace("\n--->Initializing optimizer is over...\n");
		}
		
		logger.trace(ttpairs);
		logger.trace(twpairs);
		System.exit(0);
		
		twpairs.labelSequences(trainTrees); // FIXME no errors, just alert you to pay attention to it 
		twpairs.labelSequences(testTrees); // save the search time cost by finding a specific tag-word
		twpairs.labelSequences(devTrees); // pair in in Lexicon.score(...)
		
		mrTagger = new MaxRuleTagger<List<TaggedWord>, List<String>>(ttpairs, twpairs, opts.maxslen, opts.ef1prune);
		valuator = new Valuator<List<TaggedWord>, Double>(ttpairs, twpairs, opts.maxslen, opts.ellprune);
		mvaluator = new ThreadPool(valuator, opts.nteval);
		
		double ll = Double.NEGATIVE_INFINITY;
		
		// initial likelihood of the training dataset
		logger.trace("\n-------ll of the training data initially is... ");
		long beginTime = System.currentTimeMillis();
		if (opts.eontrain) {
			ll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
			// ll = serialLL(opts, valuator, trainTrees, numberer, true);
		}
		long endTime = System.currentTimeMillis();
		logger.trace("------->" + ll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
		
		logger.info("\n---SGD CONFIG---\n[parallel: batch-" + opts.pbatch + ", grad-" + 
				opts.pgrad + ", eval-" + opts.peval + "] " + Params.toString(false) + "\n");
		
		ftrainTrees = new ArrayList<List<TaggedWord>>(trainTrees.size());
		List<Integer> idxes = new ArrayList<Integer>();
		int idx = 0, cnt = 0;
		for (List<TaggedWord> tree : trainTrees) {
			if (tree.size() <= opts.eonlylen) {
				tree.get(0).signIdx = cnt;
				ftrainTrees.add(tree);
				idxes.add(idx);
				cnt += 1;
			}
			idx += 1;
		}
		logger.trace("--only train on " + ftrainTrees.size() + " sentences.\n");
		
		lvetTagger = new LVeTTagger<List<TaggedWord>, List<Double>>(ttpairs, twpairs, opts.maxslen, opts.iosprune);
		trainer = new ThreadPool(lvetTagger, opts.ntbatch);
		
		sorter = new PriorityQueue<>(opts.bsize + 5, wcomparator);
		
		if (opts.pbatch) {
			parallelInBatch(numberer, iepoch);
		} else {
			serialInBatch(numberer, iepoch);
		}
		// kill threads
		ttpairs.shutdown();
		twpairs.shutdown();
		trainer.shutdown();
		mvaluator.shutdown();
	}
	
	protected static void debugrad(boolean debug) {
		toptimizer.debug(null, debug);
	}
	
	protected static void printGrammars() {
		logger.trace(ttpairs);
		logger.trace(twpairs);
		logger.info("\n-------saving the incorrect grammar file...");
		TaggerFile gfile = new TaggerFile(ttpairs, twpairs);
		String filename = subdatadir + opts.outGrammar + "_error.gr";
		if (gfile.save(filename)) {
			logger.info("to \'" + filename + "\' successfully.\n");
		} else {
			logger.info("to \'" + filename + "\' unsuccessfully.\n");
		}
		logger.trace("---something wrong");
		System.exit(0);
	}
	
	protected static void jointrainer(short nfailed) {
		while (!trainer.isDone()) {
			while (trainer.hasNext()) {
				List<Double> score = (List<Double>) trainer.getNext();
				if (Double.isFinite(score.get(0)) || Double.isFinite(score.get(1))) {
					logger.trace("\n~~~score: " + FunUtil.double2str(score, precision, -1, false, true) + "\n");
				} else {
					nfailed++;
					printGrammars();
				}
			}
		}
	}
	
	protected static int getBatch(int ibegin, boolean shuffle, List<List<TaggedWord>> batch) {
		if (batch != null) { batch.clear(); } 
		if (shuffle) { 
			logger.trace("\n===shuffle training samples...\n");
			Collections.shuffle(ftrainTrees, rnd4shuffle); 
		}
		int iend = ibegin + opts.bsize, nsample = ftrainTrees.size(), diff;
		iend = (diff = (iend - nsample)) >= 0 ? nsample : iend;
		for (int i = ibegin; i < iend; i++) {
			batch.add(ftrainTrees.get(i));
		}
		if (diff >= 0) {
			for (int i = 0; i < diff; i++) {
				batch.add(ftrainTrees.get(i));
			}
			iend = diff;
		}
		// sort the samples by descending sentence length
		
		sorter.clear();
		sorter.addAll(batch);
		batch.clear();
		while (!sorter.isEmpty()) {
			batch.add(sorter.poll());
		}
		
		/*
		for (List<TaggedWord> tree : batch) {
			System.out.println(tree.getYield().size() + "\t" + ibegin + "\t" + diff + "\t" + nsample);
		}
		*/
		return iend;
	}
	
	public static void parallelInBatch(Numberer numberer, int iepoch) throws Exception {
		long bTime, eTime, epochBTime, batchBTime, batchETime;
		int iprebeg, ibegin = 0, nfailed, ibatch = 0;
		List<Double> trllist = new ArrayList<>();
		List<Double> dellist = new ArrayList<>();
		List<Double> scoresOfST = new ArrayList<>(3);
		List<List<TaggedWord>> batch = new ArrayList<>(opts.bsize + 5);
		boolean shuffle = true;
		do {
			logger.trace("\n\n-------epoch " + iepoch + " begins-------\n\n");
			boolean exit = false;
			epochBTime = System.currentTimeMillis();
			while (true) {
				ibatch++;
				if ((iepoch > opts.epochskipk - 1) && (opts.nbatch == 0 || ((ibatch % opts.nbatch) == 0))) {
					exit = opts.nbatch == 0 ? true : peep(ibatch, iepoch, numberer, trllist, dellist, false);
					if (exit) { break; }
				}
				nfailed = 0;
				iprebeg = ibegin;
				ibegin = getBatch(ibegin, shuffle, batch);
				batchBTime = System.currentTimeMillis();
				for (List<TaggedWord> tree : batch) {
					trainer.execute(tree);
					while (trainer.hasNext()) {
						List<Double> score = (List<Double>) trainer.getNext();
						if (Double.isFinite(score.get(0)) || Double.isFinite(score.get(1))) {
							logger.trace("\n~~~score: " + FunUtil.double2str(score, precision, -1, false, true) + "\n");
						} else {
							nfailed++;
							printGrammars();
						}
					}
				}
				jointrainer((short) nfailed); // block the main thread until get all the feedbacks of submitted tasks
				batchETime = System.currentTimeMillis();
				
				// apply gradient descent
				logger.trace("+++Apply gradient descent for the batch " + ibatch + "... ");
				bTime = System.currentTimeMillis();
				
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(true); }
				ttpairs.applyGradientDescent(scoresOfST);
				twpairs.applyGradientDescent(scoresOfST);
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(false); }
				
				eTime = System.currentTimeMillis();
				logger.trace((eTime - bTime) / 1000.0 + "... batch time: " + (batchETime - batchBTime) / 1000.0 + "\n");
				
				if (ibegin < iprebeg) {
					shuffle = true;
					break; // epoch ends
				} else {
					shuffle = false;
				}
			}
			if (exit) { // 
				logger.info("\n---[*] exiting since the log likelihood are not increasing any more.\n");
				break;
			} else {
				endepoch(epochBTime, ibatch, iepoch, numberer, trllist, dellist);
			}
		} while (++iepoch < opts.nepoch);
		finals(trllist, dellist, numberer, ibatch, iepoch); // better finalize it in a specialized test class
	}
	
	public static void serialInBatch(Numberer numberer, int iepoch) throws Exception {
		long bTime, eTime, epochBTime, batchBTime, batchETime;
		int iprebeg, ibegin = 0, ibatch = 0, length, isample = 0;
		List<Double> trllist = new ArrayList<>();
		List<Double> dellist = new ArrayList<>();
		List<List<TaggedWord>> batch = new ArrayList<>(opts.bsize + 5);
		List<Double> scoresOfST = null;
		boolean shuffle = false;
		do {
			logger.trace("\n\n-------epoch " + iepoch + " begins-------\n\n");
			boolean exit = false;
			epochBTime = System.currentTimeMillis();
			while (true) {
				ibatch++;
				if ((iepoch > opts.epochskipk - 1) && (opts.nbatch == 0 || ((ibatch % opts.nbatch) == 0))) {
					exit = opts.nbatch == 0 ? true : peep(ibatch, iepoch, numberer, trllist, dellist, false);
					if (exit) { break; }
				}
				isample = 0;
				iprebeg = ibegin;
				ibegin = getBatch(ibegin, shuffle, batch);
				batchBTime = System.currentTimeMillis();
				for (List<TaggedWord> tree : batch) {
					isample++;
					length = tree.size(); 
					logger.trace("---Sample " + isample + "...\t");
					bTime = System.currentTimeMillis();
					
					scoresOfST = lvetTagger.evalEdgeCounts(tree, (short) 0);
					scoresOfST.add((double) length);
					
					eTime = System.currentTimeMillis();
					logger.trace((eTime - bTime) / 1000.0 + "\t");
					
					if (Double.isInfinite(scoresOfST.get(0)) || Double.isInfinite(scoresOfST.get(1))) {
						logger.trace("scores: " + FunUtil.double2str(scoresOfST, precision, -1, false, true) + "\n");
						//logger.trace("--------- " + StateTreeList.stateTreeToStringTree(tree, numberer) + "\n");
						continue;
					}
					
					logger.trace("scores: " + FunUtil.double2str(scoresOfST, precision, -1, false, true) + "\teval gradients... ");
					bTime = System.currentTimeMillis();
					
					ttpairs.evalGradients(scoresOfST);
					twpairs.evalGradients(scoresOfST);
					scoresOfST.clear();
					
					eTime = System.currentTimeMillis();
					logger.trace((eTime - bTime) / 1000.0 + "\n");
				}
				batchETime = System.currentTimeMillis();
				
				// apply gradient descent
				logger.trace("+++Apply gradient descent for the batch " + ibatch + "... ");
				bTime = System.currentTimeMillis();
				
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(true); }
				ttpairs.applyGradientDescent(scoresOfST);
				twpairs.applyGradientDescent(scoresOfST);
				if (opts.dgradnbatch > 0 && ((ibatch % opts.dgradnbatch) == 0)) { debugrad(false); }
				
				eTime = System.currentTimeMillis();
				logger.trace((eTime - bTime) / 1000.0 + "... batch time: " + (batchETime - batchBTime) / 1000.0 + "\n");
				if (ibegin < iprebeg) {
					shuffle = true;
					break; // epoch ends
				} else {
					shuffle = false;
				}
			}
			if (exit) { // 
				logger.info("\n---[*] exiting since the log likelihood are not increasing any more.\n");
				break;
			} else {
				endepoch(epochBTime, ibatch, iepoch, numberer, trllist, dellist);
			}
		} while (++iepoch < opts.nepoch);
		finals(trllist, dellist, numberer, ibatch, iepoch); // better finalize it in a specialized test class
	}
	
	public static boolean endepoch(double epochBTime, int ibatch, int iepoch,
			Numberer numberer, List<Double> trllist, List<Double> dellist) throws Exception {
		// coarse summary
		long epochETime = System.currentTimeMillis();
		logger.trace("===Average time each sample consumed is " + (epochETime - epochBTime) / (1000.0 * (ibatch * opts.bsize)) + "\n");
		
		// likelihood of the data set
		logger.trace("\n----------log-likelihood in epoch " + iepoch + " is under evaluation----------\n");
		boolean exit = peep(ibatch, iepoch, numberer, trllist, dellist, true);
		logger.trace("\n------------------------evaluation over------------------------\n");
		
		// shall we clear the inside and outside score in each state of the parse tree after the training on a sample?
		// trainTrees.reset(); // CHECK
		logger.trace("\n-------epoch " + String.format("%1$3s", iepoch) + " ends  -------\n");
		return exit;
	}
	
	public static void finals(List<Double> trllist, List<Double> dellist, Numberer numberer, int ibatch, int iepoch) {
		if (!trllist.isEmpty()) { bestTrainLL = Collections.max(trllist); }
		if (!dellist.isEmpty()) { bestDevLL = Collections.max(dellist); }
		nbatch = ibatch;
		logger.trace("Convergence Path [train]: " + trllist + "\tMAX: " + bestTrainLL + "\n");
		logger.trace("Convergence Path [ dev ]: " + dellist + "\tMAX: " + bestDevLL + "\n");
		
		logger.trace("\n----------training is over after " + iepoch + " epoches " + ibatch + " batches----------\n");
		
		if (opts.saveGrammar) {
			logger.info("\n-------saving the final grammar file...");
			TaggerFile gfile = new TaggerFile(ttpairs, twpairs);
			String filename = subdatadir + opts.outGrammar + "_final.gr";
			if (gfile.save(filename)) {
				logger.info("to \'" + filename + "\' successfully.\n");
			} else {
				logger.info("to \'" + filename + "\' unsuccessfully.\n");
			}
		}
	}
	
	public static boolean peep(int ibatch, int iepoch, Numberer numberer, List<Double> trllist, 
			List<Double> dellist, boolean ends) throws Exception {
		long beginTime, endTime;
		double trll = 1e-8, dell = 1e-8;
		// likelihood of the training set
		if (!ends && opts.eontrain) {
			logger.trace("\n-------ll of the training data after " + ibatch + " batches in epoch " + iepoch + " is... ");
			beginTime = System.currentTimeMillis();
			if (opts.peval) {
				trll = parallelLL(opts, mvaluator, trainTrees, numberer, true);
			} else {
				trll = serialLL(opts, valuator, trainTrees, numberer, true);
			}
			endTime = System.currentTimeMillis();
			logger.trace("------->" + trll + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			// trainTrees.reset();
			trllist.add(trll);
		}
		
		// check dropping count according to the log likelihood on the development set
		boolean exit = false, save = false;
		if (!ends && opts.eondev && (ibatch % opts.enbatchdev) == 0) {
			logger.trace("\n-------ll of the dev data after " + ibatch + " batches in epoch " + iepoch + " is... ");
			beginTime = System.currentTimeMillis();
			if (opts.peval) {
				dell = parallelLL(opts, mvaluator, devTrees, numberer, false);
			} else {
				dell = serialLL(opts, valuator, devTrees, numberer, false);
			}
			endTime = System.currentTimeMillis();
			logger.trace("------->" + dell + " consumed " + (endTime - beginTime) / 1000.0 + "s\n");
			// devTrees.reset();
			dellist.add(dell);
			if (dell > bestscore) {
				bestscore = dell;
				cntdrop = 0;
				save = true;
			} else {
				cntdrop++;
				exit = (cntdrop >= opts.nAllowedDrop);
			}
		}
		
		// save the intermediate grammars
		TaggerFile gfile = new TaggerFile(ttpairs, twpairs);
		if (opts.saveGrammar && save) { // always save the best grammar to the same file
			String filename = subdatadir + opts.outGrammar + "_best.gr";
			if (gfile.save(filename)) { 
				logger.info("\n------->the best [dev] grammar [iepoch = " + iepoch + ", ibatch = " + ibatch + "]\n");
			}
		}
		// save the grammar at the end of each epoch or after every # of batches
		save = (((ibatch % opts.nbatchSave) == 0) && opts.outGrammar != null);
		if (opts.saveGrammar && (ends || save)) {
			logger.info("\n-------saving grammar file...");
			String filename = (ends ? subdatadir + opts.outGrammar + "_" + iepoch + ".gr" 
					: subdatadir + opts.outGrammar + "_" + iepoch + "_" + ibatch + ".gr");
			if (gfile.save(filename)) {
				logger.info("to \'" + filename + "\' successfully.\n");
			} else {
				logger.info("to \'" + filename + "\' unsuccessfully.\n");
			}
		}
		// save the best grammar according to the log likelihood on training set
		if (trll < 0 && trll > besttrain) {
			besttrain = trll;
			String filename = subdatadir + opts.outGrammar + "_best_train.gr";
			if (gfile.save(filename)) { 
				logger.info("\n------->the best [train] grammar [iepoch = " + iepoch + ", ibatch = " + ibatch + "]\n");
			}
		}
		return exit;
	}
	
	public static double parallelLL(Options opts, ThreadPool valuator, List<List<TaggedWord>> sequences, Numberer numberer, boolean istrain) {
		double ll = 0, sumll = 0;
		int nUnparsable = 0, cnt = 0;
		
		// filterTrees(opts, stateTreeList, trees, numberer, istrain);
		
		for (List<TaggedWord> sequence: sequences) {
			valuator.execute(sequence);
			while (valuator.hasNext()) {
				ll = (double) valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll)) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
			cnt++;
		}
		while (!valuator.isDone()) {
			while (valuator.hasNext()) {
				ll = (double) valuator.getNext();
				if (Double.isInfinite(ll) || Double.isNaN(ll) || ll > 0) {
					nUnparsable++;
				} else {
					sumll += ll;
				}
			}
		}
		valuator.reset();
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + sequences.size() + "(" + cnt + ") samples]\n");
		return sumll;
	}
	
	
	public static double serialLL(Options opts, Valuator<?, ?> valuator, List<List<TaggedWord>> sequences, Numberer numberer, boolean istrain) {
		double ll = 0, sumll = 0;
		int nUnparsable = 0, cnt = 0;
		
		//filterTrees(opts, stateTreeList, trees, numberer, istrain);
		
		for (List<TaggedWord> sequence : sequences) {
			ll = valuator.probability(sequence);
			if (Double.isInfinite(ll) || Double.isNaN(ll) || ll > 0) {
				nUnparsable++;
			} else {
				sumll += ll;
			}
			cnt++;
		}
		logger.trace("\n[in calculating log likelihood " + nUnparsable + " unparsable sample(s) of " + sequences.size() + "(" + cnt + ") samples]\n");
		return sumll;
	}
}
