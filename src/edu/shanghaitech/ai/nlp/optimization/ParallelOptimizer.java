package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.GrammarRule;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public class ParallelOptimizer extends Recorder {
	private static final short THREADS_NUM = 10;
	private static Random rnd;
	private static short maxsample;
	// which contains all the rules that need optimizing, it is used to quickly index the grammar rule.
	private GrammarRule[] ruleArray;
	private Set<GrammarRule> ruleSet;
	// pseudo counts of grammar rules given the parse tree (countWithT) or the sentence (countWithS).
	private Map<GrammarRule, Batch> cntsWithT;
	private Map<GrammarRule, Batch> cntsWithS;
	// gradients
	private Map<GrammarRule, Gradient> gradients;
	private ExecutorService service;
	private List<Callable<Object>> tasks;
	
	private ParallelOptimizer() {
		this.cntsWithS = new HashMap<GrammarRule, Batch>();
		this.cntsWithT = new HashMap<GrammarRule, Batch>();
		this.ruleSet = new HashSet<GrammarRule>();
		this.gradients = new HashMap<GrammarRule, Gradient>();
	}
	
	
	public ParallelOptimizer(Random random) {
		this();
		rnd = random;
		maxsample = 2;
	}
	
	
	public ParallelOptimizer(Random random, short nsample) {
		this();
		rnd = random;
		maxsample = nsample;
	}
	
	
	public void evalGradientsParallel(List<Double> scoreSandT) {
		/*
		if (ruleArray == null) { ruleArray = ruleSet.toArray(new GrammarRule[0]); }
		ForkJoinPool pool = new ForkJoinPool(THREADS_NUM);
		pool.invoke(new ParallelForLoop(0, ruleArray.length, scoreSandT));
		*/
		
		if (tasks == null) { 
			tasks = new ArrayList<Callable<Object>>(ruleSet.size()); 
			for (GrammarRule rule : ruleSet) {
				tasks.add(new Callable<Object>() {
					@Override
					public Object call() throws Exception {
						boolean updated = false;
						Batch cntWithT = cntsWithT.get(rule);
						Batch cntWithS = cntsWithS.get(rule);
						for (short i = 0; i < Gradient.MAX_BATCH_SIZE; i++) {
							if (cntWithT.get(i) != null || cntWithS.get(i) != null) { 
								updated = true; 
								break; 
							} 
						}
						if (!updated) { return null; }
						Gradient gradient = gradients.get(rule);
						gradient.eval(rule, cntWithT, cntWithS, scoreSandT);
						// clear
						cntWithT.clear();
						cntWithS.clear();
						return null;
					}
				});
			}
		}
		try {
			service.invokeAll(tasks);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
/*		
		service = Executors.newFixedThreadPool(THREADS_NUM);
		for (GrammarRule rule : ruleSet) {
			service.submit(new Callable<Object>() {
				@Override
				public Object call() throws Exception {
					boolean updated = false;
					Batch cntWithT = cntsWithT.get(rule);
					Batch cntWithS = cntsWithS.get(rule);
					for (short i = 0; i < Gradient.MAX_BATCH_SIZE; i++) {
						if (cntWithT.get(i) != null || cntWithS.get(i) != null) { 
							updated = true; 
							break; 
						} 
					}
					if (!updated) { return null; }
					Gradient gradient = gradients.get(rule);
					gradient.eval(rule, cntWithT, cntWithS, scoreSandT);
					// clear
					cntWithT.clear();
					cntWithS.clear();
					return null;
				}
			});
		}
		service.shutdown();
*/		
	}
	
	
	/**
	 * Stochastic gradient descent.
	 * 
	 * @param scoresOfST the parse tree score (odd index) and the sentence score (even index)
	 */
	public void evalGradients(List<Double> scoreSandT, boolean parallel) {
		if (scoreSandT.size() == 0) { return; }
		if (parallel) { 
			evalGradientsParallel(scoreSandT); 
			return;
		}
		Gradient gradient;
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			boolean updated = false;
			cntWithT = cntsWithT.get(rule);
			cntWithS = cntsWithS.get(rule);
			for (short i = 0; i < Gradient.MAX_BATCH_SIZE; i++) {
				if (cntWithT.get(i) != null || cntWithS.get(i) != null) { 
					updated = true; 
					break; 
				} 
			}
			if (!updated) { continue; }
			gradient = gradients.get(rule);
			gradient.eval(rule, cntWithT, cntWithS, scoreSandT);
			// clear
			cntWithT.clear();
			cntWithS.clear();
		}
	}
	
	
	
	public void applyGradientDescent(Object placeholder) {
		Gradient gradient;
		for (GrammarRule rule : ruleSet) {
			gradient = gradients.get(rule);
			gradient.apply(rule);
		}
		reset();
	}
	
	
	/**
	 * @param rule the rule that needs optimizing.
	 */
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		Batch batchWithT = new Batch(Gradient.MAX_BATCH_SIZE);
		Batch batchWithS = new Batch(Gradient.MAX_BATCH_SIZE);
		cntsWithT.put(rule, batchWithT);
		cntsWithS.put(rule, batchWithS);
		Gradient gradient = new Gradient(rule, rnd, maxsample);
		gradients.put(rule, gradient);
	}
	
	
	/**
	 * @param rule     the grammar rule
	 * @param cnt      which contains 1) key GrammarRule.Unit.P maps to the outside score of the parent node
	 * 					2) key GrammarRule.Unit.UC/C (LC) maps to the inside score (of the left node) if the rule is unary (binary)
	 * 					3) key GrammarRule.Unit.RC maps to the inside score of the right node if the rule is binary, otherwise null
	 * @param idx      index of the sample in this batch
	 * @param withT type of the expected pseudo count
	 */
	public void addCount(GrammarRule rule, Map<String, GaussianMixture> cnt, short idx, boolean withT) {
		Batch batch = null;
		Map<GrammarRule, Batch> cnts = withT ? cntsWithT : cntsWithS;
		if (rule != null && (batch = cnts.get(rule)) != null) {
			batch.add(idx, cnt);
			return;
		}
		logger.error("Not a valid grammar rule.\n");
	}
	
	
	/**
	 * The method is used for debugging purpose.
	 * 
	 * @param rule     the grammar rule
	 * @param withT type of the expected count
	 * @return
	 */
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(GrammarRule rule, boolean withT) {
		Batch batch = null;
		Map<GrammarRule, Batch> cnts = withT ? cntsWithT : cntsWithS;
		if (rule != null && (batch = cnts.get(rule)) != null) {
			return batch.batch;
		}
		logger.error("Not a valid grammar rule or the rule was not found.\n");
		return null;
	}
	
	
	public void reset() { 
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			if ((cntWithT = cntsWithT.get(rule)) != null) { cntWithT.clear(); }
			if ((cntWithS = cntsWithS.get(rule)) != null) { cntWithS.clear(); }
		}
	}
	
	
	/**
	 * Get set of the rules.
	 * 
	 * @return 
	 */
	public Set<GrammarRule> getRuleSet() {
		return ruleSet;
	}
	
	
	class ParallelForLoop extends RecursiveAction {
		/* */
		private static final long serialVersionUID = 1L;
		
		private List<Double> score;
		private int from, to, step;
		
		public ParallelForLoop(int from, int to, List<Double> score) {
			this.step = ruleArray.length / THREADS_NUM;
			this.from = from;
			this.to = to;
			this.score = score;
			
		}
		
		@Override
		protected void compute() {
			// TODO Auto-generated method stub
			int range = to - from;
			if (range < step) {
				run(from, to);
			} else {
				int mid = (from + to) >>> 1;
				new ParallelForLoop(from, mid, score).fork();
				new ParallelForLoop(mid, to, score).fork();
			}
		}
		
		protected void run(int from, int to) {
			for (int idx = from; idx < to; idx++) {
				GrammarRule rule = ruleArray[idx];
				boolean updated = false;
				Batch cntWithT = cntsWithT.get(rule);
				Batch cntWithS = cntsWithS.get(rule);
				for (short i = 0; i < Gradient.MAX_BATCH_SIZE; i++) {
					if (cntWithT.get(i) != null || cntWithS.get(i) != null) { 
						updated = true; 
						break; 
					} 
				}
				if (!updated) { continue; }
				Gradient gradient = gradients.get(rule);
				gradient.eval(rule, cntWithT, cntWithS, score);
				// clear
				cntWithT.clear();
				cntWithS.clear();
			}
		}
		
	}
	
}
