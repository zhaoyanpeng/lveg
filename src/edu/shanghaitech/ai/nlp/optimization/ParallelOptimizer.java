package edu.shanghaitech.ai.nlp.optimization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;

import edu.shanghaitech.ai.nlp.lveg.GrammarRule;

/**
 * @author Yanpeng Zhao
 *
 */
public class ParallelOptimizer extends Optimizer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6206396492328441930L;
	protected enum ParallelMode {
		INVOKE_ALL, COMPLETION_SERVICE, CUSTOMIZED_BLOCK, FORK_JOIN
	}
	private short nthread;
	private boolean parallel;
	private ParallelMode mode;
	private Map<GrammarRule, Gradient> gradients; // gradients
	
	private transient GrammarRule[] ruleArray;
	private transient ExecutorService pool;
	private transient List<Future<Boolean>> futures;
	private transient List<Callable<Boolean>> tasks;
	private transient CompletionService<Boolean> service;
	
	
	private ParallelOptimizer() {
		this.cntsWithS = new HashMap<GrammarRule, Batch>();
		this.cntsWithT = new HashMap<GrammarRule, Batch>();
		this.ruleSet = new HashSet<GrammarRule>();
		this.gradients = new HashMap<GrammarRule, Gradient>();
		this.mode = ParallelMode.INVOKE_ALL;
		this.futures = null;
		this.tasks = null;
	}
	
	
	public ParallelOptimizer(Random random, short nthread) {
		this();
		rnd = random;
		this.nthread = nthread;
	}
	
	
	public ParallelOptimizer(Random random, short msample, short bsize, short nthread, boolean parall) {
		this();
		rnd = random;
		batchsize = bsize;
		maxsample = msample;
		this.nthread = nthread;
		this.parallel = parall;
	}
	
	
	private void evalGradientsParallel(List<Double> scoreSandT) {
		if (tasks == null) { 
			tasks = new ArrayList<Callable<Boolean>>(ruleSet.size()); 
			for (GrammarRule rule : ruleSet) {
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
				tasks.add(new Callable<Boolean>() {
					@Override
					public Boolean call() throws Exception {
						/*
						boolean updated = false;
						Batch cntWithT = cntsWithT.get(rule);
						Batch cntWithS = cntsWithS.get(rule);
						for (short i = 0; i < Gradient.MAX_BATCH_SIZE; i++) {
							if (cntWithT.get(i) != null || cntWithS.get(i) != null) { 
								updated = true; 
								break; 
							} 
						}
						if (!updated) { return false; }
						*/
						Gradient gradient = gradients.get(rule);
						boolean ichanged = gradient.eval(rule, cntWithT, cntWithS, scoreSandT);
						// clear
						cntWithT.clear();
						cntWithS.clear();
						return ichanged;
					}
				});
			}
		}
		switch (mode) {
		case COMPLETION_SERVICE: {
			useCompletionService();
			break;
		}
		case CUSTOMIZED_BLOCK: {
			useCustomizedBlock();
			break;
		}
		case INVOKE_ALL: {
			useInvokeAll();
			break;
		}
		case FORK_JOIN: {
			useForkJoin(scoreSandT);
			break;
		}
		}
		tasks = null;
	}
	
	
	private void useInvokeAll() {
		boolean exit = true;
		int nchanged = 0;
		try {
			if (pool == null) { 
				pool = Executors.newFixedThreadPool(nthread); 
			}
			List<Future<Boolean>> futures = pool.invokeAll(tasks);
			for (Future<Boolean> future : futures) {
				if (future.get()) { nchanged++; }
			}
			// we do not require the pool to exit for the purpose of reusing
			// pool.shutdown();
			// exit = pool.awaitTermination(10, TimeUnit.MILLISECONDS);
		} catch (ExecutionException | InterruptedException e) {
			e.printStackTrace();
		}
		logger.trace("exit: " + exit + ", nchanged: " + nchanged + " of " + ruleSet.size() + "..." + pool.isTerminated() + "...");
	}
	
	
	/**
	 * See comments in the method.
	 */
	private void useCompletionService() {
		boolean exit = true;
		int nchanged = 0, isdone = 0;
		pool = Executors.newFixedThreadPool(nthread);
		service = new ExecutorCompletionService<Boolean>(pool);
		for (Callable<Boolean> task : tasks) {
			service.submit(task);
		}
		try {
			pool.shutdown();
			// while (!pool.isTerminated()) { // CHECK why does it still returns false while all tasks are done?
			for (int i = 0; i < ruleSet.size(); i++) {
				Future<Boolean> future = service.take();
				if (future.get()) { nchanged++; }
				if (future.isDone()) { isdone++; }
			}
			// I found 10ms is very useful than 0ms, because it can ensure the pool is terminated but 0ms cannot?
			exit = pool.awaitTermination(10, TimeUnit.MILLISECONDS);
		} catch (ExecutionException | InterruptedException e) {
			e.printStackTrace();
		}
		logger.trace("exit: " + exit + ", nchanged: " + nchanged + " of " + ruleSet.size() + "(" + isdone + ")" + "..." + pool.isTerminated() + "...");
	}
	
	
	/**
	 * See comments in the method.
	 */
	private void useCustomizedBlock() {
		boolean exit = true;
		int nchanged = 0, isdone = 0;
		if (futures == null) { 
			futures = new ArrayList<Future<Boolean>>(ruleSet.size()); 
		}
		futures.clear();
		pool = Executors.newFixedThreadPool(nthread);
		for (Callable<Boolean> task : tasks) {
			futures.add(pool.submit(task));
		}
		try {
			pool.shutdown();
			boolean done = true;
			while (done) { 
				for (Future<Boolean> future : futures) {
					if (!future.isDone()) { done = false; } 
				}
				done = done ? false : true;
			} // I observe that enumerating the futures can ensure right outputs, why?
			// errors may occur when comment the while loop and the 'exit' line
			exit = pool.awaitTermination(10, TimeUnit.MILLISECONDS);
			for (Future<Boolean> future : futures) { // counting debugging data
				if (future.get()) { nchanged++; }
				if (future.isDone()) { isdone++; }
				// exchange the above two lines, isdone would be wrongly counted, why?
			}
		} catch (ExecutionException | InterruptedException e) {
			e.printStackTrace();
		}
		logger.trace("exit: " + exit + ", nchanged: " + nchanged + " of " + ruleSet.size() + "(" + isdone + ")" + "..." + pool.isTerminated() + "...");
	}
	
	
	/**
	 * Need to tune the size of the chunk that a thread eats, but it is somewhat memory-efficiency? 
	 * TODO Need more trials.
	 * 
	 * @param scoreSandT
	 */
	private void useForkJoin(List<Double> scoreSandT) {
		if (watch == null) { watch = new Watch(); }
		if (ruleArray == null) { ruleArray = ruleSet.toArray(new GrammarRule[0]); }
		watch.clear();
		ForkJoinPool pool = new ForkJoinPool(nthread);
		ParallelForLoop loop = new ParallelForLoop(0, ruleArray.length, scoreSandT);
		pool.invoke(loop);
		logger.trace("nchanged: " + watch.nchanged + " of " + ruleArray.length + "(" + watch.nskipped + ")" + "...");
	}
	
	
	@Override
	public void applyGradientDescent(List<Double> scoresST) {
		Gradient gradient;
		int nupdated = 0;
		for (GrammarRule rule : ruleSet) {
			gradient = gradients.get(rule);
			if (gradient.apply(rule)) {
				nupdated++;
			}
		}
		reset();
		logger.trace("nupdated=" + nupdated + " of " + ruleSet.size() + "...");
	}
	
	
	@Override
	public void evalGradients(List<Double> scoreSandT) {
		if (scoreSandT.size() == 0) { return; }
		if (parallel) { 
			evalGradientsParallel(scoreSandT); 
			return;
		}

		Batch cntWithT, cntWithS;
		int nchanged = 0, nskipped = 0;
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
			if (!updated) { nskipped++; continue; }
			Gradient gradient = gradients.get(rule);
			if (gradient.eval(rule, cntWithT, cntWithS, scoreSandT)) {
				nchanged++;
			}
			// clear
			cntWithT.clear();
			cntWithS.clear();
		}
		logger.trace("nchanged=" + nchanged + " of " + ruleSet.size() + "(" + nskipped + ")" + "...");
	}
	
	
	@Override
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		Batch batchWithT = new Batch(Gradient.MAX_BATCH_SIZE);
		Batch batchWithS = new Batch(Gradient.MAX_BATCH_SIZE);
		cntsWithT.put(rule, batchWithT);
		cntsWithS.put(rule, batchWithS);
		Gradient gradient = new Gradient(rule, rnd, maxsample, batchsize);
		gradients.put(rule, gradient);
	}
	
	
	@Override
	protected void reset() { 
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			if ((cntWithT = cntsWithT.get(rule)) != null) { cntWithT.clear(); }
			if ((cntWithS = cntsWithS.get(rule)) != null) { cntWithS.clear(); }
		}
	}
	
	
	public void shutdown() {
		if (pool != null) {
			try {
				pool.shutdown();
				pool.awaitTermination(10, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	
	/**
	 * @author Yanpeng Zhao
	 *
	 */
	protected Watch watch = new Watch();
	class Watch implements Serializable { 
		/**
		 * 
		 */
		private static final long serialVersionUID = 4040589141969741180L;
		int nchanged, nskipped; void clear() { nchanged = 0; nskipped = 0; } 
	}
	class ParallelForLoop extends RecursiveAction {
		/* */
		private static final long serialVersionUID = 1L;
		private int step = ruleArray.length / nthread;
		
		private List<Double> score;
		private int from, to;
		
		public ParallelForLoop(int from, int to, List<Double> score) {
			this.from = from;
			this.to = to;
			this.score = score;
			
		}
		
		@Override
		protected void compute() {
			int range = to - from;
			if (range <= step) {
				run(from, to);
			} else {
				int mid = (from + to) >>> 1;
				ForkJoinTask<Void> lhs = new ParallelForLoop(from, mid, score).fork();
				ForkJoinTask<Void> rhs = new ParallelForLoop(mid, to, score).fork();
				lhs.join();
				rhs.join();
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
				synchronized (watch) {
					if (!updated) { 
						watch.nskipped++; 
						continue; 
					}
				}
				Gradient gradient = gradients.get(rule);
				boolean ichanged = gradient.eval(rule, cntWithT, cntWithS, score);
				synchronized (watch) {
					if (ichanged) { watch.nchanged++; }
				}
				// clear
				cntWithT.clear();
				cntWithS.clear();
			}
		}
	}
	
}
