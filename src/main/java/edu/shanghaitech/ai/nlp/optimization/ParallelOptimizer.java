package edu.shanghaitech.ai.nlp.optimization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
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

import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.util.Executor;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

/**
 * @author Yanpeng Zhao
 *
 */
public class ParallelOptimizer extends Optimizer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6206396492328441930L;
	public enum ParallelMode {
		INVOKE_ALL, COMPLETION_SERVICE, CUSTOMIZED_BLOCK, FORK_JOIN, THREAD_POOL
	}
	private short nthread;
	private boolean verbose;
	private boolean parallel;
	private ParallelMode mode;
	private final Map<GrammarRule, Gradient> gradients; // gradients
	
	private transient ThreadPool mpool;
	private transient GrammarRule[] ruleArray;
	private transient ExecutorService pool;
	private transient List<Future<Boolean>> futures;
	private transient List<Callable<Boolean>> tasks;
	private transient CompletionService<Boolean> service;
	
	
	private ParallelOptimizer() {
		this.cntsWithS = new HashMap<>();
		this.cntsWithT = new HashMap<>();
		this.ruleSet = new HashSet<>();
		this.gradients = new HashMap<>();
		this.mode = ParallelMode.THREAD_POOL;
		this.futures = null;
		this.tasks = null;
	}
	
	
	public ParallelOptimizer(Random random, short nthread) {
		this();
		rnd = random;
		this.verbose = false;
		this.nthread = nthread;
	}
	
	
	public ParallelOptimizer(short nthread, boolean parall, ParallelMode mode, boolean verbose) {
		this();
		this.mode = mode;
		this.verbose = verbose;
		this.nthread = nthread;
		this.parallel = parall;
	}
	
	
	private void composeTasks(final List<Double> scoreSandT) {
		if (tasks == null) {
			tasks = new ArrayList<>(ruleSet.size()); 
		}
		for (final GrammarRule rule : ruleSet) {
			boolean updated = false;
			final Batch cntWithT = cntsWithT.get(rule);
			final Batch cntWithS = cntsWithS.get(rule);
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
	
	
	private void evalGradientsParallel(List<Double> scoreSandT) {
		switch (mode) {
		case COMPLETION_SERVICE: {
			composeTasks(scoreSandT);
			useCompletionService();
			tasks.clear();
			break;
		}
		case CUSTOMIZED_BLOCK: {
			composeTasks(scoreSandT);
			useCustomizedBlock();
			tasks.clear();
			break;
		}
		case THREAD_POOL: {
			useThreadPool(scoreSandT);
			break;
		}
		case INVOKE_ALL: {
			composeTasks(scoreSandT);
			useInvokeAll();
			tasks.clear();
			break;
		}
		case FORK_JOIN: {
			useForkJoin(scoreSandT);
			break;
		}
		default: {
			logger.error("unmatched parallel mode.\n");
		}
		}
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
		if (verbose) {
			logger.trace("exit: " + exit + ", nchanged: " + nchanged + " of " + ruleSet.size() + "..." + pool.isTerminated() + "...");
		}
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
			// I found 10ms very useful than 0ms, because it can ensure the pool is terminated but 0ms cannot?
			exit = pool.awaitTermination(10, TimeUnit.MILLISECONDS);
		} catch (ExecutionException | InterruptedException e) {
			e.printStackTrace();
		}
		if (verbose) {
			logger.trace("exit: " + exit + ", nchanged: " + nchanged + " of " + ruleSet.size() + "(" + isdone + ")" + "..." + pool.isTerminated() + "...");
		}
	}
	
	
	private void useThreadPool(List<Double> scoreSandT) {
		if (mpool == null) {
			SubOptimizer<?, ?> subOptimizer = new SubOptimizer<Resource, Boolean>();
			mpool = new ThreadPool(subOptimizer, nthread);
		} else {
			mpool.reset();
		}
		int nchanged = 0, nskipped = 0, nfailed = 0;
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
			if (!updated) { nskipped++; continue; }
			Gradient gradient = gradients.get(rule);
			Resource rsc = new Resource(rule, gradient, cntWithT, cntWithS, scoreSandT);
			mpool.execute(rsc);
			while (mpool.hasNext()) {
				if ((Boolean) mpool.getNext()) {
					nchanged++;
				} else {
					nfailed++;
				}
			}
		}
		while (!mpool.isDone()) {
			while (mpool.hasNext()) {
				if ((Boolean) mpool.getNext()) {
					nchanged++;
				} else {
					nfailed++;
				}
			}
		}
		if (verbose) {
			logger.trace("nchanged=" + nchanged + "(" + nfailed + ") of " + ruleSet.size() + "(" + nskipped + ")" + "...");
		}
	}
	
	
	/**
	 * See comments in the method.
	 */
	private void useCustomizedBlock() {
		boolean exit = true;
		int nchanged = 0, isdone = 0;
		if (futures == null) { 
			futures = new ArrayList<>(ruleSet.size()); 
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
		if (verbose) {
			logger.trace("exit: " + exit + ", nchanged: " + nchanged + " of " + ruleSet.size() + "(" + isdone + ")" + "..." + pool.isTerminated() + "...");
		}
	}
	
	
	/**
	 * Need to tune the size of the chunk that a thread eats, but it is somewhat memory friendly? 
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
		if (verbose) {
			logger.trace("nchanged: " + watch.nchanged + " of " + ruleArray.length + "(" + watch.nskipped + ")" + "...");
		}
	}
	
	
	@Override
	public Object debug(GrammarRule rule, boolean debug) {
		Object grads = null;
		if (rule == null || !ruleSet.contains(rule)) {
			for (GrammarRule arule : ruleSet) {
				grads = gradients.get(arule).debug(arule, debug);
				break;
			}
		} else {
			grads = gradients.get(rule).debug(rule, debug);
		}
		return grads;
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
		if (verbose) {
			logger.trace("nupdated=" + nupdated + " of " + ruleSet.size() + "...");
		}
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
		if (verbose) {
			logger.trace("nchanged=" + nchanged + " of " + ruleSet.size() + "(" + nskipped + ")" + "...");
		}
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
	
	
	@Override
	public void shutdown() {
		if (pool != null) {
			try {
				pool.shutdown();
				pool.awaitTermination(10, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		if (mpool != null) {
			mpool.shutdown();
		}
	}
	
	
	/**
	 * Type declaration of the input to the SubOptimizer.
	 *
	 */
	class Resource {
		protected GrammarRule rule;
		protected Gradient gradient;
		protected List<Double> scores;
		protected Batch iosWithT;
		protected Batch iosWithS;
		public Resource(GrammarRule rule, Gradient gradient, Batch iosWithT, Batch iosWithS, List<Double> scores) {
			this.rule = rule;
			this.gradient = gradient;
			this.iosWithT = iosWithT;
			this.iosWithS = iosWithS;
			this.scores = scores;
		}
	}
	/**
	 * @author Yanpeng Zhao
	 *
	 * @param <I> data type of the input, the resources to be consumed by threads
	 * @param <O> data type of the output, the returns returned in the method call()
	 */
	class SubOptimizer<I, O> implements Executor<I, O> {
		/**
		 * 
		 */
		private static final long serialVersionUID = 5638564355997342275L;
		protected int idx;
		
		protected I task;
		protected int itask;
		protected PriorityQueue<Meta<O>> caches;
		
		public SubOptimizer() {}
		
		private SubOptimizer(SubOptimizer<?, ?> subOptimizer) {}

		@Override
		public synchronized Object call() throws Exception {
			Resource rsc = (Resource) task;
			boolean status = rsc.gradient.eval(rsc.rule, rsc.iosWithT, rsc.iosWithS, rsc.scores);
			rsc.iosWithS.clear();
			rsc.iosWithT.clear();
			Meta<O> cache = new Meta(itask, status);
			synchronized (caches) {
				caches.add(cache);
				caches.notify();
			}
			task = null;
			return null;
		}

		@Override
		public SubOptimizer<?, ?> newInstance() {
			return new SubOptimizer<I, O>(this);
		}

		@Override
		public void setNextTask(int itask, I task) {
			this.task = task;
			this.itask = itask;
		}

		@Override
		public void setIdx(int idx, PriorityQueue<Meta<O>> caches) {
			this.idx = idx;
			this.caches = caches;
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