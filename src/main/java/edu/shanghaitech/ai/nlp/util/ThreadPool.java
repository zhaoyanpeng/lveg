package edu.shanghaitech.ai.nlp.util;

import java.io.Serializable;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import edu.shanghaitech.ai.nlp.util.Executor.Meta;

public class ThreadPool extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4870873867836614814L;
	public final static String MAIN_THREAD = "MAIN_THREAD";
	protected static Comparator<Meta<?>> idcomparator = new Comparator<Meta<?>>() {
		@Override
		public int compare(Meta<?> o1, Meta<?> o2) {
			return o1.id - o2.id;
		}
	};
	protected int lastReturn;
	protected int lastSubmission;
	
	protected int nthread;
	protected Future[] submits;
	protected Executor[] executors;
	protected ExecutorService pool;
	protected PriorityQueue<Meta<?>> scores;
	
	
	public ThreadPool(Executor<?, ?> executor, int nthread) {
		this.nthread = nthread;
		this.submits = new Future[nthread];
		this.executors = new Executor[nthread];
		this.pool = Executors.newFixedThreadPool(nthread);
		this.scores = new PriorityQueue<Meta<?>>(idcomparator);
		this.lastReturn = -1;
		this.lastSubmission = 0;
		for (int i = 0; i < nthread; i++) {
			executors[i] = executor.newInstance();
			executors[i].setIdx(i, scores);
		}
	}
	
	
	public void execute(Object task) {
		synchronized (scores) {
			while (true) {
				for (int i = 0; i < nthread; i++) {
					if (submits[i] == null || submits[i].isDone()) {
						
//						int iret = 1 << 30;
//						boolean isnull = submits[i] == null;
//						if (!isnull) {
//							try { // get the index of the finished task, should be larger than or equal to 0
//								iret = (int) submits[i].get();
//							} catch (InterruptedException | ExecutionException e) {
//								e.printStackTrace();
//								iret = 1 << 30;
//								logger.error("OOPS_BUG: get()\n");
//								throw new RuntimeException("OOPS_BUG: excute\n");
//							} 
//						}
						
//						if (MAIN_THREAD.equals(Thread.currentThread().getName())) {
//							logger.trace("\n---3------last ret: " + lastReturn + ", last submission: " + lastSubmission + 
//									", size: " + scores.size() + ", active: " + Thread.activeCount() + ", isnull: " + isnull + ", iret: " + iret);
//						}
						
						executors[i].setNextTask(lastSubmission, task);
						submits[i] = pool.submit(executors[i]);
						lastSubmission++;
						return;
					}
				}
				try {
					scores.wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	
	
	/**
	 * A safe implementation of task submission, but is less efficient.
	 * 
	 * @param task the task to be executed
	 */
	public void executeSafe(Object task) {
		synchronized (scores) {
			while (true) {
				int iworker = 0, iret = -1;
				boolean isnull = false, isdone = false, wait = false;
				for (; iworker < nthread; iworker++) {
					isnull = submits[iworker] == null;
					if (!isnull) {
						isdone = submits[iworker].isDone();
					}
					if (isnull || isdone) {
						break;
					}
				} // find the free executor
				
				if (isnull || isdone) {
					if (isdone) { 
						try { // get the index of the finished task, should be larger than or equal to 0
							iret = (int) submits[iworker].get();
						} catch (InterruptedException | ExecutionException e) {
							iret = -1;
							e.printStackTrace();
							throw new IllegalStateException("OOPS_BUG: no return value.");
						} 
						if (iret < 0) {
							wait = true;
						}
					}
					// lastSubmission: # of total submitted tasks
					// lastReturn + 1: # of total returned tasks
					// scores.size() : # of total tasks that are finished and waiting to be retrieved
					if (lastSubmission - lastReturn - 1 - scores.size()  >= nthread) {
						wait = true; // this error should be handled in Callable.call()
						throw new IllegalStateException("OOPS_BUG: Number of submissions is larger than that of available threads.");
					}
//					if (MAIN_THREAD.equals(Thread.currentThread().getName())) {
//						logger.trace("\n---3------last ret: " + lastReturn + ", last submission: " + 
//								lastSubmission + ", size: " + scores.size() + ", active: " + Thread.activeCount() + 
//								", isnull: " + isnull + ", iret: " + iret);
//					}
				} else { // no free worker
					wait = true;
				}
				
				if (wait) {
					try {
						scores.wait();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				} else {
					executors[iworker].setNextTask(lastSubmission, task);
					submits[iworker] = pool.submit(executors[iworker]);
					lastSubmission++;
					return;
				}
			}
		}
	}
	
	
	public Object getNext() {
		if (!hasNext()) { 
			throw new IllegalStateException("OOPS_BUG: Can only be invoked when there are available results to retrieve.");
		}
		synchronized (scores) {
//			if (MAIN_THREAD.equals(Thread.currentThread().getName())) {
//				logger.trace("\n---2------last ret: " + lastReturn + ", last submission: " + lastSubmission + ", size: " + scores.size());
//			}
			Meta<?> score = scores.poll();
			lastReturn++;
			return score.value();
		}
	}
	
	
	public boolean hasNext() {
		synchronized (scores) {
//			if (MAIN_THREAD.equals(Thread.currentThread().getName())) {
//				logger.trace("\n---1------last ret: " + lastReturn + ", last submission: " + lastSubmission + ", size: " + scores.size());
//			}
			if (scores.isEmpty()) { return false; }
			Meta<?> score = scores.peek();
			return score.id == (lastReturn + 1);
		}
	}
	
	
	public boolean isDone() {
		return (lastSubmission - 1) == lastReturn;
	}
	
	
	public void sleep() {
		synchronized (scores) {
			try {
				scores.wait();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	
	public void shutdown() {
		reset();
		if (pool != null) {
			try {
				pool.shutdown();
				pool.awaitTermination(10, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	
	public void reset() {
		lastReturn = -1;
		lastSubmission = 0;
		scores.clear();
	}
	
}
