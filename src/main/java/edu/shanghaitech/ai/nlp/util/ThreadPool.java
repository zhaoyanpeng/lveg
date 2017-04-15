package edu.shanghaitech.ai.nlp.util;

import java.io.Serializable;
import java.util.Comparator;
import java.util.PriorityQueue;
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
//					boolean a, b;
//					if (submits[i] == null) {
//						a = true;
//						b = true;
//					} else {
//						a = false;
//						b = submits[i].isDone();
//					}
					
					if (submits[i] == null || submits[i].isDone()) {
						executors[i].setNextTask(lastSubmission++, task);
//						 logger.trace("\n--->last-submission: " + lastSubmission + "\t" + a + "\t" + b + "\n"); // DEBUG
						submits[i] = pool.submit(executors[i]);
						return;
					}
				}
				try {
					scores.wait();
				} catch (InterruptedException e) {
					// 
				}
			}
		}
	}
	
	
	public Object getNext() {
		if (!hasNext()) { return new Double(0.0); }
		lastReturn++;
//		 logger.trace("\n--->last-ret: " + lastReturn + "\n"); // DEBUG
		synchronized (scores) {
			Meta<?> score = scores.poll();
//			 logger.trace("\n~~~>score: " + score + "\n"); // DEBUG
			scores.notifyAll();
			return score.value();
		}
	}
	
	
	public boolean hasNext() {
		synchronized (scores) {
			if (scores.isEmpty()) { return false; }
			Meta<?> score = scores.peek();
			scores.notifyAll();
			return score.id == (lastReturn + 1);
		}
	}
	
	
	public boolean isDone() {
		return (lastSubmission - 1) == lastReturn;
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
