package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Parser.Meta;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class MultiThreadedParser extends Recorder implements Serializable {
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
	protected Parser[] parsers;
	protected ExecutorService pool;
	protected PriorityQueue<Meta<?>> scores;
	
	
	public MultiThreadedParser(Parser<?> parser, int nthread) {
		this.nthread = nthread;
		this.submits = new Future[nthread];
		this.parsers = new Parser[nthread];
		this.pool = Executors.newFixedThreadPool(nthread);
		this.scores = new PriorityQueue<Meta<?>>(idcomparator);
		this.lastReturn = -1;
		this.lastSubmission = 0;
		for (int i = 0; i < nthread; i++) {
			parsers[i] = parser.newInstance();
			parsers[i].setIdx(i, scores);
		}
	}
	
	
	public void parse(Tree<State> sample) {
		synchronized (scores) {
			while (true) {
				for (int i = 0; i < nthread; i++) {
					if (submits[i] == null || submits[i].isDone()) {
						parsers[i].setNextSample(sample, lastSubmission++);
						// logger.trace("\n--->last-sub: " + lastSubmission + "\n"); // DEBUG
						submits[i] = pool.submit(parsers[i]);
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
	
	
	protected Object getNext() {
		if (!hasNext()) { return new Double(0.0); }
		lastReturn++;
		// logger.trace("\n--->last-ret: " + lastReturn + "\n"); // DEBUG
		synchronized (scores) {
			Meta<?> score = scores.poll();
			// logger.trace("\n~~~>score: " + score + "\n"); // DEBUG
			scores.notifyAll();
			return score.value();
		}
	}
	
	
	protected boolean hasNext() {
		synchronized (scores) {
			if (scores.isEmpty()) { return false; }
			Meta<?> score = scores.peek();
			scores.notifyAll();
			return score.id == (lastReturn + 1);
		}
	}
	
	
	protected boolean isDone() {
		return (lastSubmission - 1) == lastReturn;
	}
	
	
	protected void shutdown() {
		if (pool != null) {
			try {
				pool.shutdown();
				pool.awaitTermination(10, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	
	protected void reset() {
		lastReturn = -1;
		lastSubmission = 0;
		scores.clear();
	}
	
}
