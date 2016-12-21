package edu.shanghaitech.ai.nlp.lveg;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class MultiThreadedValuator extends Recorder {
	protected static class Score {
		public int id;
		public double score;
		public Score(int id, double score) {
			this.id = id;
			this.score = score;
		}
		@Override
		public String toString() {
			return "Score [id=" + id + ", score=" + score + "]";
		}
		
	}
	protected static Comparator<Score> idcomparator = new Comparator<Score>() {
		@Override
		public int compare(Score o1, Score o2) {
			return o1.id - o2.id;
		}
	};
	
	protected int lastReturn;
	protected int lastSubmission;
	
	protected int nthread;
	protected Future[] submits;
	protected ExecutorService pool;
	protected LVeGParser[] parsers;
	protected PriorityQueue<Score> scores;
	
	
	public MultiThreadedValuator(LVeGParser parser, int nthread) {
		this.nthread = nthread;
		this.pool = Executors.newFixedThreadPool(nthread);
		this.submits = new Future[nthread];
		this.parsers = new LVeGParser[nthread];
		this.scores = new PriorityQueue<Score>(idcomparator);
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
	
	
	protected double getNext() {
		if (!hasNext()) { return 0.0; }
		lastReturn++;
		// logger.trace("\n--->last-ret: " + lastReturn + "\n"); // DEBUG
		synchronized (scores) {
			Score score = scores.poll();
			// logger.trace("\n~~~>score: " + score + "\n"); // DEBUG
			scores.notifyAll();
			return score.score;
		}
	}
	
	
	protected boolean hasNext() {
		synchronized (scores) {
			if (scores.isEmpty()) { return false; }
			Score score = scores.peek();
			scores.notifyAll();
			return score.id == (lastReturn + 1);
		}
	}
	
	
	protected boolean isDone() {
		return (lastSubmission - 1) == lastReturn;
	}
	
	
	protected void shutdown() {
		pool.shutdown();
	}
	
}
