package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class Parser<T> extends Recorder implements Callable<Object>, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7112164011234304607L;
	protected static final short MAX_SENTENCE_LEN = 12;
	public static class Meta<T> {
		public int id;
		public T value;
		public Meta(int id, T value) {
			this.id = id;
			this.value = value;
		}
		public T value() { return value; }
		@Override
		public String toString() {
			return "Meta [id=" + id + ", value=" + value + "]";
		}
	}
	
	protected int idx;
	protected boolean reuse;
	protected boolean parallel;
	
	protected int isample;
	protected Chart chart;
	protected Tree<State> sample;
	protected PriorityQueue<Meta<T>> caches;
	
	
	protected abstract Parser<?> newInstance();
	
	
	protected void setIdx(int idx, PriorityQueue<Meta<T>> caches) {
		this.idx = idx;
		this.caches = caches;
	}
	
	
	protected void setNextSample(Tree<State> sample, int isample) {
		this.sample = sample;
		this.isample = isample;
	}
	
}
