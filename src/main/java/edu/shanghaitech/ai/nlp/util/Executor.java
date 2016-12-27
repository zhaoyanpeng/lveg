package edu.shanghaitech.ai.nlp.util;

import java.io.Serializable;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;

public interface Executor<I, O> extends Callable<Object>, Serializable {
	public static class Meta<O> {
		public int id;
		public O value;
		public Meta(int id, O value) {
			this.id = id;
			this.value = value;
		}
		public O value() { return value; }
		@Override
		public String toString() {
			return "Meta [id=" + id + ", value=" + value + "]";
		}
	}
	
	public abstract Executor<?, ?> newInstance();
	
	public abstract void setNextTask(int itask, I task);
	
	public abstract void setIdx(int idx, PriorityQueue<Meta<O>> caches);
}
