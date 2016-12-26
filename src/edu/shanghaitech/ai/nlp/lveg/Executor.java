package edu.shanghaitech.ai.nlp.lveg;

import java.util.PriorityQueue;

public interface Executor<I, O> {
	
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
	
	public abstract Executor<?, ?> newInstance();
	
	public abstract void setNextSample(I sample, int isample);
	
	public abstract void setIdx(int idx, PriorityQueue<Meta<O>> caches);
}
