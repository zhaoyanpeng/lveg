package edu.shanghaitech.ai.nlp.lveg;

import java.util.PriorityQueue;

import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class Parser<I, O> extends Recorder implements Executor<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7112164011234304607L;
	protected static final short MAX_SENTENCE_LEN = 12;
	
	protected int idx;
	protected boolean reuse;
	protected boolean parallel;
	
	protected int isample;
	protected Chart chart;
	protected I sample;
	protected PriorityQueue<Meta<O>> caches;
	
	@Override
	public void setIdx(int idx, PriorityQueue<Meta<O>> caches) {
		this.idx = idx;
		this.caches = caches;
	}
	
	@Override
	public void setNextSample(int isample, I sample) {
		this.sample = sample;
		this.isample = isample;
	}
	
}
