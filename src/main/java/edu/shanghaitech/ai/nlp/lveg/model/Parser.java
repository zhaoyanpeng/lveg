package edu.shanghaitech.ai.nlp.lveg.model;

import java.util.PriorityQueue;

import edu.shanghaitech.ai.nlp.lveg.model.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.util.Executor;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class Parser<I, O> extends Recorder implements Executor<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7112164011234304607L;
	protected short maxLenParsing = 120;
	
	protected int idx;
	protected boolean reuse;
	protected boolean parallel;
	
	protected I task;
	protected int itask;
	protected Chart chart;
	protected PriorityQueue<Meta<O>> caches;
	
	@Override
	public void setIdx(int idx, PriorityQueue<Meta<O>> caches) {
		this.idx = idx;
		this.caches = caches;
	}
	
	@Override
	public void setNextTask(int itask, I task) {
		this.task = task;
		this.itask = itask;
	}
	
}
