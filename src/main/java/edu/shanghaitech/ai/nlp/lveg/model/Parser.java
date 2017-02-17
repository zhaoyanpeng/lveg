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
	protected boolean iosprune;
	protected boolean cntprune;
	
	protected I task;
	protected int itask;
	protected Chart chart;
	protected PriorityQueue<Meta<O>> caches;
	
	protected Parser(short maxLenParsing, boolean reuse, boolean iosprune) {
		this.maxLenParsing = maxLenParsing;
		this.iosprune = iosprune;
		this.reuse = reuse;
	}
	
	protected Parser(short maxLenParsing, boolean reuse, boolean iosprune, boolean cntprune) {
		this.maxLenParsing = maxLenParsing;
		this.iosprune = iosprune;
		this.cntprune = cntprune;
		this.reuse = reuse;
	}
	
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
