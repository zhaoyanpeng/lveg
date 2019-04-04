package edu.shanghaitech.ai.nlp.lvet.model;

import java.util.PriorityQueue;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.util.Executor;
import edu.shanghaitech.ai.nlp.util.Recorder;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public abstract class Tagger<I, O> extends Recorder implements Executor<I, O>{
	/**
	 * 
	 */
	private static final long serialVersionUID = -8710006546132287351L;
	protected short maxslen = 120;
	
	protected int idx;
	
	protected short nthread;
	protected boolean parallel;
	protected boolean iosprune;
	protected boolean usemask;
	protected boolean usestag; // use golden tags in the sentence
	
	protected I task;
	protected int itask;
	protected Chart chart;
	protected PriorityQueue<Meta<O>> caches;
	protected Set<String>[][][] masks;
	
	protected transient ThreadPool cpool;
	
	protected Tagger(short maxslen, boolean iosprune) {
		this.maxslen = maxslen;
		this.iosprune = iosprune;
		// not supported
		this.usemask = false;
		this.usestag = false;
		this.parallel = false;
		this.nthread = -1;
	}
	
	protected Tagger(short maxslen, short nthread, boolean parallel, boolean iosprune, boolean usemask) {
		this.maxslen = maxslen;
		this.iosprune = iosprune;
		this.usemask = usemask;
		this.usestag = false;
		this.parallel = parallel;
		this.nthread = nthread < 0 ? 1 : nthread;
		if (parallel) {
			logger.error("Parallel CYKer in POS tagging is not supported.", new Throwable());
		}
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
