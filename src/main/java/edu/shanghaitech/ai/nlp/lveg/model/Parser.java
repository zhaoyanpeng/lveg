package edu.shanghaitech.ai.nlp.lveg.model;

import java.util.PriorityQueue;

import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer.InputToSubCYKer;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer.SubCYKer;
import edu.shanghaitech.ai.nlp.util.Executor;
import edu.shanghaitech.ai.nlp.util.Recorder;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public abstract class Parser<I, O> extends Recorder implements Executor<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7112164011234304607L;
	protected short maxslen = 120;
	
	protected int idx;
	protected short nthread;
	protected boolean parallel;
	protected boolean iosprune;
	protected boolean usemask;
	
	protected I task;
	protected int itask;
	protected Chart chart;
	protected PriorityQueue<Meta<O>> caches;
	
	protected transient ThreadPool cpool;
	
	protected Parser(short maxslen, short nthread, boolean parallel, boolean iosprune, boolean usemask) {
		this.maxslen = maxslen;
		this.iosprune = iosprune;
		this.usemask = usemask;
		this.parallel = parallel;
		this.nthread = nthread < 0 ? 1 : nthread;
		if (parallel) {
			SubCYKer<?, ?> subCYKer = new SubCYKer<InputToSubCYKer, Boolean>();
			this.cpool = new ThreadPool(subCYKer, nthread);
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
