package edu.shanghaitech.ai.nlp.lveg;

import java.util.PriorityQueue;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.lveg.MultiThreadedValuator.Score;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class Parser extends Recorder {
	
	protected static final short MAX_SENTENCE_LEN = 12;
	
	protected int id;
	
	protected int isample;
	protected Chart chart;
	protected Tree<State> sample;
	protected PriorityQueue<Score> scores;
	
	
	protected void setIdx(int idx, PriorityQueue<Score> scores) {
		this.id = idx;
		this.scores = scores;
	}
	
	protected void setNextSample(Tree<State> sample, int isample) {
		this.sample = sample;
		this.isample = isample;
	}
}
