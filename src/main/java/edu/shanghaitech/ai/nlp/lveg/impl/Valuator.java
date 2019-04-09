package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.syntax.State;

public class Valuator<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1069775066639086250L;
	private LVeGInferencer inferencer;
	
	
	private Valuator(Valuator<?, ?> valuator) {
		super(valuator.maxslen, valuator.nthread, valuator.parallel, valuator.iosprune, false);
		this.inferencer = valuator.inferencer;
		this.chart = new Chart(valuator.maxslen, true, false, false);
	}
	
	
	public Valuator(LVeGGrammar grammar, LVeGLexicon lexicon, short maxLenParsing, short nthread, 
			boolean parallel, boolean iosprune, boolean usemasks) {
		super(maxLenParsing, nthread, parallel, iosprune, false);
		this.inferencer = new LVeGInferencer(grammar, lexicon);
		this.chart = new Chart(maxLenParsing, true, false, false);
	}

	
	@Override
	public Valuator<?, ?> newInstance() {
		return new Valuator<I, O>(this);
	}
	
	
	@Override
	public synchronized Object call() {
		Tree<State> sample = (Tree<State>) task;
		double ll = Double.NEGATIVE_INFINITY;
		synchronized (sample) {
			ll = probability(sample);
		}
		Meta<O> cache = new Meta(itask, ll);
		synchronized (caches) {
			caches.add(cache);
			caches.notify();
		}
		task = null;
		return itask;
	}
	

	/**
	 * Compute \log p(t | s) = \log {p(t, s) / p(s)}, where s denotes the 
	 * sentence, t is the parse tree.
	 * 
	 * @param tree the parse tree
	 * @return     logarithmic conditional probability of the parse tree given the sentence 
	 */
	public double probability(Tree<State> tree) {
		double ll = Double.NEGATIVE_INFINITY;
		try { // do NOT except it to crash 
			double jointdist = scoreTree(tree);
			double partition = scoreSentence(tree);
			ll = jointdist - partition;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return ll;
	}
	
	
	/**
	 * Compute p(t, s), where s denotes the sentence, t is a parse tree.
	 * 
	 * @param tree the parse tree
	 * @return     score of the parse tree
	 */
	protected double scoreTree(Tree<State> tree) {
		LVeGInferencer.insideScoreWithTree(tree);
		double scoreT = Double.NEGATIVE_INFINITY;
		GaussianMixture score = tree.getLabel().getInsideScore();
		if (score != null) {
			scoreT = score.eval(null, true);
		}
		return scoreT;
	}
	
	
	/**
	 * Compute \sum_{t \in T} p(t, s), where T is the space of the parse tree.
	 * 
	 * @param tree in which only the sentence is used
	 * @return the sentence score
	 */
	protected double scoreSentence(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, true, false, false);
		}
		if (parallel) {
			cpool.reset();
			Inferencer.insideScore(chart, sentence, nword, iosprune, cpool);
		} else {
			Inferencer.insideScore(chart, sentence, null, nword, iosprune, false, false);
		}
		double scoreS = Double.NEGATIVE_INFINITY;
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		if (score != null) {
			scoreS = score.eval(null, true);
		}
		return scoreS;
	}

}
