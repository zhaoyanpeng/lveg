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
		if (task == null) { return null; }
		double ll = probability((Tree<State>) task);
		Meta<O> cache = new Meta(itask, ll);
		synchronized (caches) {
			caches.add(cache);
			caches.notifyAll();
		}
		task = null;
		return null;
	}
	

	/**
	 * Compute \log p(t | s) = \log {p(t, s) / p(s)}, where s denotes the 
	 * sentence, t is the parse tree.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	public double probability(Tree<State> tree) {
		double jointdist = scoreTree(tree);
		double partition = scoreSentence(tree);
//		logger.trace("\n+++jointdist: " + jointdist + "\tpartition: " + partition + "\n"); // DEBUG
		double ll = jointdist - partition; // in logarithm
		return ll;
	}
	
	
	/**
	 * Compute p(t, s), where s denotes the sentence, t is a parse tree.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	protected double scoreTree(Tree<State> tree) {
		LVeGInferencer.insideScoreWithTree(tree);
		GaussianMixture gm = tree.getLabel().getInsideScore();
		double score = gm.eval(null, true);
//		FunUtil.debugTree(tree, true, (short) -1, Inferencer.grammar.numberer); // DEBUG
//		logger.trace("\n" + score + " " + Math.exp(score) + "\n");
		return score;
	}
	
	
	/**
	 * Compute \sum_{t \in T} p(t, s), where T is the space of the parse tree.
	 * 
	 * @param tree in which only the sentence is used
	 * @return
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
			Inferencer.insideScore(chart, sentence, nword, iosprune, cpool, false);
		} else {
			Inferencer.insideScore(chart, sentence, nword, iosprune, false);
		}
		GaussianMixture gm = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double score = gm.eval(null, true);
//		FunUtil.debugChart(chart.getChart(true), (short) -1, nword); // DEBUG
//		logger.trace("\n" + score + " " + Math.exp(score) + "\n");
		return score;
	}

}
