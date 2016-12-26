package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;
import java.util.concurrent.Callable;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;

public class Valuator<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1069775066639086250L;
	private LVeGInferencer inferencer;
	
	
	private Valuator(Valuator<?, ?> valuator) {
		this.inferencer = valuator.inferencer;
		this.chart = valuator.reuse ? new Chart(MAX_SENTENCE_LEN, false) : null;
		this.reuse = valuator.reuse;
	}
	
	
	public Valuator(LVeGGrammar grammar, LVeGLexicon lexicon, boolean reuse) {
		this.inferencer = new LVeGInferencer(grammar, lexicon);
		this.chart = reuse ? new Chart(MAX_SENTENCE_LEN, false) : null;
		this.reuse = reuse;
	}

	
	@Override
	public Valuator<?, ?> newInstance() {
		return new Valuator<I, O>(this);
	}
	
	
	
	@Override
	public synchronized Object call() {
		if (sample == null) { return null; }
		double ll = probability((Tree<State>) sample);
		Meta<O> cache = new Meta(isample, ll);
		synchronized (caches) {
			caches.add(cache);
			caches.notifyAll();
		}
		sample = null;
		return null;
	}
	

	/**
	 * Compute \log p(t | s) = \log {p(t, s) / p(s)}, where s denotes the 
	 * sentence, t is the parse tree.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	protected double probability(Tree<State> tree) {
		double jointdist = scoreTree(tree);
		double partition = scoreSentence(tree);
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
		if (reuse) {
			chart.clear(nword);
		} else {
			if (chart != null) { chart.clear(-1); }
			chart = new Chart(nword, false);
		}
		LVeGInferencer.insideScore(chart, sentence, nword);
		GaussianMixture gm = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double score = gm.eval(null, true);
		return score;
	}

}
