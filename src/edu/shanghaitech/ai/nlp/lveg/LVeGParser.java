package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;
import java.util.concurrent.Callable;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Cell;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.lveg.MultiThreadedValuator.Score;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.MethodUtil;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGParser extends Parser implements Callable {
	
	private LVeGInferencer inferencer;
	private boolean reuse;
	
	
	private LVeGParser(LVeGParser parser) {
		this.inferencer = parser.inferencer;
		this.chart = parser.reuse ? new Chart(MAX_SENTENCE_LEN, false) : null;
		this.reuse = parser.reuse;
	}
	
	
	public LVeGParser(LVeGGrammar grammar, LVeGLexicon lexicon, boolean reuse) {
		this.inferencer = new LVeGInferencer(grammar, lexicon);
		this.chart = reuse ? new Chart(MAX_SENTENCE_LEN, false) : null;
		this.reuse = reuse;
	}
	
	
	public double evalRuleCount(Tree<State> tree, short isample) {
		doInsideOutside(tree); 
//		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
//		MethodUtil.debugChart(Chart.getChart(true), (short) 2); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		MethodUtil.debugChart(Chart.getChart(false), (short) 2); // DEBUG
		
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double scoreS = score.eval(null, true);
		
//		logger.trace("Sentence score in logarithm: " + scoreS + ", Margin: " + score.marginalize(false) + "\n"); // DEBUG
//		logger.trace("\nEval rule count with the sentence...\n"); // DEBUG
		
		if (Double.isInfinite(scoreS) || Double.isNaN(scoreS)) {
			System.err.println("Fatal Error: Sentence score is smaller than zero: " + scoreS);
			return -0.0;
		}
		inferencer.evalRuleCount(tree, chart, isample);
		
//		logger.trace("\nCheck rule count with the sentence...\n"); // DEBUG
//		MethodUtil.debugCount(inferencer.grammar, inferencer.lexicon, tree, chart); // DEBUG
//		logger.trace("\nEval count with the sentence over.\n"); // DEBUG
		return scoreS;
	}
	
	
	public double evalRuleCountWithTree(Tree<State> tree, short isample) {
//		logger.trace("\nInside/outside scores with the tree...\n\n"); // DEBUG
		doInsideOutsideWithTree(tree); 
//		MethodUtil.debugTree(tree, false, (short) 2); // DEBUG
		
		// the parse tree score, which should contain only weights of the components
		GaussianMixture score = tree.getLabel().getInsideScore();
		double scoreT = score.eval(null, true);
		
//		logger.trace("\nTree score: " + scoreT + "\n"); // DEBUG
//		logger.trace("\nEval rule count with the tree...\n"); // DEBUG
		
		if (Double.isInfinite(scoreT) || Double.isNaN(scoreT)) {
			System.err.println("Fatal Error: Tree score is smaller than zero: " + scoreT + "\n");
			return -0.0;
		}
		// compute the rule counts
		inferencer.evalRuleCountWithTree(tree, isample);
		
//		logger.trace("\nCheck rule count with the tree...\n"); // DEBUG
//		MethodUtil.debugCount(inferencer.grammar, inferencer.lexicon, tree); // DEBUG
//		logger.trace("\nEval count with the tree over.\n"); // DEBUG
		return scoreT;
	}
	
	
	/**
	 * @param tree the parse tree
	 * @return
	 */
	private void doInsideOutside(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		if (reuse) {
			chart.clear(nword);
		} else {
			if (chart != null) { chart.clear(-1); }
			chart = new Chart(nword, false);
		}
//		logger.trace("\nInside score...\n"); // DEBUG
		inferencer.insideScore(chart, sentence, nword);
//		MethodUtil.debugChart(Chart.iGetChart(), (short) 2); // DEBUG

//		logger.trace("\nOutside score...\n"); // DEBUG
		inferencer.setRootOutsideScore(chart);
		inferencer.outsideScore(chart, sentence, nword);
//		MethodUtil.debugChart(Chart.oGetChart(), (short) 2); // DEBUG
	}
	
	
	/**
	 * Compute the inside and outside scores for 
	 * every non-terminal in the given parse tree. 
	 * 
	 * @param tree the parse tree
	 */
	private void doInsideOutsideWithTree(Tree<State> tree) {
//		logger.trace("\nInside score with the tree...\n"); // DEBUG	
		inferencer.insideScoreWithTree(tree);
//		MethodUtil.debugTree(tree, false, (short) 2); // DEBUG
		
//		logger.trace("\nOutside score with the tree...\n"); // DEBUG
		inferencer.setRootOutsideScore(tree);
		inferencer.outsideScoreWithTree(tree);
//		MethodUtil.debugTree(tree, false, (short) 2); // DEBUG
	}
	
	
	protected LVeGParser newInstance() {
		LVeGParser parser = new LVeGParser(this);
		return parser;
	}
	
	
	@Override
	public synchronized Double call() {
		if (sample == null) { return 0.0; }
		double ll = probability(sample);
		Score score = new Score(isample, ll);
		synchronized (scores) {
			scores.add(score);
			scores.notifyAll();
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
		inferencer.insideScoreWithTree(tree);
		GaussianMixture gm = tree.getLabel().getInsideScore();
		double score = gm.eval(null, true);
		return score;
	}
	
	
	/**
	 * Compute \sum_{t \in T} p(t, s), where T is the space of the parse tree.
	 * 
	 * @param tree in which only the sentence is used.
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
		inferencer.insideScore(chart, sentence, nword);
		GaussianMixture gm = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double score = gm.eval(null, true);
		return score;
	}
	
}
