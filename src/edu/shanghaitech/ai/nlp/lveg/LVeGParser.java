package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGParser extends Recorder {
	
	private LVeGInferencer inferencer;
	
	
	public LVeGParser(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.inferencer = new LVeGInferencer(grammar, lexicon);
	}
	
	
	public double evalRuleCount(Tree<State> tree, short isample) {
		Chart chart = doInsideOutside(tree);
		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
		MethodUtil.debugChart(Chart.getChart(true), (short) 2); // DEBUG
		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
		MethodUtil.debugChart(Chart.getChart(false), (short) 2); // DEBUG
		
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double scoreS = score.eval(null, true);
		
		logger.trace("Sentence score in logarithm: " + scoreS + ", Margin: " + score.marginalize(false) + "\n"); // DEBUG
		logger.trace("\nEval rule count with the sentence...\n"); // DEBUG
		
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
	public Chart doInsideOutside(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		Chart chart = new Chart(nword);
		
//		logger.trace("\nInside score...\n"); // DEBUG
		inferencer.insideScore(chart, sentence, nword);
//		MethodUtil.debugChart(Chart.iGetChart(), (short) 2); // DEBUG

//		logger.trace("\nOutside score...\n"); // DEBUG
		inferencer.setRootOutsideScore(chart);
		inferencer.outsideScore(chart, sentence, nword);
//		MethodUtil.debugChart(Chart.oGetChart(), (short) 2); // DEBUG
		
		return chart;
	}
	
	
	/**
	 * Compute the inside and outside scores for 
	 * every non-terminal in the given parse tree. 
	 * 
	 * @param tree the parse tree
	 */
	public void doInsideOutsideWithTree(Tree<State> tree) {
//		logger.trace("\nInside score with the tree...\n"); // DEBUG	
		inferencer.insideScoreWithTree(tree);
//		MethodUtil.debugTree(tree, false, (short) 2); // DEBUG
		
//		logger.trace("\nOutside score with the tree...\n"); // DEBUG
		inferencer.setRootOutsideScore(tree);
		inferencer.outsideScoreWithTree(tree);
//		MethodUtil.debugTree(tree, false, (short) 2); // DEBUG
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
		double ll = jointdist - partition; // in logarithm
		return ll;
	}
	
	
	/**
	 * Compute p(t, s), where s denotes the sentence, t is a parse tree.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	public double scoreTree(Tree<State> tree) {
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
	public double scoreSentence(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		Inferencer.Chart chart = new Inferencer.Chart(nword);
		inferencer.insideScore(chart, sentence, nword);
		GaussianMixture gm = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double score = gm.eval(null, true);
		return score;
	}
	
}
