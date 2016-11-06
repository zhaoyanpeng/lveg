package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGParser {
	
	private Inferencer inferencer;
	
	
	public LVeGParser(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.inferencer = new Inferencer(grammar, lexicon);
	}
	
	
	/**
	 * @param tree the parse tree
	 * @return
	 */
	public Inferencer.Chart doInsideOutside(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		Inferencer.Chart chart = new Inferencer.Chart(nword);
		
		inferencer.insideScore(chart, sentence, nword, false);
		chart.resetStatus();
		inferencer.outsideScore(chart, sentence, nword, false);
		
		return chart;
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
		double ll = jointdist / partition;
		
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
		double score = gm.eval();
		
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
		inferencer.insideScore(chart, sentence, nword, false);
		
		GaussianMixture gm = chart.getInsideScore((short) 0, 0, 1);
		double score = gm.eval();
		
		return score;
	}
	
}
