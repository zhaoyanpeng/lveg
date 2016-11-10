package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
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
	
	
	public boolean evalRuleCount(Tree<State> tree) {
		Chart chart = doInsideOutside(tree);
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double sentenceScore = score.eval();
		if (sentenceScore <= 0) {
			System.err.println("Fatal Error: Sentence score is smaller than zero.");
			return false;
		}
		inferencer.evalRuleCount(tree, chart, sentenceScore);
		return true;
	}
	
	
	public boolean evalRuleCountWithTree(Tree<State> tree) {
		// inside and outside scores are stored in the non-terminals of the tree
		doInsideOutsideWithTree(tree); 
		
		// the parse tree score, which should contain only weights of the components
		GaussianMixture score = tree.getLabel().getInsideScore();
		double treeScore = score.eval();
		if (treeScore <= 0) {
			System.err.println("Fatal Error: Tree score is smaller than zero.");
			return false;
		}
		// compute the rule counts in a recursive way
		inferencer.evalRuleCountWithTree(tree, treeScore);
		return true;
	}
	
	
	/**
	 * @param tree the parse tree
	 * @return
	 */
	public Chart doInsideOutside(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		Chart chart = new Inferencer.Chart(nword);
		
		inferencer.insideScore(chart, sentence, nword, false);
		inferencer.setRootOutsideScore(chart);
		inferencer.outsideScore(chart, sentence, nword, false);
		
		return chart;
	}
	
	
	/**
	 * Compute the inside and outside scores for 
	 * every non-terminal in the given parse tree. 
	 * 
	 * @param tree the parse tree
	 */
	public void doInsideOutsideWithTree(Tree<State> tree) {
		inferencer.insideScoreWithTree(tree);
		inferencer.setRootOutsideScore(tree);
		inferencer.outsideScoreWithTree(tree);
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
		
		GaussianMixture gm = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double score = gm.eval();
		
		return score;
	}
	
}
