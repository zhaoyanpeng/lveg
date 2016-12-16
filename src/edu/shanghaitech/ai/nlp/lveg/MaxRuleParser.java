package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;

public class MaxRuleParser extends Parser {
	private MaxRuleInferencer inferencer;
	
	public MaxRuleParser(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.inferencer = new MaxRuleInferencer(grammar, lexicon);
	}
	
	
	public Tree<String> parse(Tree<State> tree) {
//		logger.trace("eval max rule counts...");
		Chart chart = evalMaxRuleCount(tree);
//		logger.trace("over\n");
		
		if (chart == null) { 
			logger.error("rule-counts chart is null.\n"); 
			return null;
		}
		Tree<String> strTree = StateTreeList.stateTreeToStringTree(tree, inferencer.grammar.tagNumberer);
		
//		logger.trace("extract max rule parse tree...");
		Tree<String> parseTree = inferencer.extractBestMaxRuleParse(chart, strTree.getYield());
//		logger.trace("over\n");
		
		return parseTree;
	}
	
	
	public Chart evalMaxRuleCount(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		Chart chart = doInsideOutside(tree, sentence, nword);
//		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
//		MethodUtil.debugChart(Chart.getChart(true), (short) 2); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		MethodUtil.debugChart(Chart.getChart(false), (short) 2); // DEBUG
		
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		double scoreS = score.eval(null, true);
		
//		logger.trace("\nSentence score in logarithm: " + scoreS + ", Margin: " + score.marginalize(false) + "\n"); // DEBUG
//		logger.trace("\nEval rule count with the sentence...\n"); // DEBUG
		
		if (Double.isInfinite(scoreS) || Double.isNaN(scoreS)) {
			System.err.println("Fatal Error: Sentence score is smaller than zero: " + scoreS);
			return null;
		}
		inferencer.evalMaxRuleCount(chart, sentence, nword, scoreS);
		return chart;
	}
	
	
	/**
	 * @param tree the parse tree
	 * @return
	 */
	public Chart doInsideOutside(Tree<State> tree, List<State> sentence, int nword) {
		Chart chart = new Chart(nword, true);
		
//		logger.trace("\nInside score...\n"); // DEBUG
		inferencer.insideScore(chart, sentence, nword);
//		MethodUtil.debugChart(Chart.iGetChart(), (short) 2); // DEBUG

//		logger.trace("\nOutside score...\n"); // DEBUG
		inferencer.setRootOutsideScore(chart);
		inferencer.outsideScore(chart, sentence, nword);
//		MethodUtil.debugChart(Chart.oGetChart(), (short) 2); // DEBUG
		
		return chart;
	}
}
