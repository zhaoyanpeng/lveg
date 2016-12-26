package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;

public class MaxRuleParser<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 514004588461969299L;
	private MaxRuleInferencer inferencer;
	
	
	private MaxRuleParser(MaxRuleParser<?, ?> parser) {
		this.inferencer = parser.inferencer;
		this.chart = parser.reuse ? new Chart(MAX_SENTENCE_LEN, true) : null;
		this.reuse = parser.reuse;
	}
	
	
	public MaxRuleParser(LVeGGrammar grammar, LVeGLexicon lexicon, boolean reuse) {
		this.inferencer = new MaxRuleInferencer(grammar, lexicon);
		this.chart = reuse ? new Chart(MAX_SENTENCE_LEN, true) : null;
		this.reuse = reuse;
	}
	

	@Override
	public synchronized Object call() throws Exception {
		if (sample == null) { return null; }
		Tree<String> tree = parse((Tree<State>) sample);
		Meta<O> cache = new Meta(isample, tree);
		synchronized (caches) {
			caches.add(cache);
			caches.notifyAll();
		}
		sample = null;
		return null;
	}
	

	@Override
	public MaxRuleParser<?, ?> newInstance() {
		return new MaxRuleParser<I, O>(this);
	}
	
	
	public Tree<String> parse(Tree<State> tree) {
//		logger.trace("eval max rule counts...");
		evalMaxRuleCount(tree);
//		logger.trace("over\n");

		Tree<String> strTree = StateTreeList.stateTreeToStringTree(tree, Inferencer.grammar.numberer);
		
//		logger.trace("extract max rule parse tree...");
		Tree<String> parseTree = inferencer.extractBestMaxRuleParse(chart, strTree.getYield());
//		logger.trace("over\n");
		
		return parseTree;
	}
	
	
	protected void evalMaxRuleCount(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		doInsideOutside(tree, sentence, nword);
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
			return;
		}
		inferencer.evalMaxRuleCount(chart, sentence, nword, scoreS);
	}
	
	
	/**
	 * @param tree the parse tree
	 * @return
	 */
	private Chart doInsideOutside(Tree<State> tree, List<State> sentence, int nword) {
		if (reuse) {
			chart.clear(nword);
		} else {
			if (chart != null) { chart.clear(-1); }
			chart = new Chart(nword, true);
		}
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
