package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;

public class MaxRuleParser<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 514004588461969299L;
	private MaxRuleInferencer inferencer;
	
	
	private MaxRuleParser(MaxRuleParser<?, ?> parser) {
		super(parser.maxLenParsing, parser.nthread, parser.parallel, parser.reuse, parser.iosprune, parser.usemasks);
		this.inferencer = parser.inferencer;
		this.chart = parser.reuse ? new Chart(maxLenParsing, true, usemasks) : null;
	}
	
	
	public MaxRuleParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxLenParsing, short nthread, 
			boolean parallel, boolean reuse, boolean iosprune, boolean usemasks) {
		super(maxLenParsing, nthread, parallel, reuse, iosprune, usemasks);
		this.inferencer = new MaxRuleInferencer(grammar, lexicon);
		this.chart = reuse ? new Chart(maxLenParsing, true, usemasks) : null;
	}
	

	@Override
	public synchronized Object call() throws Exception {
		if (task == null) { return null; }
		Tree<State> sample = (Tree<State>) task;
		evalMaxRuleCount(sample);
		Tree<String> tree = StateTreeList.stateTreeToStringTree(sample, Inferencer.grammar.numberer);
		tree = inferencer.extractBestMaxRuleParse(chart, tree.getYield());
		/*
		synchronized (inferencer) {
			tree = inferencer.extractBestMaxRuleParse(chart, tree.getYield());
		}
		*/
		Meta<O> cache = new Meta(itask, tree);
		synchronized (caches) {
			caches.add(cache);
			caches.notifyAll();
		}
		task = null;
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
//		FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
//		double scoreS = score == null ? Double.MAX_VALUE : score.eval(null, true);
		double scoreS = score.eval(null, true); // score != null
		
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
			chart = new Chart(nword, true, usemasks);
		}
		if (usemasks) {
			Inferencer.insideScoreMask(chart, sentence, nword, true,  LVeGTrainer.tgBase, LVeGTrainer.tgRatio);
			Inferencer.setRootOutsideScoreMask(chart);
			Inferencer.outsideScoreMask(chart, sentence, nword, true,  LVeGTrainer.tgBase, LVeGTrainer.tgRatio);
		}
//		logger.trace("\nInside score...\n"); // DEBUG
		if (parallel) {
			cpool.reset();
			Inferencer.insideScore(chart, sentence, nword, iosprune, cpool, usemasks);
			Inferencer.setRootOutsideScore(chart);
			cpool.reset();
			Inferencer.outsideScore(chart, sentence, nword, iosprune, cpool, usemasks);
		} else {
			Inferencer.insideScore(chart, sentence, nword, iosprune, usemasks);
//			FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG

//			logger.trace("\nOutside score...\n"); // DEBUG
			Inferencer.setRootOutsideScore(chart);
			Inferencer.outsideScore(chart, sentence, nword, iosprune, usemasks);
//			FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		}
		
		return chart;
	}
	
}
