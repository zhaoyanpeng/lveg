package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.syntax.State;

public class MaxRuleParser<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 514004588461969299L;
	private MaxRuleInferencer inferencer;
	
	
	private MaxRuleParser(MaxRuleParser<?, ?> parser) {
		super(parser.maxslen, parser.nthread, parser.parallel, parser.iosprune, false);
		this.inferencer = parser.inferencer;
		this.chart = new Chart(parser.maxslen, true, true, false);
	}
	
	
	public MaxRuleParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxLenParsing, short nthread, 
			boolean parallel, boolean iosprune, boolean usemasks) {
		super(maxLenParsing, nthread, parallel, iosprune, usemasks);
		this.inferencer = new MaxRuleInferencer(grammar, lexicon);
		this.chart = new Chart(maxLenParsing, true, true, false);
	}
	

	@Override
	public synchronized Object call() throws Exception {
		Tree<State> sample = (Tree<State>) task;
		Tree<String> parsed = null;
		synchronized (sample) {
			parsed = parse(sample);
		}
		Meta<O> cache = new Meta(itask, parsed);
		synchronized (caches) {
			caches.add(cache);
			caches.notify();
		}
		task = null;
		return itask;
	}
	

	@Override
	public MaxRuleParser<?, ?> newInstance() {
		return new MaxRuleParser<I, O>(this);
	}
	
	
	/**
	 * Dedicated to error handling.
	 * 
	 * @param tree the golden parse tree
	 * @return     parse tree given the sentence
	 */
	public Tree<String> parse(Tree<State> tree) {
		Tree<String> parsed = null;
		try { // do NOT expect it to crash
			boolean valid = evalMaxRuleCount(tree);
			if (valid) {
				parsed = StateTreeList.stateTreeToStringTree(tree, Inferencer.grammar.numberer);
				parsed = Inferencer.extractBestMaxRuleParse(chart, parsed.getYield());
			} else {
				parsed = new Tree<String>(Inferencer.DUMMY_TAG);
			}
		} catch (Exception e) {
			parsed = new Tree<String>(Inferencer.DUMMY_TAG);
			e.printStackTrace();
		}
		return parsed;
	}
	
	
	/**
	 * Compute grammar rules counts in each chart cell.
	 * 
	 * @param tree the golden parse tree
	 * @return     whether the sentence can be parsed (true) of not (false)
	 */
	protected boolean evalMaxRuleCount(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		double scoreS = doInsideOutside(tree, sentence, nword);
//		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
//		FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		
		if (Double.isFinite(scoreS)) {
//			logger.trace("\nSentence score in logarithm: " + scoreS + ", Margin: " + score.marginalize(false) + "\n"); // DEBUG
//			logger.trace("\nEval rule count with the sentence...\n"); // DEBUG
			
//			synchronized (inferencer) { // read-only, no synchronization needed in fact
				inferencer.evalMaxRuleCount(chart, sentence, nword, scoreS);
//			}
			return true;
		}
		return false;
	}
	
	
	/**
	 * @param tree     the golden parse tree
	 * @param sentence the sentence need to be parsed
	 * @param nword    length of the sentence
	 * @return
	 */
	private double doInsideOutside(Tree<State> tree, List<State> sentence, int nword) {
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, true, true, false);
		}
		if (parallel) {
			cpool.reset();
			Inferencer.insideScore(chart, sentence, nword, iosprune, cpool);
			Inferencer.setRootOutsideScore(chart);
			cpool.reset();
			Inferencer.outsideScore(chart, sentence, nword, iosprune, cpool);
		} else {
//			logger.trace("\nInside score...\n"); // DEBUG
			Inferencer.insideScore(chart, sentence, nword, iosprune, false, false);
//			FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG

			Inferencer.setRootOutsideScore(chart);
//			logger.trace("\nOutside score...\n"); // DEBUG
			Inferencer.outsideScore(chart, sentence, nword, iosprune, false, false);
//			FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		}
		double scoreS = Double.NEGATIVE_INFINITY;
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		if (score != null) {
			scoreS = score.eval(null, true);
		}
		return scoreS;
	}
	
}
