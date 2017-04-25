package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Executor;

public class PCFGMaxRuleParser<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2668797817745391783L;
	private PCFGMaxRuleInferencer inferencer;
	
	
	private PCFGMaxRuleParser(PCFGMaxRuleParser<?, ?> parser) {
		super(parser.maxslen, parser.nthread, parser.parallel, parser.iosprune, parser.usemask);
		this.inferencer = parser.inferencer; // shared by multiple threads
		this.chart = new Chart(parser.maxslen, false, true, true);
	}
	
	
	public PCFGMaxRuleParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxslen, short nthread,
			boolean parallel, boolean iosprune, boolean usemasks) {
		super(maxslen, nthread, parallel, iosprune, usemasks);
		this.inferencer = new PCFGMaxRuleInferencer(grammar, lexicon);
		this.chart = new Chart(maxslen, false, true, true);
	}

	
	@Override
	public Executor<?, ?> newInstance() {
		return new PCFGMaxRuleParser<I, O>(this);
	}

	
	@Override
	public synchronized Object call() throws Exception {
		Tree<State> sample = (Tree<State>) task;
		Tree<String> parsed = parse(sample);
		Meta<O> cache = new Meta(itask, parsed);
		synchronized (caches) {
			caches.add(cache);
			caches.notify();
		}
		task = null;
		return itask;
	}
	
	
	/**
	 * Dedicated to error handling while recovering the recorded best parse path.
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
	
	
	private boolean evalMaxRuleCount(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		double scoreS = doInsideOutside(tree, sentence, nword);
//		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
//		FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		
		if (Double.isFinite(scoreS)) {
//			logger.trace("\nEval rule count with the sentence...\n"); // DEBUG
			inferencer.evalMaxRuleProdCount(chart, sentence, nword, scoreS);
//			inferencer.evalMaxRuleSumCount(chart, sentence, nword, scoreS);
			return true;
		}
		return false;
	}
	
	
	private double doInsideOutside(Tree<State> tree, List<State> sentence, int nword) {
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, false, true, true);
		}
		
//		logger.trace("\nInside score...\n"); // DEBUG
		PCFGInferencer.insideScore(chart, sentence, nword, false, -1, -1);
//		FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG

		PCFGInferencer.setRootOutsideScore(chart);
//		logger.trace("\nOutside score...\n"); // DEBUG
		PCFGInferencer.outsideScore(chart, sentence, nword, false, -1, -1);
//		FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		
		double scoreS = Double.NEGATIVE_INFINITY;
		if (chart.containsKeyMask((short) 0, Chart.idx(0, 1), true)) {
			scoreS = chart.getInsideScoreMask((short) 0, Chart.idx(0, 1));
		}
		return scoreS;
	}

}
