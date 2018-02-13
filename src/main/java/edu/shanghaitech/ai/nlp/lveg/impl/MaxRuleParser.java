package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
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
	
	private Set<String>[][][] masks;
	
	
	private MaxRuleParser(MaxRuleParser<?, ?> parser) {
		super(parser.maxslen, parser.nthread, parser.parallel, parser.iosprune, parser.usemask);
		this.inferencer = parser.inferencer;
		this.chart = new Chart(parser.maxslen, true, true, parser.usemask);
		this.masks = parser.masks;
	}
	
	
	public MaxRuleParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxLenParsing, short nthread, 
			boolean parallel, boolean iosprune, boolean usemasks, Set<String>[][][] masks) {
		super(maxLenParsing, nthread, parallel, iosprune, usemasks);
		this.inferencer = new MaxRuleInferencer(grammar, lexicon);
		this.chart = new Chart(maxLenParsing, true, true, usemasks);
		this.masks = masks;
	}
	
	
	@Override
	public MaxRuleParser<?, ?> newInstance() {
		return new MaxRuleParser<I, O>(this);
	}
	

	@Override
	public synchronized Object call() throws Exception {
		Tree<State> sample = (Tree<State>) task;
		Tree<String> parsed = parse(sample, itask);
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
	public Tree<String> parse(Tree<State> tree, int itree) {
		Tree<String> parsed = null;
		try { // do NOT expect it to crash
			boolean valid = evalMaxRuleCount(tree, itree);
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
	 * Compute pseudo counts of grammar rules, and find a best parse path.
	 * 
	 * @param tree the golden parse tree
	 * @return     whether the sentence can be parsed (true) of not (false)
	 */
	private boolean evalMaxRuleCount(Tree<State> tree, int itree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		double scoreS = doInsideOutside(tree, sentence, itree, nword);
//		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
//		FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		
		if (Double.isFinite(scoreS)) {
//			logger.trace("\nEval rule count with the sentence...\n"); // DEBUG
			inferencer.evalMaxRuleCount(chart, sentence, nword, scoreS);
			return true;
		}
		return false;
	}
	
	
	/**
	 * Inside/outside score calculation is required by MaxRule parser.
	 * 
	 * @param tree     the golden parse tree
	 * @param sentence the sentence need to be parsed
	 * @param nword    length of the sentence
	 * @return
	 */
	private double doInsideOutside(Tree<State> tree, List<State> sentence, int itree, int nword) {
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, true, true, usemask);
		}
		if (usemask) {
			boolean status = usemask;
			if (masks != null) { // obtained from kbest PCFG parsing
				status = createKbestMask(nword, chart, masks[itree]);
			}
			if (!status || masks == null) { // posterior probability
				createPCFGMask(nword, chart, sentence);
			}
		}
		if (parallel) {
			cpool.reset();
			Inferencer.insideScore(chart, sentence, nword, iosprune, cpool);
			Inferencer.setRootOutsideScore(chart);
			cpool.reset();
			Inferencer.outsideScore(chart, sentence, nword, iosprune, cpool);
		} else {
//			logger.trace("\nInside score...\n"); // DEBUG
			Inferencer.insideScore(chart, sentence, nword, iosprune, usemask, LVeGTrainer.iomask);
//			FunUtil.debugChart(chart.getChart(true), (short) -1, tree.getYield().size()); // DEBUG

			Inferencer.setRootOutsideScore(chart);
//			logger.trace("\nOutside score...\n"); // DEBUG
			Inferencer.outsideScore(chart, sentence, nword, iosprune, usemask, LVeGTrainer.iomask);
//			FunUtil.debugChart(chart.getChart(false), (short) -1, tree.getYield().size()); // DEBUG
		}
		double scoreS = Double.NEGATIVE_INFINITY;
		GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
		if (score != null) {
			scoreS = score.eval(null, true);
		} else if (usemask && masks != null) { // re-parse using pcfg pruning, assuming k-best pruning fails
			chart.clear(nword);
			
			createPCFGMask(nword, chart, sentence);
		
			if (parallel) {
				cpool.reset();
				Inferencer.insideScore(chart, sentence, nword, iosprune, cpool);
				Inferencer.setRootOutsideScore(chart);
				cpool.reset();
				Inferencer.outsideScore(chart, sentence, nword, iosprune, cpool);
			} else {
				Inferencer.insideScore(chart, sentence, nword, iosprune, usemask, LVeGTrainer.iomask);
				Inferencer.setRootOutsideScore(chart);
				Inferencer.outsideScore(chart, sentence, nword, iosprune, usemask, LVeGTrainer.iomask);
			}
			score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
			if (score != null) {
				scoreS = score.eval(null, true);
			}
		}
		return scoreS;
	}
	
	
	private static boolean createKbestMask(int nword, Chart chart, Set<String>[][] mask) {
		int len = mask.length, idx, layer;
		if (nword != len) { return false; }
		for (int i = 0; i < len; i++) {
			for (int j = i; j < len; j++) {
				layer = nword - j + i; // nword - (j - i)
				idx = layer * (layer - 1) / 2 + i; // (nword - 1 + 1)(nword - 1) / 2 
				for (String label : mask[i][j]) {
					short ikey = (short) Inferencer.grammar.numberer.number(label);
					chart.addPosteriorMask(ikey, idx);
				}
			}
		}
		return true;
	}
	
	
	private static void createPCFGMask(int nword, Chart chart, List<State> sentence) {
		PCFGInferencer.insideScore(chart, sentence, nword, LVeGTrainer.iomask, LVeGTrainer.tgBase, LVeGTrainer.tgRatio);
		PCFGInferencer.setRootOutsideScore(chart);
		PCFGInferencer.outsideScore(chart, sentence, nword, LVeGTrainer.iomask,  LVeGTrainer.tgBase, LVeGTrainer.tgRatio);
		if (!LVeGTrainer.iomask) { // not use inside/outside score masks
			double score = chart.getInsideScoreMask((short) 0, Chart.idx(0, 1));
			PCFGInferencer.createPosteriorMask(nword, chart, score, LVeGTrainer.tgProb);
		}
	}
	
}
