package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Debugger;
import edu.shanghaitech.ai.nlp.util.FunUtil;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGParser<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1363406979999225830L;
	private LVeGInferencer inferencer;
	
	
	private LVeGParser(LVeGParser<?, ?> parser) {
		super(parser.maxslen, parser.nthread, parser.parallel, parser.iosprune, parser.usemask);
		this.inferencer = parser.inferencer;
		this.chart = new Chart(parser.maxslen, true, false, parser.usemask);
	}
	
	
	public LVeGParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxLenParsing, short nthread, 
			boolean parallel, boolean iosprune, boolean usemasks) {
		super(maxLenParsing, nthread, parallel, iosprune, usemasks);
		this.inferencer = new LVeGInferencer(grammar, lexicon);
		this.chart = new Chart(maxLenParsing, true, false, usemasks);
	}
	
	
	@Override
	public synchronized Object call() throws Exception {
		Tree<State> sample = (Tree<State>) task;
		double scoreT = Double.NEGATIVE_INFINITY;
		double scoreS = Double.NEGATIVE_INFINITY;
		List<Double> scores = new ArrayList<>(3);
//		synchronized (sample) { // why is it necessary to synchronize sample?
			scoreT = doInsideOutsideWithTree(sample); 
			scoreS = doInsideOutside(sample); 
			scores.add(scoreT);
			scores.add(scoreS);
			scores.add((double) sample.getYield().size());
//		}
		
		if (Double.isFinite(scoreT) && Double.isFinite(scoreS)) {
			try { // do NOT expect it to crash
				synchronized (inferencer) {
					inferencer.evalRuleCountWithTree(sample, (short) 0);
					inferencer.evalRuleCount(sample, chart, (short) 0, false);
					inferencer.evalGradients(scores);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		Meta<O> cache = new Meta(itask, scores);
		synchronized (caches) {
			caches.add(cache);
			caches.notify();
		}
		task = null;
		return itask;
	}


	@Override
	public LVeGParser<?, ?> newInstance() {
		return new LVeGParser<I, O>(this);
	}
	
	
	public List<Double> evalRuleCounts(Tree<State> tree, short isample) {
		double scoreT = doInsideOutsideWithTree(tree); 
//		logger.trace("\nInside/outside scores with the tree...\n\n"); // DEBUG
//		logger.trace(Debugger.debugTree(tree, false, (short) 2, Inferencer.grammar.numberer, false) + "\n"); // DEBUG
		
		double scoreS = doInsideOutside(tree); 
//		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
//		Debugger.debugChart(chart.getChart(true), (short) -1, tree.getYield().size(), Inferencer.grammar.numberer); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		Debugger.debugChart(chart.getChart(false), (short) -1, tree.getYield().size(), Inferencer.grammar.numberer); // DEBUG
		
		List<Double> scores = new ArrayList<>(3);
		scores.add(scoreT);
		scores.add(scoreS);
		
		if (Double.isFinite(scoreT) && Double.isFinite(scoreS)) {
			try { // do NOT expect it to crash
				synchronized (inferencer) {
					inferencer.evalRuleCountWithTree(tree, isample);
//					logger.trace("\nCheck rule count with the tree...\n"); // DEBUG
//					Debugger.debugCount(Inferencer.grammar, Inferencer.lexicon, tree); // DEBUG
//					logger.trace("\nEval count with the tree over.\n"); // DEBUG
					
					inferencer.evalRuleCount(tree, chart, isample, false);
//					logger.trace("\nCheck rule count with the sentence...\n"); // DEBUG
//					Debugger.debugCount(Inferencer.grammar, Inferencer.lexicon, tree, chart); // DEBUG
//					logger.trace("\nEval count with the sentence over.\n"); // DEBUG
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		return scores;
	}
	
	
	/**
	 * @param tree the golden parse tree
	 * @return     score of the sentence
	 */
	public double doInsideOutside(Tree<State> tree) {
		double scoreS = Double.NEGATIVE_INFINITY;
		try { // do NOT expect it to crash
			List<State> sentence = tree.getYield();
			int nword = sentence.size();
			if (chart != null) {
				chart.clear(nword);
			} else {
				chart = new Chart(maxslen, true, false, usemask);
			}
			if (usemask) {
//				logger.trace("\nInside score masks...\n"); // DEBUG
				PCFGInferencer.insideScore(chart, sentence, nword, LVeGTrainer.iomask, LVeGTrainer.tgBase, LVeGTrainer.tgRatio);
//				FunUtil.debugChart(chart.getChartMask(true), (short) -1, tree.getYield().size(), Inferencer.grammar.numberer); // DEBUG
				
//				logger.trace("\nOutside score masks...\n"); // DEBUG
				PCFGInferencer.setRootOutsideScore(chart);
				PCFGInferencer.outsideScore(chart, sentence, nword, LVeGTrainer.iomask,  LVeGTrainer.tgBase, LVeGTrainer.tgRatio);
//				FunUtil.debugChart(chart.getChartMask(false), (short) -1, tree.getYield().size(), Inferencer.grammar.numberer); // DEBUG
				
				if (!LVeGTrainer.iomask) { // not use inside/outside score masks
					double score = chart.getInsideScoreMask((short) 0, Chart.idx(0, 1));
					PCFGInferencer.createPosteriorMask(nword, chart, score, LVeGTrainer.tgProb);
//					logger.trace("\nSENTENCE SCORE: " + scoreS + "\n"); // DEBUG
//					logger.trace("\nPosterior masks...\n"); // DEBUG
//					FunUtil.debugChart(chart.getChartTmask(), (short) -1, tree.getYield().size(), Inferencer.grammar.numberer); // DEBUG
				}
			}
			if (parallel) {
				cpool.reset();
				Inferencer.insideScore(chart, sentence, nword, iosprune, cpool);
				Inferencer.setRootOutsideScore(chart);
				cpool.reset();
				Inferencer.outsideScore(chart, sentence, nword, iosprune, cpool);
			} else {
//				logger.trace("\nInside score...\n"); // DEBUG
				Inferencer.insideScore(chart, sentence, nword, iosprune, usemask, LVeGTrainer.iomask);
//				Debugger.debugChart(chart.getChart(true), (short) -1, tree.getYield().size(), Inferencer.grammar.numberer); // DEBUG
		
//				logger.trace("\nOutside score...\n"); // DEBUG
				Inferencer.setRootOutsideScore(chart);
				Inferencer.outsideScore(chart, sentence, nword, iosprune, usemask, LVeGTrainer.iomask);
//				Debugger.debugChart(chart.getChart(false), (short) -1, tree.getYield().size(), Inferencer.grammar.numberer); // DEBUG
			}
			
//			System.exit(0);
			
			GaussianMixture score = chart.getInsideScore((short) 0, Chart.idx(0, 1));
			if (score != null) {
				scoreS = score.eval(null, true);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return scoreS;
	}
	
	
	/**
	 * Compute the inside and outside scores for very non-terminal in the given parse tree. 
	 * 
	 * @param tree the golden parse tree
	 * @return     score of the parse tree
	 */
	public double doInsideOutsideWithTree(Tree<State> tree) {
		double scoreT = Double.NEGATIVE_INFINITY;
		try { // do NOT expect it to crash
//			logger.trace("\nInside score with the tree...\n"); // DEBUG	
			LVeGInferencer.insideScoreWithTree(tree);
//			Debugger.debugTree(tree, false, (short) 2, Inferencer.grammar.numberer, false); // DEBUG
			
//			logger.trace("\nOutside score with the tree...\n"); // DEBUG
			LVeGInferencer.setRootOutsideScore(tree);
			LVeGInferencer.outsideScoreWithTree(tree);
//			Debugger.debugTree(tree, false, (short) 2, Inferencer.grammar.numberer, false); // DEBUG
		} catch (Exception e) {
			e.printStackTrace();
		}
		// the parse tree score, which should contain only weights of the components
		GaussianMixture score = tree.getLabel().getInsideScore();
		if (score != null) {
			scoreT = score.eval(null, true);
		}
//		logger.trace("\nTree score: " + scoreT + "\n"); // DEBUG
//		logger.trace("\nEval rule count with the tree...\n"); // DEBUG
		return scoreT;
	}
	
}
