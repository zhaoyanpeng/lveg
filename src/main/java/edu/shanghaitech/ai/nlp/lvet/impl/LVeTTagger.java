package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lvet.model.Inferencer;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.lvet.model.Tagger;
import edu.shanghaitech.ai.nlp.util.Debugger;
import edu.shanghaitech.ai.nlp.util.Executor;

public class LVeTTagger<I, O> extends Tagger<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4220883980665219868L;
	private LVeTInferencer inferencer;
	
	
	private LVeTTagger(LVeTTagger<?, ?> tagger) {
		super(tagger.maxslen, tagger.iosprune);
		this.inferencer = tagger.inferencer;
		this.chart = new Chart(tagger.maxslen, true, false, tagger.usemask, true);
		this.masks = tagger.masks;
	}
	
	public LVeTTagger(TagTPair ttpair, TagWPair twpair, short maxLenParsing, boolean iosprune) {
		super(maxLenParsing, iosprune);
		this.inferencer = new LVeTInferencer(ttpair, twpair);
		this.chart = new Chart(maxLenParsing, true, false, false, true);
	}
	
	@Override
	public synchronized Object call() throws Exception {
		List<TaggedWord> sample = (List<TaggedWord>) task;
		double scoreT = Double.NEGATIVE_INFINITY;
		double scoreS = Double.NEGATIVE_INFINITY;
		List<Double> scores = new ArrayList<>(3);
		int itree = masks == null ? -1 : Integer.valueOf(sample.get(0).signIdx);
//		synchronized (sample) { // why is it necessary to synchronize sample?
			scoreT = doForwardBackwardWithTags(sample); 
			scoreS = doForwardBackward(sample, itree); 
			scores.add(scoreT);
			scores.add(scoreS);
			scores.add((double) sample.size());
//		}
		
		if (Double.isFinite(scoreT) && Double.isFinite(scoreS)) {
			try { // do NOT expect it to crash
				synchronized (inferencer) {
					inferencer.evalEdgeCountWithTags(sample, (short) 0);
					inferencer.evalEdgeCount(sample, chart, (short) 0);
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
	public Executor<?, ?> newInstance() {
		return new LVeTTagger<I, O>(this);
	}
	
	public List<Double> evalEdgeCounts(List<TaggedWord> sequence, short isequence) {
		double scoreT = doForwardBackwardWithTags(sequence);
//		logger.trace("\nInside/outside scores with the tree...\n\n"); // DEBUG
//		logger.trace(Debugger.debugSequence(sequence, false, (short) 5, Inferencer.ttpair.numberer) + "\n");

		double scoreS = doForwardBackward(sequence, isequence);
//		logger.trace("\nInside scores with the sentence...\n\n"); // DEBUG
//		Debugger.debugChart(chart.getChart(true), (short) 5, sequence.size(), Inferencer.ttpair.numberer); // DEBUG
//		logger.trace("\nOutside scores with the sentence...\n\n"); // DEBUG
//		Debugger.debugChart(chart.getChart(false), (short) 5, sequence.size(), Inferencer.ttpair.numberer); // DEBUG
		
		List<Double> scores = new ArrayList<>(3);
		scores.add(scoreT);
		scores.add(scoreS);
		
		if (Double.isFinite(scoreT) && Double.isFinite(scoreS)) {
			try { // do NOT expect it to crash
				synchronized (inferencer) {
					inferencer.evalEdgeCountWithTags(sequence, isequence);
//					logger.trace("\nCheck rule count with the tree...\n"); // DEBUG
//					Debugger.debugCount(Inferencer.ttpair, Inferencer.twpair, true);
//					logger.trace("\nEval count with the tree over.\n"); // DEBUG
					
					inferencer.evalEdgeCount(sequence, chart, isequence);
//					logger.trace("\nCheck rule count with the sentence...\n"); // DEBUG
//					Debugger.debugCount(Inferencer.ttpair, Inferencer.twpair, false);
//					logger.trace("\nEval count with the sentence over.\n"); // DEBUG
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return scores;
	}
	
	public double doForwardBackwardWithTags(List<TaggedWord> sequence) {
		double scoreT = Double.NEGATIVE_INFINITY;
		try { // two scores should be equal
			scoreT = LVeTInferencer.forwardWithTags(sequence);
//			logger.info("\nForward scoreT :" + scoreT); 
			scoreT = LVeTInferencer.backwardWithTags(sequence);
//			logger.info("\nBackward scoreT:" + scoreT + "\n"); 
		} catch (Exception e) {
			e.printStackTrace();
		}
		return scoreT;
	}
	
	public double doForwardBackward(List<TaggedWord> sequence, int isequence) {
		double scoreS = Double.NEGATIVE_INFINITY;
		try {
			GaussianMixture score = null;
			int nword = sequence.size();
			if (chart != null) {
				chart.clear(nword);
			} else {
				chart = new Chart(maxslen, true, false, usemask, true);
			}
			Inferencer.forward(chart, sequence, nword); 
			score = chart.getInsideScore((short) Pair.ENDING_IDX, nword - 1, Inferencer.LEVEL_ONE);
			if (score != null) {
				scoreS = score.eval(null, true);
			}
//			logger.info("\nForward scoreS :" + scoreS);
			
			scoreS = Double.NEGATIVE_INFINITY;
			Inferencer.backward(chart, sequence, nword);
			score = chart.getOutsideScore((short) Pair.LEADING_IDX, 0, Inferencer.LEVEL_ONE);
			if (score != null) {
				scoreS = score.eval(null, true);
			}
//			logger.info("\nBackward scoreS:" + scoreS + "\n");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return scoreS;
	}
}
