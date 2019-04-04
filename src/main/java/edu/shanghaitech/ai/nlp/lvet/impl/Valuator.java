package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.List;

import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lvet.model.Inferencer;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lvet.model.Tagger;

public class Valuator<I, O> extends Tagger<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1779944422969244044L;
	private LVeTInferencer inferencer;
	
	private Valuator(Valuator<?, ?> valuator) {
		super(valuator.maxslen, valuator.nthread, valuator.parallel, valuator.iosprune, false);
		this.inferencer = valuator.inferencer;
		this.chart = new Chart(valuator.maxslen, true, false, valuator.usemask, true);
	}
	
	
	public Valuator(TagTPair ttpair, TagWPair twpair, short maxLenParsing, boolean iosprune) {
		super(maxLenParsing, iosprune);
		this.inferencer = new LVeTInferencer(ttpair, twpair);
		this.chart = new Chart(maxLenParsing, true, false, false, true);
	}

	@Override
	public Valuator<?, ?> newInstance() {
		return new Valuator<I, O>(this);
	}

	@Override
	public synchronized Object call() throws Exception {
		List<TaggedWord> sample = (List<TaggedWord>) task;
		double ll = Double.NEGATIVE_INFINITY;
		synchronized (sample) {
			ll = probability(sample);
		}
		Meta<O> cache = new Meta(itask, ll);
		synchronized (caches) {
			caches.add(cache);
			caches.notify();
		}
		task = null;
		return itask;
	}
	
	public double probability(List<TaggedWord> sequence) {
		double ll = Double.NEGATIVE_INFINITY;
		try { // do NOT except it to crash 
			double jointdist = scoreTags(sequence);
			double partition = scoreSentence(sequence);
			ll = jointdist - partition;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return ll;
	}
	
	protected double scoreTags(List<TaggedWord> sequence) {
		double scoreT = LVeTInferencer.forwardWithTags(sequence);
//		logger.info("\nForward scoreT :" + scoreT);
//		scoreT = LVeTInferencer.forwardWithTags(sequence);
//		logger.info("\nForward scoreT :" + scoreT);
//		scoreT = LVeTInferencer.forwardWithTags(sequence);
//		logger.info("\nForward scoreT :" + scoreT);
		
//		scoreT = LVeTInferencer.backwardWithTags(sequence);
//		logger.info("\nBackward scoreT:" + scoreT);
//		scoreT = LVeTInferencer.backwardWithTags(sequence);
//		logger.info("\nBackward scoreT:" + scoreT);
//		scoreT = LVeTInferencer.backwardWithTags(sequence);
//		logger.info("\nBackward scoreT:" + scoreT + "\n");
		
		return scoreT;
	}
	
	protected double scoreSentence(List<TaggedWord> sequence) {
		int nword = sequence.size();
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, true, true, usemask, true);
		}
		Inferencer.forward(chart, sequence, nword);

		double scoreS = Double.NEGATIVE_INFINITY;
		GaussianMixture score = chart.getInsideScore((short) Pair.ENDING_IDX, nword - 1, Inferencer.LEVEL_ONE);
		if (score != null) {
			scoreS = score.eval(null, true);
		}
		return scoreS;
	}
	
	public double probability(List<TaggedWord> sequence, boolean reverse) {
		double ll = Double.NEGATIVE_INFINITY;
		try { // do NOT except it to crash 
			double jointdist = scoreTags(sequence, reverse);
			double partition = scoreSentence(sequence, reverse);
			ll = jointdist - partition;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return ll;
	}
	
	protected double scoreTags(List<TaggedWord> sequence, boolean reverse) {
		return reverse ? LVeTInferencer.backwardWithTags(sequence)
				: LVeTInferencer.forwardWithTags(sequence);
	}
	
	protected double scoreSentence(List<TaggedWord> sequence, boolean reverse) {
		int nword = sequence.size();
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, true, true, usemask, true);
		}
		
		GaussianMixture score = null;
		if (reverse) {
			Inferencer.backward(chart, sequence, nword);
			score = chart.getOutsideScore((short) Pair.LEADING_IDX, 0, Inferencer.LEVEL_ONE);
		} else {
			Inferencer.forward(chart, sequence, nword);
			score = chart.getInsideScore((short) Pair.ENDING_IDX, nword - 1, Inferencer.LEVEL_ONE);
		}
		
		double scoreS = Double.NEGATIVE_INFINITY;
		if (score != null) {
			scoreS = score.eval(null, true);
		}
		return scoreS;
	}
}
