package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lvet.model.Inferencer;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.lvet.model.Tagger;

public class MaxRuleTagger<I, O> extends Tagger<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -117180047284812720L;
	private MaxRuleInferencer inferencer;
	
	private MaxRuleTagger(MaxRuleTagger<?, ?> tagger) {
		super(tagger.maxslen, tagger.iosprune);
		this.inferencer = tagger.inferencer;
		this.chart = new Chart(tagger.maxslen, true, false, tagger.usemask, true);
		this.masks = tagger.masks;
	}
	
	public MaxRuleTagger(TagTPair ttpair, TagWPair twpair, short maxLenParsing, boolean iosprune) {
		super(maxLenParsing, iosprune);
		this.inferencer = new MaxRuleInferencer(ttpair, twpair);
		this.chart = new Chart(maxLenParsing, true, false, false, true);
	}

	@Override
	public MaxRuleTagger<?, ?> newInstance() {
		return new MaxRuleTagger<I, O>(this);
	}

	@Override
	public synchronized Object call() throws Exception {
		List<TaggedWord> sample = (List<TaggedWord>) task;
		List<String> parsed = parse(sample, itask); 
		Meta<O> cache = new Meta(itask, parsed);
		synchronized (caches) {
			caches.add(cache);
			caches.notify();
		}
		task = null;
		return itask;
	}
	
	public List<String> parse(List<TaggedWord> sequence, int isequence) {
		List<String> parsed = null;
		try {
			boolean valid = evalMaxRuleCount(sequence, isequence);
			if (valid) {
				parsed = Inferencer.extractBestMaxRuleTags(chart, sequence);
			} else {
				parsed = new ArrayList<String>();
				parsed.add(Inferencer.DUMMY_TAG);
			}
		} catch (Exception e) {
			parsed = new ArrayList<String>();
			parsed.add(Inferencer.DUMMY_TAG);
			e.printStackTrace();
		}
		return parsed;
	}
	
	private boolean evalMaxRuleCount(List<TaggedWord> sequence, int isequence) {
		double scoreS = doForwardBackward(sequence, isequence);
		if (Double.isFinite(scoreS)) {
			inferencer.evalMaxRuleCount(chart, sequence, scoreS);
			return true;
		}
		return false;
	}
	
	private double doForwardBackward(List<TaggedWord> sequence, int isequence) {
		int nword = sequence.size();
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, true, true, usemask, true);
		}
		Inferencer.forward(chart, sequence, nword);
		Inferencer.backward(chart, sequence, nword);
		
		double scoreS = Double.NEGATIVE_INFINITY;
		GaussianMixture score = chart.getOutsideScore((short) Pair.LEADING_IDX, 0, Inferencer.LEVEL_ONE);
		if (score != null) {
			scoreS = score.eval(null, true);
		}
		return scoreS;
	}

}
