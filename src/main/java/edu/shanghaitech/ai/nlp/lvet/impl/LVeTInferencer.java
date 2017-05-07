package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lvet.model.Inferencer;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;

public class LVeTInferencer extends Inferencer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1171380507772132745L;
	
	public LVeTInferencer(TagTPair attpair, TagWPair atwpair) {
		ttpair = attpair;
		twpair = atwpair;
	}
	
	public static double forwardWithTags(List<TaggedWord> sequence) {
		int nword;
		double scoreT = Double.NEGATIVE_INFINITY;
		if ((nword = sequence.size()) <= 0) { return scoreT; }
		GaussianMixture pinScore, cinScore, eedgeW, tedgeW;
		TaggedWord word, preword;
		// time step 1
		word = sequence.get(0);
		cinScore = word.getInsideScore();
		eedgeW = twpair.score(word, (short) word.tagIdx);
		tedgeW = ttpair.getEdgeWeight((short) Pair.LEADING_IDX, word.tagIdx, GrammarRule.RHSPACE, false);
		cinScore = tedgeW.mulAndMarginalize(eedgeW, cinScore, GrammarRule.Unit.C, true);
		word.setInsideScore(cinScore);
		// time step [2, T]
		preword = word;
		for (int i = 1; i < nword; i++) {
			word = sequence.get(i);
			cinScore = word.getInsideScore();
			pinScore = preword.getInsideScore();
			eedgeW = twpair.score(word, (short) word.tagIdx);
			tedgeW = ttpair.getEdgeWeight((short) preword.tagIdx, word.tagIdx, GrammarRule.LRURULE, false);
			cinScore = tedgeW.mulAndMarginalize(pinScore, cinScore, GrammarRule.Unit.P, true);
			cinScore = cinScore.mulAndMarginalize(eedgeW, cinScore, GrammarRule.Unit.UC, false);
			word.setInsideScore(cinScore);
			preword = word;
		}
		// time step T + 1
		pinScore = preword.getInsideScore();
		tedgeW = ttpair.getEdgeWeight((short) preword.tagIdx, Pair.ENDING_IDX, GrammarRule.LHSPACE, false);
		cinScore = tedgeW.mulAndMarginalize(pinScore, null, GrammarRule.Unit.P, true);
		if (cinScore != null) {
			scoreT = cinScore.eval(null, true);
		}
		return scoreT;
	}
	
	public static void backwardWithTags(List<TaggedWord> sequence) {
		int nword;
		if ((nword = sequence.size()) <= 0) { return; }
		GaussianMixture poutScore, coutScore, eedgeW, tedgeW;
		TaggedWord word, postword;
		// time step T
		word = sequence.get(nword - 1);
		tedgeW = ttpair.getEdgeWeight((short) word.tagIdx, Pair.ENDING_IDX, GrammarRule.LHSPACE, false);
		poutScore = tedgeW.copy(true);
		word.setOutsideScore(poutScore);
		postword =  word;
		// time step [1, T]
		for (int i = nword - 2; i <= 0; i--) {
			word = sequence.get(i);
			poutScore = word.getOutsideScore();
			coutScore = postword.getOutsideScore();
			eedgeW = twpair.score(postword, (short) postword.tagIdx);
			tedgeW = ttpair.getEdgeWeight((short) word.tagIdx, postword.tagIdx, GrammarRule.LRURULE, false);
			coutScore = eedgeW.mulAndMarginalize(coutScore, null, GrammarRule.Unit.P, true);
			poutScore = tedgeW.mulAndMarginalize(coutScore, poutScore, GrammarRule.Unit.UC, true);
			word.setOutsideScore(poutScore);
			postword = word;
		}
	}
	
	public static void evalEdgeCountWithTags(List<TaggedWord> sequence) {
		int nword;
		if ((nword = sequence.size()) <= 0) { return; }
		GaussianMixture outScore, pinScore, eedgeW, tedgeW;
		Map<String, GaussianMixture> scores;
		TaggedWord word, preword;
		// time step 1
		word = sequence.get(0);
		outScore = word.getOutsideScore();
		scores = new HashMap<String, GaussianMixture>(2, 1);
		eedgeW = twpair.score(word, (short) word.tagIdx);

	}
	
	public static void evalEdgeCount(List<TaggedWord> sequence, Chart chart) {
		
	}
	
	public static void setEndBackwardScore(List<TaggedWord> sequence) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		sequence.get(sequence.size() - 1).setOutsideScore(gm);
	}

}
