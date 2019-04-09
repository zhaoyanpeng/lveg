package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.EnumMap;
import java.util.List;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lvet.model.Inferencer;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;

/**
 * This implementation is somewhat counter-intuitive in the sense that the forward procedure
 * traverses the input sentence from root node to leaf nodes, which is determined by the 
 * spanning direction of production rules. Even though it can be modified to be read-friendly,
 * I do not think this is necessary cause I prefer to understand the algorithm from different 
 * points of view.
 */
public class LVeTInferencer extends Inferencer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1171380507772132745L;
	
	public LVeTInferencer(TagTPair attpair, TagWPair atwpair) {
		ttpair = attpair;
		twpair = atwpair;
	}
	
	/**
	 * Forward procedure accidently computes outside scores. 
	 * <p>
	 * Ther are two types of outside scores: 
	 * <p>
	 * t_1 -> t_2: without word, outside scores at t_2 with key RuleUnit.C or RuleUnit.UC
	 * <p>
	 * t_1 -> t_2 -> w_2: with word, outside scores at t_2 with key RuleUnit.P
	 * <p>
	 * For emission rules, only after forward and backward procedures, can we compute their outside scores.
	 * @param sequence
	 * @return
	 */
	public static double forwardWithTags(List<TaggedWord> sequence) {
		int nword;
		double scoreT = Double.NEGATIVE_INFINITY;
		if ((nword = sequence.size()) <= 0) { return scoreT; }
		GaussianMixture pinScore, cinScore, eedgeW, tedgeW;
		TaggedWord word, preword;
		// time step 1
		word = sequence.get(0);
		eedgeW = twpair.score(word, (short) word.tagIdx);
		tedgeW = ttpair.getEdgeWeight((short) Pair.LEADING_IDX, word.tagIdx, RuleType.RHSPACE, false);
		
		cinScore = tedgeW.copy(true);
		word.setInsideScore(cinScore, false); // C remains
		
		pinScore = word.getInsideScore(true);
		pinScore = eedgeW.mul(cinScore, pinScore, RuleUnit.P);
		word.setInsideScore(pinScore, true);
		// time step [2, T]
		preword = word;
		for (int i = 1; i < nword; i++) {
			word = sequence.get(i);
			eedgeW = twpair.score(word, (short) word.tagIdx);
			tedgeW = ttpair.getEdgeWeight((short) preword.tagIdx, word.tagIdx, RuleType.LRURULE, false);
			
			pinScore = preword.getInsideScore(true);
			
			cinScore = word.getInsideScore(false);
			cinScore = tedgeW.mulAndMarginalize(pinScore, cinScore, RuleUnit.P, true); // UC remains
			word.setInsideScore(cinScore, false);
			
			pinScore = word.getInsideScore(true);
			pinScore = eedgeW.mul(cinScore, pinScore, RuleUnit.P); // P remains
			word.setInsideScore(pinScore, true);
			preword = word;
		}
		// time step T + 1
		pinScore = preword.getInsideScore(true);
		tedgeW = ttpair.getEdgeWeight((short) preword.tagIdx, Pair.ENDING_IDX, RuleType.LHSPACE, false);
		cinScore = tedgeW.mulAndMarginalize(pinScore, null, RuleUnit.P, true);
		if (cinScore != null) {
			scoreT = cinScore.eval(null, true);
		}
		return scoreT;
	}
	
	/**
	 * Backward procedure accidently computes inside scores.
	 * <p>
	 * There are two types of inside scores.
	 * <p>
	 * t_1 -> t_2: without word, outside scores at t_1 with key RuleUnit.P
	 * <p>
	 * t_1 -> t_2 / w_1: with word, outside scores at t_1 with key RuleUnit.P
	 * <p>
	 * @param sequence
	 * @return
	 */
	public static double backwardWithTags(List<TaggedWord> sequence) {
		int nword;
		double scoreT = Double.NEGATIVE_INFINITY;
		if ((nword = sequence.size()) <= 0) { return scoreT; }
		GaussianMixture poutScore, coutScore, eedgeW, tedgeW;
		TaggedWord word, postword;
		// time step T
		word = sequence.get(nword - 1);
		eedgeW = twpair.score(word, (short) word.tagIdx);
		tedgeW = ttpair.getEdgeWeight((short) word.tagIdx, Pair.ENDING_IDX, RuleType.LHSPACE, false);
		
		coutScore = tedgeW.copy(true);
		word.setOutsideScore(coutScore, false);
		
		poutScore = word.getOutsideScore(true);
		poutScore = eedgeW.mul(coutScore, poutScore, RuleUnit.P);
		word.setOutsideScore(poutScore, true);
		// time step [1, T]
		postword =  word;
		for (int i = nword - 2; i >= 0; i--) {
			word = sequence.get(i);
			eedgeW = twpair.score(word, (short) word.tagIdx);
			tedgeW = ttpair.getEdgeWeight((short) word.tagIdx, postword.tagIdx, RuleType.LRURULE, false);
			
			poutScore = postword.getOutsideScore(true);
			
			coutScore = word.getOutsideScore(false);
			coutScore = tedgeW.mulAndMarginalize(poutScore, coutScore, RuleUnit.UC, true); // P remains
			word.setOutsideScore(coutScore, false);
			
			poutScore = word.getOutsideScore(true);
			poutScore = eedgeW.mul(coutScore, poutScore, RuleUnit.P); // P remains
			word.setOutsideScore(poutScore, true);
			postword = word;
		}
		// time step 0
		poutScore = postword.getOutsideScore(true);
		tedgeW = ttpair.getEdgeWeight((short) Pair.LEADING_IDX, postword.tagIdx, RuleType.RHSPACE, false);
		coutScore = tedgeW.mulAndMarginalize(poutScore, null, RuleUnit.C, true);
		if (coutScore != null) {
			scoreT = coutScore.eval(null, true);
		}
		return scoreT;
	}
	
	public void evalEdgeCount(List<TaggedWord> sequence, Chart chart, short isample) {
		int nword;
		if ((nword = sequence.size()) <= 0) { return; }
		GaussianMixture cinScore, loutScore, routScore, outScore;
		EnumMap<RuleUnit, GaussianMixture> scores = null;
		List<GrammarRule> tedges, eedges; // transition & emission
		Set<Short> tkeys;
		// time step [1, T]
		for (int i = 0; i < nword; i++) {
			eedges = twpair.getEdgesWithWord(sequence.get(i));
			for (GrammarRule rule : eedges) {
				UnaryGrammarRule eedge = (UnaryGrammarRule) rule;
				// transition rule		
				if (i > 0) {
					tkeys = chart.keySet(i - 1, true, LEVEL_ZERO);
					if (tkeys == null || tkeys.size() == 0) {
						throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (i - 1) + " should not be empty.");
					}
					tedges = ttpair.getEdgeWithC(eedge.lhs);
					for (GrammarRule tedge : tedges) {
						if (!tkeys.contains(tedge.lhs)) { continue; }
						if ((cinScore = chart.getOutsideScore(eedge.lhs, i, LEVEL_ZERO)) == null || 
								(outScore = chart.getInsideScore(tedge.lhs, i - 1, LEVEL_ZERO)) == null) {
							continue;
						}
						scores = new EnumMap<>(RuleUnit.class);
						scores.put(RuleUnit.P, outScore);
						scores.put(RuleUnit.UC, cinScore);
						ttpair.addCount(tedge.lhs, eedge.lhs, scores, RuleType.LRURULE, isample, false);
					}
				} else { 
					if (ttpair.getEdge((short) Pair.LEADING_IDX, eedge.lhs, RuleType.RHSPACE) != null) {
						if ((cinScore = chart.getOutsideScore(eedge.lhs, 0, LEVEL_ZERO)) == null ||
								(outScore = chart.getInsideScore(eedge.lhs, 0, LEVEL_ONE)) == null ) {
							continue;
						}
						scores = new EnumMap<>(RuleUnit.class);
						scores.put(RuleUnit.P, outScore);
						scores.put(RuleUnit.C, cinScore);
						ttpair.addCount((short) Pair.LEADING_IDX, eedge.lhs, scores, RuleType.RHSPACE, isample, false);
					}
				}
				// emission rule
				if ((loutScore = chart.getInsideScore(eedge.lhs, i, LEVEL_ONE)) != null &&
						(routScore = chart.getOutsideScore(eedge.lhs, i, LEVEL_ONE)) != null) {
					outScore = routScore.mul(loutScore, null, RuleUnit.P);
					cinScore = eedge.weight.copy(true);
					
					scores = new EnumMap<>(RuleUnit.class);
					scores.put(RuleUnit.P, outScore);
					scores.put(RuleUnit.C, cinScore);
					twpair.addCount(eedge.lhs, eedge.rhs, scores, RuleType.LHSPACE, isample, false);
				} // there must be at least one possible path for an emission rule
			}
		}
		// time step T + 1
		tkeys = chart.keySet(nword - 1, true, LEVEL_ZERO);
		if (tkeys == null || tkeys.size() == 0) {
			throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (nword - 1) + " should not be empty.");
		}
		for (Short tkey : tkeys) {
			if (ttpair.getEdge(tkey, Pair.ENDING_IDX, RuleType.LHSPACE) == null) {
				continue;
			}
			if ((cinScore = chart.getOutsideScore(tkey, nword - 1, LEVEL_ONE)) == null ||
					(outScore = chart.getInsideScore(tkey, nword - 1, LEVEL_ZERO)) == null) {
				continue;
			}
			scores = new EnumMap<>(RuleUnit.class);
			scores.put(RuleUnit.P, outScore);
			scores.put(RuleUnit.C, cinScore);
			ttpair.addCount(tkey, Pair.ENDING_IDX, scores, RuleType.LHSPACE, isample, false);
		}
	}
	
	public void evalEdgeCountWithTags(List<TaggedWord> sequence, short isample) {
		int nword;
		if ((nword = sequence.size()) <= 0) { return; }
		GaussianMixture cinScore, loutScore, routScore, outScore;
		EnumMap<RuleUnit, GaussianMixture> scores = null;
		TaggedWord word, preword = null;
		// time step [1, T]
		for (int i = 0; i < nword; i++) {
			word = sequence.get(i);
			cinScore = word.getOutsideScore(true);
			// transition rule
			scores = new EnumMap<>(RuleUnit.class);
			if (preword != null) {
				outScore = preword.getInsideScore(true);
				scores.put(RuleUnit.P, outScore);
				scores.put(RuleUnit.UC, cinScore);
				ttpair.addCount((short) preword.tagIdx, word.tagIdx, scores, RuleType.LRURULE, isample, true);
			} else {
				outScore = word.getInsideScore(false); // edge weight itself
				scores.put(RuleUnit.P, outScore);
				scores.put(RuleUnit.C, cinScore);
				ttpair.addCount((short) Pair.LEADING_IDX, word.tagIdx, scores, RuleType.RHSPACE, isample, true);				
			}
			// emission rule
			loutScore = word.getInsideScore(false);
			routScore = word.getOutsideScore(false);
			outScore = routScore.mul(loutScore, null, RuleUnit.P);
			cinScore = twpair.score(word, (short) word.tagIdx).copy(true); // edge weight itself
			
			scores = new EnumMap<>(RuleUnit.class);
			scores.put(RuleUnit.P, outScore);
			scores.put(RuleUnit.C, cinScore);
			twpair.addCount((short) word.tagIdx, word.wordIdx, scores, RuleType.LHSPACE, isample, true);
			preword = word;
		}
		// time step T + 1
		outScore = preword.getInsideScore(true); 
		cinScore = preword.getOutsideScore(false); // edge weight itself
		scores = new EnumMap<>(RuleUnit.class);
		scores.put(RuleUnit.P, outScore);
		scores.put(RuleUnit.C, cinScore);
		ttpair.addCount((short) preword.tagIdx, Pair.ENDING_IDX, scores, RuleType.LHSPACE, isample, true);
	}
	
}
