package edu.shanghaitech.ai.nlp.lvet.model;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lvet.impl.TagTPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;

public class Inferencer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2077360985644905333L;
	
	public static TagWPair twpair;
	public static TagTPair ttpair;
	public static final short LEVEL_ZERO = 0;
	
	public void evalGradients(List<Double> scores) {
		twpair.evalGradients(scores);
		ttpair.evalGradients(scores);
	}
	
	public static void forward(Chart chart, List<TaggedWord> sequence, int nword) {
		List<GrammarRule> tedges, eedges; // transition & emission
		GaussianMixture pinScore, cinScore;
		Set<Short> tkeys;
		// time step 1
		eedges = twpair.getRulesWithWord(sequence.get(0));
		for (GrammarRule eedge : eedges) {
			GrammarRule tedge = null;
			if ((tedge = ttpair.getEdge((short) Pair.LEADING_IDX, eedge.lhs, GrammarRule.RHSPACE)) == null) {
				continue;
			}
			cinScore = tedge.weight.mulForInsideOutside(eedge.weight, GrammarRule.Unit.C, true);
			chart.addInsideScore(eedge.lhs, 0, cinScore, LEVEL_ZERO);
		}
		// time step [2, T]
		for (int i = 1; i < nword; i++) {
			tkeys = chart.keySet(i - 1, true, LEVEL_ZERO);
			if (tkeys == null || tkeys.size() == 0) {
				throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (i - 1) + " should not be empty.");
			}
			eedges = twpair.getRulesWithWord(sequence.get(i));
			for (GrammarRule eedge : eedges) {
				tedges = ttpair.getEdgeWithC(eedge.lhs);
				for (GrammarRule tedge : tedges) {
					if (!tkeys.contains(tedge.lhs)) { continue; }
					pinScore = chart.getInsideScore(tedge.lhs, i - 1, LEVEL_ZERO);
					cinScore = tedge.weight.mulForInsideOutside(pinScore, GrammarRule.Unit.P, true);
					cinScore = cinScore.mulForInsideOutside(eedge.weight, GrammarRule.Unit.UC, false);
					chart.addInsideScore(eedge.lhs, i, cinScore, LEVEL_ZERO);
				}
			}
		}
		// last step T + 1
		tkeys = chart.keySet(nword - 1, true, LEVEL_ZERO);
		if (tkeys == null || tkeys.size() == 0) {
			throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (nword - 1) + " should not be empty.");
		}
		for (Short tkey : tkeys) {
			GrammarRule tedge = null;
			if ((tedge = ttpair.getEdge(tkey, Pair.ENDING_IDX, GrammarRule.LHSPACE)) == null) {
				continue;
			}
			pinScore = chart.getInsideScore(tkey, nword - 1, LEVEL_ZERO);
			cinScore = tedge.weight.mulForInsideOutside(pinScore, GrammarRule.Unit.P, true);
			chart.addInsideScore((short) Pair.ENDING_IDX, nword, cinScore, LEVEL_ZERO);
		}
	}
	
	public static void backward(Chart chart, List<TaggedWord> sequence, int nword) {
		List<GrammarRule> tedges, eedges; // transition & emission
		GaussianMixture poutScore, coutScore;
		// time step T + 1
		// setEndBackwardScore(...)
		// time step T
		tedges = ttpair.getEdgeWithC(Pair.ENDING_IDX);
		for (GrammarRule tedge : tedges) {
			chart.addOutsideScore(tedge.lhs, nword - 1, tedge.weight.copy(true), LEVEL_ZERO);
		}
		// time step [1, T - 2]
		for (int i = nword - 2; i >= 0; i--) {
			Set<Short> tkeys = chart.keySet(i + 1, false, LEVEL_ZERO);
			if (tkeys == null || tkeys.size() == 0) {
				throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (i - 1) + " should not be empty.");
			}
			eedges = twpair.getRulesWithWord(sequence.get(i + 1));
			for (GrammarRule eedge : eedges) {
				if (!tkeys.contains(eedge.lhs)) { continue; }
				coutScore = chart.getOutsideScore(eedge.lhs, i + 1, LEVEL_ZERO);
				tedges = ttpair.getEdgeWithC(eedge.lhs);
				for (GrammarRule tedge : tedges) {
					coutScore = eedge.weight.mulForInsideOutside(coutScore, GrammarRule.Unit.P, true);
					poutScore = tedge.weight.mulForInsideOutside(coutScore, GrammarRule.Unit.UC, true);
					chart.addOutsideScore(tedge.lhs, i, poutScore, LEVEL_ZERO);
				}
			}
		}
	}
	
	public static void setEndBackwardScore(Chart chart, List<TaggedWord> sequence) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		chart.addOutsideScore((short) Pair.ENDING_IDX, sequence.size(), gm, LEVEL_ZERO);
	}
}
