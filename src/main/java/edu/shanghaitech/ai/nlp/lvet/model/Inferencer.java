package edu.shanghaitech.ai.nlp.lvet.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
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
	public static final short LEVEL_ONE = 1;
	public final static String DUMMY_TAG = "OOPS_ROOT";
	
	public void evalGradients(List<Double> scores) {
		twpair.evalGradients(scores);
		ttpair.evalGradients(scores);
	}
	
	public static void forward(Chart chart, List<TaggedWord> sequence, int nword) {
		List<GrammarRule> tedges, eedges; // transition & emission
		GaussianMixture pinScore, cinScore;
		Set<Short> tkeys;
		// time step 1
		eedges = twpair.getEdgesWithWord(sequence.get(0));
		for (GrammarRule eedge : eedges) {
			GrammarRule tedge = null;
			if ((tedge = ttpair.getEdge((short) Pair.LEADING_IDX, eedge.lhs, RuleType.RHSPACE)) == null) {
				continue;
			}
			
			cinScore = tedge.weight.copy(true);
			chart.addInsideScore(eedge.lhs, 0, cinScore, LEVEL_ONE); // C remains
			
			pinScore = eedge.weight.mul(cinScore, null, RuleUnit.P);
			chart.addInsideScore(eedge.lhs, 0, pinScore, LEVEL_ZERO);
		}
		// time step [2, T]
		for (int i = 1; i < nword; i++) {
			tkeys = chart.keySet(i - 1, true, LEVEL_ZERO);
			if (tkeys == null || tkeys.size() == 0) {
				throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (i - 1) + " should not be empty.");
			}
			eedges = twpair.getEdgesWithWord(sequence.get(i));
			for (GrammarRule eedge : eedges) {
				tedges = ttpair.getEdgeWithC(eedge.lhs);
				for (GrammarRule tedge : tedges) {
					if (!tkeys.contains(tedge.lhs)) { continue; }
					pinScore = chart.getInsideScore(tedge.lhs, i - 1, LEVEL_ZERO);
					
					cinScore = tedge.weight.mulAndMarginalize(pinScore, null, RuleUnit.P, true); // UC remains
					chart.addInsideScore(eedge.lhs, i, cinScore, LEVEL_ONE);
					
					pinScore = eedge.weight.mul(cinScore, null, RuleUnit.P); // P remains
					chart.addInsideScore(eedge.lhs, i, pinScore, LEVEL_ZERO);
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
			if ((tedge = ttpair.getEdge(tkey, Pair.ENDING_IDX, RuleType.LHSPACE)) == null) {
				continue;
			}
			pinScore = chart.getInsideScore(tkey, nword - 1, LEVEL_ZERO);
			cinScore = tedge.weight.mulAndMarginalize(pinScore, null, RuleUnit.P, true);
			chart.addInsideScore((short) Pair.ENDING_IDX, nword - 1, cinScore, LEVEL_ONE);
		}
	}
	
	public static void backward(Chart chart, List<TaggedWord> sequence, int nword) {
		List<GrammarRule> tedges, eedges; // transition & emission
		GaussianMixture poutScore, coutScore;
		Set<Short> tkeys;
		// time step T
		eedges = twpair.getEdgesWithWord(sequence.get(nword - 1));
		for (GrammarRule eedge : eedges) {
			GrammarRule tedge = null;
			if ((tedge = ttpair.getEdge(eedge.lhs, Pair.ENDING_IDX, RuleType.LHSPACE)) == null) {
				continue;
			}
			coutScore = tedge.weight.copy(true);
			chart.addOutsideScore(eedge.lhs, nword - 1, coutScore, LEVEL_ONE);
			
			poutScore = eedge.weight.mul(coutScore, null, RuleUnit.P);
			chart.addOutsideScore(tedge.lhs, nword - 1, poutScore, LEVEL_ZERO);
		}
		// time step [1, T - 1]
		for (int i = nword - 2; i >= 0; i--) {
			tkeys = chart.keySet(i + 1, false, LEVEL_ZERO);
			if (tkeys == null || tkeys.size() == 0) {
				throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (i - 1) + " should not be empty.");
			}
			eedges = twpair.getEdgesWithWord(sequence.get(i));
			for (GrammarRule eedge : eedges) {
				tedges = ttpair.getEdgeWithP(eedge.lhs);
				for (GrammarRule tedge : tedges) {
					Short rhs = (short) ((UnaryGrammarRule) tedge).rhs;
					if (!tkeys.contains(rhs)) { continue; }
					poutScore = chart.getOutsideScore(rhs, i + 1, LEVEL_ZERO);
					
					coutScore = tedge.weight.mulAndMarginalize(poutScore, null, RuleUnit.UC, true); // P remains
					chart.addOutsideScore(eedge.lhs, i, coutScore, LEVEL_ONE);
					
					poutScore = eedge.weight.mul(coutScore, null, RuleUnit.P); // P remains
					chart.addOutsideScore(eedge.lhs, i, poutScore, LEVEL_ZERO);
				}
			}
		}
		// last time step 0
		Set<Short> keys = chart.keySet(0, false, LEVEL_ZERO);
		if (keys == null || keys.size() == 0) {
			throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (nword - 1) + " should not be empty.");
		}
		tkeys = new HashSet<Short>(keys); // to avoid `java.util.ConcurrentModificationException` since we need to add LEADING_IDX to the cell
		for (Short tkey : tkeys) {
			GrammarRule tedge = null;
			if ((tedge = ttpair.getEdge((short) Pair.LEADING_IDX, tkey, RuleType.RHSPACE)) == null) {
				continue;
			}
			coutScore = chart.getOutsideScore(tkey, 0, LEVEL_ZERO);
			poutScore = tedge.weight.mulAndMarginalize(coutScore, null, RuleUnit.C, true);
			chart.addOutsideScore((short) Pair.LEADING_IDX, 0, poutScore, LEVEL_ONE);
		}
	}
	
	public static List<String> extractBestMaxRuleTags(Chart chart, List<TaggedWord> sequence) {
		List<String> parsed = new ArrayList<String>();
		extractBestMaxRuleTags(chart, parsed, (short) Pair.ENDING_IDX, sequence.size() - 1);
		return parsed;
	}
	
	private static void extractBestMaxRuleTags(Chart chart, List<String> parsed, short key, int idx) {
		if (key == Pair.LEADING_IDX) {
			return;
		}
		
		short son = (short) chart.getMaxRuleSon(key, idx);
		idx = key != Pair.ENDING_IDX ? idx - 1 : idx;
		if (son == Pair.LEADING_IDX) {
			return;
		}
		
		extractBestMaxRuleTags(chart, parsed, son, idx);
		
		String tag = (String) ttpair.numberer.object(son);
		parsed.add(tag);
	}
	
}
