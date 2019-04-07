package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lvet.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lvet.model.Inferencer;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;

public class MaxRuleInferencer extends Inferencer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4929906393973221172L;
	
	public MaxRuleInferencer(TagTPair attpair, TagWPair atwpair) {
		ttpair = attpair;
		twpair = atwpair;
	}
	
	protected void evalMaxRuleCount(Chart chart, List<TaggedWord> sequence, double scoreS) {
		int nword;
		if ((nword = sequence.size()) <= 0) { return; }
		EnumMap<RuleUnit, GaussianMixture> scores = null;
		GaussianMixture cinScore, outScore;
		double lcount, rcount, maxcnt, newcnt;
		List<GrammarRule> tedges, eedges; // transition & emission
		Set<Short> tkeys;
		// time step 0
		eedges = twpair.getEdgesWithWord(sequence.get(0));
		for (GrammarRule eedge : eedges) {
			GrammarRule tedge = null;
			if ((tedge = ttpair.getEdge((short) Pair.LEADING_IDX, eedge.lhs, RuleType.RHSPACE)) == null) {
				continue;
			}
			if ((cinScore = chart.getOutsideScore(eedge.lhs, 0, LEVEL_ZERO)) == null) {
					continue;
			} // the emission rule must be valid
			
			scores = new EnumMap<>(RuleUnit.class);
			scores.put(RuleUnit.C, cinScore);
			lcount = tedge.weight.mulAndMarginalize(scores) - scoreS;
			
			// count for the emission rule; these scores must exist cause it has passed the check
			outScore = chart.getInsideScore(eedge.lhs, 0, LEVEL_ZERO); // is always indexed by P, word combined
			cinScore = chart.getOutsideScore(eedge.lhs, 0, LEVEL_ONE); // is always indexed by P, word excluded
			scores = new EnumMap<>(RuleUnit.class);
			scores.put(RuleUnit.P, outScore);
			rcount = cinScore.mulAndMarginalize(scores) - scoreS;
			
			newcnt = lcount + rcount;
			chart.addMaxRuleCount(eedge.lhs, 0, newcnt, tedge.lhs, (short) -1, LEVEL_ZERO);
		}
		// time step [1, T]
		for (int i = 1; i < nword; i++) {
			eedges = twpair.getEdgesWithWord(sequence.get(i));
			// find constraints
			tkeys = chart.keySetMaxRule(i - 1, LEVEL_ZERO);
			if (tkeys == null || tkeys.size() == 0) {
				throw new RuntimeException("OOPS_BUG: key set of tags in time step " + (i - 1) + " should not be empty.");
			}
			// transition and emission rule 
			for (GrammarRule rule : eedges) {
				UnaryGrammarRule eedge = (UnaryGrammarRule) rule;
				if (!chart.containsKey(eedge.lhs, i, true, LEVEL_ZERO) || // LEVEL_ZERO and LEVEL_ONE imply each other
					!chart.containsKey(eedge.lhs, i, false, LEVEL_ONE)) {
					continue;
				} // the emission rule must be valid
				tedges = ttpair.getEdgeWithC(eedge.lhs);
				for (GrammarRule tedge : tedges) {
					if (!tkeys.contains(tedge.lhs)) {
						continue;
					} // the transition rule must be valid
					
					newcnt = chart.getMaxRuleCount(tedge.lhs, i - 1);
					if ((maxcnt = chart.getMaxRuleCount(eedge.lhs, i)) > newcnt) {
						continue;
					} // skip if impossible
					
					// compute counts for the left-to-right transition rule and the current emission rule
					
					// count for the transition rule
					cinScore = chart.getOutsideScore(eedge.lhs, i, LEVEL_ZERO); // must exist because of mutual implication
					outScore = chart.getInsideScore(tedge.lhs, i - 1, LEVEL_ZERO); // must exist, just think about it
					scores = new EnumMap<>(RuleUnit.class);
					scores.put(RuleUnit.P, outScore);
					scores.put(RuleUnit.UC, cinScore);
					lcount = tedge.weight.mulAndMarginalize(scores) - scoreS;
					
					// count for the emission rule; these scores must exist cause it has passed the check
					outScore = chart.getInsideScore(eedge.lhs, i, LEVEL_ZERO); // is always indexed by P, word combined
					cinScore = chart.getOutsideScore(eedge.lhs, i, LEVEL_ONE); // is always indexed by P, word excluded
					scores = new EnumMap<>(RuleUnit.class);
					scores.put(RuleUnit.P, outScore);
					rcount = cinScore.mulAndMarginalize(scores) - scoreS;
					
					newcnt = newcnt + lcount + rcount;
					if (i == 0 || newcnt > maxcnt) {
						chart.addMaxRuleCount(eedge.lhs, i, newcnt, tedge.lhs, (short) -1, LEVEL_ZERO);
					}
				}
			}
		}
		// time step T + 1
		Set<Short> keys = chart.keySetMaxRule(nword - 1, LEVEL_ZERO);
		if (keys == null || keys.size() == 0) {
			throw new RuntimeException("OOPS_BUG: key set of tags in time step " + nword + " should not be empty.");
		}
		tkeys = new HashSet<Short>(keys); // to avoid `java.util.ConcurrentModificationException` since we need to add ENDING_IDX to the cell
		for (Short tkey : tkeys) {
			if (ttpair.getEdge(tkey, Pair.ENDING_IDX, RuleType.LHSPACE) == null) {
				continue;
			}
			newcnt = chart.getMaxRuleCount(tkey, nword - 1);
			if ((maxcnt = chart.getMaxRuleCount((short) Pair.ENDING_IDX, nword - 1, (short) 0)) > newcnt) {
				continue;
			}
			if ((outScore = chart.getInsideScore(tkey, nword - 1, LEVEL_ZERO)) == null ||
					(cinScore = chart.getOutsideScore(tkey, nword - 1, LEVEL_ONE)) == null) {
				continue;
			}
			scores = new EnumMap<>(RuleUnit.class);
			scores.put(RuleUnit.P, outScore);
			newcnt = newcnt + cinScore.mulAndMarginalize(scores);
			if (newcnt > maxcnt) {
				chart.addMaxRuleCount((short) Pair.ENDING_IDX, nword - 1, newcnt, tkey, (short) -1, LEVEL_ZERO);
			}
		}
	}
}
