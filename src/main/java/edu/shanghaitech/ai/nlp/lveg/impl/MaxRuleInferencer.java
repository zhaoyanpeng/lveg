package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * All the values are stored in logarithmic form.
 * 
 * @author Yanpeng Zhao
 *
 */
public class MaxRuleInferencer extends Inferencer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3844565735030401845L;

	
	public MaxRuleInferencer(LVeGGrammar agrammar, LVeGLexicon alexicon) {
		grammar = agrammar;
		lexicon = alexicon;
	}
	
	
	/**
	 * Compute the inside score given the sentence and grammar rules.
	 * 
	 * @param chart  [in/out]-side score container
	 * @param tree   in which only the sentence is used
	 * @param nword  # of words in the sentence
	 * @param scoreS sentence score in logarithmic form
	 */
	protected void evalMaxRuleCount(Chart chart, List<State> sentence, int nword, double scoreS) {
		List<GrammarRule> rules;
		int x0, y0, x1, y1, c0, c1, c2;
		double lcount, rcount, maxcnt, newcnt;
		GaussianMixture linScore, rinScore, outScore, cinScore;
		// lexicons
		for (int i = 0; i < nword; i++) {
			State word = sentence.get(i);
			int iCell = Chart.idx(i, nword);
			rules = lexicon.getRulesWithWord(word);
			for (GrammarRule rule : rules) {
				if (chart.containsKey(rule.lhs, iCell, false)) {
					cinScore = lexicon.score(word, rule.lhs);
					outScore = chart.getOutsideScore(rule.lhs, iCell);
					EnumMap<RuleUnit, GaussianMixture> scores = new EnumMap<>(RuleUnit.class);
					scores.put(RuleUnit.P, outScore);
					newcnt = cinScore.mulAndMarginalize(scores) - scoreS;
					chart.addMaxRuleCount(rule.lhs, iCell, newcnt, 0, (short) -1, (short) 0);
				}
			}
			maxRuleCountForUnaryRule(chart, iCell, scoreS);
		}		
		
		// binary rules
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (short itag = 0; itag < grammar.ntag; itag++) {
					if ((outScore = chart.getOutsideScore(itag, c2)) == null) { continue; }
					rules = grammar.getBRuleWithP(itag);
					for (GrammarRule arule : rules) {
						BinaryGrammarRule rule = (BinaryGrammarRule) arule;
						for (int right = left; right < left + ilayer; right++) {
							y0 = right;
							x1 = right + 1;
							c0 = Chart.idx(x0, nword - (y0 - x0));
							c1 = Chart.idx(x1, nword - (y1 - x1));
							if ((linScore = chart.getInsideScore(rule.lchild, c0)) == null || 
									(rinScore = chart.getInsideScore(rule.rchild, c1)) == null) {
								continue;
							}
							if ((lcount = chart.getMaxRuleCount(rule.lchild, c0)) == Double.NEGATIVE_INFINITY || 
									(rcount = chart.getMaxRuleCount(rule.rchild, c1)) == Double.NEGATIVE_INFINITY) {
								continue;
							}
							newcnt = lcount + rcount;
							if ((maxcnt = chart.getMaxRuleCount(rule.lhs, c2)) > newcnt) { continue; }
							EnumMap<RuleUnit, GaussianMixture> scores = new EnumMap<>(RuleUnit.class);
							scores.put(RuleUnit.P, outScore);
							scores.put(RuleUnit.LC, linScore);
							scores.put(RuleUnit.RC, rinScore);
							newcnt = newcnt + rule.weight.mulAndMarginalize(scores) - scoreS;
							if (newcnt > maxcnt) {
								// the negative, higher 2 bytes (lchild, sign bit exclusive) <- lower 2 bytes (rchild)
								int sons = (1 << 31) + (rule.lchild << 16) + rule.rchild;
								chart.addMaxRuleCount(rule.lhs, c2, newcnt, sons, (short) right, (short) 0);
							}
						}
					}
				}
				
				// unary rules
				maxRuleCountForUnaryRule(chart, c2, scoreS);
			}
		}
	}
	
	
	/**
	 * Discriminate unary rules in the chain.
	 */
	@SuppressWarnings("unused")
	private void maxRuleCountForUnaryRuleLevel(Chart chart, int idx, double scoreS) {
		double count, newcnt, maxcnt;
		GaussianMixture outScore, cinScore, ruleW;
		Map<Short, Short>  midsons = new HashMap<Short, Short>(99, 1);
		Map<Short, Double> midcnts = new HashMap<Short, Double>(99, 1);
		// first level
		Set<Short> ikeyLevel = chart.keySet(idx, true, (short) 0);
		Set<Short> okeyLevel = chart.keySet(idx, false, (short) 1);
		if (ikeyLevel != null && okeyLevel != null) {
			for (Short ikey : ikeyLevel) {
				if ((count = chart.getMaxRuleCount(ikey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				for (Short okey : okeyLevel) {
					maxcnt = Double.NEGATIVE_INFINITY;
					if (okey == ROOT || ikey == okey) { continue; }
					if (midcnts.containsKey(okey) && (maxcnt = midcnts.get(okey)) > count) { continue; }
					if ((cinScore = chart.getInsideScore(ikey, idx, (short) 0)) == null || 
							(outScore = chart.getOutsideScore(okey, idx, (short) 1)) == null) {
						continue;
					}
					if ((ruleW = grammar.getURuleWeight(okey, ikey, RuleType.LRURULE, true)) == null) { continue; }
					EnumMap<RuleUnit, GaussianMixture> scores = new EnumMap<>(RuleUnit.class);
					scores.put(RuleUnit.P, outScore);
					scores.put(RuleUnit.UC, cinScore);
					newcnt = count + ruleW.mulAndMarginalize(scores) - scoreS;
					if (newcnt > maxcnt) {
						midcnts.put(okey, newcnt);
						midsons.put(okey, ikey);
					}
				}
			}
		}
		// second level
		ikeyLevel = midcnts.keySet();
		okeyLevel = chart.keySet(idx, false, (short) 0);
		if (!midcnts.isEmpty() && okeyLevel != null) {
			for (Short ikey : ikeyLevel) {
				count = midcnts.get(ikey);
				int son = midsons.get(ikey);
				for (Short okey : okeyLevel) {
					if (okey == ROOT || ikey == okey || son == okey) { continue; }
					if ((maxcnt = chart.getMaxRuleCount(okey, idx)) > count) { continue; }
					if ((cinScore = chart.getInsideScore(ikey, idx, (short) 1)) == null || 
							(outScore = chart.getOutsideScore(okey, idx, (short) 0)) == null) {
						continue;
					}
					if ((ruleW = grammar.getURuleWeight(okey, ikey, RuleType.LRURULE, true)) == null) { continue; }
					EnumMap<RuleUnit, GaussianMixture> scores = new EnumMap<>(RuleUnit.class);
					scores.put(RuleUnit.P, outScore);
					scores.put(RuleUnit.UC, cinScore);
					newcnt = count + ruleW.mulAndMarginalize(scores) - scoreS;
					if (newcnt > maxcnt) {
						int sons = (son << 16) + ikey;
						chart.addMaxRuleCount(okey, idx, newcnt, sons, (short) -1, (short) 2);
					}
				}
			}
			
		}
		midsons.clear();
		midcnts.clear();
	}
	
	
	/**
	 * Treat the chain unary rule as a unary rule.
	 */
	private void maxRuleCountForUnaryRuleChain(Chart chart, int idx, double scoreS) {
		double count, newcnt, maxcnt;
		GaussianMixture outScore, cinScore, w0, w1;
		Set<Short> ikeyLevel0 = chart.keySet(idx, true, (short) 0);
		Set<Short> ikeyLevel1 = chart.keySet(idx, true, (short) 1);
		Set<Short> okeyLevel0 = chart.keySet(idx, false, (short) 0);
		if (ikeyLevel0 != null && ikeyLevel1 != null && okeyLevel0 != null) { // ROOT in cell 0 and in level 2 should be allowed? No
			for (Short ikey : ikeyLevel0) {
				if ((count = chart.getMaxRuleCount(ikey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				for (Short okey : okeyLevel0) {
					if (okey == ROOT || ikey == okey) { continue; } // ROOT is excluded, nonsense when ikey == okey
					if ((maxcnt = chart.getMaxRuleCount(okey, idx)) > count) { continue; }
					if ((cinScore = chart.getInsideScore(ikey, idx, (short) 0)) == null || 
							(outScore = chart.getOutsideScore(okey, idx, (short) 0)) == null) {
						continue;
					}
					for (Short mid : ikeyLevel1) {
						if ((w0 = grammar.getURuleWeight(mid, ikey, RuleType.LRURULE, true)) == null ||
								(w1 = grammar.getURuleWeight(okey, mid, RuleType.LRURULE, true)) == null) {
							continue;
						}
						cinScore = w0.mulAndMarginalize(cinScore, null, RuleUnit.UC, true);
						EnumMap<RuleUnit, GaussianMixture> scores = new EnumMap<>(RuleUnit.class);
						scores.put(RuleUnit.P, outScore);
						scores.put(RuleUnit.UC, cinScore);
						newcnt = count + w1.mulAndMarginalize(scores) - scoreS;
						if (newcnt > maxcnt) {
							int sons = (ikey << 16) + mid; // higher 2 bytes (grandson) <- lower 2 bytes (child)
							chart.addMaxRuleCount(okey, idx, newcnt, sons, (short) -1, (short) 2);
						}
					}
				}
			}
		}
	}
	
	
	/**
	 * @param chart  CYK chart
	 * @param idx    index of the cell in the chart
	 * @param scoreS sentence score
	 */
	private void maxRuleCountForUnaryRule(Chart chart, int idx, double scoreS) {
		List<GrammarRule> rules;
		double count, newcnt, maxcnt;
		GaussianMixture outScore, cinScore;
		// chain unary rule of length 1
		Set<Short> mkeyLevel0 = chart.keySetMaxRule(idx, (short) 0);
		if (mkeyLevel0 != null) { // ROOT in cell 0 and in level 1 should be allowed? No
			for (short mkey : mkeyLevel0) {
				if ((cinScore = chart.getInsideScore(mkey, idx, (short) 0)) == null) { continue; }
				if ((count = chart.getMaxRuleCount(mkey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				rules = grammar.getURuleWithC(mkey);
				Iterator<GrammarRule> iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (rule.type == RuleType.RHSPACE) { continue; } // ROOT is excluded
					if ((maxcnt = chart.getMaxRuleCount(rule.lhs, idx)) > count) { continue; }
					if ((outScore = chart.getOutsideScore(rule.lhs, idx, (short) 0)) == null) { continue; }
					EnumMap<RuleUnit, GaussianMixture> scores = new EnumMap<>(RuleUnit.class);
					scores.put(RuleUnit.P, outScore);
					scores.put(RuleUnit.UC, cinScore);
					newcnt = count + rule.weight.mulAndMarginalize(scores) - scoreS;
					if (newcnt > maxcnt) {
						chart.addMaxRuleCount(rule.lhs, idx, newcnt, mkey, (short) -1, (short) 1);
					}
				}
			}
		}
		
		// chain unary rule of length 2
		maxRuleCountForUnaryRuleChain(chart, idx, scoreS);
//		maxRuleCountForUnaryRuleLevel(chart, idx, scoreS);
		
		// ROOT treated as a specific 'binary' rule, I think we should not consider ROOT in the above two cases, and 
		// only consider it in the following case, only in this way will we keep the count calculation consist. Here
		// we can probably construct ROOT->A->B->C; ROOT->B->C; ROOT->C;
		if (idx == 0 && (outScore = chart.getOutsideScore(ROOT, idx)) != null) {
			rules = grammar.getURuleWithP(ROOT); // outside score should be 1
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) { // CHECK need to check again
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next(); 
				if ((cinScore = chart.getInsideScore((short) rule.rhs, idx)) == null ||
						(count = chart.getMaxRuleCount((short) rule.rhs, idx)) == Double.NEGATIVE_INFINITY ||
						(maxcnt = chart.getMaxRuleCount(ROOT, idx)) > count) {
					continue;
				}
				EnumMap<RuleUnit, GaussianMixture> scores = new EnumMap<>(RuleUnit.class);
				scores.put(RuleUnit.C, cinScore);
				newcnt = count + rule.weight.mulAndMarginalize(scores) - scoreS;
				if (newcnt > maxcnt) {
					chart.addMaxRuleCount(ROOT, idx, newcnt, rule.rhs, (short) -1, (short) 0); // a specific 'binary' rule
				}
			}	
		}
	}
	
}