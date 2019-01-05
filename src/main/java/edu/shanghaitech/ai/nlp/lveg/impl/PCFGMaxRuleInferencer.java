package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil;

public class PCFGMaxRuleInferencer extends PCFGInferencer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7545305823426970355L;
	
	
	public PCFGMaxRuleInferencer(LVeGGrammar agrammar, LVeGLexicon alexicon) {
		super(agrammar, alexicon);
	}
	
	
	protected void evalMaxRuleProdCount(Chart chart, List<State> sentence, int nword, double scoreS) {
		List<GrammarRule> rules;
		int x0, y0, x1, y1, c0, c1, c2;
		double lcount, rcount, maxcnt, newcnt;
		Double linScore, rinScore, outScore, cinScore;
		// lexicons
		for (int i = 0; i < nword; i++) {
			State word = sentence.get(i);
			int iCell = Chart.idx(i, nword);
			rules = lexicon.getRulesWithWord(word);
			for (GrammarRule rule : rules) {
				if (chart.containsKeyMask(rule.lhs, iCell, false)) {
					cinScore = rule.weight.getProb();;
					outScore = chart.getOutsideScoreMask(rule.lhs, iCell);
					
					newcnt = cinScore + outScore - scoreS;
					chart.addMaxRuleCount(rule.lhs, iCell, newcnt, 0, (short) -1, (short) 0);
				}
			}
			maxRuleProdCountForUnaryRule(chart, iCell, scoreS);
		}	
		
		// binary rules
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (short itag = 0; itag < grammar.ntag; itag++) {
					
					if (!chart.containsKeyMask(itag, c2, false)) { continue; }
					outScore = chart.getOutsideScoreMask(itag, c2);
					
					rules = grammar.getBRuleWithP(itag);
					for (GrammarRule arule : rules) {
						BinaryGrammarRule rule = (BinaryGrammarRule) arule;
						for (int right = left; right < left + ilayer; right++) {
							y0 = right;
							x1 = right + 1;
							c0 = Chart.idx(x0, nword - (y0 - x0));
							c1 = Chart.idx(x1, nword - (y1 - x1));
							
							if (!chart.containsKeyMask(rule.lchild, c0, true) ||
									!chart.containsKeyMask(rule.rchild, c1, true)) {
								continue;
							}
							linScore = chart.getInsideScoreMask(rule.lchild, c0);
							rinScore = chart.getInsideScoreMask(rule.rchild, c1);
							
							if ((lcount = chart.getMaxRuleCount(rule.lchild, c0)) == Double.NEGATIVE_INFINITY || 
									(rcount = chart.getMaxRuleCount(rule.rchild, c1)) == Double.NEGATIVE_INFINITY) {
								continue;
							}
							newcnt = lcount + rcount;
							if ((maxcnt = chart.getMaxRuleCount(rule.lhs, c2)) > newcnt) { continue; }
							
							newcnt = newcnt + rule.weight.getProb() + outScore + linScore + rinScore - scoreS;
							if (newcnt > maxcnt) {
								// the negative, higher 2 bytes (lchild, sign bit exclusive) <- lower 2 bytes (rchild)
								int sons = (1 << 31) + (rule.lchild << 16) + rule.rchild;
								chart.addMaxRuleCount(rule.lhs, c2, newcnt, sons, (short) right, (short) 0);
							}
						}
					}
				}
				
				// unary rules
				maxRuleProdCountForUnaryRule(chart, c2, scoreS);
			}
		}
	}
	
	
	private void maxRuleProdCountForUnaryRule(Chart chart, int idx, double scoreS) {
		List<GrammarRule> rules;
		double count, newcnt, maxcnt;
		Double outScore, cinScore;
		// chain unary rule of length 1
		Set<Short> mkeyLevel0 = chart.keySetMaxRule(idx, (short) 0);
		if (mkeyLevel0 != null) { // ROOT in cell 0 and in level 1 should be allowed? No
			for (short mkey : mkeyLevel0) {
				
				if (!chart.containsKeyMask(mkey, idx, true, (short) 0)) { continue; }
				cinScore = chart.getInsideScoreMask(mkey, idx, (short) 0);
				
				if ((count = chart.getMaxRuleCount(mkey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				rules = grammar.getURuleWithC(mkey);
				Iterator<GrammarRule> iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (rule.type == RuleType.RHSPACE) { continue; } // ROOT is excluded
					if ((maxcnt = chart.getMaxRuleCount(rule.lhs, idx)) > count) { continue; }
					
					if (!chart.containsKeyMask(rule.lhs, idx, false, (short) 0)) { continue; }
					outScore = chart.getOutsideScoreMask(rule.lhs, idx, (short) 0);
					
					newcnt = count + rule.weight.getProb() + outScore + cinScore - scoreS;
					if (newcnt > maxcnt) {
						chart.addMaxRuleCount(rule.lhs, idx, newcnt, mkey, (short) -1, (short) 1);
					}
				}
			}
		}
		
		// chain unary rule of length 2
		maxRuleProdCountForUnaryRuleChain(chart, idx, scoreS);
//		maxRuleCountForUnaryRuleLevel(chart, idx, scoreS);
		
		// ROOT treated as a specific 'binary' rule, I think we should not consider ROOT in the above two cases, and 
		// only consider it in the following case, only in this way will we keep the count calculation consist. Here
		// we can probably construct ROOT->A->B->C; ROOT->B->C; ROOT->C;
		if (idx == 0 && (outScore = chart.getOutsideScoreMask(ROOT, idx)) != null) {
			rules = grammar.getURuleWithP(ROOT); // outside score should be 1
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) { // CHECK need to check again
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next(); 
				
				if (!chart.containsKeyMask((short) rule.rhs, idx, true) ||
						(count = chart.getMaxRuleCount((short) rule.rhs, idx)) == Double.NEGATIVE_INFINITY ||
						(maxcnt = chart.getMaxRuleCount(ROOT, idx)) > count) {
					continue;
				}
				cinScore = chart.getInsideScoreMask((short) rule.rhs, idx);
				
				newcnt = count + rule.weight.getProb() + cinScore - scoreS;
				if (newcnt > maxcnt) {
					chart.addMaxRuleCount(ROOT, idx, newcnt, rule.rhs, (short) -1, (short) 0); // a specific 'binary' rule
				}
			}	
		}
	}
	
	
	private void maxRuleProdCountForUnaryRuleChain(Chart chart, int idx, double scoreS) {
		double count, newcnt, maxcnt;
		Double outScore, cinScore;
		GaussianMixture w0, w1;
		Set<Short> ikeyLevel0 = chart.keySetMask(idx, true, (short) 0);
		Set<Short> ikeyLevel1 = chart.keySetMask(idx, true, (short) 1);
		Set<Short> okeyLevel0 = chart.keySetMask(idx, false, (short) 0);
		if (ikeyLevel0 != null && ikeyLevel1 != null && okeyLevel0 != null) { // ROOT in cell 0 and in level 2 should be allowed? No
			for (Short ikey : ikeyLevel0) {
				if ((count = chart.getMaxRuleCount(ikey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				for (Short okey : okeyLevel0) {
					if (okey == ROOT || ikey == okey) { continue; } // ROOT is excluded, nonsense when ikey == okey
					if ((maxcnt = chart.getMaxRuleCount(okey, idx)) > count) { continue; }
					
					if (!chart.containsKeyMask(ikey, idx, true, (short) 0) ||
							!chart.containsKeyMask(okey, idx, false, (short) 0)) {
						continue;
					}
					cinScore = chart.getInsideScoreMask(ikey, idx, (short) 0);
					outScore = chart.getOutsideScoreMask(okey, idx, (short) 0);
					
					for (Short mid : ikeyLevel1) {
						if ((w0 = grammar.getURuleWeight(mid, ikey, RuleType.LRURULE, true)) == null ||
								(w1 = grammar.getURuleWeight(okey, mid, RuleType.LRURULE, true)) == null) {
							continue;
						}
						
						cinScore = w0.getProb() + cinScore;
						newcnt = count + w1.getProb() + outScore + cinScore - scoreS; 
						if (newcnt > maxcnt) {
							int sons = (ikey << 16) + mid; // higher 2 bytes (grandson) <- lower 2 bytes (child)
							chart.addMaxRuleCount(okey, idx, newcnt, sons, (short) -1, (short) 2);
						}
					}
				}
			}
		}
	}
	
	
	protected void evalMaxRuleSumCount(Chart chart, List<State> sentence, int nword, double scoreS) {
		List<GrammarRule> rules;
		int x0, y0, x1, y1, c0, c1, c2;
		double lcount, rcount, maxcnt, newcnt, rulecnt;
		Double linScore, rinScore, outScore, cinScore;
		// lexicons
		for (int i = 0; i < nword; i++) {
			State word = sentence.get(i);
			int iCell = Chart.idx(i, nword);
			rules = lexicon.getRulesWithWord(word);
			for (GrammarRule rule : rules) {
				if (chart.containsKeyMask(rule.lhs, iCell, false)) {
					cinScore = rule.weight.getProb();
					outScore = chart.getOutsideScoreMask(rule.lhs, iCell);
					
					newcnt = cinScore + outScore - scoreS;
					chart.addMaxRuleCount(rule.lhs, iCell, newcnt, 0, (short) -1, (short) 0);
				}
			}
			maxRuleSumCountForUnaryRule(chart, iCell, scoreS);
		}	
		
		// binary rules
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (short itag = 0; itag < grammar.ntag; itag++) {
					
					if (!chart.containsKeyMask(itag, c2, false)) { continue; }
					outScore = chart.getOutsideScoreMask(itag, c2);
					
					rules = grammar.getBRuleWithP(itag);
					for (GrammarRule arule : rules) {
						BinaryGrammarRule rule = (BinaryGrammarRule) arule;
						for (int right = left; right < left + ilayer; right++) {
							y0 = right;
							x1 = right + 1;
							c0 = Chart.idx(x0, nword - (y0 - x0));
							c1 = Chart.idx(x1, nword - (y1 - x1));
							
							if (!chart.containsKeyMask(rule.lchild, c0, true) ||
									!chart.containsKeyMask(rule.rchild, c1, true)) {
								continue;
							}
							linScore = chart.getInsideScoreMask(rule.lchild, c0);
							rinScore = chart.getInsideScoreMask(rule.rchild, c1);
							
							if ((lcount = chart.getMaxRuleCount(rule.lchild, c0)) == Double.NEGATIVE_INFINITY || 
									(rcount = chart.getMaxRuleCount(rule.rchild, c1)) == Double.NEGATIVE_INFINITY) {
								continue;
							}
							
							// newcnt = lcount + rcount;
							newcnt = FunUtil.logAdd(lcount, rcount);
							if ((maxcnt = chart.getMaxRuleCount(rule.lhs, c2)) > newcnt) { continue; }
							
							// newcnt = newcnt + rule.weight.getProb() + outScore + linScore + rinScore - scoreS;
							rulecnt = rule.weight.getProb() + outScore + linScore + rinScore - scoreS;
							newcnt = FunUtil.logAdd(newcnt, rulecnt);
							if (newcnt > maxcnt) {
								// the negative, higher 2 bytes (lchild, sign bit exclusive) <- lower 2 bytes (rchild)
								int sons = (1 << 31) + (rule.lchild << 16) + rule.rchild;
								chart.addMaxRuleCount(rule.lhs, c2, newcnt, sons, (short) right, (short) 0);
							}
						}
					}
				}
				
				// unary rules
				maxRuleSumCountForUnaryRule(chart, c2, scoreS);
			}
		}
	}
	
	
	private void maxRuleSumCountForUnaryRule(Chart chart, int idx, double scoreS) {
		List<GrammarRule> rules;
		double count, newcnt, maxcnt, rulecnt;
		Double outScore, cinScore;
		// chain unary rule of length 1
		Set<Short> mkeyLevel0 = chart.keySetMaxRule(idx, (short) 0);
		if (mkeyLevel0 != null) { // ROOT in cell 0 and in level 1 should be allowed? No
			for (short mkey : mkeyLevel0) {
				
				if (!chart.containsKeyMask(mkey, idx, true, (short) 0)) { continue; }
				cinScore = chart.getInsideScoreMask(mkey, idx, (short) 0);
				
				if ((count = chart.getMaxRuleCount(mkey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				rules = grammar.getURuleWithC(mkey);
				Iterator<GrammarRule> iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (rule.type == RuleType.RHSPACE) { continue; } // ROOT is excluded
					if ((maxcnt = chart.getMaxRuleCount(rule.lhs, idx)) > count) { continue; }
					
					if (!chart.containsKeyMask(rule.lhs, idx, false, (short) 0)) { continue; }
					outScore = chart.getOutsideScoreMask(rule.lhs, idx, (short) 0);
					
					// newcnt = count + rule.weight.getProb() + outScore + cinScore - scoreS;
					rulecnt = rule.weight.getProb() + outScore + cinScore - scoreS;
					newcnt = FunUtil.logAdd(count, rulecnt);
					if (newcnt > maxcnt) {
						chart.addMaxRuleCount(rule.lhs, idx, newcnt, mkey, (short) -1, (short) 1);
					}
				}
			}
		}
		
		// chain unary rule of length 2
		maxRuleSumCountForUnaryRuleChain(chart, idx, scoreS);
//		maxRuleCountForUnaryRuleLevel(chart, idx, scoreS);
		
		// ROOT treated as a specific 'binary' rule, I think we should not consider ROOT in the above two cases, and 
		// only consider it in the following case, only in this way will we keep the count calculation consist. Here
		// we can probably construct ROOT->A->B->C; ROOT->B->C; ROOT->C;
		if (idx == 0 && (outScore = chart.getOutsideScoreMask(ROOT, idx)) != null) {
			rules = grammar.getURuleWithP(ROOT); // outside score should be 1
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) { // CHECK need to check again
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next(); 
				
				if (!chart.containsKeyMask((short) rule.rhs, idx, true) ||
						(count = chart.getMaxRuleCount((short) rule.rhs, idx)) == Double.NEGATIVE_INFINITY ||
						(maxcnt = chart.getMaxRuleCount(ROOT, idx)) > count) {
					continue;
				}
				cinScore = chart.getInsideScoreMask((short) rule.rhs, idx);
				
				// newcnt = count + rule.weight.getProb() + cinScore - scoreS;
				rulecnt = rule.weight.getProb() + cinScore - scoreS;
				newcnt = FunUtil.logAdd(count, rulecnt);
				if (newcnt > maxcnt) {
					chart.addMaxRuleCount(ROOT, idx, newcnt, rule.rhs, (short) -1, (short) 0); // a specific 'binary' rule
				}
			}	
		}
	}
	
	
	private void maxRuleSumCountForUnaryRuleChain(Chart chart, int idx, double scoreS) {
		double count, newcnt, maxcnt, rulecnt;
		double outScore, cinScore;
		GaussianMixture w0, w1;
		Set<Short> ikeyLevel0 = chart.keySetMask(idx, true, (short) 0);
		Set<Short> ikeyLevel1 = chart.keySetMask(idx, true, (short) 1);
		Set<Short> okeyLevel0 = chart.keySetMask(idx, false, (short) 0);
		if (ikeyLevel0 != null && ikeyLevel1 != null && okeyLevel0 != null) { // ROOT in cell 0 and in level 2 should be allowed? No
			for (Short ikey : ikeyLevel0) {
				if ((count = chart.getMaxRuleCount(ikey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				for (Short okey : okeyLevel0) {
					if (okey == ROOT || ikey == okey) { continue; } // ROOT is excluded, nonsense when ikey == okey
					if ((maxcnt = chart.getMaxRuleCount(okey, idx)) > count) { continue; }
					
					if (!chart.containsKeyMask(ikey, idx, true, (short) 0) ||
							!chart.containsKeyMask(okey, idx, false, (short) 0)) {
						continue;
					}
					cinScore = chart.getInsideScoreMask(ikey, idx, (short) 0);
					outScore = chart.getOutsideScoreMask(okey, idx, (short) 0);
					
					for (Short mid : ikeyLevel1) {
						if ((w0 = grammar.getURuleWeight(mid, ikey, RuleType.LRURULE, true)) == null ||
								(w1 = grammar.getURuleWeight(okey, mid, RuleType.LRURULE, true)) == null) {
							continue;
						}
						
						cinScore = w0.getProb() + cinScore;
						// newcnt = count + w1.getProb() + outScore + cinScore - scoreS; 
						rulecnt = w1.getProb() + outScore + cinScore - scoreS;
						newcnt = FunUtil.logAdd(count, rulecnt);
						if (newcnt > maxcnt) {
							int sons = (ikey << 16) + mid; // higher 2 bytes (grandson) <- lower 2 bytes (child)
							chart.addMaxRuleCount(okey, idx, newcnt, sons, (short) -1, (short) 2);
						}
					}
				}
			}
		}
	}
}
