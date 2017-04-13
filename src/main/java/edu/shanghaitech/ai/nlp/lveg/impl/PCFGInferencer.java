package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;

public class PCFGInferencer extends Inferencer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7545305823426970355L;


	public static void insideScoreMask(Chart chart, List<State> sentence, int nword, boolean prune, int base, double ratio) {
		int x0, y0, x1, y1, c0, c1, c2;
		double ruleScore, linScore, rinScore, pinScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(sentence.get(i));
			for (GrammarRule rule : rules) {
				chart.addInsideScoreMask(rule.lhs, iCell, rule.weight.getProb(), (short) 0, false);
			}
//			if (prune) { chart.pruneInsideScoreMask(iCell, (short) 0); }
			insideScoreForUnaryRuleMask(chart, iCell, prune);
			if (prune) { chart.pruneInsideScoreMask(iCell, (short) -1, base, ratio); }
		}
		
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
					BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
					for (int right = left; right < left + ilayer; right++) {
						y0 = right;
						x1 = right + 1;
						c0 = Chart.idx(x0, nword - (y0 - x0));
						c1 = Chart.idx(x1, nword - (y1 - x1));
						
						if (chart.containsKeyMask(rule.lchild, c0, true) && chart.containsKeyMask(rule.rchild, c1, true)) {
							ruleScore = rule.weight.getProb();
							linScore = chart.getInsideScoreMask(rule.lchild, c0);
							rinScore = chart.getInsideScoreMask(rule.rchild, c1);
							
							pinScore = linScore + ruleScore + rinScore; // in logarithmic form
							chart.addInsideScoreMask(rule.lhs, c2, pinScore, (short) 0, false);
						}
					}
				}
//				if (prune) { chart.pruneInsideScoreMask(c2, (short) 0); }
				insideScoreForUnaryRuleMask(chart, c2, prune);
				if (prune) { chart.pruneInsideScoreMask(c2, (short) -1, base, ratio); }
			}
		}
	}
	
	
	public static void outsideScoreMask(Chart chart, List<State> sentence, int nword, boolean prune, int base, double ratio) {
		int x0, y0, x1, y1, c0, c1, c2;
		double poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				x1 = left + ilayer + 1; 
				c2 = Chart.idx(left, nword - ilayer);
				for (int right = left + ilayer + 1; right < nword; right++) {
					y0 = right;
					y1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKeyMask(rule.lhs, c0, false) && chart.containsKeyMask(rule.rchild, c1, true)) {
							ruleScore = rule.weight.getProb();
							poutScore = chart.getOutsideScoreMask(rule.lhs, c0);
							rinScore = chart.getInsideScoreMask(rule.rchild, c1);
							
							loutScore = ruleScore + poutScore + rinScore;
							chart.addOutsideScoreMask(rule.lchild, c2, loutScore, (short) 0, false);
						}
					}
				}
				
				y0 = left + ilayer;
				y1 = left - 1;
				for (int right = 0; right < left; right++) {
					x0 = right; 
					x1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKeyMask(rule.lhs, c0, false) && chart.containsKeyMask(rule.lchild, c1, true)) {
							ruleScore = rule.weight.getProb();
							poutScore = chart.getOutsideScoreMask(rule.lhs, c0);
							linScore = chart.getInsideScoreMask(rule.lchild, c1);
							
							routScore = ruleScore + poutScore + linScore;
							chart.addOutsideScoreMask(rule.rchild, c2, routScore, (short) 0, false);
						}
					}
				}
//				if (prune) { chart.pruneOutsideScoreMask(c2, (short) 0); }
				outsideScoreForUnaryRuleMask(chart, c2, prune);
				if (prune) { chart.pruneOutsideScoreMask(c2, (short) -1, base, ratio); }	
			}
		}
	}
	
	
	
	private static void insideScoreForUnaryRuleMask(Chart chart, int idx, boolean prune) {
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		double pinScore, cinScore;
		while (level < LENGTH_UCHAIN && (set = chart.keySetMask(idx, true, level)) != null) {
			for (Short idTag : set) {
				rules = grammar.getURuleWithC(idTag); // ROOT is excluded, and is not considered in level 0
				Iterator<GrammarRule> iterator = rules.iterator();
				cinScore = chart.getInsideScoreMask(idTag, idx, level);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; } // ROOT is allowed only when it is in cell 0 and is in level 1 or 2
					pinScore = rule.weight.getProb() + cinScore;
					chart.addInsideScoreMask(rule.lhs, idx, pinScore, (short) (level + 1), false);
				}
			}
			level++;
//			if (prune) { chart.pruneInsideScore(idx, level); } // CHECK
		}
		// have to process ROOT node specifically, ROOT is in cell 0 and is in level 3
		if (idx == 0 && (set = chart.keySetMask(idx, true, LENGTH_UCHAIN)) != null) {
			for (Short idTag : set) { // the maximum inside level below ROOT
				rules = grammar.getURuleWithC(idTag);
				Iterator<GrammarRule> iterator = rules.iterator();
				cinScore = chart.getInsideScoreMask(idTag, idx, LENGTH_UCHAIN);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (rule.type != GrammarRule.RHSPACE) { continue; } // only consider ROOT in level 3
					pinScore = rule.weight.getProb() + cinScore;
					chart.addInsideScoreMask(rule.lhs, idx, pinScore, (short) (LENGTH_UCHAIN + 1), false);
				}
			}
//			if (prune) { chart.pruneInsideScore(idx, (short) (LENGTH_UCHAIN + 1)); } // CHECK
		}
	}
	
	
	private static void outsideScoreForUnaryRuleMask(Chart chart, int idx, boolean prune) {
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		double poutScore, coutScore;
		// have to process ROOT node specifically
		if (idx == 0 && (set = chart.keySetMask(idx, false, (short) (LENGTH_UCHAIN + 1))) != null) {
			for (Short idTag : set) { // can only contain ROOT
				if (idTag != ROOT) { continue; }
				rules = grammar.getURuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator(); // see set ROOT's outside score
				poutScore = chart.getOutsideScoreMask(idTag, idx, (short) (LENGTH_UCHAIN + 1)); // 1
				while (iterator.hasNext()) { // CHECK
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					coutScore = rule.weight.getProb(); // since OS(ROOT) = 1
					chart.addOutsideScoreMask((short) rule.rhs, idx, coutScore, level, false);
				}
			}
//			if (prune) { chart.pruneOutsideScore(idx, level); } // CHECK
		}
		while(level < LENGTH_UCHAIN && (set = chart.keySetMask(idx, false, level)) != null) {
			for (Short idTag : set) {
				rules = grammar.getURuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator();
				poutScore = chart.getOutsideScoreMask(idTag, idx, level);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					coutScore = rule.weight.getProb() + poutScore;
					chart.addOutsideScoreMask((short) rule.rhs, idx, coutScore, (short) (level + 1), false);
				}
			}
			level++;
//			if (prune) { chart.pruneOutsideScore(idx, level); } // CHECK
		}
	}
	
	
	public static void makeMask(int nword, Chart chart, double scoreS, double threshold) {
		int idx;
		short level;
		Set<Short> set;
		double oscore, iscore, count;
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				level = 0;
				idx = Chart.idx(left, nword - ilayer);
				
				
				
				while (level < (LENGTH_UCHAIN + 1) && (set = chart.keySetMask(idx, false, level)) != null) {
					for (Short idTag : set) {
						oscore = chart.getOutsideScoreMask(idTag, idx, level);
						iscore = chart.getInsideScoreMask(idTag, idx, (short) (LENGTH_UCHAIN - level));
						if (oscore != Double.NEGATIVE_INFINITY && iscore != Double.NEGATIVE_INFINITY) {
							count = iscore + oscore - scoreS; // in logarithmic form
							if (count > threshold) {
								chart.addMask(idTag, idx, level); // top-down from perspective of outside scores
							}
						}						
					}
					level++;
				}
				
				
				
			}
		}
	}
	
	
	public static void setRootOutsideScoreMask(Chart chart) {
		chart.addOutsideScoreMask((short) 0, Chart.idx(0, 1), 0, (short) (LENGTH_UCHAIN + 1), false);
	}
	
}
