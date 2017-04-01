package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.impl.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Executor;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public abstract class Inferencer extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3449371510125004187L;
	protected final static short ROOT = 0;
	protected final static short LENGTH_UCHAIN = 2;
	protected static ChainUrule chainurule;
	
	public static LVeGLexicon lexicon;
	public static LVeGGrammar grammar;
	
	public enum ChainUrule {
		ALL_POSSIBLE_PATH, PRE_COMPUTE_CHAIN, NOT_PRE_ADD_INTER, NOT_PRE_NOT_INTER, DEFAULT,
	}
	
	
	/**
	 * Accumulate gradients.
	 * 
	 * @param scores   score of the parse tree and score of the sentence
	 * @param parallel parallel (true) or not (false)
	 */
	public void evalGradients(List<Double> scores) {
		grammar.evalGradients(scores);
		lexicon.evalGradients(scores);
	}
	
	
	/**
	 * Compute the inside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used
	 * @param nword # of words in the sentence
	 */
	public static void insideScore(Chart chart, List<State> sentence, int nword, boolean prune, ThreadPool cpool, boolean usemasks) {
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(sentence.get(i));
			// preterminals
			for (GrammarRule rule : rules) {
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true), (short) 0, false);
			}
			// DEBUG unary grammar rules
//			logger.trace("Cell [" + i + ", " + (i + 0) + "]="+ iCell + "\t is being estimated. # " );
//			long start = System.currentTimeMillis();
			if (prune) { chart.pruneInsideScore(iCell, (short) 0); }
			insideScoreForUnaryRule(chart, iCell, chainurule, prune, usemasks);
			if (prune) { chart.pruneInsideScore(iCell, (short) -1); }
//			long ttime = System.currentTimeMillis() - start;
//			logger.trace("\tafter chain unary\t" + chart.size(iCell, true) + "\ttime: " + ttime / 1000 + "\n");
		}		
		
		// inside score
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				int c2 = Chart.idx(left, nword - ilayer);
				Cell cell = chart.get(c2, true); 
				
				// binary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
					InputToSubCYKer input = new InputToSubCYKer(ilayer, left, nword, cell, chart, rmap.getValue(), true);
					cpool.execute(input);
					while (cpool.hasNext()) {
						cpool.getNext();
					}
				}
				while (!cpool.isDone()) {
					while (cpool.hasNext()) {
						cpool.getNext();
					}
				}
				// DEBUG unary grammar rules
//				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # ");
//				long start = System.currentTimeMillis();
				if (prune) { chart.pruneInsideScore(c2, (short) 0); }
				insideScoreForUnaryRule(chart, c2, chainurule, prune, usemasks);
				if (prune) { chart.pruneInsideScore(c2, (short) -1); }
//				long ttime = System.currentTimeMillis() - start;
//				logger.trace("\tafter chain unary\t" + chart.size(c2, true) + "\ttime: " + ttime / 1000 + "\n");
			}
		}
	}
	
	
	/**
	 * Compute the outside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used.
	 * @param nword # of words in the sentence
	 */
	public static void outsideScore(Chart chart, List<State> sentence, int nword, boolean prune, ThreadPool cpool, boolean usemasks) {
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				int c2 = Chart.idx(left, nword - ilayer);
				Cell cell = chart.get(c2, false); 
				
				// binary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
					InputToSubCYKer input = new InputToSubCYKer(ilayer, left, nword, cell, chart, rmap.getValue(), false);
					cpool.execute(input);
					while (cpool.hasNext()) {
						cpool.getNext();
					}
				}
				while (!cpool.isDone()) {
					while (cpool.hasNext()) {
						cpool.getNext();
					}
				}
				// DEBUG unary grammar rules
//				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # ");
//				long start = System.currentTimeMillis();
				if (prune) { chart.pruneOutsideScore(c2, (short) 0); }
				outsideScoreForUnaryRule(chart, c2, chainurule, prune, usemasks);
				if (prune) { chart.pruneOutsideScore(c2, (short) -1); }
//				long ttime = System.currentTimeMillis() - start;
//				logger.trace("\tafter chain unary\t" + chart.size(c2, false) + "\ttime: " + ttime / 1000 + "\n");
			}
		}
	}
	
	
	public static void insideScoreMask(Chart chart, List<State> sentence, int nword, boolean prune, int base, double ratio) {
		int x0, y0, x1, y1, c0, c1, c2;
		double ruleScore, linScore, rinScore, pinScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(sentence.get(i));
			for (GrammarRule rule : rules) {
				chart.addInsideScoreMask(rule.lhs, iCell, rule.weight.prob, (short) 0, false);
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
							ruleScore = rule.weight.prob;
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
							ruleScore = rule.weight.prob;
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
							ruleScore = rule.weight.prob;
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
	
	
	/**
	 * Compute the inside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used
	 * @param nword # of words in the sentence
	 */
	public static void insideScore(Chart chart, List<State> sentence, int nword, boolean prune, boolean usemasks) {
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture pinScore, linScore, rinScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(sentence.get(i));
			// preterminals
			for (GrammarRule rule : rules) {
				if (usemasks) { if (!chart.isAllowed(rule.lhs, iCell, true)) { continue; } } // CHECK
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true), (short) 0, false);
			}
			// DEBUG unary grammar rules
//			logger.trace("Cell [" + i + ", " + (i + 0) + "]="+ iCell + "\t is being estimated. # " );
//			long start = System.currentTimeMillis();
			if (prune) { chart.pruneInsideScore(iCell, (short) 0); }
			insideScoreForUnaryRule(chart, iCell, chainurule, prune, usemasks);
			if (prune) { chart.pruneInsideScore(iCell, (short) -1); }
//			long ttime = System.currentTimeMillis() - start;
//			logger.trace("\tafter chain unary\t" + chart.size(iCell, true) + "\ttime: " + ttime / 1000 + "\n");
		}		
		
		// inside score
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
					BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
					if (usemasks) { if (!chart.isAllowed(rule.lhs, c2, true)) { continue; } } // CHECK
					for (int right = left; right < left + ilayer; right++) {
						y0 = right;
						x1 = right + 1;
						c0 = Chart.idx(x0, nword - (y0 - x0));
						c1 = Chart.idx(x1, nword - (y1 - x1));
					
						if (chart.containsKey(rule.lchild, c0, true) && chart.containsKey(rule.rchild, c1, true)) {
							ruleScore = rule.getWeight();
							linScore = chart.getInsideScore(rule.lchild, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
							pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addInsideScore(rule.lhs, c2, pinScore, (short) 0, false);
						}
					}
				}
				// DEBUG unary grammar rules
//				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # ");
//				long start = System.currentTimeMillis();
				if (prune) { chart.pruneInsideScore(c2, (short) 0); }
				insideScoreForUnaryRule(chart, c2, chainurule, prune, usemasks);
				if (prune) { chart.pruneInsideScore(c2, (short) -1); }
//				long ttime = System.currentTimeMillis() - start;
//				logger.trace("\tafter chain unary\t" + chart.size(c2, true) + "\ttime: " + ttime / 1000 + "\n");
			}
		}
	}
	
	
	/**
	 * Compute the outside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used.
	 * @param nword # of words in the sentence
	 */
	public static void outsideScore(Chart chart, List<State> sentence, int nword, boolean prune, boolean usemasks) {
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				x1 = left + ilayer + 1; 
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
					BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
					if (usemasks) { if (!chart.isAllowed(rule.lchild, c2, false)) { continue; } } // CHECK
					for (int right = left + ilayer + 1; right < nword; right++) {
						y0 = right;
						y1 = right;
						c0 = Chart.idx(x0, nword - (y0 - x0));
						c1 = Chart.idx(x1, nword - (y1 - x1));
					
						if (chart.containsKey(rule.lhs, c0, false) && chart.containsKey(rule.rchild, c1, true)) {
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addOutsideScore(rule.lchild, c2, loutScore, (short) 0, false);
						}
					}
				}
				
				y0 = left + ilayer;
				y1 = left - 1;
				// binary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
					BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
					if (usemasks) { if (!chart.isAllowed(rule.rchild, c2, false)) { continue; } } // CHECK
					for (int right = 0; right < left; right++) {
						x0 = right; 
						x1 = right;
						c0 = Chart.idx(x0, nword - (y0 - x0));
						c1 = Chart.idx(x1, nword - (y1 - x1));
					
						if (chart.containsKey(rule.lhs, c0, false) && chart.containsKey(rule.lchild, c1, true)) {
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, c0);
							linScore = chart.getInsideScore(rule.lchild, c1);
							
							routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
							chart.addOutsideScore(rule.rchild, c2, routScore, (short) 0, false);
						}
					}
				}
				// DEBUG unary grammar rules
//				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # ");
//				long start = System.currentTimeMillis();
				if (prune) { chart.pruneOutsideScore(c2, (short) 0); }
				outsideScoreForUnaryRule(chart, c2, chainurule, prune, usemasks);
				if (prune) { chart.pruneOutsideScore(c2, (short) -1); }
//				long ttime = System.currentTimeMillis() - start;
//				logger.trace("\tafter chain unary\t" + chart.size(c2, false) + "\ttime: " + ttime / 1000 + "\n");
			}
		}
	}
	
	
	private static void outsideScoreForUnaryRule(Chart chart, int idx, ChainUrule identifier, boolean prune, boolean usemasks) {
		switch (identifier) {
		case DEFAULT: {
			outsideScoreForUnaryRuleDefault(chart, idx, prune, usemasks);
			break;
		}
		default:
			logger.error("Invalid unary-rule-processing-method. ");
		}
	}
	
	
	private static void insideScoreForUnaryRule(Chart chart, int idx, ChainUrule identifier, boolean prune, boolean usemasks) {
		switch (identifier) {
		case DEFAULT: {
			insideScoreForUnaryRuleDefault(chart, idx, prune, usemasks);
			break;
		}
		default:
			logger.error("Invalid unary-rule-processing-method. ");
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
					coutScore = rule.weight.prob; // since OS(ROOT) = 1
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
					coutScore = rule.weight.prob + poutScore;
					chart.addOutsideScoreMask((short) rule.rhs, idx, coutScore, (short) (level + 1), false);
				}
			}
			level++;
//			if (prune) { chart.pruneOutsideScore(idx, level); } // CHECK
		}
	}
	
	private static void outsideScoreForUnaryRuleDefault(Chart chart, int idx, boolean prune, boolean usemasks) {
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		String rmKey = GrammarRule.Unit.P;
		GaussianMixture poutScore, coutScore;
		// have to process ROOT node specifically
		if (idx == 0 && (set = chart.keySet(idx, false, (short) (LENGTH_UCHAIN + 1))) != null) {
			for (Short idTag : set) { // can only contain ROOT
				if (idTag != ROOT) { continue; }
				rules = grammar.getURuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator(); // see set ROOT's outside score
				poutScore = chart.getOutsideScore(idTag, idx, (short) (LENGTH_UCHAIN + 1)); // 1
				while (iterator.hasNext()) { // CHECK
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (usemasks) { if (!chart.isAllowed((short) rule.rhs, idx, false)) { continue; } } // CHECK
					coutScore = rule.weight.copy(true); // since OS(ROOT) = 1
					// coutScore = rule.weight.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore((short) rule.rhs, idx, coutScore, level, false);
				}
			}
			if (prune) { chart.pruneOutsideScore(idx, level); }
		}
		while(level < LENGTH_UCHAIN && (set = chart.keySet(idx, false, level)) != null) {
			for (Short idTag : set) {
				rules = grammar.getURuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator();
				poutScore = chart.getOutsideScore(idTag, idx, level);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (usemasks) { if (!chart.isAllowed((short) rule.rhs, idx, false)) { continue; } } // CHECK
					coutScore = rule.weight.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore((short) rule.rhs, idx, coutScore, (short) (level + 1), false);
				}
			}
			level++;
			if (prune) { chart.pruneOutsideScore(idx, level); }
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
					pinScore = rule.weight.prob + cinScore;
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
					pinScore = rule.weight.prob + cinScore;
					chart.addInsideScoreMask(rule.lhs, idx, pinScore, (short) (LENGTH_UCHAIN + 1), false);
				}
			}
//			if (prune) { chart.pruneInsideScore(idx, (short) (LENGTH_UCHAIN + 1)); } // CHECK
		}
	}
	
	private static void insideScoreForUnaryRuleDefault(Chart chart, int idx, boolean prune, boolean usemasks) {
		String rmKey;
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		GaussianMixture pinScore, cinScore;
		while (level < LENGTH_UCHAIN && (set = chart.keySet(idx, true, level)) != null) {
			for (Short idTag : set) {
				rules = grammar.getURuleWithC(idTag); // ROOT is excluded, and is not considered in level 0
				Iterator<GrammarRule> iterator = rules.iterator();
				cinScore = chart.getInsideScore(idTag, idx, level);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (usemasks) { if (!chart.isAllowed(rule.lhs, idx, true)) { continue; } } // CHECK
					if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; } // ROOT is allowed only when it is in cell 0 and is in level 1 or 2
					rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
					pinScore = rule.weight.mulForInsideOutside(cinScore, rmKey, true);
					chart.addInsideScore(rule.lhs, idx, pinScore, (short) (level + 1), false);
				}
			}
			level++;
			if (prune) { chart.pruneInsideScore(idx, level); }
		}
		// have to process ROOT node specifically, ROOT is in cell 0 and is in level 3
		if (idx == 0 && (set = chart.keySet(idx, true, LENGTH_UCHAIN)) != null) {
			for (Short idTag : set) { // the maximum inside level below ROOT
				rules = grammar.getURuleWithC(idTag);
				Iterator<GrammarRule> iterator = rules.iterator();
				cinScore = chart.getInsideScore(idTag, idx, LENGTH_UCHAIN);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (rule.type != GrammarRule.RHSPACE) { continue; } // only consider ROOT in level 3
					pinScore = rule.weight.mulForInsideOutside(cinScore, GrammarRule.Unit.C, true);
					chart.addInsideScore(rule.lhs, idx, pinScore, (short) (LENGTH_UCHAIN + 1), false);
				}
			}
			if (prune) { chart.pruneInsideScore(idx, (short) (LENGTH_UCHAIN + 1)); }
		}
	}
	
	public static void setRootOutsideScoreMask(Chart chart) {
		chart.addOutsideScoreMask((short) 0, Chart.idx(0, 1), 0, (short) (LENGTH_UCHAIN + 1), false);
	}
	
	/**
	 * @param chart 
	 */
	public static void setRootOutsideScore(Chart chart) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
//		GaussianMixture gm = DiagonalGaussianMixture.borrowObject((short) 1); // POOL
		gm.marginalizeToOne();
		chart.addOutsideScore((short) 0, Chart.idx(0, 1), gm, (short) (LENGTH_UCHAIN + 1), false);
	}
	
	
	public static class InputToSubCYKer {
		protected int ilayer;
		protected int nword;
		protected int left;
		protected Chart chart;
		protected Cell cell;
		protected boolean inside;
		protected GrammarRule rule;
		public InputToSubCYKer(int ilayer, int left, int nword,
				Cell cell, Chart chart, GrammarRule rule, boolean inside) {
			this.inside = inside;
			this.ilayer = ilayer;
			this.nword = nword;
			this.chart = chart;
			this.left = left;
			this.cell = cell;
			this.rule = rule;
		}
	}
	
	public static class SubCYKer<I, O> implements Executor<I, O> {
		/**
		 * 
		 */
		private static final long serialVersionUID = 3714235072505026471L;
		protected int idx;
		
		protected I task;
		protected int itask;
		protected PriorityQueue<Meta<O>> caches;
		
		public SubCYKer() {}
		private SubCYKer(SubCYKer<?, ?> subCYKer) {}

		@Override
		public synchronized Object call() throws Exception {
			if (task == null) { return null; }
			InputToSubCYKer input = (InputToSubCYKer) task;
			int x0, x1, y0, y1, c0, c1;
			int left = input.left, ilayer = input.ilayer, nword = input.nword;
			if (input.inside) { // inside scores with binary rules
				x0 = left;
				y1 = left + ilayer;
				GaussianMixture pinScore, linScore, rinScore, ruleScore;
				BinaryGrammarRule rule = (BinaryGrammarRule) (input.rule);
				for (int right = left; right < left + ilayer; right++) {
					y0 = right;
					x1 = right + 1;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
				
					if (input.chart.containsKey(rule.lchild, c0, true) && input.chart.containsKey(rule.rchild, c1, true)) {
						ruleScore = rule.getWeight();
						linScore = input.chart.getInsideScore(rule.lchild, c0);
						rinScore = input.chart.getInsideScore(rule.rchild, c1);
						
						pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
						pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
						input.cell.addScore(rule.lhs, pinScore, (short) 0, false);
					}
				}
			} else { // outside scores with binary rules
				x0 = left;
				x1 = left + ilayer + 1; 
				GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
				BinaryGrammarRule rule = (BinaryGrammarRule) (input.rule);
				for (int right = left + ilayer + 1; right < nword; right++) {
					y0 = right;
					y1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
				
					if (input.chart.containsKey(rule.lhs, c0, false) && input.chart.containsKey(rule.rchild, c1, true)) {
						ruleScore = rule.getWeight();
						poutScore = input.chart.getOutsideScore(rule.lhs, c0);
						rinScore = input.chart.getInsideScore(rule.rchild, c1);
						
						loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
						loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
						input.cell.addScore(rule.lchild, loutScore, (short) 0, false);
					}
				}
				y0 = left + ilayer;
				y1 = left - 1;
				for (int right = 0; right < left; right++) {
					x0 = right; 
					x1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					if (input.chart.containsKey(rule.lhs, c0, false) && input.chart.containsKey(rule.lchild, c1, true)) {
						ruleScore = rule.getWeight();
						poutScore = input.chart.getOutsideScore(rule.lhs, c0);
						linScore = input.chart.getInsideScore(rule.lchild, c1);
						
						routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
						routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
						input.cell.addScore(rule.rchild, routScore, (short) 0, false); 
					}
				}
			}
			Meta<O> cache = new Meta(itask, true); // currently nothing returns, may add try...catch.. clause
			synchronized (caches) {
				caches.add(cache);
				caches.notifyAll();
			}
			task = null;
			return null;
		}

		@Override
		public Executor<?, ?> newInstance() {
			return new SubCYKer<I, O>(this);
		}

		@Override
		public void setNextTask(int itask, I task) {
			this.task = task;
			this.itask = itask;
		}

		@Override
		public void setIdx(int idx, PriorityQueue<edu.shanghaitech.ai.nlp.util.Executor.Meta<O>> caches) {
			this.idx = idx;
			this.caches = caches;
		}
		
	}
	
	
	/**
	 * Manage cells in the chart.
	 * </p>
	 * TODO If we know the maximum length of the sentence, we can pre-allocate 
	 * the memory space and reuse it. And static type is prefered.
	 * </p>
	 * 
	 * @author Yanpeng Zhao
	 *
	 */
	public static class Chart {
		private List<Cell> ochart = null;
		private List<Cell> ichart = null;
		private List<Cell> mchart = null; // for max-rule parser
		
		private List<Cell> omasks = null; // for treebank grammars
		private List<Cell> imasks = null;
		
		private PriorityQueue<Double> queue = null; // owned by this chart, need to make it thread safe
		
		protected Chart() { // TODO make it static? See clear().
			if (ichart == null || ichart.size() != LVeGLearner.maxlength) {
				initialize(LVeGLearner.maxlength, false, false);
			} else {
				clear(-1);
			}
		}
		
		public Chart(int n, boolean maxrule) {
			queue = new PriorityQueue<Double>();
			initialize(n, maxrule, false);
		}
		
		public Chart(int n, boolean maxrule, boolean usemask) {
			queue = new PriorityQueue<Double>();
			initialize(n, maxrule, usemask);
		}
		
		private void initialize(int n, boolean maxrule, boolean usemask) {
			clear(n); // empty the old memory
			int size = n * (n + 1) / 2;
			ochart = new ArrayList<Cell>(size);
			ichart = new ArrayList<Cell>(size);
			for (int i = 0; i < size; i++) {
				ochart.add(new Cell());
				ichart.add(new Cell());
			}
			if (maxrule) {
				mchart = new ArrayList<Cell>(size);
				for (int i = 0; i < size; i++) {
					mchart.add(new Cell(maxrule));
				}
			}
			if (usemask) {
				imasks = new ArrayList<Cell>(size);
				omasks = new ArrayList<Cell>(size);
				for (int i = 0; i < size; i++) {
					imasks.add(new Cell(maxrule, usemask));
					omasks.add(new Cell(maxrule, usemask));
				}
			}
		}
		
		/**
		 * Map the index to the real memory address. Imagine the upper right 
		 * triangle chart as the pyramid. E.g., see the pyramid below, which 
		 * could represent a sentence of length 6.
		 * <pre>
		 * + + + + + +
		 *   + + + + +
		 *     + + + +
		 *       + + +
		 *         + +
		 *           + 
		 * </pre>
		 * 
		 * @param i      index of the row
		 * @param ilayer layer in the pyramid, from top (1) to bottom (ilayer), loc = ilayer * (ilayer - 1) / 2 + i;
		 * 				  loc = (n + n - ilayer + 1) * ilayer / 2 + i if ilayer ranges from bottom (0, 0)->0 to top (ilayer).
		 * @return
		 */
		public static int idx(int i, int ilayer) {
			return ilayer * (ilayer - 1) / 2 + i;
		}
		
		public void setStatus(int idx, boolean status, boolean inside) {
			if (inside) {
				ichart.get(idx).setStatus(status);
			} else {
				ochart.get(idx).setStatus(status);
			}
		}
		
		public void addMaxRuleCount(short key, int idx, double count, int sons, Short splitpoint, short level) {
			mchart.get(idx).addMaxRuleCount(key, count, sons, splitpoint, level);
		}
		
		public int getMaxRuleSon(short key, int idx, short level) {
			return mchart.get(idx).getMaxRuleSon(key, level);
		}
		
		public int getMaxRuleSon(short key, int idx) {
			return mchart.get(idx).getMaxRuleSon(key);
		}
		
		public double getMaxRuleCount(short key, int idx, short level) {
			return mchart.get(idx).getMaxRuleCount(key, level);
		}
		
		public double getMaxRuleCount(short key, int idx) {
			return mchart.get(idx).getMaxRuleCount(key);
		}
		
		public short getSplitPoint(short key, int idx) {
			return mchart.get(idx).getSplitPoint(key);
		}
		
		public Set<Short> keySetMaxRule(int idx, short level) {
			return mchart.get(idx).keySetMaxRule(level);
		}
		
		public boolean getStatus(int idx, boolean inside) {
			return inside ? ichart.get(idx).getStatus() : ochart.get(idx).getStatus();
		}
		
		public List<Cell> getChart(boolean inside) {
			return inside ? ichart : ochart;
		}
		
		public Cell get(int idx, boolean inside) {
			return inside ? ichart.get(idx) : ochart.get(idx);
		}
		
		public int size(int idx, boolean inside) {
			return inside ? ichart.get(idx).size() : ochart.get(idx).size();
		}
		
		public Set<Short> keySetMask(int idx, boolean inside, short level) {
			return inside ? imasks.get(idx).keySetMask(level) : omasks.get(idx).keySetMask(level);
		}
		
		public Set<Short> keySet(int idx, boolean inside, short level) {
			return inside ? ichart.get(idx).keySet(level) : ochart.get(idx).keySet(level);
		}
		
		public Set<Short> keySetMask(int idx, boolean inside) {
			return inside ? imasks.get(idx).keySetMask() : omasks.get(idx).keySetMask();
		}
		
		public Set<Short> keySet(int idx, boolean inside) {
			return inside ? ichart.get(idx).keySet() : ochart.get(idx).keySet();
		}
		
		public boolean containsKeyMask(short key, int idx, boolean inside, short level) {
			return inside ? imasks.get(idx).containsKeyMask(key, level) : omasks.get(idx).containsKeyMask(key, level);
		}
		
		public boolean containsKey(short key, int idx, boolean inside, short level) {
			return inside ? ichart.get(idx).containsKey(key, level) : ochart.get(idx).containsKey(key, level);
		}
		
		public boolean containsKeyMask(short key, int idx, boolean inside) {
			return inside ? imasks.get(idx).containsKeyMask(key) : omasks.get(idx).containsKeyMask(key);
		}
		
		public boolean containsKey(short key, int idx, boolean inside) {
			return inside ? ichart.get(idx).containsKey(key) : ochart.get(idx).containsKey(key);
		}
		
		public boolean isAllowed(short key, int idx, boolean inside) {
			return inside ? imasks.get(idx).isAllowed(key) : omasks.get(idx).isAllowed(key);
		}
		
		public void addInsideScoreMask(short key, int idx, double score, short level, boolean prune) {
			imasks.get(idx).addScoreMask(key, score, level, prune);
		}
		
		public void addInsideScore(short key, int idx, GaussianMixture gm, short level, boolean prune) {
			ichart.get(idx).addScore(key, gm, level, prune);
		}
		
		public double getInsideScoreMask(short key, int idx, short level) {
			return imasks.get(idx).getScoreMask(key, level);
		}
		
		public GaussianMixture getInsideScore(short key, int idx, short level) {
			return ichart.get(idx).getScore(key, level);
		}
		
		public double getInsideScoreMask(short key, int idx) {
			return imasks.get(idx).getScoreMask(key);
		}
		
		public GaussianMixture getInsideScore(short key, int idx) {
			return ichart.get(idx).getScore(key);
		}
		
		public void addOutsideScoreMask(short key, int idx, double score, short level, boolean prune) {
			omasks.get(idx).addScoreMask(key, score, level, prune);
		}
		
		public void addOutsideScore(short key, int idx, GaussianMixture gm, short level, boolean prune) {
			ochart.get(idx).addScore(key, gm, level, prune);
		}
		
		public double getOutsideScoreMask(short key, int idx, short level) {
			return omasks.get(idx).getScoreMask(key, level);
		}
		
		public GaussianMixture getOutsideScore(short key, int idx, short level) {
			return ochart.get(idx).getScore(key, level);
		}
		
		public double getOutsideScoreMask(short key, int idx) {
			return omasks.get(idx).getScoreMask(key);
		}
		
		public GaussianMixture getOutsideScore(short key, int idx) {
			return ochart.get(idx).getScore(key);
		}
		
		public void pruneOutsideScoreMask(int idx, short level, int base, double ratio) {
			if (level < 0) {
				omasks.get(idx).pruneScoreMask(queue, base, ratio);
			} else {
				omasks.get(idx).pruneScoreMask(level, queue, base, ratio);
			}
		}
		
		public void pruneOutsideScore(int idx, short level) {
			if (level < 0) {
				ochart.get(idx).pruneScore();
			} else {
				ochart.get(idx).pruneScore(level);
			}
		}
		
		public void pruneInsideScoreMask(int idx, short level, int base, double ratio) {
			if (level < 0) {
				imasks.get(idx).pruneScoreMask(queue, base, ratio);
			} else {
				imasks.get(idx).pruneScoreMask(level, queue, base, ratio);
			}
		}
		
		public void pruneInsideScore(int idx, short level) {
			if (level < 0) {
				ichart.get(idx).pruneScore();
			} else {
				ichart.get(idx).pruneScore(level);
			}
		}
		
		/**
		 * @deprecated
		 */
		public void addInsideScore(short key, int idx, GaussianMixture gm, boolean prune) {
			ichart.get(idx).addScore(key, gm, prune);
		}
		
		/**
		 * @deprecated
		 */
		public void addOutsideScore(short key, int idx, GaussianMixture gm, boolean prune) {
			ochart.get(idx).addScore(key, gm, prune);
		}
		
		public void clear(int n) {
			int cnt, max = n > 0 ? (n * (n + 1) / 2) : ichart.size();
			if (ichart != null) {
				cnt = 0;
				for (Cell cell : ichart) {
					if (++cnt > max) { break; }
					if (cell != null) { cell.clear(); }
				}
				if (n < 0) { ichart.clear(); }
			}
			if (ochart != null) {
				cnt = 0;
				for (Cell cell : ochart) {
					if (++cnt > max) { break; }
					if (cell != null) { cell.clear(); }
				}
				if (n < 0) { ochart.clear(); }
			}
			if (mchart != null) {
				cnt = 0;
				for (Cell cell : mchart) {
					if (++cnt > max) { break; }
					if (cell != null) { cell.clear(); }
				}
				if (n < 0) { mchart.clear(); }
			}
			if (imasks != null) {
				cnt = 0;
				for (Cell cell : imasks) {
					if (++cnt > max) { break; }
					if (cell != null) { cell.clear(); }
				}
				if (n < 0) { imasks.clear(); }
			}
			if (omasks != null) {
				cnt = 0;
				for (Cell cell : omasks) {
					if (++cnt > max) { break; }
					if (cell != null) { cell.clear(); }
				}
				if (n < 0) { omasks.clear(); }
			}
		}
		
		@Override
		public String toString() {
			return "Chart [ichart=" + ichart + ", ochart=" + ochart + "]";
		}
	}
	
	
	/**
	 * Cells of the chart used in calculating inside and outside scores.
	 * 
	 * @author Yanpeng Zhao
	 *
	 */
	public static class Cell {
		// key word "private" does not make any difference, outer class can access all the fields 
		// of the inner class through the instance of the inner class or in the way of the static 
		// fields accessing.
		private boolean status;
		private Map<Short, GaussianMixture> totals;
		private Map<Short, Map<Short, GaussianMixture>> scores;
		
		private Map<Short, Map<Short, Double>> maxRuleCnts;
		private Map<Short, Map<Short, Integer>> maxRuleSons;
		private Map<Short, Integer> maxRuleSon;
		private Map<Short, Short> maxRulePos;
		private Map<Short, Short> splitPoint;
		// using masks
		private Set<Short> masks; 
		private Map<Short, Double> mtotals;
		private Map<Short, Map<Short, Double>> mscores;
		
		
		private Cell() {
			this.status = false;
			this.totals = new HashMap<Short, GaussianMixture>();
			this.scores = new HashMap<Short, Map<Short, GaussianMixture>>();
		}
		
		public Cell(boolean maxrule) {
//			this(); // CHECK
			if (maxrule) {
				this.maxRuleCnts = new HashMap<Short, Map<Short, Double>>(3, 1);
				this.maxRuleSons = new HashMap<Short, Map<Short, Integer>>(3, 1);
				this.maxRulePos = new HashMap<Short, Short>();
				this.maxRuleSon = new HashMap<Short, Integer>(); // low 2 bytes are used
				this.splitPoint = new HashMap<Short, Short>();
			}
		}
		
		public Cell(boolean maxrule, boolean usemask) {
//			this(maxrule); // CHECK
			if (usemask) {
				this.masks = new HashSet<Short>();
				this.mtotals = new HashMap<Short, Double>();
				this.mscores = new HashMap<Short, Map<Short, Double>>();
			}
		}
		
		protected void setStatus(boolean status) {
			this.status = status;
		}
		
		protected boolean getStatus() {
			return status;
		}
		
		protected int size() {
			return totals.size();
		}
		
		protected Set<Short> keySetMask() {
			return mtotals.keySet();
		}
		
		protected Set<Short> keySet() {
			return totals.keySet();
		}
		
		protected Set<Short> keySetMask(short level) {
			return mscores.get(level) == null ? null : mscores.get(level).keySet();
		}
		
		protected Set<Short> keySet(short level) {
			return scores.get(level) == null ? null : scores.get(level).keySet();
		}
		
		protected Set<Short> keySetMaxRule(short level) {
			return maxRuleCnts.get(level) == null ? null : maxRuleCnts.get(level).keySet();
		}
		
		protected boolean containsKeyMask(short key) {
			return mtotals.containsKey(key);
		}
		
		protected boolean containsKey(short key) {
			return totals.containsKey(key);
		}
		
		protected boolean containsKeyMask(short key, short level) {
			return mscores.get(level) == null ? false : mscores.get(level).containsKey(key);
		}
		
		protected boolean containsKey(short key, short level) {
			return scores.get(level) == null ? false : scores.get(level).containsKey(key);
		}
		
		protected boolean isAllowed(short key) {
			return masks.contains(key);
		}
		
		protected void pruneScoreMask(PriorityQueue<Double> queue, int base, double ratio) {
			if (mtotals.size() > base) {
				int k = (int) (base + Math.floor(mtotals.size() * ratio));
				double kval;
				queue.clear();
				Collection<Double> scores = mtotals.values();
				for (Double d : scores) { 
					queue.offer(d);
					if (queue.size() > k) { queue.poll(); }
				}
				kval = queue.peek(); // k-th largest value
				for (Map.Entry<Short, Double> score : mtotals.entrySet()) {
					if (score.getValue() >= kval) { masks.add(score.getKey()); }
				}
				
			} else {
				masks.addAll(mtotals.keySet());
			}
		}
		
		protected void pruneScore() {
			Collection<GaussianMixture> scores = totals.values();
			for (GaussianMixture score : scores) {
				score.delTrivia();
			}
		}
		
		protected void pruneScoreMask(short level, PriorityQueue<Double> queue, int base, double ratio) {
			// TODO
		}
		
		protected void pruneScore(short level) {
			Map<Short, GaussianMixture> lscores = scores.get(level);
			if (lscores != null) {
				Collection<GaussianMixture> lscore = lscores.values();
				for (GaussianMixture score : lscore) {
					score.delTrivia();
				}
			}
		}
		
		protected void addMaxRuleCount(short key, double count, int sons, Short splitpoint, short level) {
			// cnts for the same nonterminals in different levels
			Map<Short, Double> lcnts = maxRuleCnts.get(level);
			if (lcnts == null) {
				lcnts = new HashMap<Short, Double>();
				maxRuleCnts.put(level, lcnts);
			}
			Double cnt = lcnts.get(key); // double check
			if (cnt != null && cnt > count) { return; }
			lcnts.put(key, count);
			// sons for the same nonterminals in different levels
			Map<Short, Integer> lsons = maxRuleSons.get(level);
			if (lsons == null) {
				lsons = new HashMap<Short, Integer>();
				maxRuleSons.put(level, lsons);
			}
			lsons.put(key, sons);
			// the max one
			maxRulePos.put(key, level);
			maxRuleSon.put(key, sons);
			if (level == 0) { // binary rules or ROOT
				splitPoint.put(key, splitpoint);
			}
		}
		
		protected double getMaxRuleCount(short key, short level) {
			Double cnt = maxRuleCnts.get(level) == null ? null : maxRuleCnts.get(level).get(key);
			return cnt == null ? Double.NEGATIVE_INFINITY : cnt;
		}
		
		protected double getMaxRuleCount(short key) {
			Short lkey = maxRulePos.get(key);
			return lkey == null ? Double.NEGATIVE_INFINITY : maxRuleCnts.get(lkey).get(key);
		}
		
		protected int getMaxRuleSon(short key, short level) {
			Integer son = maxRuleSons.get(level) == null ? null : maxRuleSons.get(level).get(key);
			return son == null ? -1 : son;
			/*
			Map<Short, Integer> lsons = null;
			if ((lsons = maxRuleSons.get(level)) != null) {
				return lsons.get(key);
			}
			return -1;
			*/
		}
		
		protected int getMaxRuleSon(short key) {
			return maxRuleSon.get(key);
		}
		
		protected short getSplitPoint(short key) {
			Short split = splitPoint.get(key);
			return split == null ? -1 : split;
		}
		
		
		protected synchronized void addScoreMask(short key, double score, short level, boolean prune) {
			Map<Short, Double> lscore = mscores.get(level);
			if (lscore == null) {
				lscore = new HashMap<Short, Double>();
				mscores.put(level, lscore);
			}
			Double ascore = lscore.get(key);
			if (ascore == null) {
				lscore.put(key, score);
			} else {
				lscore.put(key, FunUtil.logAdd(ascore, score));
			}
			addScoreMask(key, score, prune);
		}
		
		private synchronized void addScoreMask(short key, double score, boolean prune) {
			if (containsKeyMask(key)) {
				mtotals.put(key, FunUtil.logAdd(mtotals.get(key), score));
			} else {
				mtotals.put(key, score);
			}
		}
		
		protected synchronized void addScore(short key, GaussianMixture gm, short level, boolean prune) {
			Map<Short, GaussianMixture> lscore = scores.get(level);
			if (lscore == null) {
				lscore = new HashMap<Short, GaussianMixture>();
				scores.put(level, lscore);
			}
			GaussianMixture agm = lscore.get(key);
			if (agm == null) {
				lscore.put(key, gm);
			} else {
				agm.add(gm, prune);
			}
			addScore(key, gm, prune);
		}
		
		private synchronized void addScore(short key, GaussianMixture gm, boolean prune) {
			if (containsKey(key)) { 
				// gm is passed into this method by addScore(short, GaussianMixture, short, boolean),
				// before that it has been added into Cell.scores, and is filtered when 
				// GaussianMixture.add(GaussianMixture, boolean) is called. Here gm may be filtered
				// again by calling totals.get(key).add(...), in which some components of gm may be 
				// cleared through the reference and further modify Cell.scores. The safe practice is
				// copying gm and adding into Cell.totals, but that results in unnecessary memory
				// overhead, so I choose not to clear the filtered component in GaussianMixture.add()
				totals.get(key).add(gm, prune);
				/*totals.get(key).add(gm.copy(true), prune);*/
			} else {
				// it should own its own memory space, so that the score in a 
				// specific level could not be modified through the reference
				totals.put(key, gm.copy(true));
			}
		}
		
		protected double getScoreMask(short key, short level) {
			return mscores.get(level) == null ? Double.NEGATIVE_INFINITY : mscores.get(level).get(key);
		}
		
		protected GaussianMixture getScore(short key, short level) {
			return scores.get(level) == null ? null : scores.get(level).get(key);
		}
		
		protected double getScoreMask(short key) {
			return mtotals.get(key);
		}
		
		protected GaussianMixture getScore(short key) {
			return totals.get(key);
		}
		
		protected void clear() {
			status = false;
			if (scores != null) {
//				for (Map.Entry<Short, Map<Short, GaussianMixture>> level : scores.entrySet()) {
//					for (Map.Entry<Short, GaussianMixture> entry : level.getValue().entrySet()) {
//						if (entry.getValue() != null) { 
//							entry.getValue().clear(); 
////							GaussianMixture.returnObject(entry.getValue()); // POOL
//						}
//					}
//					level.getValue().clear();
//				}
				scores.clear();
			}
			if (totals != null) {
//				for (Map.Entry<Short, GaussianMixture> entry : totals.entrySet()) {
//					if (entry.getValue() != null) { 
//						entry.getValue().clear(); 
////						GaussianMixture.returnObject(entry.getValue()); // POOL
//					}
//				}
				totals.clear();
			}
			// the following is for max rule parser
			if (maxRuleCnts != null) {
//				for (Map.Entry<Short, Map<Short, Double>> entry : maxRuleCnts.entrySet()) {
//					if (entry.getValue() != null) { entry.getValue().clear(); }
//				}
				maxRuleCnts.clear();
			}
			if (maxRuleSon != null) { maxRuleSon.clear(); }
			if (maxRulePos != null) { maxRulePos.clear(); }
			if (splitPoint != null) { splitPoint.clear(); }
			
			if (mscores != null) { mscores.clear(); }
			if (mtotals != null) { mtotals.clear(); }
		}
		
		public String toString(boolean simple, int nfirst, boolean quantity) {
			if (simple) {
				String name;
				StringBuffer sb = new StringBuffer();
				sb.append("Cell [status=" + status + ", size=" + totals.size());
				
				for (Map.Entry<Short, GaussianMixture> score : totals.entrySet()) {
					name = (String) grammar.numberer.object(score.getKey());
					if (quantity) {
						sb.append(", " + name + "(nc)=" + score.getValue().ncomponent);
					} else {
						sb.append(", " + name + "=" + score.getValue().toString(simple, nfirst));
					}
				}
				
				sb.append("]");
				
				sb.append("\n\n--- details in each level---\n");
				for (Map.Entry<Short, Map<Short, GaussianMixture>> level : scores.entrySet()) {
					sb.append("\n------>level " + level.getKey() + " ntag = " + level.getValue().size() + "\n");
					for (Map.Entry<Short, GaussianMixture> detail : level.getValue().entrySet()) {
						name = (String) grammar.numberer.object(detail.getKey());
						if (quantity) {
							sb.append("\nid=" + detail.getKey() + ", " + name + "(nc)=" + detail.getValue().ncomponent);
						} else {
							sb.append("\nid=" + detail.getKey() + ", " + name + "=" + detail.getValue().toString(simple, nfirst));
						}
					}
					sb.append("\n");
				}
				
				return sb.toString();
			} else {
				return toString();
			}
		}
		
		@Override
		public String toString() {
			String name;
			StringBuffer sb = new StringBuffer();
			sb.append("Cell [status=" + status + ", size=" + totals.size());
			for (Map.Entry<Short, GaussianMixture> score : totals.entrySet()) {
				name = (String) grammar.numberer.object(score.getKey());
				sb.append(", " + name + "=" + score.getValue());
			}
			sb.append("]");
			return sb.toString();
		}
	}
}
