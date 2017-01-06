package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
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
		chainurule = ChainUrule.DEFAULT;
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
		int x0, y0, x1, y1, c0, c1, c2;
		double lcount, rcount, maxcnt, newcnt;
		GaussianMixture linScore, rinScore, outScore, cinScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		// lexicons
		for (int i = 0; i < nword; i++) {
			State word = sentence.get(i);
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(word);
			for (GrammarRule rule : rules) {
				if (chart.containsKey(rule.lhs, iCell, false)) {
					cinScore = lexicon.score(word, rule.lhs);
					outScore = chart.getOutsideScore(rule.lhs, iCell);
					newcnt = outScore.marginalize(true) + cinScore.marginalize(true) - scoreS;
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
				for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
					BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
					if ((outScore = chart.getOutsideScore(rule.lhs, c2)) == null) { continue; }
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
						newcnt = newcnt + rule.weight.marginalize(true) + outScore.marginalize(true) + 
								linScore.marginalize(true) + rinScore.marginalize(true) - scoreS; // CHECK
						if (newcnt > maxcnt) {
							// the negative, higher 2 bytes (lchild, sign bit exclusive) <- lower 2 bytes (rchild)
							int sons = (1 << 31) + (rule.lchild << 16) + rule.rchild;
							chart.addMaxRuleCount(rule.lhs, c2, newcnt, sons, (short) right, (short) 0);
						}
					}
				}
				// unary rules
				maxRuleCountForUnaryRule(chart, c2, scoreS);
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
		GaussianMixture outScore, cinScore, w0, w1;
		// chain unary rule of length 1
		Set<Short> mkeyLevel0 = chart.keySetMaxRule(idx, (short) 0);
		for (short mkey : mkeyLevel0) {
			if ((cinScore = chart.getInsideScore(mkey, idx, (short) 0)) == null) { continue; }
			if ((count = chart.getMaxRuleCount(mkey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
			rules = grammar.getURuleWithC(mkey);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				if (rule.type == GrammarRule.RHSPACE) { continue; } // ROOT in cell 0 is NOT allowed
				if ((maxcnt = chart.getMaxRuleCount(rule.lhs, idx)) > count) { continue; }
				if ((outScore = chart.getOutsideScore(rule.lhs, idx, (short) 0)) == null) { continue; }
				newcnt = count + rule.weight.marginalize(true) + cinScore.marginalize(true) + outScore.marginalize(true) - scoreS;
				if (newcnt > maxcnt) {
					chart.addMaxRuleCount(rule.lhs, idx, newcnt, mkey, (short) -1, (short) 1);
				}
			}
		}
		// chain unary rule of length 2
		Set<Short> ikeyLevel0 = chart.keySet(idx, true, (short) 0);
		Set<Short> ikeyLevel1 = chart.keySet(idx, true, (short) 1);
		Set<Short> okeyLevel0 = chart.keySet(idx, false, (short) 0);
		if (ikeyLevel0 != null && ikeyLevel1 != null && okeyLevel0 != null) {
			for (Short ikey : ikeyLevel0) {
				if ((count = chart.getMaxRuleCount(ikey, idx, (short) 0)) == Double.NEGATIVE_INFINITY) { continue; }
				for (Short okey : okeyLevel0) {
					if (ikey == okey) { continue; } // nonsense
					if ((maxcnt = chart.getMaxRuleCount(okey, idx)) > count) { continue; }
					if ((cinScore = chart.getInsideScore(ikey, idx, (short) 0)) == null || 
							(outScore = chart.getOutsideScore(okey, idx, (short) 0)) == null) {
						continue;
					}
					for (Short mid : ikeyLevel1) {
						if ((w0 = grammar.getURuleWeight(mid, ikey, GrammarRule.LRURULE)) == null ||
								(w1 = grammar.getURuleWeight(okey, mid, GrammarRule.LRURULE)) == null) {
							continue;
						}
						newcnt = count + cinScore.marginalize(true) + w0.marginalize(true) + 
								w1.marginalize(true) + outScore.marginalize(true) - scoreS;
						if (newcnt > maxcnt) {
							int sons = (ikey << 16) + mid; // higher 2 bytes (grandson) <- lower 2 bytes (child)
							chart.addMaxRuleCount(okey, idx, newcnt, sons, (short) -1, (short) 2);
						}
					}
				}
			}
		}
		// ROOT
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
				newcnt = count + rule.weight.marginalize(true) + cinScore.marginalize(true) + outScore.marginalize(true) - scoreS;
				if (newcnt > maxcnt) {
					chart.addMaxRuleCount(ROOT, idx, newcnt, rule.rhs, (short) -1, (short) 0);
				}
			}	
		}
	}
	
	protected Tree<String> extractBestMaxRuleParse(Chart chart, List<String> sentence) {
		return extractBestMaxRuleParse(chart, 0, sentence.size() - 1, sentence.size(), (short) 0, sentence);
	}
	
	/**
	 * @param chart
	 * @param left     0
	 * @param right    sentence.size() - 1
	 * @param nword
	 * @param sentence the sentence
	 * @return
	 */
	protected Tree<String> extractBestMaxRuleParse(Chart chart, int left, int right, int nword, List<String> sentence) {
		return extractBestMaxRuleParse(chart, left, right, nword, (short) 0, sentence);
	}
	
	/**
	 * @param chart
	 * @param left     0
	 * @param right    sentence.size() - 1
	 * @param nword
	 * @param idtag
	 * @param sentence the sentence
	 * @return
	 */
	private Tree<String> extractBestMaxRuleParse(Chart chart, int left, int right, int nword, short idtag, List<String> sentence) {
		int idx = Chart.idx(left, nword - (right - left));
		int son = chart.getMaxRuleSon(idtag, idx);
		if (son <= 0) { // sons = (1 << 31) + (rule.lchild << 16) + rule.rchild; or sons = 0;
			return extractBestMaxRuleParseBinary(chart, left, right, nword, idtag, sentence);
		} else {
			short idGrandson = (short) (son >>> 16);
			short idChild = (short) ((son << 16) >>> 16);
			List<Tree<String>> child = new ArrayList<Tree<String>>();
			String pname = (String) grammar.numberer.object(idtag);
			if (pname.endsWith("^g")) { pname = pname.substring(0, pname.length() - 2); }
			if (idx == 0 && idtag == 0) { // ROOT->A->B->C; ROOT->B->C; ROOT->C;
				if (idGrandson != 0) { logger.error("There must be something wrong in the max rule parse\n."); }
				child.add(extractBestMaxRuleParse(chart, left, right, nword, idChild, sentence));
				return new Tree<String>(pname, child);
			}
			if (idGrandson == 0) {
				child.add(extractBestMaxRuleParseBinary(chart, left, right, nword, idChild, sentence));
				return new Tree<String>(pname, child);
			} else {
				child.add(extractBestMaxRuleParseBinary(chart, left, right, nword, idGrandson, sentence));
				List<Tree<String>> chainChild = new ArrayList<Tree<String>>();
				String cname = (String) grammar.numberer.object(idChild);
				if (cname.endsWith("^g")) { cname = cname.substring(0, cname.length() - 2); }
				chainChild.add(new Tree<String>(cname, child));
				return new Tree<String>(pname, chainChild);
			}
		}
	}
	
	private Tree<String> extractBestMaxRuleParseBinary(Chart chart, int left, int right, int nword, short idtag, List<String> sentence) {
		List<Tree<String>> children = new ArrayList<Tree<String>>();
		String pname = (String) grammar.numberer.object(idtag);
		if (pname.endsWith("^g")) { pname = pname.substring(0, pname.length() - 2); }
		int idx = Chart.idx(left, nword - (right - left));
		int son = ((chart.getMaxRuleSon(idtag, idx) << 1) >>> 1);
		if (right  == left) {
			if (son == 0) {
				children.add(new Tree<String>(sentence.get(left)));
			} else {
				logger.error("must be somthing wrong.\n");
			}
		} else {
			int splitpoint = chart.getSplitPoint(idtag, idx);
			if (splitpoint == -1) {
				logger.error("\n---holly shit---\n");
				logger.error("do you want to know what is wrong?\n");
				return new Tree<String>("ROOT");
			}
			short lchild = (short) (son >>> 16);
			short rchild = (short) ((son << 16) >> 16);
			Tree<String> lchildTree = extractBestMaxRuleParse(chart, left, splitpoint, nword, lchild, sentence);
			Tree<String> rchildTree = extractBestMaxRuleParse(chart, splitpoint + 1, right, nword, rchild, sentence);
			children.add(lchildTree);
			children.add(rchildTree);
		}
		return new Tree<String>(pname, children);
	}
	
}