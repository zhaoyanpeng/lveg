package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import org.apache.log4j.Logger;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.MethodUtil;

/**
 * Compute the inside and outside scores and store them in a chart. 
 * Imagine the upper right triangle chart as the pyramid. E.g., see
 * the pyramid below, which could represent a sentence of length 6.
 * <pre>
 * + + + + + +
 *   + + + + +
 *     + + + +
 *       + + +
 *         + +
 *           + 
 * </pre>
 * @author Yanpeng Zhao
 *
 */
public class Inferencer {
	
	protected LVeGLexicon lexicon;
	protected LVeGGrammar grammar;
	
	
	public Inferencer(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.grammar = grammar;
		this.lexicon = lexicon;
	}
	
	
	/**
	 * Compute the inside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used.
	 * @param nword # of words in the sentence
	 */
	protected void insideScore(Chart chart, List<State> sentence, int nword) {
		int x0, y0, x1, y1, c0, c1, c2;
		Queue<Short> tagQueue = new LinkedList<Short>(); 
		GaussianMixture pinScore, linScore, rinScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			int wordIdx = sentence.get(i).wordIdx; 
			List<GrammarRule> rules = lexicon.getRulesWithWord(wordIdx);
			// pre-terminals
			for (GrammarRule rule : rules) {
				tagQueue.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true));
			}
			// unary grammar rules
			insideScoreForUnaryRule(chart, tagQueue, iCell, false);
		}		
		
		// inside score
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				for (int right = left; right < left + ilayer; right++) {				
					y0 = right;
					x1 = right + 1;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					// to ensure...
					tagQueue.clear();
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.iContainsKey(rule.lchild, c0) && chart.iContainsKey(rule.rchild, c1)) {
							tagQueue.offer(rule.lhs);
							ruleScore = rule.getWeight();
							linScore = chart.getInsideScore(rule.lchild, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
							pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addInsideScore(rule.lhs, c2, pinScore);
						}
					}
					// unary grammar rules
					insideScoreForUnaryRule(chart, tagQueue, c2, false);
				}
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
	protected void outsideScore(Chart chart, List<State> sentence, int nword) {
		
		int x0, y0, x1, y1, c0, c1, c2;
		Queue<Short> tagQueue = new LinkedList<Short>(); 
		GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		tagQueue.offer((short) 0);
		outsideScoreForUnaryRule(chart, tagQueue, Chart.idx(0, 1), false);
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				// to ensure...
				tagQueue.clear();
				
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
						if (chart.oContainsKey(rule.lhs, c0) && chart.iContainsKey(rule.rchild, c1)) {
							tagQueue.offer(rule.lchild);
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addOutsideScore(rule.lchild, c2, loutScore);
						}
					}
				}
				
				y0 = left + ilayer;
				y1 = left - 1;
				// c2 = Chart.idx(left, nword - ilayer);
				for (int right = 0; right < left; right++) {
					x0 = right; 
					x1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.oContainsKey(rule.lhs, c0) && chart.iContainsKey(rule.lchild, c1)) {
							tagQueue.offer(rule.rchild);
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, c0);
							linScore = chart.getInsideScore(rule.lchild, c1);
							
							routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
							chart.addOutsideScore(rule.rchild, c2, routScore);
						}
					}
				}
//				LVeGLearner.logger.trace("Can u hear me?\t"); // DEBUG
				// unary grammar rules
				outsideScoreForUnaryRule(chart, tagQueue, c2, false);
//				LVeGLearner.logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t has been estimated."); // DEBUG
			}
		}
	}
	
	
	/**
	 * Compute the inside score with the parse tree known.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	protected void insideScoreWithTree(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			insideScoreWithTree(child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture cinScore = lexicon.score(word, idParent);
			// LVeGLearner.logger.trace("Word\trule: [" + idParent + "] " + cinScore); // DEBUG
			parent.setInsideScore(cinScore.copy(true));
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				GaussianMixture ruleScore, cinScore, pinScore;
				State child = children.get(0).getLabel();
				cinScore = child.getInsideScore();
				short idChild = child.getId();
				
				if (idParent != 0) {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.GENERAL);
					pinScore = ruleScore.mulForInsideOutside(cinScore, GrammarRule.Unit.UC, true);
				} else { // root, inside score of the root node is a constant in double
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
					pinScore = ruleScore.mulForInsideOutside(cinScore, GrammarRule.Unit.C, true);
				}
				// LVeGLearner.logger.trace("Unary\trule: [" + idParent + ", " + idChild + "] " + ruleScore); // DEBUG
				parent.setInsideScore(pinScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, pinScore, linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
				pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
				pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
				
				// LVeGLearner.logger.trace("Binary\trule: [" + idParent + ", " + idlChild + ", " + idrChild + "] " + ruleScore); // DEBUG
				parent.setInsideScore(pinScore);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	/**
	 * Compute the outside score with the parse tree known.
	 * 
	 * @param tree the parse tree
	 */
	protected void outsideScoreWithTree(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		State parent = tree.getLabel();
		short idParent = parent.getId();
		
		if (tree.isPreTerminal()) {
			// nothing to do
		} else {
			GaussianMixture poutScore = parent.getOutsideScore();
			
			switch (children.size()) {
			case 0:
				break;
			case 1: {
				GaussianMixture ruleScore, coutScore;
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				if (idParent != 0) {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.GENERAL);
				} else { // root
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
				}
				// rule: p(root->nonterminal) does not contain "P" part, so no removing occurs when
				// the current parent node is the root node
				coutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				
				// LVeGLearner.logger.trace("Unary\trule: [" + idParent + ", " + idChild + "] " + ruleScore); // DEBUG
				child.setOutsideScore(coutScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, loutScore, routScore;
				GaussianMixture linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
				loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
				
				routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
				
				// LVeGLearner.logger.trace("Binary\trule: [" + idParent + ", " + idlChild + ", " + idrChild + "] " + ruleScore); // DEBUG
				lchild.setOutsideScore(loutScore);
				rchild.setOutsideScore(routScore);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);
			}
			
			for (Tree<State> child : children) {
				outsideScoreWithTree(child);
			}
		}
	}
	
	
	protected void evalRuleCount(Tree<State> tree, Chart chart, short isample, double sentenceScore) {
		double count = 0.0;
		List<State> sentence = tree.getYield();
		int x0, x1, y0, y1, c0, c1, c2, nword = sentence.size();
		GaussianMixture outScore, cinScore, linScore, rinScore, ruleScore;
		
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				
				// unary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) rmap.getValue();
					if (chart.oContainsKey(rule.lhs, c2) && chart.iContainsKey(rule.rhs, c2)) {
						ruleScore = rule.getWeight();
						cinScore = chart.getInsideScore(rule.rhs, c2);
						outScore = chart.getOutsideScore(rule.lhs, c2);
						
						String key = rule.lhs == 0 ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
						char ruleType = rule.lhs == 0 ? GrammarRule.RHSPACE : GrammarRule.GENERAL;
						Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
						scores.put(String.valueOf(isample), null);
						scores.put(GrammarRule.Unit.P, outScore);
						scores.put(key, cinScore);
						grammar.addCount(rule.lhs, rule.rhs, ruleType, scores, false);
					}
				}
				
				// lexicons
				if (x0 == y1) {
					evalUnaryRuleCount(chart, c2, nword, isample, sentenceScore, sentence);
					continue; // not necessary
				}
				
				for (int split = left; split < left + ilayer; split++) {
					y0 = split;
					x1 = split + 1;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.oContainsKey(rule.lhs, c2) && 
							chart.iContainsKey(rule.lchild, c0) && 
							chart.iContainsKey(rule.rchild, c1)) {
							outScore = chart.getOutsideScore(rule.lhs, c2);
							linScore = chart.getInsideScore(rule.lchild, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
							scores.put(String.valueOf(isample), null);
							scores.put(GrammarRule.Unit.P, outScore);
							scores.put(GrammarRule.Unit.LC, linScore);
							scores.put(GrammarRule.Unit.RC, rinScore);
							grammar.addCount(rule.lhs, rule.lchild, rule.rchild, scores, false);
						}
					}
				}
			}
		}
	}
	
	
	/**
	 * Eval the rule counts with the parse tree known.
	 * 
	 * @param tree      the parse tree
	 * @param treeScore the score of the tree
	 */
	protected void evalRuleCountWithTree(Tree<State> tree, short isample, double treeScore) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			evalRuleCountWithTree(child, isample, treeScore);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		GaussianMixture outScore = parent.getOutsideScore();
		Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
		scores.put(GrammarRule.Unit.P, outScore);
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture cinScore = parent.getInsideScore();
			scores.put(GrammarRule.Unit.C, cinScore);
			lexicon.addCount(idParent, (short) word.wordIdx, GrammarRule.LHSPACE, scores, true);
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				GaussianMixture cinScore = child.getInsideScore();
				// root, if (idParent == 0) is true
				char type = idParent == 0 ? GrammarRule.RHSPACE : GrammarRule.GENERAL;
				String key = idParent == 0 ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
				scores.put(key, cinScore);
				grammar.addCount(idParent, idChild, type, scores, true);
				break;
			}
			case 2: {
				GaussianMixture linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				scores.put(GrammarRule.Unit.LC, linScore);
				scores.put(GrammarRule.Unit.RC, rinScore);
				grammar.addCount(idParent, idlChild, idrChild, scores, true);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	/**
	 * Compute the inside score, the recursive version.
	 * 
	 * @param chart    which stores the parse info
	 * @param sentence the sentence
	 * @param begin    left bound of the spanning range
	 * @param end      right bound of the spanning range
	 */
	protected void insideScore(Chart chart, List<State> sentence, int begin, int end) {
		int nword = sentence.size();
		
		if (begin == end) {
			int index = sentence.get(begin).wordIdx;
			int iCell = Chart.idx(begin, nword);
			Queue<Short> tagQueue = new LinkedList<Short>(); 
			List<GrammarRule> rules = lexicon.getRulesWithWord(index);
			for (GrammarRule rule : rules) {
				tagQueue.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true));
			}
			// unary grammar rules
			insideScoreForUnaryRule(chart, tagQueue, iCell, false);
		}
		
		int c2 = Chart.idx(begin, nword- (end - begin));
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int split = begin; split < end; split++) {
			
			int x0 = begin, y0 = split;
			int x1 = split + 1, y1 = end;
			int c0 = Chart.idx(x0, nword - (y0 - x0));
			int c1 = Chart.idx(x1, nword - (y1 - x1));
			
			if (!chart.iGetStatus(c0)) {
				insideScore(chart, sentence, begin, split);
			}
			
			if (!chart.iGetStatus(c1)) {
				insideScore(chart, sentence, split + 1, end);
			}
			
			GaussianMixture ruleScore, linScore, rinScore, pinScore;
			for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
				BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
				if (chart.iContainsKey(rule.lchild, c0) && chart.iContainsKey(rule.rchild, c1)) {
					ruleScore = rule.getWeight();
					linScore = chart.getInsideScore(rule.lchild, c0);
					rinScore = chart.getInsideScore(rule.rchild, c1);
					pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
					pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
					chart.addInsideScore(rule.lhs, c2, pinScore);
				}
			}
		}
		chart.iSetStatus(c2, true);
	}
	
	
	/**
	 * Compute the outside score, the recursive version.
	 * 
	 * @param chart    which stores the parse info
	 * @param sentence the sentence
	 * @param begin    left bound of the spanning range
	 * @param end      right bound of the spanning range
	 */
	protected void outsideScore(Chart chart, List<State> sentence, int begin, int end) {
		
	}
	
	
	private void evalUnaryRuleCount(Chart chart, int idx, int nword, short isample, double sentenceScore, List<State> sentence) {  		
		for (int i = 0; i < nword; i++) {
			int wordIdx = sentence.get(i).wordIdx;
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(wordIdx);
			// pre-terminals
			for (GrammarRule rule : rules) {
				GaussianMixture cinScore = chart.getInsideScore(rule.lhs, iCell);
				GaussianMixture outScore = chart.getOutsideScore(rule.lhs, iCell);
				Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
				scores.put(String.valueOf(isample), null);
				scores.put(GrammarRule.Unit.P, outScore);
				scores.put(GrammarRule.Unit.C, cinScore);
				lexicon.addCount(rule.lhs, (short) wordIdx, GrammarRule.LHSPACE, scores, false);				
			}
		}
	}
	
	
	private void insideScoreForUnaryRule(Chart chart, Queue<Short> tagQueue, int idx, boolean identifier) {
		if (identifier) {
			insideScoreForUnaryRule(chart, tagQueue, idx);
		} else {
			if (tagQueue != null) {
				short idTag;
				String ruleType;
				List<GrammarRule> rules;
				GaussianMixture pinScore, cinScore, ruleScore;
				while (!tagQueue.isEmpty()) {
					idTag = tagQueue.poll();
					rules = grammar.getChainSumUnaryRulesWithC(idTag);
					cinScore = chart.getInsideScore(idTag, idx);
					Iterator<GrammarRule> iterator = rules.iterator();
					while (iterator.hasNext()) {
						UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
						ruleScore = rule.getWeight();
						// Root->X is valid if idx == 0, otherwise we may skip such kind of rules
						if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; }
						// double check, in case the rule.lhs is the root node
						ruleType = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
						pinScore = ruleScore.mulForInsideOutside(cinScore, ruleType, true);
						chart.addInsideScore(rule.lhs, idx, pinScore);
					}
				}
			}
		}
	}
	
	
	private void outsideScoreForUnaryRule(Chart chart, Queue<Short> tagQueue, int idx, boolean identifier) {
		if (identifier) {
			outsideScoreForUnaryRule(chart, tagQueue, idx);
		} else {
			if (tagQueue != null) {
				short idTag;
				String ruleType;
				List<GrammarRule> rules;
				GaussianMixture poutScore, coutScore, ruleScore;
				while (!tagQueue.isEmpty()) {
					idTag = tagQueue.poll();
					rules = grammar.getChainSumUnaryRulesWithP(idTag);
					poutScore = chart.getOutsideScore(idTag, idx);
					Iterator<GrammarRule> iterator = rules.iterator();
					while (iterator.hasNext()) {
						UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
						ruleScore = rule.getWeight();
						// ROOT->X is valid if and only if idx == 0
						ruleType = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
						coutScore = ruleScore.mulForInsideOutside(poutScore, ruleType, true);
						chart.addOutsideScore(rule.rhs, idx, coutScore);
					}
				}
			}
		}
	}
	
	
	/**
	 * Compute the outside score for nonterminals of unary rules.
	 * 
	 * @param chart    which records the inside and outside scores 
	 * @param tagQueue which stores the right hand sides of candidate unary rules 
	 * @param idx      the real index of the cell
	 */
	private void outsideScoreForUnaryRule(Chart chart, Queue<Short> tagQueue, int idx) {
		if (tagQueue != null) {
			short idTag;
			String ruleType;
			List<GrammarRule> rules;
			GaussianMixture poutScore, coutScore, ruleScore;
			while (!tagQueue.isEmpty()) {
				idTag = tagQueue.poll();
				rules = grammar.getUnaryRuleWithP(idTag);
				poutScore = chart.getOutsideScore(idTag, idx);
				Iterator<GrammarRule> iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					tagQueue.offer(rule.rhs);
					ruleScore = rule.getWeight();
					ruleType = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
					coutScore = ruleScore.mulForInsideOutside(poutScore, ruleType, true);
					chart.addOutsideScore(rule.rhs, idx, coutScore);
				}
			}
		}
	}
	
	
	/**
	 * Compute the inside score for nonterminals of unary rules.
	 * 
	 * @param chart    which records the inside and outside scores 
	 * @param tagQueue which stores the right hand sides of candidate unary rules 
	 * @param idx      the real index of the cell
	 */
	private void insideScoreForUnaryRule(Chart chart, Queue<Short> tagQueue, int idx) {
		if (tagQueue != null) {
			short idTag;
			String ruleType;
			List<GrammarRule> rules;
			GaussianMixture pinScore, cinScore, ruleScore;
			// consider rules A->B->w and B->C->w. So p(w | B) = p(B->w) + p(B->C) * p(C->w).
			// TODO Could it be a dead loop? BUG.
			// DONE Grammar rules are checked before the training. See MethodUtil.checkUnaryRuleCircle().
			while (!tagQueue.isEmpty()) {
				idTag = tagQueue.poll();
				rules = grammar.getUnaryRuleWithC(idTag);
				cinScore = chart.getInsideScore(idTag, idx);
				Iterator<GrammarRule> iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					tagQueue.offer(rule.lhs);
					ruleScore = rule.getWeight();
					// in case when the rule.lhs is the root node
					ruleType = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
					pinScore = ruleScore.mulForInsideOutside(cinScore, ruleType, true);
					chart.addInsideScore(rule.lhs, idx, pinScore);
				}
			}
		}
	}
	
	
	/**
	 * @param outScore  outside score of the parent
	 * @param linScore  inside score of the left child
	 * @param rinScore  inside score of the right child
	 * @param ruleScore binary rule score
	 * @return
	 */
	private double computeBinaryRuleCount(
			GaussianMixture outScore, 
			GaussianMixture linScore, 
			GaussianMixture rinScore, 
			GaussianMixture ruleScore) {
		double part0 = outScore == null ? 1.0 : outScore.marginalize();
		double part1 = linScore == null ? 1.0 : linScore.marginalize();
		double part2 = rinScore == null ? 1.0 : rinScore.marginalize();
		double part3 = ruleScore == null ? 1.0 : ruleScore.marginalize();
		// LVeGLearner.logger.trace("+++[" + part0 + ", " + part1 + ", " + part2 + ", " + part3 + "]"); // DEBUG
		return part0 * part1 * part2 * part3;
	}
	
	
	/**
	 * @param outScore  outside score of the parent
	 * @param cinScore  inside score of the child
	 * @param ruleScore unary rule score
	 * @return
	 */
	private double computeUnaryRuleCount(
			GaussianMixture outScore, 
			GaussianMixture cinScore, 
			GaussianMixture ruleScore) {
		double part0 = outScore == null ? 1.0 : outScore.marginalize();
		double part1 = cinScore == null ? 1.0 : cinScore.marginalize();
		double part2 = ruleScore == null ? 1.0 : ruleScore.marginalize();
		// LVeGLearner.logger.trace("---[" + part0 + ", " + part1 + ", " + part2 + "]"); // DEBUG
		return part0 * part1 * part2;
	}
	
	
	/**
	 * Set the outside score of the root node to 1.
	 * 
	 * @param tree the parse tree
	 */
	protected void setRootOutsideScore(Tree<State> tree) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		tree.getLabel().setOutsideScore(gm);
	}
	
	
	/**
	 * @param chart 
	 */
	protected void setRootOutsideScore(Chart chart) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		chart.addOutsideScore((short) 0, Chart.idx(0, 1), gm);
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
		private static List<Cell> ochart = null;
		private static List<Cell> ichart = null;
		
		
		public Chart() {
			if (ichart == null || ichart.size() != LVeGLearner.maxlength) {
				initialize(LVeGLearner.maxlength);
			} else {
				clear();
			}
		}
		
		
		public Chart(int n) {
			initialize(n);
		}
		
		
		private static void initialize(int n) {
			clear(); // empty the old memory
			int size = n * (n + 1) / 2;
			ochart = new ArrayList<Cell>(size);
			ichart = new ArrayList<Cell>(size);
			for (int i = 0; i < size; i++) {
				ochart.add(new Cell());
				ichart.add(new Cell());
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
		 * @param ilayer layer in the pyramid, from top (1) to bottom (ilayer);
		 * 				  (n + n - ilayer + 1) * ilayer / 2 + i if ilayer ranges from bottom (0, 0)->0 to top (ilayer).
		 * @return
		 */
		public static int idx(int i, int ilayer) {
			return ilayer * (ilayer - 1) / 2 + i;
		}
		
		
		public static List<Cell> iGetChart() {
			return ichart;
		}
		
		
		public static List<Cell> oGetChart() {
			return ochart;
		}
		
		
		public Cell oGet(int idx) {
			return ochart.get(idx);
		}
		
		
		public Cell iGet(int idx) {
			return ichart.get(idx);
		}


		public boolean iContainsKey(short key, int idx) {
			return ichart.get(idx).containsKey(key);
		}
		
		
		public boolean oContainsKey(short key, int idx) {
			return ochart.get(idx).containsKey(key);
		}
		
		
		public void iSetStatus(int idx, boolean status) {
			ichart.get(idx).setStatus(status);
		}
		
		
		public boolean iGetStatus(int idx) {
			return ichart.get(idx).getStatus();
		}
		
		
		public void oSetStatus(int idx, boolean status) {
			ochart.get(idx).setStatus(status);
		}
		
		
		public boolean oGetStatus(int idx) {
			return ochart.get(idx).getStatus();
		}
		
		
		public void addInsideScore(short key, int idx, GaussianMixture gm) {
			ichart.get(idx).addScore(key, gm);
		}
		
		
		public GaussianMixture getInsideScore(short key, int idx) {
			return ichart.get(idx).getScore(key);
		}
		
		
		public void addOutsideScore(short key, int idx, GaussianMixture gm) {
			ochart.get(idx).addScore(key, gm);
		}
		
		
		public GaussianMixture getOutsideScore(short key, int idx) {
			return ochart.get(idx).getScore(key);
		}
		
		
		public static void clear() {
			if (ichart != null) {
				for (Cell cell : ichart) {
					if (cell != null) { cell.clear(); }
				}
				// ichart.clear();
			}
			if (ochart != null) {
				for (Cell cell : ochart) {
					if (cell != null) { cell.clear(); }
				}
				// ochart.clear();
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
		private Map<Short, GaussianMixture> scores;
		
		public Cell() {
			this.status = false;
			this.scores = new HashMap<Short, GaussianMixture>();
		}
		
		
		protected void setStatus(boolean status) {
			this.status = status;
		}
		
		
		protected boolean getStatus() {
			return status;
		}
		
		
		protected boolean containsKey(short key) {
			return scores.containsKey(key);
		}
		
		
		protected void addScore(short key, GaussianMixture gm) {
			if (containsKey(key)) {
				scores.get(key).add(gm);
			} else {
				scores.put(key, gm);
			}
		}
		
		
		protected GaussianMixture getScore(short key) {
			return scores.get(key);
		}
		
		
		protected void clear() {
			status = false;
			for (Map.Entry<Short, GaussianMixture> map : scores.entrySet()) {
				GaussianMixture gm = map.getValue();
				if (gm != null) { gm.clear(); }
			}
			scores.clear();
		}
		
		
		public String toString(boolean simple, int nfirst) {
			if (simple) {
				String name;
				StringBuffer sb = new StringBuffer();
				sb.append("Cell [status=" + status + ", size=" + scores.size());
				for (Map.Entry<Short, GaussianMixture> score : scores.entrySet()) {
					name = (String) Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET).object(score.getKey());
					sb.append(", " + name + "=" + score.getValue().toString(simple, nfirst));
				}
				sb.append("]");
				return sb.toString();
			} else {
				return toString();
			}
		}
		
		
		@Override
		public String toString() {
			String name;
			StringBuffer sb = new StringBuffer();
			sb.append("Cell [status=" + status + ", size=" + scores.size());
			for (Map.Entry<Short, GaussianMixture> score : scores.entrySet()) {
				name = (String) Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET).object(score.getKey());
				sb.append(", " + name + "=" + score.getValue());
			}
			sb.append("]");
			return sb.toString();
		}
	}
}
