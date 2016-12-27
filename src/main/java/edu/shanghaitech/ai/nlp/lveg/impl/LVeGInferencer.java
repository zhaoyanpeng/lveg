package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.syntax.State;

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
public class LVeGInferencer extends Inferencer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3716227615216124859L;


	public LVeGInferencer(LVeGGrammar agrammar, LVeGLexicon alexicon) {
		grammar = agrammar;
		lexicon = alexicon;
		chainurule = ChainUrule.DEFAULT;
	}
	
	
	/**
	 * Compute the inside score with the parse tree known.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	public static void insideScoreWithTree(Tree<State> tree) {
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
					ruleScore = grammar.getURuleWeight(idParent, idChild, GrammarRule.LRURULE);
					pinScore = ruleScore.mulForInsideOutside(cinScore, GrammarRule.Unit.UC, true);
				} else { // root, inside score of the root node is a constant in double
					ruleScore = grammar.getURuleWeight(idParent, idChild, GrammarRule.RHSPACE);
					pinScore = ruleScore.mulForInsideOutside(cinScore, GrammarRule.Unit.C, true);
				}
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
				ruleScore = grammar.getBRuleWeight(idParent, idlChild, idrChild);
				
				pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
				pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
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
	protected static void outsideScoreWithTree(Tree<State> tree) {
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
					ruleScore = grammar.getURuleWeight(idParent, idChild, GrammarRule.LRURULE);
				} else { // root
					ruleScore = grammar.getURuleWeight(idParent, idChild, GrammarRule.RHSPACE);
				}
				// rule: p(root->nonterminal) does not contain "P" part, so no removing occurs when
				// the current parent node is the root node
				coutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
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
				ruleScore = grammar.getBRuleWeight(idParent, idlChild, idrChild);
				
				loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
				
				routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
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
	
	
	protected void evalRuleCount(Tree<State> tree, Chart chart, short isample) {
		List<State> sentence = tree.getYield();
		int x0, x1, y0, y1, c0, c1, c2, nword = sentence.size();
		GaussianMixture outScore, linScore, rinScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				
				// general unary grammar rules
				evalUnaryRuleCount(chart, c2, isample, null);
				
				// unary grammar rules containing lexicons
				if (x0 == y1) {
					evalUnaryRuleCount(chart, c2, isample, sentence.get(x0));
					continue; // not necessary, think about it...
				}
				
				for (int split = left; split < left + ilayer; split++) {
					y0 = split;
					x1 = split + 1;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lhs, c2, false) && 
							chart.containsKey(rule.lchild, c0, true) && 
							chart.containsKey(rule.rchild, c1, true)) {
							outScore = chart.getOutsideScore(rule.lhs, c2);
							linScore = chart.getInsideScore(rule.lchild, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
							scores.put(GrammarRule.Unit.P, outScore);
							scores.put(GrammarRule.Unit.LC, linScore);
							scores.put(GrammarRule.Unit.RC, rinScore);
							grammar.addCount(rule.lhs, rule.lchild, rule.rchild, scores, isample, false);
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
	protected void evalRuleCountWithTree(Tree<State> tree, short isample) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			evalRuleCountWithTree(child, isample);
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
			lexicon.addCount(idParent, word.wordIdx, scores, GrammarRule.LHSPACE, isample, true);
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
				String key = idParent == 0 ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
				byte type = idParent == 0 ? GrammarRule.RHSPACE : GrammarRule.LRURULE;
				scores.put(key, cinScore);
				grammar.addCount(idParent, idChild, scores, type, isample, true);
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
				grammar.addCount(idParent, idlChild, idrChild, scores, isample, true);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	private void evalUnaryRuleCount(Chart chart, int idx, short isample, State word) {
		Set<Short> set;
		List<GrammarRule> rules;
		GaussianMixture outScore, cinScore;
		// have to process ROOT node specifically
		if (idx == 0 && (set = chart.keySet(idx, false, (short) (LENGTH_UCHAIN + 1))) != null) {
			for (Short idTag : set) { // can only contain ROOT
				rules = grammar.getURuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator(); // see set ROOT's outside score
				outScore = chart.getOutsideScore(idTag, idx, (short) (LENGTH_UCHAIN + 1)); // 1
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (!chart.containsKey((short) rule.rhs, idx, true)) { continue; }
					cinScore = chart.getInsideScore((short) rule.rhs, idx); // CHECK not correct? Yes, it's correct.
					Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
					scores.put(GrammarRule.Unit.P, outScore);
					scores.put(GrammarRule.Unit.C, cinScore);
					grammar.addCount(rule.lhs, rule.rhs, scores, GrammarRule.RHSPACE, isample, false);
				}
			}
		}
		// general unary grammar rules
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getURuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
			UnaryGrammarRule rule = (UnaryGrammarRule) rmap.getValue();
			if (rule.type == GrammarRule.RHSPACE) { continue; }
			if (!chart.containsKey(rule.lhs, idx, false) || !chart.containsKey((short) rule.rhs, idx, true)) { continue; }
			Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
			mergeUnaryRuleCount(chart, idx, rule, scores, (short) 0, (short) (0)); // O_{0}(X) I_{0}(Y)
			mergeUnaryRuleCount(chart, idx, rule, scores, (short) 0, (short) (1)); // O_{0}(X) I_{1}(Y)
			mergeUnaryRuleCount(chart, idx, rule, scores, (short) 1, (short) (0)); // O_{1}(X) I_{0}(Y)
			if (!scores.isEmpty()) { grammar.addCount(rule.lhs, rule.rhs, scores, GrammarRule.LRURULE, isample, false); }
		}
		// have to process unary rules containing LEXICONS specifically
		if (word != null) {
			rules = lexicon.getRulesWithWord(word);
			for (GrammarRule rule : rules) {
//				if (chart.containsKey(rule.lhs, idx, true, (short) 0) && chart.containsKey(rule.lhs, idx, false)) {
//					cinScore = chart.getInsideScore(rule.lhs, idx, (short) 0);
				if (chart.containsKey(rule.lhs, idx, false)) {
					Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
					cinScore = lexicon.score(word, rule.lhs);
					outScore = chart.getOutsideScore(rule.lhs, idx);
					scores.put(GrammarRule.Unit.P, outScore);
					scores.put(GrammarRule.Unit.C, cinScore);
					lexicon.addCount(rule.lhs, word.wordIdx, scores, GrammarRule.LHSPACE, isample, false);
				}
			}
		}
	}
	
	private void mergeUnaryRuleCount(
			Chart chart, int idx, UnaryGrammarRule rule, Map<String, GaussianMixture> scores, short olevel, short ilevel) {
		if (chart.containsKey(rule.lhs, idx, false, olevel) && chart.containsKey((short) rule.rhs, idx, true, ilevel)) {
			GaussianMixture cinScore = chart.getInsideScore((short) rule.rhs, idx, ilevel);
			GaussianMixture outScore = chart.getOutsideScore((short) rule.lhs, idx, olevel);
			if (scores.get(GrammarRule.Unit.P) != null) {
				scores.get(GrammarRule.Unit.P).add(outScore);
				scores.get(GrammarRule.Unit.UC).add(cinScore);
			} else { // new memory space
				scores.put(GrammarRule.Unit.P, outScore.copy(true));
				scores.put(GrammarRule.Unit.UC, cinScore.copy(true));
			}
		}
	}
	
	
	/**
	 * Set the outside score of the root node to 1.
	 * 
	 * @param tree the parse tree
	 */
	protected static void setRootOutsideScore(Tree<State> tree) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		tree.getLabel().setOutsideScore(gm);
	}
	
}
