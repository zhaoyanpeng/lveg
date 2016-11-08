package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import edu.berkeley.nlp.syntax.Tree;
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
public class Inferencer {
	
	private LVeGLexicon lexicon;
	private LVeGGrammar grammar;
	
	
	public Inferencer(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.grammar = grammar;
		this.lexicon = lexicon;
	}
	
	
	/**
	 * Compute the inside score given the sentence and grammar rules.
	 * 
	 * @param tree in which only the sentence is used.
	 * @return
	 */
	protected void insideScore(Chart chart, List<State> sentence, int nword, boolean recursive) {
		int x0, y0, x1, y1, l0, l1, l2;
		Queue<Short> tagQueue = new LinkedList<Short>(); 
		GaussianMixture pinScore, linScore, rinScore, ruleScore;
		
		// base case
		for (int i = 0; i < nword; i++) {
			int wordIdx = sentence.get(i).wordIdx;
			List<GrammarRule> rules = lexicon.getRulesWithWord(wordIdx);
			// pre-terminals
			for (GrammarRule rule : rules) {
				tagQueue.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, i, nword, rule.getWeight().copy(true));
			}
			// unary grammar rules
			insideScoreForUnaryRule(chart, tagQueue, i, nword);
		}		
		
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		// inside score
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				for (int right = left; right < left + ilayer; right++) {				
					x0 = left;
					y0 = right;
					x1 = right + 1;
					y1 = left + ilayer;
					l0 = nword - (y0 - x0);
					l1 = nword - (y1 - x1);
					l2 = nword - ilayer;
					// to ensure...
					tagQueue.clear();
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lchild, x0, l0) && chart.containsKey(rule.rchild, x1, l1)) {
							tagQueue.offer(rule.lhs);
							ruleScore = rule.getWeight();
							linScore = chart.getInsideScore(rule.lchild, x0, l0);
							rinScore = chart.getInsideScore(rule.rchild, x1, l1);
							
							pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
							pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addInsideScore(rule.lhs, left, l2, pinScore);
						}
					}
					// unary grammar rules
					insideScoreForUnaryRule(chart, tagQueue, l2, nword);
				}
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
	 * Compute the outside score given the sentence and grammar rules.
	 * 
	 * @param tree in which only the sentence is used.
	 */
	protected void outsideScore(Chart chart, List<State> sentence, int nword, boolean recursive) {
		
		int x0, y0, x1, y1, l0, l1, l2;
		GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
		
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				
				// unary grammar rules
				outsideScoreForUnaryRule(chart, left, nword - ilayer);
				
				// left child
				for (int right = left + ilayer + 1; right < nword; right++) {
					x0 = left; 
					y0 = right;
					x1 = left + ilayer + 1; 
					y1 = right;
					l0 = nword - (y0 - x0); 
					l1 = nword - (y1 - x1);
					l2 = nword - ilayer;

					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lhs, x0, l0) && chart.containsKey(rule.rchild, x1, l1)) {
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, x0, l0);
							rinScore = chart.getInsideScore(rule.rchild, x1, l1);
							
							loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addInsideScore(rule.lhs, left, l2, loutScore);
						}
					}
				}
				
				// right child
				for (int right = 0; right < left; right++) {
					x0 = right; 
					y0 = left + ilayer;
					x1 = right;
					y1 = left - 1;
					l0 = nword - (y0 - x0); 
					l1 = nword - (y1 - x1);
					l2 = nword - ilayer;
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lhs, x0, l0) && chart.containsKey(rule.lchild, x1, l1)) {
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, x0, l0);
							linScore = chart.getInsideScore(rule.lchild, x1, l1);
							
							routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
							chart.addInsideScore(rule.lhs, left, l2, routScore);
						}
					}
				}
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
				
				child.setOutsideScore(coutScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, loutScore, routScore;
				GaussianMixture linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(0).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
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
	
	
	protected void evalRuleCount(Chart chart, List<State> sentence, int nword, double sentenceScore) {
		double count = 0.0;
		GaussianMixture outScore, cinScore, linScore, rinScore, ruleScore;
		
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		// base case
		for (int i = 0; i < nword; i++) {
			int wordIdx = sentence.get(i).wordIdx;
			List<GrammarRule> rules = lexicon.getRulesWithWord(wordIdx);
			// pre-terminals
			for (GrammarRule rule : rules) {
				cinScore = chart.getInsideScore(rule.lhs, i, nword);
				outScore = chart.getOutsideScore(rule.lhs, i, nword);
				count = computeUnaryRuleCount(outScore, cinScore, null) / sentenceScore;
				lexicon.addCount(rule.lhs, (short) wordIdx, GrammarRule.LHSPACE, count, false);
			}
			
			// consider rules A->B->w and B->C->w. So p(w | B) = p(B->w) + p(B->C) * p(C->w).
			// TODO Could it be a dead loop? BUG.
			// unary grammar rules
			for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) rmap.getValue();
				if (chart.containsKey(rule.lhs, i, nword) && chart.containsKey(rule.rhs, i, nword)) {
					outScore = chart.getOutsideScore(rule.lhs, i, nword);
					cinScore = chart.getInsideScore(rule.rhs, i, nword);
					ruleScore = rule.getWeight();
					count = computeUnaryRuleCount(outScore, cinScore, ruleScore) / sentenceScore;
					grammar.addCount(rule.lhs, rule.rhs, GrammarRule.GENERAL, count, false);
				}
			}
		}		
		
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				for (int right = left; right < left + ilayer; right++) {				
					int x0 = left, y0 = right;
					int x1 = right + 1, y1 = left + ilayer;
					int l0 = nword - (y0 - x0), l1 = nword - (y1 - x1), l2 = nword - ilayer;

					// unary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
						UnaryGrammarRule rule = (UnaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lhs, left, l2) && chart.containsKey(rule.rhs, left, l2)) {
							outScore = chart.getOutsideScore(rule.lhs, left, l2);
							cinScore = chart.getInsideScore(rule.rhs, left, l2);
							ruleScore = rule.getWeight();
							count = computeUnaryRuleCount(outScore, cinScore, ruleScore) / sentenceScore;
							grammar.addCount(rule.lhs, rule.rhs, GrammarRule.GENERAL, count, false);
						}
					}
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lchild, x0, l0) && chart.containsKey(rule.rchild, x1, l1)) {
							outScore = chart.getOutsideScore(rule.lhs, left, l2);
							linScore = chart.getInsideScore(rule.lchild, x0, l0);
							rinScore = chart.getInsideScore(rule.rchild, x1, l1);
							ruleScore = rule.getWeight();
							
							count = computeBinaryRuleCount(outScore, linScore, rinScore, ruleScore) / sentenceScore;
							grammar.addCount(rule.lhs, rule.lchild, rule.rchild, count, false);
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
	protected void evalRuleCountWithTree(Tree<State> tree, double treeScore) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			evalRuleCountWithTree(child, treeScore);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		GaussianMixture outScore = parent.getOutsideScore();
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture cinScore = parent.getInsideScore();
			double count = computeUnaryRuleCount(outScore, cinScore, null) / treeScore;
			lexicon.addCount(idParent, (short) word.wordIdx, GrammarRule.LHSPACE, count, true);
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				GaussianMixture ruleScore, cinScore;
				State child = children.get(0).getLabel();
				cinScore = child.getInsideScore();
				short idChild = child.getId();
				
				// root, if (idParent == 0) is true
				char type = idParent == 0 ? GrammarRule.RHSPACE : GrammarRule.GENERAL;
				ruleScore = grammar.getUnaryRuleScore(idParent, idChild, type);
				
				double count = computeUnaryRuleCount(outScore, cinScore, ruleScore) / treeScore;
				grammar.addCount(idParent, idChild, type, count, true);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
				double count = computeBinaryRuleCount(outScore, linScore, rinScore, ruleScore) / treeScore;
				grammar.addCount(idParent, idlChild, idrChild, count, true);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	
	private void evalRuleCountForUnaryRule() {
		
	}
	
	
	private void outsideScoreForUnaryRule(Chart chart, int i, int ilayer) {
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
			UnaryGrammarRule rule = (UnaryGrammarRule) rmap.getValue();
			
		}
	}
	
	
	/**
	 * @param chart    which records the inside and outside scores 
	 * @param tagQueue which stores the right hand sides of candidate unary rules 
	 * @param i        index of the row
	 * @param ilayer   layer in the pyramid, from top (1) to bottom (ilayer);
	 */
	private void insideScoreForUnaryRule(Chart chart, Queue<Short> tagQueue, int i, int ilayer) {
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
				Iterator<GrammarRule> iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					tagQueue.offer(rule.lhs);
					ruleScore = rule.getWeight();
					cinScore = chart.getInsideScore(rule.rhs, i, ilayer);
					// in case when the rule.lhs is the root node
					ruleType = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
					pinScore = ruleScore.mulForInsideOutside(cinScore, ruleType, true);
					chart.addInsideScore(rule.lhs, i, ilayer, pinScore);
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
		return part0 * part1 * part2;
	}
	
	
	/**
	 * Set the outside score of the root node to 1.
	 * 
	 * @param tree the parse tree
	 */
	protected void setRootOutsideScore(Tree<State> tree) {
		GaussianMixture gm = tree.getLabel().getOutsideScore();
		gm.marginalizeToOne();
	}
	
	
	/**
	 * @param chart 
	 */
	protected void setRootOutsideScore(Chart chart) {
		GaussianMixture gm = chart.getOutsideScore((short) 0, (short) 0, (short) 1);
		gm.marginalizeToOne();
	}
	
	
	/**
	 * Manage cells in the chart.
	 * </p>
	 * TODO If we know the maximum length of the sentence, we can pre-allocate 
	 * the memory space and reuse it. And static type is prefered.
	 * </p>
	 * TODO Why do I prefer list?
	 * 
	 * @author Yanpeng Zhao
	 *
	 */
	protected static class Chart {
		private static int nword = 0;
		private static List<Cell> chart = null;
		
		public Chart() {
			if (chart == null && chart.size() != LVeGLearner.maxlength) {
				initialize(LVeGLearner.maxlength);
			} else {
				clear();
			}
		}
		
		
		public Chart(int n) {
			initialize(n);
		}
		
		
		public static void initialize(int n) {
			
			clear(); // empty the old memory
			nword = n;
			int size = n * (n + 1) / 2;
			chart = new ArrayList<Cell>(size);
			
			for (int i = 0; i < size; i++) {
				chart.add(new Cell());
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
		 * @param n      # of rows (words)
		 * @return
		 */
		public int getIdx(int i, int ilayer) {
			return ilayer * (ilayer - 1) / 2 + i;
		}
		
		
		public boolean containsKey(
				short key, int i, int ilayer) {
			int idx = getIdx(i, ilayer);
			return chart.get(idx).containsKey(key);
		}
		
		
		public void setStatus(int i, int ilayer, boolean status) {
			int idx = getIdx(i, ilayer);
			chart.get(idx).status = status;
		}
		
		
		public boolean getStatus(int i, int ilayer) {
			int idx = getIdx(i, ilayer);
			return chart.get(idx).status;
		}
		
		
		public void addInsideScore(
				short key, int i, int ilayer, GaussianMixture gm) {
			int idx = getIdx(i, ilayer);
			chart.get(idx).addInsideScore(key, gm);
		}
		
		
		public GaussianMixture getInsideScore(
				short key, int i, int ilayer) {
			int idx = getIdx(i, ilayer);
			return chart.get(idx).getInsideScore(key);
		}
		
		
		public void addOutsideScore(
				short key, int i, int ilayer, GaussianMixture gm) {
			int idx = getIdx(i, ilayer);
			chart.get(idx).addOutsideScore(key, gm);
		}
		
		
		public GaussianMixture getOutsideScore(
				short key, int i, int ilayer) {
			int idx = getIdx(i, ilayer);
			return chart.get(idx).getOutsideScore(key);
		}
		
		
		public void resetStatus() {
			for (Cell cell : chart) {
				if (cell != null) { cell.setStatus(false); }
			}
		}
		
		
		public static void clear() {
			if (chart == null) { return; }
			for (Cell cell : chart) {
				if (cell != null) { cell.clear(); }
			}
			// chart.clear();
		}
	}
	
	
	/**
	 * Cells of the chart used in calculating inside and outside scores.
	 * 
	 * @author Yanpeng Zhao
	 *
	 */
	private static class Cell {
		// key word "private" does not make any difference, outer class can access all the fields 
		// of the inner class through the instance of the inner class or in the way of the static 
		// fields accessing.
		private boolean status;
		private Map<Short, List<GaussianMixture>> scores;
		
		public Cell() {
			this.status = false;
			this.scores = new HashMap<Short, List<GaussianMixture>>();
		}

		
		protected boolean isEmpty() {
			return scores.isEmpty();
		}
		
		
		protected void setStatus(boolean status) {
			this.status = status;
		}
		
		
		protected boolean containsKey(short key) {
			return scores.containsKey(key);
		}
		
		
		protected void addInsideScore(short key, GaussianMixture gm) {
			if (containsKey(key)) {
				List<GaussianMixture> gms = scores.get(key);
				gms.get(0).add(gm);
			} else {
				List<GaussianMixture> gms = new ArrayList<GaussianMixture>(2);
				gms.add(gm);
				gms.add(null);
				scores.put(key, gms);
			}
		}
		
		
		protected GaussianMixture getInsideScore(short key) {
			if (containsKey(key)) {
				return scores.get(key).get(0);
			} else {
				return null;
			}
		}
		
		
		protected void addOutsideScore(short key, GaussianMixture gm) {
			if (containsKey(key)) {
				List<GaussianMixture> gms = scores.get(key);
				gms.get(1).add(gm);
			} else {
				List<GaussianMixture> gms = new ArrayList<GaussianMixture>(2);
				gms.add(null);
				gms.add(gm);
				scores.put(key, gms);
			}
		}
		
		
		protected GaussianMixture getOutsideScore(short key) {
			if (containsKey(key)) {
				return scores.get(key).get(1);
			} else {
				return null;
			}
		}
		
		
		protected void clear() {
			for (Map.Entry<Short, List<GaussianMixture>> map : scores.entrySet()) {
				List<GaussianMixture> gms = map.getValue();
				for (GaussianMixture gm : gms) {
					if (gm != null) { gm.clear(); }
				}
			}
			scores.clear();
		}
	}
	
	
	/**
	 * Compute the inside score. Recursive version.
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
			
			Queue<Short> newTags = new LinkedList<Short>(); 
			GaussianMixture inside, prule, linside, rinside;
			
			List<GrammarRule> rules = lexicon.getRulesWithWord(index);
			for (GrammarRule rule : rules) {
				newTags.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, begin, nword, rule.getWeight().copy(true));
			}
			// consider rules A->B->w and B->C->w. So p(w | B) = p(B->w) + p(B->C) * p(C->w).
			// TODO Could it be a dead loop? BUG.
			while (!newTags.isEmpty()) {
				index = newTags.poll();
				rules = grammar.getUnaryRuleWithC(index);
				
				Iterator<GrammarRule> iterator= rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					newTags.offer(rule.lhs);		
					prule = rule.getWeight();
					inside = chart.getInsideScore(rule.rhs, begin, nword);
					prule.mulForInsideOutside(inside, GrammarRule.Unit.UC, true);
					chart.addInsideScore(rule.lhs, begin, nword, prule);
				}
			}
		}
		
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int split = begin; split < end; split++) {
			
			int x0 = begin, y0 = split;
			int x1 = split + 1, y1 = end;
			int l0 = nword - (y0 - x0), l1 = nword - (y1 - x1);
			if (!chart.getStatus(x0,  l0)) {
				insideScore(chart, sentence, begin, split);
			}
			if (!chart.getStatus(x1, l1)) {
				insideScore(chart, sentence, split + 1, end);
			}
			
			GaussianMixture inside, prule, linside, rinside;
			for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
				BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
				if (chart.containsKey(rule.lchild, x0, l0) && chart.containsKey(rule.rchild, x1, l1)) {
					prule = rule.getWeight();
					linside = chart.getInsideScore(rule.lchild, x0, l0);
					rinside = chart.getInsideScore(rule.rchild, x1, l1);
					
					inside = prule.mulForInsideOutside(linside, GrammarRule.Unit.LC, true);
					inside = prule.mulForInsideOutside(rinside, GrammarRule.Unit.RC, false);
					chart.addInsideScore(rule.lhs, begin, nword - (end - begin), inside);
				}
			}
		}
		chart.setStatus(begin, nword - (end - begin), true);
	}
	
}
