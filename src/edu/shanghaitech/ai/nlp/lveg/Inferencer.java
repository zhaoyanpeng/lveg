package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
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
	 * Compute the inside score.
	 * 
	 * @param tree in which only the sentence is used.
	 * @return
	 */
	protected void insideScore(Chart chart, List<State> sentence, int nword, boolean recursive) {
		
		Queue<Short> newTags = new LinkedList<Short>(); 
		GaussianMixture inside, prule, linside, rinside;
		
		// base case
		for (int i = 0; i < nword; i++) {
			int index = sentence.get(i).wordIdx;
			List<UnaryGrammarRule> rules = lexicon.getRules(index);
			for (UnaryGrammarRule rule : rules) {
				newTags.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, i, nword, rule.getWeight().copy(true));
			}
			// consider rules A->B->w and B->C->w. So p(w | B) = p(B->w) + p(B->C) * p(C->w).
			// TODO Could it be a dead loop? BUG.
			while (!newTags.isEmpty()) {
				index = newTags.poll();
				rules = grammar.getUnaryRuleWithC(index);
				for (UnaryGrammarRule rule : rules) {
					newTags.offer(rule.lhs);		
					prule = rule.getWeight();
					inside = chart.getInsideScore(rule.rhs, i, nword);
					prule.mulForInsideOutside(inside, GrammarRule.Unit.UC, true);
					chart.addInsideScore(rule.lhs, i, nword, prule);
				}
			}
		}		
		
		Map<UnaryGrammarRule, UnaryGrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		Map<BinaryGrammarRule, BinaryGrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		// inside score
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				for (int right = left; right < left + ilayer; right++) {				
					int x0 = left, y0 = right;
					int x1 = right + 1, y1 = left + ilayer;
					int l0 = nword - (y0 - x0), l1 = nword - (y1 - x1);

					// unary grammar rules
					for (Map.Entry<UnaryGrammarRule, UnaryGrammarRule> rmap : uRuleMap.entrySet()) {
						
					}
					
					// binary grammar rules
					for (Map.Entry<BinaryGrammarRule, BinaryGrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = rmap.getValue();
						if (chart.containsKey(rule.lchild, x0, l0) && chart.containsKey(rule.rchild, x1, l1)) {
							prule = rule.getWeight();
							linside = chart.getInsideScore(rule.lchild, x0, l0);
							rinside = chart.getInsideScore(rule.rchild, x1, l1);
							
							prule = prule.mulForInsideOutside(linside, GrammarRule.Unit.LC, true);
							prule = prule.mulForInsideOutside(rinside, GrammarRule.Unit.RC, false);
							chart.addInsideScore(rule.lhs, left, nword - ilayer, prule);
						}
					}
				}
			}
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
			
			List<UnaryGrammarRule> rules = lexicon.getRules(index);
			for (UnaryGrammarRule rule : rules) {
				newTags.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, begin, nword, rule.getWeight().copy(true));
			}
			// consider rules A->B->w and B->C->w. So p(w | B) = p(B->w) + p(B->C) * p(C->w).
			// TODO Could it be a dead loop? BUG.
			while (!newTags.isEmpty()) {
				index = newTags.poll();
				rules = grammar.getUnaryRuleWithC(index);
				for (UnaryGrammarRule rule : rules) {
					newTags.offer(rule.lhs);		
					prule = rule.getWeight();
					inside = chart.getInsideScore(rule.rhs, begin, nword);
					prule.mulForInsideOutside(inside, GrammarRule.Unit.UC, true);
					chart.addInsideScore(rule.lhs, begin, nword, prule);
				}
			}
		}
		
		Map<UnaryGrammarRule, UnaryGrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		Map<BinaryGrammarRule, BinaryGrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
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
			for (Map.Entry<BinaryGrammarRule, BinaryGrammarRule> rmap : bRuleMap.entrySet()) {
				BinaryGrammarRule rule = rmap.getValue();
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
			GaussianMixture inScore = lexicon.score(word, idParent);
			
			parent.setInsideScore(inScore.copy(true));
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				GaussianMixture ruleScore, inScore, childInScore;
				State child = children.get(0).getLabel();
				childInScore = child.getInsideScore();
				short idChild = child.getId();
				
				if (idParent != 0) {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.GENERAL);
					inScore = ruleScore.mulForInsideOutside(childInScore, GrammarRule.Unit.UC, true);
				} else { // root, inside score of the root node is a constant in double
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
					inScore = ruleScore.mulForInsideOutside(childInScore, GrammarRule.Unit.C, true);
				}
				
				parent.setInsideScore(inScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, inScore, lchildInScore, rchildInScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				lchildInScore = lchild.getInsideScore();
				rchildInScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
				inScore = ruleScore.mulForInsideOutside(lchildInScore, GrammarRule.Unit.LC, true);
				inScore = inScore.mulForInsideOutside(rchildInScore, GrammarRule.Unit.RC, false);
				
				parent.setInsideScore(inScore);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	/**
	 * Compute the outside score.
	 * 
	 * @param tree in which only the sentence is used.
	 */
	protected void outsideScore(Chart chart, List<State> sentence, int nword, boolean recursive) {
		
		int x0, y0, x1, y1, l0, l1;
		GaussianMixture outside, prule, linside, rinside;
		
		Map<UnaryGrammarRule, UnaryGrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		Map<BinaryGrammarRule, BinaryGrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				
				for (int right = left + ilayer + 1; right < nword; right++) {
					x0 = left; 
					y0 = right;
					x1 = left + ilayer + 1; 
					y1 = right;
					l0 = nword - (y0 - x0); 
					l1 = nword - (y1 - x1);
					
					for (Map.Entry<BinaryGrammarRule, BinaryGrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = rmap.getValue();
						if (chart.containsKey(rule.lhs, x0, l0) && chart.containsKey(rule.rchild, x1, l1)) {
							prule = rule.getWeight();
							outside = chart.getOutsideScore(rule.lhs, x0, l0);
							rinside = chart.getInsideScore(rule.rchild, x1, l1);
							
							prule = prule.mulForInsideOutside(outside, GrammarRule.Unit.P, true);
							prule = prule.mulForInsideOutside(rinside, GrammarRule.Unit.RC, false);
							chart.addInsideScore(rule.lhs, left, nword - ilayer, prule);
						}
					}
				}
				
				for (int right = 0; right < left; right++) {
					x0 = right; 
					y0 = left + ilayer;
					x1 = right; 
					y1 = left - 1;
					l0 = nword - (y0 - x0); 
					l1 = nword - (y1 - x1);
					
					for (Map.Entry<BinaryGrammarRule, BinaryGrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = rmap.getValue();
						if (chart.containsKey(rule.lhs, x0, l0) && chart.containsKey(rule.lchild, x1, l1)) {
							prule = rule.getWeight();
							outside = chart.getOutsideScore(rule.lhs, x0, l0);
							linside = chart.getInsideScore(rule.lchild, x1, l1);
							
							prule = prule.mulForInsideOutside(outside, GrammarRule.Unit.P, true);
							prule = prule.mulForInsideOutside(linside, GrammarRule.Unit.LC, false);
							chart.addInsideScore(rule.lhs, left, nword - ilayer, prule);
						}
					}
				}
			}
		}
	}
	
	
	/**
	 * @param recursive recursive version
	 * @return
	 */
	protected void outsideScore(boolean recursive) {
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
			GaussianMixture parentOutScore = parent.getOutsideScore();
			
			switch (children.size()) {
			case 0:
				break;
			case 1: {
				GaussianMixture ruleScore, outScore;
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				if (idParent != 0) {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.GENERAL);
				} else { // root
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
				}
				// rule: p(root->nonterminal) does not contain "P" part, so no removing occurs when
				// the current parent node is the root node
				outScore = ruleScore.mulForInsideOutside(parentOutScore, GrammarRule.Unit.P, true);
				
				child.setOutsideScore(outScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, loutScore, routScore;
				GaussianMixture lchildInScore, rchildInScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(0).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				lchildInScore = lchild.getInsideScore();
				rchildInScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
				loutScore = ruleScore.mulForInsideOutside(parentOutScore, GrammarRule.Unit.P, true);
				loutScore = loutScore.mulForInsideOutside(rchildInScore, GrammarRule.Unit.RC, false);
				
				routScore = ruleScore.mulForInsideOutside(parentOutScore, GrammarRule.Unit.P, true);
				routScore = routScore.mulForInsideOutside(lchildInScore, GrammarRule.Unit.LC, false);
				
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
	
	
	/**
	 * Set the outside score of the root node to 1.
	 * 
	 * @param tree the parse tree
	 */
	protected void setRootOutsideScore(Tree<State> tree) {
		GaussianMixture gm = tree.getLabel().getOutsideScore();
		double weight = 1.0 / gm.ncomponent;
		for (int i = 0; i < gm.ncomponent; i++) {
			gm.weights.set(i, weight);
			gm.mixture.get(i).clear();
		}
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
		 * Imagine the upper right triangle chart as the pyramid.
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
	
}
