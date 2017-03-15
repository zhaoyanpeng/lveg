package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.impl.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Recorder;

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
	public static void insideScore(Chart chart, List<State> sentence, int nword, boolean prune) {
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture pinScore, linScore, rinScore, ruleScore;
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
			insideScoreForUnaryRule(chart, iCell, chainurule, prune);
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
				insideScoreForUnaryRule(chart, c2, chainurule, prune);
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
	public static void outsideScore(Chart chart, List<State> sentence, int nword, boolean prune) {
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
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
				for (int right = 0; right < left; right++) {
					x0 = right; 
					x1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
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
				outsideScoreForUnaryRule(chart, c2, chainurule, prune);
				if (prune) { chart.pruneOutsideScore(c2, (short) -1); }
//				long ttime = System.currentTimeMillis() - start;
//				logger.trace("\tafter chain unary\t" + chart.size(c2, false) + "\ttime: " + ttime / 1000 + "\n");
			}
		}
	}
	
	
	private static void outsideScoreForUnaryRule(Chart chart, int idx, ChainUrule identifier, boolean prune) {
		switch (identifier) {
		case DEFAULT: {
			outsideScoreForUnaryRuleDefault(chart, idx, prune);
			break;
		}
		default:
			logger.error("Invalid unary-rule-processing-method. ");
		}
	}
	
	
	private static void insideScoreForUnaryRule(Chart chart, int idx, ChainUrule identifier, boolean prune) {
		switch (identifier) {
		case DEFAULT: {
			insideScoreForUnaryRuleDefault(chart, idx, prune);
			break;
		}
		default:
			logger.error("Invalid unary-rule-processing-method. ");
		}
	}
	
	private static void outsideScoreForUnaryRuleDefault(Chart chart, int idx, boolean prune) {
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		String rmKey = GrammarRule.Unit.P;
		GaussianMixture poutScore, coutScore;
		// have to process ROOT node specifically
		if (idx == 0 && (set = chart.keySet(idx, false, (short) (LENGTH_UCHAIN + 1))) != null) {
			for (Short idTag : set) { // can only contain ROOT
				rules = grammar.getURuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator(); // see set ROOT's outside score
				poutScore = chart.getOutsideScore(idTag, idx, (short) (LENGTH_UCHAIN + 1)); // 1
				while (iterator.hasNext()) { // CHECK
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					coutScore = rule.weight.mulForInsideOutside(poutScore, rmKey, true);
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
					coutScore = rule.weight.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore((short) rule.rhs, idx, coutScore, (short) (level + 1), false);
				}
			}
			level++;
			if (prune) { chart.pruneOutsideScore(idx, level); }
		}
	}
	
	private static void insideScoreForUnaryRuleDefault(Chart chart, int idx, boolean prune) {
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
	
	/**
	 * @param chart 
	 */
	public static void setRootOutsideScore(Chart chart) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
//		GaussianMixture gm = DiagonalGaussianMixture.borrowObject((short) 1); // POOL
		gm.marginalizeToOne();
		chart.addOutsideScore((short) 0, Chart.idx(0, 1), gm, (short) (LENGTH_UCHAIN + 1), false);
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
		private List<Cell> mchart = null; // for max rule
		
		protected Chart() { // TODO make it static? See clear().
			if (ichart == null || ichart.size() != LVeGLearner.maxlength) {
				initialize(LVeGLearner.maxlength, false);
			} else {
				clear(-1);
			}
		}
		
		public Chart(int n, boolean maxrule) {
			initialize(n, maxrule);
		}
		
		private void initialize(int n, boolean maxrule) {
			clear(n); // empty the old memory
			int size = n * (n + 1) / 2;
			ochart = new ArrayList<Cell>(size);
			ichart = new ArrayList<Cell>(size);
			for (int i = 0; i < size; i++) {
				ochart.add(new Cell(maxrule));
				ichart.add(new Cell(maxrule));
			}
			if (maxrule) {
				mchart = new ArrayList<Cell>(size);
				for (int i = 0; i < size; i++) {
					mchart.add(new Cell(maxrule));
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
		
		public Set<Short> keySet(int idx, boolean inside, short level) {
			return inside ? ichart.get(idx).keySet(level) : ochart.get(idx).keySet(level);
		}
		
		public Set<Short> keySet(int idx, boolean inside) {
			return inside ? ichart.get(idx).keySet() : ochart.get(idx).keySet();
		}
		
		public boolean containsKey(short key, int idx, boolean inside, short level) {
			return inside ? ichart.get(idx).containsKey(key, level) : ochart.get(idx).containsKey(key, level);
		}
		
		public boolean containsKey(short key, int idx, boolean inside) {
			return inside ? ichart.get(idx).containsKey(key) : ochart.get(idx).containsKey(key);
		}
		
		public void addInsideScore(short key, int idx, GaussianMixture gm, short level, boolean prune) {
			ichart.get(idx).addScore(key, gm, level, prune);
		}
		
		public GaussianMixture getInsideScore(short key, int idx, short level) {
			return ichart.get(idx).getScore(key, level);
		}
		
		public GaussianMixture getInsideScore(short key, int idx) {
			return ichart.get(idx).getScore(key);
		}
		
		public void addOutsideScore(short key, int idx, GaussianMixture gm, short level, boolean prune) {
			ochart.get(idx).addScore(key, gm, level, prune);
		}
		
		public GaussianMixture getOutsideScore(short key, int idx, short level) {
			return ochart.get(idx).getScore(key, level);
		}
		
		public GaussianMixture getOutsideScore(short key, int idx) {
			return ochart.get(idx).getScore(key);
		}
		
		public void pruneOutsideScore(int idx, short level) {
			if (level < 0) {
				ochart.get(idx).pruneScore();
			} else {
				ochart.get(idx).pruneScore(level);
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
		
		
		private Cell() {
			this.status = false;
			this.totals = new HashMap<Short, GaussianMixture>();
			this.scores = new HashMap<Short, Map<Short, GaussianMixture>>();
		}
		
		public Cell(boolean maxrule) {
			this();
			if (maxrule) {
				this.maxRuleCnts = new HashMap<Short, Map<Short, Double>>(3, 1);
				this.maxRuleSons = new HashMap<Short, Map<Short, Integer>>(3, 1);
				this.maxRulePos = new HashMap<Short, Short>();
				this.maxRuleSon = new HashMap<Short, Integer>(); // low 2 bytes are used
				this.splitPoint = new HashMap<Short, Short>();
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
		
		protected Set<Short> keySet() {
			return totals.keySet();
		}
		
		protected Set<Short> keySet(short level) {
			return scores.get(level) == null ? null : scores.get(level).keySet();
		}
		
		protected Set<Short> keySetMaxRule(short level) {
			return maxRuleCnts.get(level) == null ? null : maxRuleCnts.get(level).keySet();
		}
		
		protected boolean containsKey(short key) {
			return totals.containsKey(key);
		}
		
		protected boolean containsKey(short key, short level) {
			return scores.get(level) == null ? false : scores.get(level).containsKey(key);
		}
		
		protected void pruneScore() {
			Collection<GaussianMixture> scores = totals.values();
			for (GaussianMixture score : scores) {
				score.delTrivia();
			}
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
		
		protected void addScore(short key, GaussianMixture gm, short level, boolean prune) {
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
		
		private void addScore(short key, GaussianMixture gm, boolean prune) {
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
		
		protected GaussianMixture getScore(short key, short level) {
			return scores.get(level) == null ? null : scores.get(level).get(key);
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
