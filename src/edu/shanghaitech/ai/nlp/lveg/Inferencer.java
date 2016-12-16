package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class Inferencer extends Recorder {
	
	protected LVeGLexicon lexicon;
	protected LVeGGrammar grammar;
	
	protected ChainUrule chainurule;
	
	protected final static short ROOT = 0;
	protected final static short LENGTH_UCHAIN = 2;
	
	enum ChainUrule {
		ALL_POSSIBLE_PATH, PRE_COMPUTE_CHAIN, NOT_PRE_ADD_INTER, NOT_PRE_NOT_INTER, DEFAULT,
	}
	
	
	/**
	 * Compute the inside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used
	 * @param nword # of words in the sentence
	 */
	protected void insideScore(Chart chart, List<State> sentence, int nword) {
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture pinScore, linScore, rinScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			int wordIdx = sentence.get(i).wordIdx; 
			List<GrammarRule> rules = lexicon.getRulesWithWord(wordIdx);
			// preterminals
			for (GrammarRule rule : rules) {
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true), (short) 0);
			}
			// DEBUG unary grammar rules
//			logger.trace("Cell [" + i + ", " + (i + 0) + "]="+ iCell + "\t is being estimated. # " );
//			long start = System.currentTimeMillis();
			insideScoreForUnaryRule(chart, iCell, chainurule);
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
							chart.addInsideScore(rule.lhs, c2, pinScore, (short) 0);
						}
					}
				}
				// DEBUG unary grammar rules
//				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # ");
//				long start = System.currentTimeMillis();
				insideScoreForUnaryRule(chart, c2, chainurule);
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
	protected void outsideScore(Chart chart, List<State> sentence, int nword) {
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
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
							chart.addOutsideScore(rule.lchild, c2, loutScore, (short) 0);
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
							chart.addOutsideScore(rule.rchild, c2, routScore, (short) 0);
						}
					}
				}
				// DEBUG unary grammar rules
//				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # ");
//				long start = System.currentTimeMillis();
				outsideScoreForUnaryRule(chart, c2, chainurule);
//				long ttime = System.currentTimeMillis() - start;
//				logger.trace("\tafter chain unary\t" + chart.size(c2, false) + "\ttime: " + ttime / 1000 + "\n");
			}
		}
	}
	
	private void outsideScoreForUnaryRule(Chart chart, int idx, ChainUrule identifier) {
		switch (identifier) {
		case DEFAULT: {
			outsideScoreForUnaryRuleDefault(chart, idx);
			break;
		}
		default:
			logger.error("Invalid unary-rule-processing-method. ");
		}
	}
	
	
	private void insideScoreForUnaryRule(Chart chart, int idx, ChainUrule identifier) {
		switch (identifier) {
		case DEFAULT: {
			insideScoreForUnaryRuleDefault(chart, idx);
			break;
		}
		default:
			logger.error("Invalid unary-rule-processing-method. ");
		}
	}
	
	private void outsideScoreForUnaryRuleDefault(Chart chart, int idx) {
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		String rmKey = GrammarRule.Unit.P;
		GaussianMixture poutScore, coutScore;
		// have to process ROOT node specifically
		if (idx == 0 && (set = chart.keySet(idx, false, (short) (LENGTH_UCHAIN + 1))) != null) {
			for (Short idTag : set) { // can only contain ROOT
				rules = grammar.getUnaryRuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator(); // see set ROOT's outside score
				poutScore = chart.getOutsideScore(idTag, idx, (short) (LENGTH_UCHAIN + 1)); // 1
				while (iterator.hasNext()) { // CHECK
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					coutScore = rule.weight.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore(rule.rhs, idx, coutScore, level);
				}
			}
		}
		while(level < LENGTH_UCHAIN && (set = chart.keySet(idx, false, level)) != null) {
			for (Short idTag : set) {
				rules = grammar.getUnaryRuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator();
				poutScore = chart.getOutsideScore(idTag, idx, level);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					coutScore = rule.weight.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore(rule.rhs, idx, coutScore, (short) (level + 1));
				}
			}
			level++;
		}
	}
	
	private void insideScoreForUnaryRuleDefault(Chart chart, int idx) {
		String rmKey;
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		GaussianMixture pinScore, cinScore;
		while (level < LENGTH_UCHAIN && (set = chart.keySet(idx, true, level)) != null) {
			for (Short idTag : set) {
				rules = grammar.getUnaryRuleWithC(idTag); // ROOT is excluded
				Iterator<GrammarRule> iterator = rules.iterator();
				cinScore = chart.getInsideScore(idTag, idx, level);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; } // ROOT is allowed
					rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
					pinScore = rule.weight.mulForInsideOutside(cinScore, rmKey, true);
					chart.addInsideScore(rule.lhs, idx, pinScore, (short) (level + 1));
				}
			}
			level++;
		}
		// have to process ROOT node specifically
		if (idx == 0 && (set = chart.keySet(idx, true, LENGTH_UCHAIN)) != null) {
			for (Short idTag : set) { // the maximum inside level below ROOT
				rules = grammar.getUnaryRuleWithC(idTag);
				Iterator<GrammarRule> iterator = rules.iterator();
				cinScore = chart.getInsideScore(idTag, idx, LENGTH_UCHAIN);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (rule.type != GrammarRule.RHSPACE) { continue; } 
					pinScore = rule.weight.mulForInsideOutside(cinScore, GrammarRule.Unit.C, true);
					chart.addInsideScore(rule.lhs, idx, pinScore, (short) (LENGTH_UCHAIN + 1));
				}
			}
		}
	}
	
	/**
	 * @param chart 
	 */
	protected void setRootOutsideScore(Chart chart) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		chart.addOutsideScore((short) 0, Chart.idx(0, 1), gm, (short) (LENGTH_UCHAIN + 1));
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
		
		public Chart() {
			if (ichart == null || ichart.size() != LVeGLearner.maxlength) {
				initialize(LVeGLearner.maxlength, false);
			} else {
				clear();
			}
		}
		
		public Chart(int n, boolean maxrule) {
			initialize(n, maxrule);
		}
		
		private void initialize(int n, boolean maxrule) {
			clear(); // empty the old memory
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
		
		protected int getMaxRuleSon(short key, int idx) {
			return mchart.get(idx).getMaxRuleSon(key);
		}
		
		public double getMaxRuleCount(short key, int idx, short level) {
			return mchart.get(idx).getMaxRuleCount(key, level);
		}
		
		public double getMaxRuleCount(short key, int idx) {
			return mchart.get(idx).getMaxRuleCount(key);
		}
		
		protected short getSplitPoint(short key, int idx) {
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
		
		public void addInsideScore(short key, int idx, GaussianMixture gm, short level) {
			ichart.get(idx).addScore(key, gm, level);
		}
		
		public GaussianMixture getInsideScore(short key, int idx, short level) {
			return ichart.get(idx).getScore(key, level);
		}
		
		public GaussianMixture getInsideScore(short key, int idx) {
			return ichart.get(idx).getScore(key);
		}
		
		public void addOutsideScore(short key, int idx, GaussianMixture gm, short level) {
			ochart.get(idx).addScore(key, gm, level);
		}
		
		public GaussianMixture getOutsideScore(short key, int idx, short level) {
			return ochart.get(idx).getScore(key, level);
		}
		
		public GaussianMixture getOutsideScore(short key, int idx) {
			return ochart.get(idx).getScore(key);
		}
		
		/**
		 * @deprecated
		 */
		public void addInsideScore(short key, int idx, GaussianMixture gm) {
			ichart.get(idx).addScore(key, gm);
		}
		
		/**
		 * @deprecated
		 */
		public void addOutsideScore(short key, int idx, GaussianMixture gm) {
			ochart.get(idx).addScore(key, gm);
		}
		
		public void clear() {
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
		private Map<Short, GaussianMixture> totals;
		private Map<Short, Map<Short, GaussianMixture>> scores;
		
		private Map<Short, Map<Short, Double>> maxRuleCnt;
		private Map<Short, Integer> maxRuleSon;
		private Map<Short, Short> maxRulePos;
		private Map<Short, Short> splitPoint;
		
		
		public Cell() {
			this.status = false;
			this.totals = new HashMap<Short, GaussianMixture>();
			this.scores = new HashMap<Short, Map<Short, GaussianMixture>>();
		}
		
		public Cell(boolean maxrule) {
			this();
			if (maxrule) {
				this.maxRuleCnt = new HashMap<Short, Map<Short, Double>>();
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
			return maxRuleCnt.get(level) == null ? null : maxRuleCnt.get(level).keySet();
		}
		
		protected boolean containsKey(short key) {
			return totals.containsKey(key);
		}
		
		protected boolean containsKey(short key, short level) {
			return scores.get(level) == null ? false : scores.get(level).containsKey(key);
		}
		
		protected void addMaxRuleCount(short key, double count, int sons, Short splitpoint, short level) {
			Map<Short, Double> lscore = maxRuleCnt.get(level);
			if (lscore == null) {
				lscore = new HashMap<Short, Double>();
				maxRuleCnt.put(level, lscore);
			}
			Double cnt = lscore.get(key);
			if (cnt != null && cnt > count) { return; }
			lscore.put(key, count);
			maxRulePos.put(key, level);
			maxRuleSon.put(key, sons);
			if (level == 0) { // binary rules
				splitPoint.put(key, splitpoint);
			}
		}
		
		protected double getMaxRuleCount(short key, short level) {
			Double cnt = maxRuleCnt.get(level) == null ? null : maxRuleCnt.get(level).get(key);
			return cnt == null ? Double.NEGATIVE_INFINITY : cnt;
		}
		
		protected double getMaxRuleCount(short key) {
			Short lkey = maxRulePos.get(key);
			return lkey == null ? Double.NEGATIVE_INFINITY : maxRuleCnt.get(lkey).get(key);
		}
		
		protected int getMaxRuleSon(short key) {
			return maxRuleSon.get(key);
		}
		
		protected short getSplitPoint(short key) {
			Short split = splitPoint.get(key);
			return split == null ? -1 : split;
		}
		
		protected void addScore(short key, GaussianMixture gm, short level) {
			Map<Short, GaussianMixture> lscore = scores.get(level);
			if (lscore == null) {
				lscore = new HashMap<Short, GaussianMixture>();
				scores.put(level, lscore);
			}
			GaussianMixture agm = lscore.get(key);
			if (agm == null) {
				lscore.put(key, gm);
			} else {
				agm.add(gm);
			}
			addScore(key, gm);
		}
		
		private void addScore(short key, GaussianMixture gm) {
			if (containsKey(key)) { 
				totals.get(key).add(gm);
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
			for (Map.Entry<Short, GaussianMixture> map : totals.entrySet()) {
				GaussianMixture gm = map.getValue();
				if (gm != null) { gm.clear(); }
			}
			scores.clear();
		}
		
		public String toString(boolean simple, int nfirst) {
			if (simple) {
				String name;
				StringBuffer sb = new StringBuffer();
				sb.append("Cell [status=" + status + ", size=" + totals.size());
				
				for (Map.Entry<Short, GaussianMixture> score : totals.entrySet()) {
					name = (String) Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET).object(score.getKey());
					// sb.append(", " + name + "=" + score.getValue().toString(simple, nfirst));
					sb.append(", " + name + ":nc=" + score.getValue().ncomponent);
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
			sb.append("Cell [status=" + status + ", size=" + totals.size());
			for (Map.Entry<Short, GaussianMixture> score : totals.entrySet()) {
				name = (String) Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET).object(score.getKey());
				sb.append(", " + name + "=" + score.getValue());
			}
			sb.append("]");
			return sb.toString();
		}
	}
}
