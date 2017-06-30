package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.impl.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Cell;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Executor;
import edu.shanghaitech.ai.nlp.util.Recorder;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public abstract class Inferencer extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3449371510125004187L;
	protected final static short ROOT = 0;
	protected final static short LENGTH_UCHAIN = 2;
	public final static String DUMMY_TAG = "OOPS_ROOT";
	
	public static LVeGLexicon lexicon;
	public static LVeGGrammar grammar;
	
	
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
	 * Compute the inside score given the sentence and grammar rules, parallel version.
	 * <p>
	 * TODO need to update it to be consistent with the serial version
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used
	 * @param nword # of words in the sentence
	 */
	public static void insideScore(Chart chart, List<State> sentence, int nword, boolean prune, ThreadPool cpool) {
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(sentence.get(i));
			// preterminals
			for (GrammarRule rule : rules) {
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true), (short) 0);
			}
			if (prune) { chart.pruneInsideScore(iCell, (short) 0); }
			insideScoreForUnaryRule(chart, iCell, prune, false, false);
			if (prune) { chart.pruneInsideScore(iCell, (short) -1); }
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
				if (prune) { chart.pruneInsideScore(c2, (short) 0); }
				insideScoreForUnaryRule(chart, c2, prune, false, false);
				if (prune) { chart.pruneInsideScore(c2, (short) -1); }
			}
		}
	}
	
	
	/**
	 * Compute the outside score given the sentence and grammar rules, parallel version.
	 * <p>
	 * TODO need to update it to be consistent with the serial version
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used.
	 * @param nword # of words in the sentence
	 */
	public static void outsideScore(Chart chart, List<State> sentence, int nword, boolean prune, ThreadPool cpool) {
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
				if (prune) { chart.pruneOutsideScore(c2, (short) 0); }
				outsideScoreForUnaryRule(chart, c2, prune, false, false);
				if (prune) { chart.pruneOutsideScore(c2, (short) -1); }
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
	public static void insideScore(Chart chart, List<State> sentence, int nword, boolean prune, boolean usemask, boolean iomask) {
		List<GrammarRule> rules;
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture pinScore, linScore, rinScore, ruleScore;
		
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			rules = lexicon.getRulesWithWord(sentence.get(i));
			for (GrammarRule rule : rules) {
				if (usemask && iomask) {
					if (!chart.isAllowed(rule.lhs, iCell, true)) { continue; } 
				} else if (usemask) {
					if (!chart.isPosteriorAllowed(rule.lhs, iCell)) { continue; }
				}
				
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true), (short) 0);
			}
			
			if (prune) { chart.pruneInsideScore(iCell, (short) 0); }
			insideScoreForUnaryRule(chart, iCell, prune, usemask, iomask);
			if (prune) { chart.pruneInsideScore(iCell, (short) -1); }
		}		
		
		// inside score
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (short itag = 0; itag < grammar.ntag; itag++) {
					if (usemask && iomask) {
						if (!chart.isAllowed(itag, c2, true)) { continue; } 
					} else if (usemask) {
						if (!chart.isPosteriorAllowed(itag, c2)) { continue; }
					}
					
					rules = grammar.getBRuleWithP(itag);
					for (GrammarRule arule : rules) {
						BinaryGrammarRule rule = (BinaryGrammarRule) arule;
						for (int right = left; right < left + ilayer; right++) {
							y0 = right;
							x1 = right + 1;
							c0 = Chart.idx(x0, nword - (y0 - x0));
							c1 = Chart.idx(x1, nword - (y1 - x1));
						
							if (chart.containsKey(rule.lchild, c0, true) && chart.containsKey(rule.rchild, c1, true)) {
								ruleScore = rule.getWeight();
								linScore = chart.getInsideScore(rule.lchild, c0);
								rinScore = chart.getInsideScore(rule.rchild, c1);
								
								pinScore = ruleScore.mulForInsideOutside(linScore, RuleUnit.LC, true);
								pinScore = pinScore.mulForInsideOutside(rinScore, RuleUnit.RC, false);
								chart.addInsideScore(rule.lhs, c2, pinScore, (short) 0);
							}
						}
					}
				}
				
				if (prune) { chart.pruneInsideScore(c2, (short) 0); }
				insideScoreForUnaryRule(chart, c2, prune, usemask, iomask);
				if (prune) { chart.pruneInsideScore(c2, (short) -1); }
			}
		}
	}
	
	
	/**
	 * Compute the outside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used
	 * @param nword # of words in the sentence
	 */
	public static void outsideScore(Chart chart, List<State> sentence, int nword, boolean prune, boolean usemask, boolean iomask) {
		List<GrammarRule> rules;
		int x0, y0, x1, y1, c0, c1, c2;
		GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				x1 = left + ilayer + 1; 
				c2 = Chart.idx(left, nword - ilayer);
				// binary grammar rules
				for (short itag = 0; itag < grammar.ntag; itag++) {
					if (usemask && iomask) {
						if (!chart.isAllowed(itag, c2, false)) { continue; } 
					} else if (usemask) {
						if (!chart.isPosteriorAllowed(itag, c2)) { continue; }
					}
					
					rules = grammar.getBRuleWithLC(itag);
					for (GrammarRule arule : rules) {
						BinaryGrammarRule rule = (BinaryGrammarRule) arule;
						for (int right = left + ilayer + 1; right < nword; right++) {
							y0 = right;
							y1 = right;
							c0 = Chart.idx(x0, nword - (y0 - x0));
							c1 = Chart.idx(x1, nword - (y1 - x1));
						
							if (chart.containsKey(rule.lhs, c0, false) && chart.containsKey(rule.rchild, c1, true)) {
								ruleScore = rule.getWeight();
								poutScore = chart.getOutsideScore(rule.lhs, c0);
								rinScore = chart.getInsideScore(rule.rchild, c1);
								
								loutScore = ruleScore.mulForInsideOutside(poutScore, RuleUnit.P, true);
								loutScore = loutScore.mulForInsideOutside(rinScore, RuleUnit.RC, false);
								chart.addOutsideScore(rule.lchild, c2, loutScore, (short) 0);
							}
						}
					}
				}
				
				y0 = left + ilayer;
				y1 = left - 1;
				// binary grammar rules
				for (short itag = 0; itag < grammar.ntag; itag++) {
					if (usemask && iomask) {
						if (!chart.isAllowed(itag, c2, false)) { continue; } 
					} else if (usemask) {
						if (!chart.isPosteriorAllowed(itag, c2)) { continue; }
					}				
					
					rules = grammar.getBRuleWithRC(itag);
					for (GrammarRule arule : rules) {
						BinaryGrammarRule rule = (BinaryGrammarRule) arule;
						for (int right = 0; right < left; right++) {
							x0 = right; 
							x1 = right;
							c0 = Chart.idx(x0, nword - (y0 - x0));
							c1 = Chart.idx(x1, nword - (y1 - x1));
						
							if (chart.containsKey(rule.lhs, c0, false) && chart.containsKey(rule.lchild, c1, true)) {
								ruleScore = rule.getWeight();
								poutScore = chart.getOutsideScore(rule.lhs, c0);
								linScore = chart.getInsideScore(rule.lchild, c1);
								
								routScore = ruleScore.mulForInsideOutside(poutScore, RuleUnit.P, true);
								routScore = routScore.mulForInsideOutside(linScore, RuleUnit.LC, false);
								chart.addOutsideScore(rule.rchild, c2, routScore, (short) 0);
							}
						}
					}
				}
				
				if (prune) { chart.pruneOutsideScore(c2, (short) 0); }
				outsideScoreForUnaryRule(chart, c2, prune, usemask, iomask);
				if (prune) { chart.pruneOutsideScore(c2, (short) -1); }
			}
		}
	}
	
	
	private static void outsideScoreForUnaryRule(Chart chart, int idx, boolean prune, boolean usemask, boolean iomask) {
		Set<Short> set;
		short level = 0;
		List<GrammarRule> rules;
		RuleUnit rmKey = RuleUnit.P;
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
					if (usemask && iomask) {
						if (!chart.isAllowed((short) rule.rhs, idx, false)) { continue; } 
					} else if (usemask) {
						if (!chart.isPosteriorAllowed((short) rule.rhs, idx)) { continue; }
					}
					
					coutScore = rule.weight.copy(true); // since OS(ROOT) = 1
					chart.addOutsideScore((short) rule.rhs, idx, coutScore, level);
				}
			}
			if (prune) { chart.pruneOutsideScore(idx, level); }
		}
		while (level < LENGTH_UCHAIN && (set = chart.keySet(idx, false, level)) != null) {
			for (Short idTag : set) {
				rules = grammar.getURuleWithP(idTag);
				Iterator<GrammarRule> iterator = rules.iterator();
				poutScore = chart.getOutsideScore(idTag, idx, level);
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					if (usemask && iomask) {
						if (!chart.isAllowed((short) rule.rhs, idx, false)) { continue; } 
					} else if (usemask) {
						if (!chart.isPosteriorAllowed((short) rule.rhs, idx)) { continue; }
					}
					
					coutScore = rule.weight.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore((short) rule.rhs, idx, coutScore, (short) (level + 1));
				}
			}
			level++;
			if (prune) { chart.pruneOutsideScore(idx, level); }
		}
	}
	
	
	private static void insideScoreForUnaryRule(Chart chart, int idx, boolean prune, boolean usemask, boolean iomask) {
		RuleUnit rmKey;
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
					if (usemask && iomask) {
						if (!chart.isAllowed(rule.lhs, idx, true)) { continue; } 
					} else if (usemask) {
						if (!chart.isPosteriorAllowed(rule.lhs, idx)) { continue; }
					}
					
					if (idx != 0 && rule.type == RuleType.RHSPACE) { continue; } // ROOT is allowed only when it is in cell 0 and is in level 1 or 2
					rmKey = rule.type == RuleType.RHSPACE ? RuleUnit.C : RuleUnit.UC;
					pinScore = rule.weight.mulForInsideOutside(cinScore, rmKey, true);
					chart.addInsideScore(rule.lhs, idx, pinScore, (short) (level + 1));
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
					if (rule.type != RuleType.RHSPACE) { continue; } // only consider ROOT in level 3
					pinScore = rule.weight.mulForInsideOutside(cinScore, RuleUnit.C, true);
					chart.addInsideScore(rule.lhs, idx, pinScore, (short) (LENGTH_UCHAIN + 1));
				}
			}
			if (prune) { chart.pruneInsideScore(idx, (short) (LENGTH_UCHAIN + 1)); }
		}
	}
	
	
	/**
	 * Set the outside score of ROOT to 1.
	 * 
	 * @param chart the chart storing inside/outside scores
	 */
	public static void setRootOutsideScore(Chart chart) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		chart.addOutsideScore((short) 0, Chart.idx(0, 1), gm, (short) (LENGTH_UCHAIN + 1));
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
						
						pinScore = ruleScore.mulForInsideOutside(linScore, RuleUnit.LC, true);
						pinScore = pinScore.mulForInsideOutside(rinScore, RuleUnit.RC, false);
						input.cell.addScore(rule.lhs, pinScore, (short) 0);
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
						
						loutScore = ruleScore.mulForInsideOutside(poutScore, RuleUnit.P, true);
						loutScore = loutScore.mulForInsideOutside(rinScore, RuleUnit.RC, false);
						input.cell.addScore(rule.lchild, loutScore, (short) 0);
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
						
						routScore = ruleScore.mulForInsideOutside(poutScore, RuleUnit.P, true);
						routScore = routScore.mulForInsideOutside(linScore, RuleUnit.LC, false);
						input.cell.addScore(rule.rchild, routScore, (short) 0); 
					}
				}
			}
			Meta<O> cache = new Meta(itask, true); // currently nothing returns, may add try...catch.. clause
			synchronized (caches) {
				caches.add(cache);
				caches.notifyAll();
			}
			task = null;
			return itask;
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
	
	
	public static Tree<String> extractBestMaxRuleParse(Chart chart, List<String> sentence) {
		return extractBestMaxRuleParse(chart, 0, sentence.size() - 1, sentence.size(), (short) 0, sentence);
	}
	
	
	protected static Tree<String> extractBestMaxRuleParse(Chart chart, int left, int right, int nword, List<String> sentence) {
		return extractBestMaxRuleParse(chart, left, right, nword, (short) 0, sentence);
	}
	
	
	private static Tree<String> extractBestMaxRuleParse(Chart chart, int left, int right, int nword, short idtag, List<String> sentence) {
		int idx = Chart.idx(left, nword - (right - left));
		int son = chart.getMaxRuleSon(idtag, idx);
		if (son <= 0) { // sons = (1 << 31) + (rule.lchild << 16) + rule.rchild; or sons = 0;
			return extractBestMaxRuleParseBinary(chart, left, right, nword, idtag, sentence);
		} else {
			short idGrandson = (short) (son >>> 16);
			short idChild = (short) ((son << 16) >>> 16);
			List<Tree<String>> child = new ArrayList<>();
			String pname = null;
			pname = (String) grammar.numberer.object(idtag);
			if (pname == null) {
				throw new IllegalStateException("OOPS_BUG: id: " + idtag + ", son: " + idChild + ", grandson: " + idGrandson);
			}
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
				List<Tree<String>> chainChild = new ArrayList<>();
				String cname = (String) grammar.numberer.object(idChild);
				if (cname.endsWith("^g")) { cname = cname.substring(0, cname.length() - 2); }
				chainChild.add(new Tree<String>(cname, child));
				return new Tree<String>(pname, chainChild);
			}
		}
	}
	
	
	private static Tree<String> extractBestMaxRuleParseBinary(Chart chart, int left, int right, int nword, short idtag, List<String> sentence) {
		List<Tree<String>> children = new ArrayList<>();
		String pname = null;
		pname = (String) grammar.numberer.object(idtag);
		if (pname == null) { 
			throw new IllegalStateException("OOPS_BUG: id: " + idtag);
		}
		if (pname.endsWith("^g")) { 
			pname = pname.substring(0, pname.length() - 2); 
		}
		int idx = Chart.idx(left, nword - (right - left));
		int son = chart.getMaxRuleSon(idtag, idx, (short) 0); // can only exist in level 0
		if (right  == left) {
			if (son == 0) {
				children.add(new Tree<String>(sentence.get(left)));
			} else {
				logger.error("There must be somthing wrong in the preterminal layer.\n");
			}
		} else {
			int splitpoint = chart.getSplitPoint(idtag, idx);
			if (splitpoint == -1) {
				logger.error("It is not the binary rule since there is no split point.\n");
				return new Tree<String>(DUMMY_TAG);
			}
			if (son > 0) {
				logger.error("It is not the binary rule since the son is larger than 0.\n");
				return new Tree<String>(DUMMY_TAG);
			}
			son = ((son << 1) >>> 1);
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
