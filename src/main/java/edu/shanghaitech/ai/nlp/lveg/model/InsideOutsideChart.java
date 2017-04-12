package edu.shanghaitech.ai.nlp.lveg.model;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;

public class InsideOutsideChart {	
	
	public static class LVeGCell {
		private boolean status;
		private Map<Short, GaussianMixture> totals;
		private Map<Short, Map<Short, GaussianMixture>> scores;
		
		private Map<Short, Map<Short, Double>> maxRuleCnts;
		private Map<Short, Map<Short, Integer>> maxRuleSons;
		private Map<Short, Integer> maxRuleSon;
		private Map<Short, Short> maxRulePos;
		private Map<Short, Short> splitPoint;
		
		public LVeGCell() {
			this.status = false;
			this.totals = new HashMap<Short, GaussianMixture>();
			this.scores = new HashMap<Short, Map<Short, GaussianMixture>>();
		}
		
		public LVeGCell(boolean maxrule) {
//			this(); // CHECK
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
			if (maxRuleSons != null) {
//				for (Map.Entry<Short, Map<Short, Integer>> entry : maxRuleSons.entrySet()) {
//					if (entry.getValue() != null) { entry.getValue().clear(); }
//				}
				maxRuleSons.clear();
			}
			if (maxRuleSon != null) { maxRuleSon.clear(); }
			if (maxRulePos != null) { maxRulePos.clear(); }
			if (splitPoint != null) { splitPoint.clear(); }
		}
		
		public String toString(boolean simple, int nfirst, boolean quantity, Numberer numberer) {
			if (simple) {
				String name;
				StringBuffer sb = new StringBuffer();
				sb.append("Cell [status=" + status + ", size=" + totals.size());
				
				for (Map.Entry<Short, GaussianMixture> score : totals.entrySet()) {
					name = (String) numberer.object(score.getKey());
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
						name = (String) numberer.object(detail.getKey());
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
		
		public String toString(Numberer numberer) {
			String name;
			StringBuffer sb = new StringBuffer();
			sb.append("Cell [status=" + status + ", size=" + totals.size());
			for (Map.Entry<Short, GaussianMixture> score : totals.entrySet()) {
				name = (String) numberer.object(score.getKey());
				sb.append(", " + name + "=" + score.getValue());
			}
			sb.append("]");
			return sb.toString();
		}
	}
	
	public static class PCFGCell {
		// key word "private" does not make any difference, outer class can access all the fields 
		// of the inner class through the instance of the inner class or in the way of the static 
		// fields accessing.
		private Set<Short> masks; 
		private Map<Short, Double> mtotals;
		private Map<Short, Set<Short>> mtags;
		private Map<Short, Map<Short, Double>> mscores;
		
		public PCFGCell(boolean maxrule, boolean usemask) {
			if (usemask) {
				this.masks = new HashSet<Short>();
				this.mtags = new HashMap<Short, Set<Short>>();
				this.mtotals = new HashMap<Short, Double>();
				this.mscores = new HashMap<Short, Map<Short, Double>>();
			}
		}
		
		protected Set<Short> keySetMask() {
			return mtotals.keySet();
		}
		
		protected Set<Short> keySetMask(short level) {
			return mscores.get(level) == null ? null : mscores.get(level).keySet();
		}
		
		protected boolean containsKeyMask(short key) {
			return mtotals.containsKey(key);
		}
		
		protected boolean containsKeyMask(short key, short level) {
			return mscores.get(level) == null ? false : mscores.get(level).containsKey(key);
		}
		
		protected boolean isAllowed(short key) {
			return masks.contains(key);
		}
		
		protected boolean isAllowed(short key, short level) {
			Set<Short> keys = mtags.get(level);
			if (keys != null && keys.contains(key)) {
				return true;
			}
			return false;
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
		
		protected void pruneScoreMask(short level, PriorityQueue<Double> queue, int base, double ratio) {
			// TODO
		}
		
		protected void addMask(short key, short level) {
			Set<Short> keys = mtags.get(level);
			if (keys == null) {
				keys = new HashSet<Short>();
				mtags.put(level, keys);
			}
			keys.add(key);
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
		
		protected double getScoreMask(short key, short level) {
			return mscores.get(level) == null ? Double.NEGATIVE_INFINITY : mscores.get(level).get(key);
		}
		
		protected double getScoreMask(short key) {
			return mtotals.get(key);
		}
		
		protected void clear() {
			if (mscores != null) { mscores.clear(); }
			if (mtotals != null) { mtotals.clear(); }
			if (masks != null) { masks.clear(); }
			if (mtags != null) { mtags.clear(); }
		}
		
	}

}
