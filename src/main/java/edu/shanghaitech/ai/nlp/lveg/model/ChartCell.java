package edu.shanghaitech.ai.nlp.lveg.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Numberer;

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
 * We store such chart in an array, the mapping from the position in 
 * the chart to the index of the array is established as the follows
 * <p>
 * Layer: from 1, the top of the pyramid, to # of words, the bottom of 
 * the pyramid. 
 * <p>
 * Index: calculated as layer * (layer - 1) / 2 + offset, where offset
 * is counted starting from left to right for each layer.
 * <p>
 * If layer ranges from bottom (0) to top (# of words), index would be
 * (# + # - layer + 1) * layer / 2 + offset. We used the first schema.
 */
public class ChartCell {	
	
	public enum CellType {
		DEFAULT, MAX_RULE, PCFG, MASK
	}
	
	public static Cell getCell(CellType type) {
		Cell cell = new Cell();
		switch (type) {
		case MAX_RULE: {
			cell.maxRulePos = new HashMap<Short, Short>();
			cell.maxRuleSon = new HashMap<Short, Integer>();
			cell.splitPoint = new HashMap<Short, Short>();
			cell.maxRuleCnts = new HashMap<Short, Map<Short, Double>>(3, 1);
			cell.maxRuleSons = new HashMap<Short, Map<Short, Integer>>(3, 1);
			break;
		}
		case PCFG: {
			cell.masks = new HashSet<Short>();
			cell.mtags = new HashMap<Short, Set<Short>>(3, 1);
			cell.mtotals = new HashMap<Short, Double>();
			cell.mscores = new HashMap<Short, Map<Short, Double>>(3, 1);
			break;
		}
		case MASK: {
			cell.masks = new HashSet<Short>();
			cell.mtags = new HashMap<Short, Set<Short>>(3, 1);
			break;
		}
		case DEFAULT: {
			cell.totals = new HashMap<Short, GaussianMixture>();
			cell.scores = new HashMap<Short, Map<Short, GaussianMixture>>(3, 1);
			break;
		}
		default: {
			// TODO
		}
		}
		return cell;
	}
	
	
	public static class Chart {
		private List<Cell> ochart = null;
		private List<Cell> ichart = null;
		private List<Cell> mchart = null; // for max-rule parser
		
		private List<Cell> omasks = null; // for treebank grammars
		private List<Cell> imasks = null;
		private List<Cell> tmasks = null; // tag mask
		
		private PriorityQueue<Double> queue = null; // owned by this chart, need to make it thread safe
		
		public Chart(int n, boolean defchoice, boolean maxrule, boolean usemask) {
			queue = new PriorityQueue<Double>();
			initialize(n, defchoice, maxrule, usemask);
		}
		
		private void initialize(int n, boolean defchoice, boolean maxrule, boolean usemask) {
			int size = n * (n + 1) / 2;
			if (defchoice) {
				ochart = new ArrayList<Cell>(size);
				ichart = new ArrayList<Cell>(size);
				for (int i = 0; i < size; i++) {
					ochart.add(getCell(CellType.DEFAULT));
					ichart.add(getCell(CellType.DEFAULT));
				}
			}
			if (maxrule) {
				mchart = new ArrayList<Cell>(size);
				for (int i = 0; i < size; i++) {
					mchart.add(getCell(CellType.MAX_RULE));
				}
			}
			if (usemask) {
				imasks = new ArrayList<Cell>(size);
				omasks = new ArrayList<Cell>(size);
				tmasks = new ArrayList<Cell>(size);
				for (int i = 0; i < size; i++) {
					imasks.add(getCell(CellType.PCFG));
					omasks.add(getCell(CellType.PCFG));
					tmasks.add(getCell(CellType.MASK));
				}
			}
		}
		
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
		
		public List<Cell> getChartTmask() {
			return tmasks;
		}
		
		public List<Cell> getChartMask(boolean inside) {
			return inside ? imasks : omasks;
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
		
		public void addInsideScoreMask(short key, int idx, double score, short level) {
			imasks.get(idx).addScoreMask(key, score, level);
		}
		
		public void addInsideScore(short key, int idx, GaussianMixture gm, short level) {
			ichart.get(idx).addScore(key, gm, level);
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
		
		public void addOutsideScoreMask(short key, int idx, double score, short level) {
			omasks.get(idx).addScoreMask(key, score, level);
		}
		
		public void addOutsideScore(short key, int idx, GaussianMixture gm, short level) {
			ochart.get(idx).addScore(key, gm, level);
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
		
		public void pruneOutsideScoreMask(int idx, short level, int base, double ratio, boolean retainall) {
			synchronized (queue) {
				if (level < 0) {
					omasks.get(idx).pruneScoreMask(queue, base, ratio, retainall);
				} else {
					omasks.get(idx).pruneScoreMask(level, queue, base, ratio);
				}
			}
		}
		
		public void pruneOutsideScore(int idx, short level) {
			if (level < 0) {
				ochart.get(idx).pruneScore();
			} else {
				ochart.get(idx).pruneScore(level);
			}
		}
		
		public void pruneInsideScoreMask(int idx, short level, int base, double ratio, boolean retainall) {
			synchronized (queue) {
				if (level < 0) {
					imasks.get(idx).pruneScoreMask(queue, base, ratio, retainall);
				} else {
					imasks.get(idx).pruneScoreMask(level, queue, base, ratio);
				}
			}
		}
		
		public void pruneInsideScore(int idx, short level) {
			if (level < 0) {
				ichart.get(idx).pruneScore();
			} else {
				ichart.get(idx).pruneScore(level);
			}
		}
		
		public boolean isAllowed(short key, int idx, boolean inside) {
			return inside ? imasks.get(idx).isAllowed(key) : omasks.get(idx).isAllowed(key);
		}
		
		public boolean isPosteriorAllowed(short key, int idx) {
			return tmasks.get(idx).isAllowed(key);
		}
		
		public boolean isPosteriorAllowed(short key, int idx, short level) {
			return tmasks.get(idx).isAllowed(key, level);
		}
		
		public void addPosteriorMask(short key, int idx) {
			tmasks.get(idx).addPosteriorMask(key);
		}
		
		public void addPosteriorMask(short key, int idx, short level) {
			tmasks.get(idx).addPosteriorMask(key, level);
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
			if (tmasks != null) {
				cnt = 0;
				for (Cell cell : tmasks) {
					if (++cnt > max) { break; }
					if (cell != null) { cell.clear(); }
				}
			}
		}
		
		@Override
		public String toString() {
			return "Chart [ichart=" + ichart + ", ochart=" + ochart + "]";
		}
	}
	
	
	public static class Cell {
		// LVeG parser
		private boolean status;
		private Map<Short, GaussianMixture> totals;
		private Map<Short, Map<Short, GaussianMixture>> scores;
		// for parsing
		private Map<Short, Map<Short, Double>> maxRuleCnts;
		private Map<Short, Map<Short, Integer>> maxRuleSons;
		private Map<Short, Integer> maxRuleSon;
		private Map<Short, Short> maxRulePos;
		private Map<Short, Short> splitPoint;
		// using masks
		private Set<Short> masks; 
		private Map<Short, Double> mtotals;
		private Map<Short, Set<Short>> mtags;
		private Map<Short, Map<Short, Double>> mscores;
		
		private Cell() {
			this.status = false;
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
		
		protected void pruneScoreMask(PriorityQueue<Double> queue, int base, double ratio, boolean retainall) {
			if (!retainall && mtotals.size() > base) {
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
		
		protected void addPosteriorMask(short key) {
			masks.add(key);
		}
		
		protected void addPosteriorMask(short key, short level) {
			Set<Short> keys = mtags.get(level);
			if (keys == null) {
				keys = new HashSet<Short>();
				mtags.put(level, keys);
			}
			keys.add(key);
		}
		
		protected synchronized void addScoreMask(short key, double score, short level) {
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
			addScoreMask(key, score);
		}
		
		private synchronized void addScoreMask(short key, double score) {
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
		
		
		/*******************************************/
		/**            Hello, Master!             **/
		/**                                       **/
		/*******************************************/
		
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
		}
		
		protected int getMaxRuleSon(short key) {
			return maxRuleSon.get(key);
		}
		
		protected short getSplitPoint(short key) {
			Short split = splitPoint.get(key);
			return split == null ? -1 : split;
		}
		
		protected synchronized void addScore(short key, GaussianMixture gm, short level) {
			Map<Short, GaussianMixture> lscore = scores.get(level);
			if (lscore == null) {
				lscore = new HashMap<Short, GaussianMixture>();
				scores.put(level, lscore);
			}
			GaussianMixture agm = lscore.get(key);
			if (agm == null) {
				lscore.put(key, gm);
			} else {
				agm.add(gm, false);
			}
			addScore(key, gm);
		}
		
		private synchronized void addScore(short key, GaussianMixture gm) {
			if (containsKey(key)) { 
				// gm is passed into this method by addScore(short, GaussianMixture, short, boolean),
				// before that it has been added into Cell.scores, and is filtered when 
				// GaussianMixture.add(GaussianMixture, boolean) is called. Here gm may be filtered
				// again by calling totals.get(key).add(...), in which some components of gm may be 
				// cleared through the reference and further modify Cell.scores. The safe practice is
				// copying gm and adding into Cell.totals, but that results in unnecessary memory
				// overhead, so I choose not to clear the filtered component in GaussianMixture.add()
				totals.get(key).add(gm, false);
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
			if (scores != null) { scores.clear(); }
			if (totals != null) { totals.clear(); }
			
			if (maxRuleCnts != null) { maxRuleCnts.clear(); }
			if (maxRuleSons != null) { maxRuleSons.clear(); }
			if (maxRuleSon != null) { maxRuleSon.clear(); }
			if (maxRulePos != null) { maxRulePos.clear(); }
			if (splitPoint != null) { splitPoint.clear(); }
			
			if (mscores != null) { mscores.clear(); }
			if (mtotals != null) { mtotals.clear(); }
			if (masks != null) { masks.clear(); }
			if (mtags != null) { mtags.clear(); }
		}
		
		public String toString(boolean simple, int nfirst, boolean quantity, Numberer numberer) {
			if (simple) {
				String name;
				StringBuffer sb = new StringBuffer();
				if (totals != null) {
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
				}
				
				if (scores != null) {
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
				}
				
				if (masks != null) {
					sb.append("\n\n--- masks of this cell---\n");
					sb.append("[size=" + masks.size());
					for (Short tag : masks) {
						sb.append(", " + tag);
					}
					sb.append("]");
					if (mtotals != null) {
						sb.append("\nScores [size=" + mtotals.size());
						for (Map.Entry<Short, Double> score : mtotals.entrySet()) {
							name = (String) numberer.object(score.getKey());
							sb.append(", id=" + score.getKey() + ", " + name + "=" + score.getValue() );
						}
						sb.append("]\n");
					}
				}
				
				return sb.toString();
			} else {
				return toString();
			}
		}
	}

}
