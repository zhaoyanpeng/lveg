package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class Inferencer extends Recorder {
	
	protected final static short LENGTH_UCHAIN = 2;
	
	enum ChainUrule {
		ALL_POSSIBLE_PATH, PRE_COMPUTE_CHAIN, NOT_PRE_ADD_INTER, NOT_PRE_NOT_INTER, DEFAULT,
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
		
		public boolean getStatus(int idx, boolean inside) {
			return inside ? ichart.get(idx).getStatus() : ochart.get(idx).getStatus();
		}
		
		public static List<Cell> getChart(boolean inside) {
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
		private Map<Short, GaussianMixture> totals;
		private Map<Short, Map<Short, GaussianMixture>> scores;
		
		public Cell() {
			this.status = false;
			this.totals = new HashMap<Short, GaussianMixture>();
			this.scores = new HashMap<Short, Map<Short, GaussianMixture>>();
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
		
		protected boolean containsKey(short key) {
			return totals.containsKey(key);
		}
		
		protected boolean containsKey(short key, short level) {
			return scores.get(level) == null ? false : scores.get(level).containsKey(key);
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
				totals.put(key, gm);
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
				sb.append("Cell [status=" + status + ", size=" + scores.size());
				
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
			sb.append("Cell [status=" + status + ", size=" + scores.size());
			for (Map.Entry<Short, GaussianMixture> score : totals.entrySet()) {
				name = (String) Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET).object(score.getKey());
				sb.append(", " + name + "=" + score.getValue());
			}
			sb.append("]");
			return sb.toString();
		}
	}
}
