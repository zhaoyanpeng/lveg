package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;

public class Batch {
	protected Map<Short, List<Map<String, GaussianMixture>>> batch;
	
	/**
	 * @param maxsize initialized by default if lower than 0, otherwise initialized with the specified capacity
	 */
	public Batch(int maxsize) {
		if (maxsize > 0) {
			batch = new HashMap<Short, List<Map<String, GaussianMixture>>>(maxsize, 1);
		} else {
			batch = new HashMap<Short, List<Map<String, GaussianMixture>>>();
		}
	}
	
	protected void add(short idx, Map<String, GaussianMixture> cnt) {
		List<Map<String, GaussianMixture>> cnts = null;
		if ((cnts = batch.get(idx)) != null) {
			cnts.add(cnt);
		} else {
			cnts = new ArrayList<Map<String, GaussianMixture>>();
			cnts.add(cnt);
			batch.put(idx, cnts);
		}
	}
	
	protected List<Map<String, GaussianMixture>> get(short i) {
		return batch.get(i);
	}
	
	protected boolean containsKey(short i) {
		return batch.containsKey(i);
	}
	
	protected Set<Short> keySet() {
		return batch.keySet();
	}
	
	protected void clear() {
		batch.clear();
	}
	
	protected int size() {
		return batch.size();
	}
}
