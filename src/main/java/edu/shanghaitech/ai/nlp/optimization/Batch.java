package edu.shanghaitech.ai.nlp.optimization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;

public class Batch implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5018573183821755031L;
	protected Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> batch;
	
	/**
	 * @param maxsize initialized by default if lower than 0, otherwise initialized with the specified capacity
	 */
	public Batch(int maxsize) {
		if (maxsize > 0) {
			batch = new HashMap<>(maxsize, 1);
		} else {
			batch = new HashMap<>();
		}
	}
	
	protected void add(short idx, EnumMap<RuleUnit, GaussianMixture> cnt) {
		List<EnumMap<RuleUnit, GaussianMixture>> cnts = null;
		if ((cnts = batch.get(idx)) != null) {
			cnts.add(cnt);
		} else {
			cnts = new ArrayList<>();
			cnts.add(cnt);
			batch.put(idx, cnts);
		}
	}
	
	protected List<EnumMap<RuleUnit, GaussianMixture>> get(short i) {
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
