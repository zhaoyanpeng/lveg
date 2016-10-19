package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Map grammar rules to their counts. Implementation in the generic way could be better.
 * 
 * @author Yanpeng Zhao
 *
 */
public class RuleTable<T> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	Class<T> type;
	Map<GrammarRule, GaussianMixture> table;
	
	
	public RuleTable(Class<T> type) {
		this.table = new HashMap<GrammarRule, GaussianMixture>();
		this.type = type;
	}
	
	
	public int size() {
		return table.size();
	}
	
	
	public void clear() {
		table.clear();
	}
	
	
	public boolean isEmpty() {
		return size() == 0;
	}
	
	
	public Set<GrammarRule> keySet() {
		return table.keySet();
	}
	
	
	public boolean containsKey(GrammarRule key) {
		return table.containsKey(key);
	}
	
	
	/**
	 * Type-specific instance.
	 * 
	 * @param key search keyword
	 * @return
	 * 
	 */
	public boolean isCompatible(GrammarRule key) {
		return type.isInstance(key);
	}
	
	
	public GaussianMixture getCount(GrammarRule key) {
		return table.get(key);
	}
	
	
	public void setCount(GrammarRule key, GaussianMixture value) {
		if (isCompatible(key)) {
			table.put(key, value);
		}
	}
	
	
	public void increaseCount(GrammarRule key, double increment) {
		GaussianMixture count = getCount(key);
		if (count == null) {
			GaussianMixture gm = new GaussianMixture();
			gm.add(increment);
			setCount(key, gm);
			return;
		}
		count.add(increment);
	}

	
	public void increaseCount(GrammarRule key, GaussianMixture increment) {
		GaussianMixture count = getCount(key);
		if (count == null) {
			GaussianMixture gm = new GaussianMixture();
			gm.add(increment);
			setCount(key, gm);
			return;
		}
		count.add(increment);
	}
}
