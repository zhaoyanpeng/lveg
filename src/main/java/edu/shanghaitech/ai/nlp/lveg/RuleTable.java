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
	private static final long serialVersionUID = 7379371632425330796L;
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
	
	
	public void addCount(GrammarRule key, double increment) {
		GaussianMixture count = getCount(key);
		if (count == null) {
			GaussianMixture gm = new DiagonalGaussianMixture();
			gm.add(increment);
			setCount(key, gm);
			return;
		}
		count.add(increment);
	}

	
	public void addCount(GrammarRule key, GaussianMixture increment) {
		GaussianMixture count = getCount(key);
		if (count == null) {
			GaussianMixture gm = new DiagonalGaussianMixture();
			gm.add(increment);
			setCount(key, gm);
			return;
		}
		count.add(increment);
	}
	
	
	/**
	 * @param deep deep copy or shallow copy
	 * @return
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public RuleTable copy(boolean deep) {
		RuleTable ruleTable = new RuleTable(type);
		for (GrammarRule rule : table.keySet()) {
			// copy key by reference, when only the count (value) varies
			if (!deep) {
				ruleTable.addCount(rule, null);
			} else {
				ruleTable.addCount(rule.copy(), table.get(rule).copy(true));
			}
		}
		return ruleTable;
	}
}
