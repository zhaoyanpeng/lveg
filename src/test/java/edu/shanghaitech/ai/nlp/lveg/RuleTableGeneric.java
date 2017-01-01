package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;

/**
 * Map grammar rules to their counts. {@link #isCompatible(Object)} must 
 * work well since it is the key. This kind of implementation does not 
 * have the practical significance. See the test case in RuleTableTest.
 * 
 * @author Yanpeng Zhao
 *
 */
public class RuleTableGeneric<T> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	Class<T> type;
	Map<T, GaussianMixture> table;
	
	
	public RuleTableGeneric(Class<T> type) {
		this.table = new HashMap<T, GaussianMixture>();
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
	
	
	public Set<T> keySet() {
		return table.keySet();
	}
	
	
	public boolean containsKey(T key) {
		return table.containsKey(key);
	}
	
	
	/**
	 * Type-specific instance.
	 * 
	 * @param key search keyword
	 * @return
	 * 
	 */
	public boolean isCompatible(T key) {
		return type.isInstance(key);
	}
	
	
	public GaussianMixture getCount(T key) {
		return table.get(key);
	}
	
	
	public void setCount(T key, GaussianMixture value) {
		if (isCompatible(key)) {
			table.put(key, value);
		}
	}
	
	
	public void increaseCount(T key, double increment) {
		GaussianMixture count = getCount(key);
		if (count == null) {
			GaussianMixture mog = new DiagonalGaussianMixture((short) 0);
			mog.add(increment);
			setCount(key, mog);
			return;
		}
		count.add(increment);
	}

	
	public void increaseCount(T key, GaussianMixture increment, boolean prune) {
		GaussianMixture count = getCount(key);
		if (count == null) {
			GaussianMixture mog = new DiagonalGaussianMixture((short) 0);
			mog.add(increment, prune);
			setCount(key, mog);
			return;
		}
		count.add(increment, prune);
	}
}
