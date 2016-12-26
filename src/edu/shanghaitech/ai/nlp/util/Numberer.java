package edu.shanghaitech.ai.nlp.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * Rewrite, changing static fields to non-static ones, to be able to save to the object.
 * 
 * @author Dan Klein
 * @author Yanpeng Zhao
 *
 */
public class Numberer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2309277618797328501L;
	private Map numbererMap = new HashMap();
	private boolean locked = false;
	private MutableInteger idx;
	private Map idx2obj;
	private Map obj2idx;
	private int count;
	
	public Numberer() {}
	
	public Numberer(boolean placeholder) {
		this.count = 0;
		this.idx2obj = new HashMap();
		this.obj2idx = new HashMap();
		this.idx = new MutableInteger();
	}
	
	public void put(String key, Object o) {
		numbererMap.put(key, o);
	}
	
	public Numberer getGlobalNumberer(String key) {
		Numberer value = (Numberer) numbererMap.get(key);
		if (value == null) {
			value = new Numberer(true);
			numbererMap.put(key, value);
		}
		return value;
	}
	
	public int translate(String src, String des, int i) {
		return getGlobalNumberer(des).number(
				getGlobalNumberer(src).object(i));
	}
	
	public boolean containsIdx(int i) {
		return idx2obj.containsKey(i);
	}
	
	public boolean containsObj(Object o) {
		return obj2idx.containsKey(o); // CHECK
	}
	
	public int number(String key, Object o) {
		return getGlobalNumberer(key).number(o);
	}
	
	
	public Object object(String key, int i) {
		return getGlobalNumberer(key).object(i);
	}
	
	
	public void setNumbererMap(Map numbererMap) {
		this.numbererMap = numbererMap;
	}
	
	
	public int number(Object o) {
		MutableInteger anidx = (MutableInteger) obj2idx.get(o);
		if (anidx == null) {
			if (locked) {
				throw new NoSuchElementException("no object: " + o);
			} 
			anidx = new MutableInteger(count);
			count++;
			obj2idx.put(o, anidx);
			idx2obj.put(anidx, o);
		}
		return anidx.intValue();
	}
	
	public Object object(int i) {
		idx.set(i); // CHECK must be individually initialized?
		return idx2obj.get(idx);
	}
	
	public Map getNumbererMap() {
		return numbererMap;
	}
	
	public Set objects() {
		return obj2idx.keySet();
	}
	
	public void lock() {
		this.locked = true;
	}
	
	public int size() {
		return count;
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("[");
		for (int i = 0; i < count; i++) {
			sb.append(i);
			sb.append("->");
			sb.append(object(i));
			if (i < count - 1) {
				sb.append(", ");
			}
		}
		sb.append("]");
		return sb.toString();
	}
	
}
