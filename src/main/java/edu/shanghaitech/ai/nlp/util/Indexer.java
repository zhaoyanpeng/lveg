package edu.shanghaitech.ai.nlp.util;

import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Indexer<E> extends AbstractList<E> implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private boolean locked = false;
	private List<E> objects;
	private Map<E, Integer> indexes;
	
	public Indexer() {
		objects = new ArrayList<E>();
		indexes = new HashMap<E, Integer>();
	}
	
	public Indexer(Collection<? extends E> c) {
		this();
		for (E e : c) { index(e); }
	}
	
	/**
	 * Look up the index of the given element, would be added 
	 * if the element does not exist.
	 * 
	 * @param e the element to be looked up
	 * @return  the index of the given element
	 */
	public int index(E e) {
		if (e == null) { return -1; }
		Integer idx = indexes.get(e);
		if (idx == null) {
			if (locked) { return -1; }
			idx = size();
			objects.add(e);
			indexes.put(e, idx);
		}
		return idx;
	}
	
	public int indexof(Object o) {
		Integer idx = indexes.get(o);
		return idx == null ? -1 : idx;
	}
	
	public boolean add(E e) {
		if (locked) {
			throw new IllegalStateException("Tried to add to locked indexer");
		}
		if (contains(e)) { return false; }
		indexes.put(e, size());
		objects.add(e);
		return true;
	}
	
	public boolean contains(Object o) {
		return indexes.containsKey(o);
	}
	
	@Override
	public E get(int index) {
		return objects.get(index);
	}

	@Override
	public int size() {
		return objects.size();
	}
	
	@Override
	public void clear() {
		objects.clear();
		indexes.clear();
	}
	
	public List<E> getObjects() {
		return objects;
	}
	
	public void lock() {
		this.locked = true;
	}
	
	public void unlock() {
		this.locked = false;
	}

	@Override
	public String toString() {
		return "Indexer [locked=" + locked + ", objects=" + objects + ", indexes=" + indexes + "]";
	}
	
}
