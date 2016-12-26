package edu.shanghaitech.ai.nlp.util;

/**
 * @author Dan Klein
 *
 */
public class MutableInteger extends Number implements Comparable<Object> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6612815442080956970L;
	private int i;
	
	public MutableInteger() {
		this(0);
	}
	
	public MutableInteger(int i) {
		this.i = i;
	}
	
	public void set(int i) {
		this.i = i;
	}
	
	public int compareTo(MutableInteger mi) {
		return (i < mi.i ? -1 : (i == mi.i ? 0 : 1));
	}

	@Override
	public int compareTo(Object o) {
		return compareTo((MutableInteger) o);
	}
	
	@Override
	public int hashCode() {
		return i;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof MutableInteger) {
			return i == ((MutableInteger) obj).i;
		}
		return false;
	}

	@Override
	public int intValue() {
		return i;
	}
	
	@Override
	public byte byteValue() {
		return (byte) i;
	}

	@Override
	public long longValue() {
		return i;
	}
	
	@Override
	public short shortValue() {
		return (short) i;
	}

	@Override
	public float floatValue() {
		return i;
	}

	@Override
	public double doubleValue() {
		return i;
	}
	
	@Override
	public String toString() {
		return Integer.toString(i);
	}
}
