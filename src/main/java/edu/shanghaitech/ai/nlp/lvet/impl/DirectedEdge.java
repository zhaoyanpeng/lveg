package edu.shanghaitech.ai.nlp.lvet.impl;

import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;

public class DirectedEdge extends GrammarRule implements Comparable<Object> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5899459705430874090L;
	public int rhs;
	
	public DirectedEdge(short src, int des, byte type) {
		this.lhs = src;
		this.rhs = des;
		this.type = type;
	}

	@Override
	public void initializeWeight(byte type, short ncomponent, short ndim) {
		weight = rndRuleWeight(type, ncomponent, ndim);
	}
	
	@Override
	public GrammarRule copy() {
		DirectedEdge rule = new DirectedEdge(lhs, rhs, type);
		rule.weight = weight.copy(true);
		return rule;
	}

	@Override
	public boolean isUnary() {
		return true;
	}
	
	@Override
	public int hashCode() {
		return (lhs << 18) ^ (rhs);
	}
	
	@Override
	public boolean equals(Object o) {
		if (this == o) { return true; }
		
		if (o instanceof DirectedEdge) {
			DirectedEdge rule = (DirectedEdge) o;
			if (lhs == rule.lhs && rhs == rule.rhs && type == rule.type) {
				return true;
			}
		}
		return false;
	}

	@Override
	public int compareTo(Object o) {
		DirectedEdge rule = (DirectedEdge) o;
		if (lhs < rule.lhs) { return -1; }
		if (lhs > rule.lhs) { return 1; }
		if (rhs < rule.rhs) { return -1; }
		if (rhs > rule.rhs) { return 1; }
		return 0;
	}
	
	@Override
	public String toString() {
		return "D-Edge [P: " + lhs +", C: " + rhs + ", T: " + (short) type + "]";
	}
	
}
