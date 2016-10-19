package edu.shanghaitech.ai.nlp.lveg;

import java.util.Map;
import java.util.Set;
import java.util.HashMap;
import java.util.HashSet;

public class BinaryGrammarRule extends GrammarRule implements Comparable<Object> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * the IDs of the two right-hand side nonterminals
	 */
	protected short rhsLeft;
	protected short rhsRight;
	
	
	public BinaryGrammarRule() {
		// TODO
	}
	
	
	public BinaryGrammarRule(short lhs, short rhsLeft, short rhsRight) {
		this.lhs = lhs;
		this.rhsLeft = rhsLeft;
		this.rhsRight = rhsRight;
		
		initializeWeight();
	}
	
	
	public BinaryGrammarRule(short lhs, short rhsLeft, short rhsRight, boolean flag) {
		this.lhs = lhs;
		this.rhsLeft = rhsLeft;
		this.rhsRight = rhsRight;
	}
	
	
	public BinaryGrammarRule(short lhs, short rhsLeft, short rhsRight, GaussianMixture weight) {
		this.lhs = lhs;
		this.rhsLeft = rhsLeft;
		this.rhsRight = rhsRight;
		this.weight = weight;
	}
	
	
	private void initializeWeight() {
		weight = new GaussianMixture(LVeGLearner.ncomponent);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set2 = new HashSet<GaussianDistribution>();
			set0.add(new GaussianDistribution());
			set1.add(new GaussianDistribution());
			set2.add(new GaussianDistribution());
			map.put(Unit.P, set0);
			map.put(Unit.LC, set1);
			map.put(Unit.RC, set2);
			weight.add(i, map);
		}
	}


	public short getRhsLeft() {
		return rhsLeft;
	}


	public void setRhsLeft(short rhsLeft) {
		this.rhsLeft = rhsLeft;
	}


	public short getRhsRight() {
		return rhsRight;
	}


	public void setRhsRight(short rhsRight) {
		this.rhsRight = rhsRight;
	}


	@Override
	public boolean isUnary() {
		return false;
	}
	
	
	@Override
	public int hashCode() {
		return (lhs << 16) ^ (rhsLeft << 8) ^ (rhsRight);
	}

	
	@Override
	public boolean equals(Object o) {
		if (this == o) { return true; }
		
		if (o instanceof BinaryGrammarRule) {
			BinaryGrammarRule rule = (BinaryGrammarRule) o;
			if (lhs == rule.lhs && rhsLeft == rule.rhsLeft && rhsRight == rule.rhsRight) {
				return true;
			}
		}
		return false;
	}


	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		BinaryGrammarRule rule = (BinaryGrammarRule) o;
		if (lhs < rule.lhs) { return -1; }
		if (lhs > rule.lhs) { return 1; }
		if (rhsLeft < rule.rhsLeft) { return -1; }
		if (rhsLeft > rule.rhsLeft) { return 1; }
		if (rhsRight < rule.rhsRight) { return -1; }
		if (rhsRight > rule.rhsRight) { return 1; }
		return 0;
	}
	
	
	@Override
	public String toString() {
		return "BinaryGrammarRule [P: " + lhs +", LC: " + rhsLeft + ", RC: " + rhsRight + "]";
	}
	
}
