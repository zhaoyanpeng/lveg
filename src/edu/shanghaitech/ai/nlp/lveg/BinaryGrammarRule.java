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
	protected short lchild;
	protected short rchild;
	
	
	public BinaryGrammarRule() {
		// TODO
	}
	
	
	public BinaryGrammarRule(short lhs, short lchild, short rchild) {
		this.lhs = lhs;
		this.lchild = lchild;
		this.rchild = rchild;
	}
	
	
	/**
	 * Constructor with the rule weight initialized.
	 * 
	 * @param lhs    id of the tag on the left hand side
	 * @param lchild id of the tag, representing the left child
	 * @param rchild id of the tag, representing the right child
	 * @param flag   to distinguish from the above constructor (without weight initialized)
	 */
	public BinaryGrammarRule(short lhs, short lchild, short rchild, boolean flag) {
		this.lhs = lhs;
		this.lchild = lchild;
		this.rchild = rchild;
		initializeWeight();
	}
	
	
	public BinaryGrammarRule(short lhs, short lchild, short rchild, GaussianMixture weight) {
		this.lhs = lhs;
		this.lchild = lchild;
		this.rchild = rchild;
		this.weight = weight;
	}
	
	
	private void initializeWeight() {
		weight = new GaussianMixture(LVeGLearner.ncomponent);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set2 = new HashSet<GaussianDistribution>();
			set0.add(new GaussianDistribution(LVeGLearner.dim));
			set1.add(new GaussianDistribution(LVeGLearner.dim));
			set2.add(new GaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, set0);
			map.put(Unit.LC, set1);
			map.put(Unit.RC, set2);
			weight.add(i, map);
		}
	}


	public short getLchild() {
		return lchild;
	}


	public void setLchild(short lchild) {
		this.lchild = lchild;
	}


	public short getRchild() {
		return rchild;
	}


	public void setRchild(short rchild) {
		this.rchild = rchild;
	}


	@Override
	public boolean isUnary() {
		return false;
	}
	
	
	@Override
	public int hashCode() {
		return (lhs << 16) ^ (lchild << 8) ^ (rchild);
	}

	
	@Override
	public boolean equals(Object o) {
		if (this == o) { return true; }
		
		if (o instanceof BinaryGrammarRule) {
			BinaryGrammarRule rule = (BinaryGrammarRule) o;
			if (lhs == rule.lhs && lchild == rule.lchild && rchild == rule.rchild) {
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
		if (lchild < rule.lchild) { return -1; }
		if (lchild > rule.lchild) { return 1; }
		if (rchild < rule.rchild) { return -1; }
		if (rchild > rule.rchild) { return 1; }
		return 0;
	}
	
	
	@Override
	public String toString() {
		return "BinaryGrammarRule [P: " + lhs +", LC: " + lchild + ", RC: " + rchild + "]";
	}
	
}
