package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;

import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Yanpeng Zhao
 *
 */
public class BinaryGrammarRule extends GrammarRule implements Comparable<Object> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7092883784519182728L;
	/**
	 * the IDs of the two right-hand side nonterminals
	 */
	public short lchild;
	public short rchild;
	
	
	public BinaryGrammarRule() {}
	
	
	public BinaryGrammarRule(short lhs, short lchild, short rchild) {
		this.lhs = lhs;
		this.lchild = lchild;
		this.rchild = rchild;
		this.type = LRBRULE;
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
		this.type = LRBRULE;
		initializeWeight();
	}
	
	
	public BinaryGrammarRule(short lhs, short lchild, short rchild, GaussianMixture weight) {
		this.lhs = lhs;
		this.lchild = lchild;
		this.rchild = rchild;
		this.weight = weight;
		this.type = LRBRULE;
	}
	
	
	private void initializeWeight() {
		weight = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set2 = new HashSet<GaussianDistribution>();
			set0.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			set1.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			set2.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, set0);
			map.put(Unit.LC, set1);
			map.put(Unit.RC, set2);
			weight.add(i, map);
		}
	}

	
	public GrammarRule copy() {
		BinaryGrammarRule rule = new BinaryGrammarRule(lhs, lchild, rchild);
		rule.weight = weight.copy(true);
		return rule;
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
		return "B-Rule [P: " + lhs +", LC: " + lchild + ", RC: " + rchild + "]";
	}
	
}
