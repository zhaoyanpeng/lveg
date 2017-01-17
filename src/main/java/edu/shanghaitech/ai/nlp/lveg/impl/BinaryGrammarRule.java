package edu.shanghaitech.ai.nlp.lveg.impl;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;

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
	
	
	public BinaryGrammarRule(short lhs, short lchild, short rchild) {
		this.lhs = lhs;
		this.lchild = lchild;
		this.rchild = rchild;
		this.type = LRBRULE;
	}
	
	
	public BinaryGrammarRule(short lhs, short lchild, short rchild, boolean init) {
		this(lhs, lchild, rchild);
		if (init) { initializeWeight(GrammarRule.LRBRULE); }
	}
	
	
	public BinaryGrammarRule(short lhs, short lchild, short rchild, GaussianMixture weight) {
		this(lhs, lchild, rchild);
		this.weight = weight;
	}
	
	
	public void initializeWeight(byte type) {
		weight = rndRuleWeight(GrammarRule.LRBRULE);
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
			if (lhs == rule.lhs && lchild == rule.lchild && rchild == rule.rchild && type == rule.type) {
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
