package edu.shanghaitech.ai.nlp.lveg;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;


/**
 * @author Yanpeng Zhao
 *
 */
public class UnaryGrammarRule extends GrammarRule implements Comparable<Object> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * the ID of the right-hand side nonterminal
	 */
	protected short rhs;
	protected char type;
	
	
	public UnaryGrammarRule() {
		// TODO
	}
	
	
	/**
	 * This constructor returns an instance without initializing the weight.
	 * 
	 * @param lhs  id of the tag on the left hand side
	 * @param rhs  id of the tag on the right hand side 
	 * 
	 */
	public UnaryGrammarRule(short lhs, short rhs) {
		this.lhs = lhs;
		this.rhs = rhs;
		this.type = GENERAL;
		
	}
	
	
	public UnaryGrammarRule(short lhs, short rhs, char type) {
		this.lhs = lhs;
		this.rhs = rhs;
		this.type = type;
		initializeWeight(type);
	}
	
	
	public UnaryGrammarRule(short lhs, short rhs, GaussianMixture weight) {
		this.lhs = lhs;
		this.rhs = rhs;
		this.type = GENERAL;
		this.weight = weight;
	}
	
	
	private void initializeWeight(int type) {
		weight = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		
		switch (type) {
		case RHSPACE: // rules for the root since it does not have subtypes
			for (int i = 0; i < LVeGLearner.ncomponent; i++) {
				Set<GaussianDistribution> set = new HashSet<GaussianDistribution>();
				set.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
				weight.add(i, Unit.C, set);
			}
			break;
		case LHSPACE: // rules in the preterminal layer (discarded)
			for (int i = 0; i < LVeGLearner.ncomponent; i++) {
				Set<GaussianDistribution> set = new HashSet<GaussianDistribution>();
				set.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
				weight.add(i, Unit.P, set);
			}
			break;
		case GENERAL: // general unary rules 
			for (int i = 0; i < LVeGLearner.ncomponent; i++) {
				Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
				Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
				Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();
				set0.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
				set1.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
				map.put(Unit.P, set0);
				map.put(Unit.UC, set1);
				weight.add(i, map);
			}
			break;
		default:
			System.err.println("Not consistent with any unary rule type.");
		}
	}
	
	
	public GrammarRule copy() {
		UnaryGrammarRule rule = new UnaryGrammarRule(lhs, rhs);
		rule.weight = weight.copy(true);
		rule.type = type;
		return rule;
	}
	
	
	public short getRhs() {
		return rhs;
	}


	public void setRhs(short rhs) {
		this.rhs = rhs;
	}


	public char getType() {
		return type;
	}


	public void setType(char type) {
		this.type = type;
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
		
		if (o instanceof UnaryGrammarRule) {
			UnaryGrammarRule rule = (UnaryGrammarRule) o;
			if (lhs == rule.lhs && rhs == rule.rhs && type == rule.type) {
				return true;
			}
		}
		return false;
	}


	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		UnaryGrammarRule rule = (UnaryGrammarRule) o;
		if (lhs < rule.lhs) { return -1; }
		if (lhs > rule.lhs) { return 1; }
		if (rhs < rule.rhs) { return -1; }
		if (rhs > rule.rhs) { return 1; }
		return 0;
	}
	
	
	@Override
	public String toString() {
		return "U-Rule [P: " + lhs +", UC: " + rhs + ", T: " + (short) type + "]";
	}

}
