package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;

public class GrammarRule implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public final static char RHSPACE = 2;
	public final static char LHSPACE = 1;
	public final static char GENERAL = 0;

	/**
	 * the ID of the left-hand side nonterminal
	 */
	protected short lhs;

	protected GaussianMixture weight;
	
	
	public GrammarRule() {
		// TODO
	}
	
	
	public static class Unit {
		public final static String P = "p";
		public final static String C = "c";
		public final static String LC = "lc";
		public final static String RC = "rc";
		public final static String UC = "uc";
	}
	
	
	public boolean isUnary() {
		return false;
	}
	
	
	public short getLhs() {
		return lhs;
	}
	
	
	public void setLhs(short lhs) {
		this.lhs = lhs;
	}

	
	public GaussianMixture getWeight() {
		return weight;
	}


	public void setWeight(GaussianMixture weight) {
		this.weight = weight;
	}
}
