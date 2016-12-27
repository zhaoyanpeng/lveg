package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;

/**
 * @author Yanpeng Zhao
 *
 */
public abstract class GrammarRule implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1464935410648668068L;
	public final static byte LRBRULE = 3; // left and right hand sides, binary rule
	public final static byte RHSPACE = 2; 
	public final static byte LHSPACE = 1; 
	public final static byte LRURULE = 0; // left and right hand sides, unary rule

	/**
	 * the ID of the left-hand side nonterminal
	 */
	public short lhs;
	public byte type;
	public GaussianMixture weight;
	
	
	public GrammarRule() {
		// TODO
	}
	
	
	public static class Unit {
		public final static String P = "p";
		public final static String C = "c";
		public final static String LC = "lc";
		public final static String RC = "rc";
		public final static String UC = "uc";
		public final static String RM = "rm";
	}
	
	
	public abstract boolean isUnary();
	public abstract GrammarRule copy();
	
	
	public byte getType() {
		return type;
	}


	public void setType(byte type) {
		this.type = type;
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
