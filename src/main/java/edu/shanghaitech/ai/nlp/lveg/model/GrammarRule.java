package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;

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
	public abstract void initializeWeight(byte type, short ncomponent, short ndim);
	
	
	public void addWeightComponent(byte type, short increment, short ndim) {
		short defNcomp = increment > 0 ? increment : GaussianMixture.defNcomponent;
		short defNdim = ndim > 0 ? ndim : GaussianDistribution.defNdimension;
		if (weight == null) {
			weight = rndRuleWeight(type, defNcomp, defNdim);
		} else {
			GaussianMixture aweight = new DiagonalGaussianMixture(defNcomp);
			rndRuleWeight(type, defNcomp, defNdim, aweight);
			weight.add(aweight, false);
			weight.rectifyId(); // required
		}
	}
	
	
	public static GaussianMixture rndRuleWeight(byte type, short ncomponent, short ndim) {
		short defNcomp = ncomponent > 0 ? ncomponent : GaussianMixture.defNcomponent;
		short defNdim = ndim > 0 ? ndim : GaussianDistribution.defNdimension;
		GaussianMixture aweight = new DiagonalGaussianMixture(defNcomp);
		rndRuleWeight(type, defNcomp, defNdim, aweight);
		return aweight;
	}
	
	
	private static void rndRuleWeight(byte type, short ncomponent, short dim, GaussianMixture weight) {
		switch (type) {
		case RHSPACE: // rules for the root since it does not have subtypes
			for (int i = 0; i < ncomponent; i++) {
				Set<GaussianDistribution> set = new HashSet<GaussianDistribution>(1, 1);
				set.add(new DiagonalGaussianDistribution(dim));
				weight.add(i, Unit.C, set);
			}
			break;
		case LHSPACE: // rules in the preterminal layer (discarded)
			for (int i = 0; i < ncomponent; i++) {
				Set<GaussianDistribution> set = new HashSet<GaussianDistribution>(1, 1);
				set.add(new DiagonalGaussianDistribution(dim));
				weight.add(i, Unit.P, set);
			}
			break;
		case LRURULE: // general unary rules 
			for (int i = 0; i < ncomponent; i++) {
				Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>(2, 1);
				Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>(1, 1);
				Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>(1, 1);
				set0.add(new DiagonalGaussianDistribution(dim));
				set1.add(new DiagonalGaussianDistribution(dim));
				map.put(Unit.P, set0);
				map.put(Unit.UC, set1);
				weight.add(i, map);
			}
			break;
		case LRBRULE: // general binary rules
			for (int i = 0; i < ncomponent; i++) {
				Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>(3, 1);
				Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>(1, 1);
				Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>(1, 1);
				Set<GaussianDistribution> set2 = new HashSet<GaussianDistribution>(1, 1);
				set0.add(new DiagonalGaussianDistribution(dim));
				set1.add(new DiagonalGaussianDistribution(dim));
				set2.add(new DiagonalGaussianDistribution(dim));
				map.put(Unit.P, set0);
				map.put(Unit.LC, set1);
				map.put(Unit.RC, set2);
				weight.add(i, map);
			}
			break;
		default:
			throw new RuntimeException("Not consistent with any grammar rule type. Type: " + type);
		}
	}	

	
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
