package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.EnumMap;
import java.util.HashSet;
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
	public enum RuleType {
		LRURULE(0), LHSPACE(1), RHSPACE(2), LRBRULE(3);
		private final int id;
		
		private RuleType(int id) {
			this.id = id;
		}
		
		public String id() {
			return String.valueOf(id);
		}
		
		@Override
		public String toString() {
			return String.valueOf(id);
		}
	}
	
	public enum RuleUnit {
		P, C, LC, RC, UC, RM
	}

	/**
	 * the ID of the left-hand side nonterminal
	 */
	public short lhs;
	public RuleType type;
	public GaussianMixture weight;
	
	/*
	public final static byte LRBRULE = 3; // left and right hand sides, binary rule
	public final static byte RHSPACE = 2; 
	public final static byte LHSPACE = 1; 
	public final static byte LRURULE = 0; // left and right hand sides, unary rule
	*/
	
	/*
	public static class Unit {
		public final static String P = "p";
		public final static String C = "c";
		public final static String LC = "lc";
		public final static String RC = "rc";
		public final static String UC = "uc";
		public final static String RM = "rm";
	}
	*/
	
	public GrammarRule() {
		// TODO
	}
	
	public abstract boolean isUnary();
	public abstract GrammarRule copy();
	public abstract void initializeWeight(RuleType type, short ncomponent, short ndim);
	
	
	public void addWeightComponent(RuleType type, short increment, short ndim) {
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
	
	
	public static GaussianMixture rndRuleWeight(RuleType type, short ncomponent, short ndim) {
		short defNcomp = ncomponent > 0 ? ncomponent : GaussianMixture.defNcomponent;
		short defNdim = ndim > 0 ? ndim : GaussianDistribution.defNdimension;
		GaussianMixture aweight = new DiagonalGaussianMixture(defNcomp);
		rndRuleWeight(type, defNcomp, defNdim, aweight);
		return aweight;
	}
	
	
	private static void rndRuleWeight(RuleType type, short ncomponent, short dim, GaussianMixture weight) {
		switch (type) {
		case RHSPACE: // rules for the root since it does not have subtypes
			for (int i = 0; i < ncomponent; i++) {
				Set<GaussianDistribution> set = new HashSet<>(1, 1);
				set.add(new DiagonalGaussianDistribution(dim));
				weight.add(i, RuleUnit.C, set);
			}
			break;
		case LHSPACE: // rules in the preterminal layer (discarded)
			for (int i = 0; i < ncomponent; i++) {
				Set<GaussianDistribution> set = new HashSet<>(1, 1);
				set.add(new DiagonalGaussianDistribution(dim));
				weight.add(i, RuleUnit.P, set);
			}
			break;
		case LRURULE: // general unary rules 
			for (int i = 0; i < ncomponent; i++) {
				EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
				Set<GaussianDistribution> set0 = new HashSet<>(1, 1);
				Set<GaussianDistribution> set1 = new HashSet<>(1, 1);
				set0.add(new DiagonalGaussianDistribution(dim));
				set1.add(new DiagonalGaussianDistribution(dim));
				map.put(RuleUnit.P, set0);
				map.put(RuleUnit.UC, set1);
				weight.add(i, map);
			}
			break;
		case LRBRULE: // general binary rules
			for (int i = 0; i < ncomponent; i++) {
				EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
				Set<GaussianDistribution> set0 = new HashSet<>(1, 1);
				Set<GaussianDistribution> set1 = new HashSet<>(1, 1);
				Set<GaussianDistribution> set2 = new HashSet<>(1, 1);
				set0.add(new DiagonalGaussianDistribution(dim));
				set1.add(new DiagonalGaussianDistribution(dim));
				set2.add(new DiagonalGaussianDistribution(dim));
				map.put(RuleUnit.P, set0);
				map.put(RuleUnit.LC, set1);
				map.put(RuleUnit.RC, set2);
				weight.add(i, map);
			}
			break;
		default:
			throw new RuntimeException("Not consistent with any grammar rule type. Type: " + type);
		}
	}	

	
	public RuleType getType() {
		return type;
	}


	public void setType(RuleType type) {
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
