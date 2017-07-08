package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

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
		
		public int id() {
			return id;
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
	
	public GrammarRule() {
		// TODO
	}
	
	public abstract boolean isUnary();
	public abstract GrammarRule copy();
	public abstract void initializeWeight(RuleType type, short ncomponent, short ndim);
	
	
	public static GaussianMixture rndRuleWeight(RuleType type, short ncomponent, short ndim) {
		short defNcomp = ncomponent > 0 ? ncomponent : GaussianMixture.defNcomponent;
		short defNdim = ndim > 0 ? ndim : GaussianDistribution.defNdimension;
		GaussianMixture aweight = new DiagonalGaussianMixture(defNcomp);
		rndRuleWeight(type, defNcomp, defNdim, aweight);
		aweight.setBinding(type); // CHECK
		aweight.buildSimpleView();
		return aweight;
	}
	
	
	private static void rndRuleWeight(RuleType type, short ncomponent, short dim, GaussianMixture weight) {
		int ncomp = 0;
		switch (type) {
		case RHSPACE: { // rules for the root since it does not have subtypes
			List<GaussianDistribution> list = new ArrayList<>(5);
			for (int i = 0; i < ncomponent; i++) {
				list.add(new DiagonalGaussianDistribution(dim));
			}
			weight.add(RuleUnit.C, list);
			ncomp = ncomponent;
			break;
		}
		case LHSPACE: { // rules in the preterminal layer (discarded)
			List<GaussianDistribution> list = new ArrayList<>(5);
			for (int i = 0; i < ncomponent; i++) {
				list.add(new DiagonalGaussianDistribution(dim));
			}
			weight.add(RuleUnit.P, list);
			ncomp = ncomponent;
			break;
		}
		case LRURULE: { // general unary rules 
			List<GaussianDistribution> list0 = new ArrayList<>(5);
			List<GaussianDistribution> list1 = new ArrayList<>(5);
			for (int i = 0; i < ncomponent; i++) {
				list0.add(new DiagonalGaussianDistribution(dim));
				list1.add(new DiagonalGaussianDistribution(dim));
			}
			weight.add(RuleUnit.P, list0);
			weight.add(RuleUnit.UC, list1);
			ncomp = ncomponent * ncomponent;
			break;
		}
		case LRBRULE: { // general binary rules
			List<GaussianDistribution> list0 = new ArrayList<>(5);
			List<GaussianDistribution> list1 = new ArrayList<>(5);
			List<GaussianDistribution> list2 = new ArrayList<>(5);
			for (int i = 0; i < ncomponent; i++) {
				list0.add(new DiagonalGaussianDistribution(dim));
				list1.add(new DiagonalGaussianDistribution(dim));
				list2.add(new DiagonalGaussianDistribution(dim));
			}
			weight.add(RuleUnit.P, list0);
			weight.add(RuleUnit.LC, list1);
			weight.add(RuleUnit.RC, list2);
			ncomp = ncomponent * ncomponent * ncomponent;
			break;
		}
		default:
			throw new RuntimeException("Not consistent with any grammar rule type. Type: " + type);
		}
		weight.initMixingW(ncomp); // CHECK
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
