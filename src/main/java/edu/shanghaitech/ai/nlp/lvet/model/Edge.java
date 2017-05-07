package edu.shanghaitech.ai.nlp.lvet.model;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.Unit;
import edu.shanghaitech.ai.nlp.lvet.LVeTTrainer;

public abstract class Edge implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 92060200995406282L;
	public int src;
	public int des;
	public EdgeType type;
	public GaussianMixture weight;
	
	public static enum EdgeType {
		LRURULE, LHSPACE
	}
	public static enum NodeType {
		P, C, UC
	}
	
	
	public Edge(int src, int des, EdgeType type) {
		this.src = src;
		this.des = des;
		this.type = type;
	}
	
	
	public abstract void initializeWeight(EdgeType type, short ncomponent, short ndim);
	
	
	public void addWeightComponent(EdgeType type, short increment, short ndim) {
		short defNcomp = increment > 0 ? increment : LVeTTrainer.ncomp;
		short defNdim = ndim > 0 ? ndim : LVeTTrainer.ndim;
		if (weight == null) {
			weight = rndRuleWeight(type, defNcomp, defNdim);
		} else {
			GaussianMixture aweight = new DiagonalGaussianMixture(defNcomp);
			rndRuleWeight(type, defNcomp, defNdim, aweight);
			weight.add(aweight, false);
			weight.rectifyId(); // required
		}
	}
	
	
	public static GaussianMixture rndRuleWeight(EdgeType type, short ncomponent, short ndim) {
		short defNcomp = ncomponent > 0 ? ncomponent : LVeTTrainer.ncomp;
		short defNdim = ndim > 0 ? ndim : LVeTTrainer.ndim;
		GaussianMixture aweight = new DiagonalGaussianMixture(defNcomp);
		rndRuleWeight(type, defNcomp, defNdim, aweight);
		return aweight;
	}
	
	
	private static void rndRuleWeight(EdgeType type, short ncomponent, short dim, GaussianMixture weight) {
		switch (type) {
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
		default:
			throw new RuntimeException("Not consistent with any grammar rule type. Type: " + type);
		}
	}	
	
}
