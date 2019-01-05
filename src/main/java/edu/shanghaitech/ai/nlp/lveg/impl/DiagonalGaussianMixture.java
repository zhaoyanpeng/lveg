package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.EnumMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.util.FunUtil;

/**
 * This one only differs from the GaussianMixture when in the need of creating the new specific instance.
 * 
 * @author Yanpeng Zhao
 *
 */
public class DiagonalGaussianMixture extends GaussianMixture {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1083077972374093199L;

	public DiagonalGaussianMixture() {
		super((short) 0);
	}
	
	public DiagonalGaussianMixture(short ncomponent) {
		super(ncomponent);
		initialize();
	}
	
	public DiagonalGaussianMixture(short ncomponent, boolean init) {
		super(ncomponent);
		if (init) { initialize(); }
	}
	
	@Override
	public GaussianMixture instance(short ncomponent, boolean init) {
		return new DiagonalGaussianMixture(ncomponent, init);
	}
	
	public DiagonalGaussianMixture(
			short ncomponent, List<Double> weights, List<EnumMap<RuleUnit, Set<GaussianDistribution>>> mixture) {
		this();
		this.ncomponent = ncomponent;
		for (int i = 0; i < weights.size(); i++) {
			this.components.add(new Component((short) i, weights.get(i), mixture.get(i)));
		}
	}
	
	
	public static DiagonalGaussianMixture borrowObject(short ncomponent) {
		GaussianMixture obj = null;
		try {
			obj = defObjectPool.borrowObject(ncomponent);
		} catch (Exception e) {
			logger.error("---------Borrow GM " + e + "\n");
			try {
				LVeGTrainer.mogPool.invalidateObject(ncomponent, obj);
			} catch (Exception e1) {
				logger.error("---------Borrow GM(invalidate) " + e + "\n");
			}
			ncomponent = ncomponent == -1 ? defNcomponent : ncomponent;
			obj = new DiagonalGaussianMixture(ncomponent);
		}
		return (DiagonalGaussianMixture) obj;
	}
	
	@Override
	protected void initialize() {
		for (int i = 0; i < ncomponent; i++) {
			double weight = (defRnd.nextDouble() - defNegWRatio) * defMaxmw;
			EnumMap<RuleUnit, Set<GaussianDistribution>> multivnd = new EnumMap<>(RuleUnit.class);
			 weight = /*-0.69314718056*/ 0; // mixing weight 0.5, 1, 2
			components.add(new Component((short) i, weight, multivnd));
		}
	}
	
	@Override
	public DiagonalGaussianMixture copy(boolean deep) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		copy(gm, deep);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture replaceKeys(EnumMap<RuleUnit, RuleUnit> keys) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		replaceKeys(gm, keys);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture replaceAllKeys(RuleUnit newkey) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		replaceAllKeys(gm, newkey);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture multiply(GaussianMixture multiplier) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		multiply(gm, multiplier);
		return gm;
	}
	
	@Override
	public GaussianMixture mulAndMarginalize(GaussianMixture gm, GaussianMixture des, RuleUnit key, boolean deep) {
		// 'des' is exactly the same as 'this' when deep is false
		if (des != null) { // placeholder
			if (deep) { 
				des.clear(false); 
				copy(des, true);
			} 
		} else { // new memo space
			des = deep ? copy(true) : this;
		}
		// the following is the general case
		for (Component comp : des.components()) {
			double logsum = Double.NEGATIVE_INFINITY;
			GaussianDistribution gd = comp.squeeze(key);
			// w(ROOT->X) has no P portion in computing outside score
			if (gd == null) { continue; } 
			for (Component comp1 : gm.components()) {
				GaussianDistribution gd1 = comp1.squeeze(null);
				double logcomp = comp1.getWeight() + gd.mulAndMarginalize(gd1);
				logsum = FunUtil.logAdd(logsum, logcomp);
			}
			comp.setWeight(comp.getWeight() + logsum);
			comp.getMultivnd().remove(key);
		}
		return des;
	}

	@Override
	public double mulAndMarginalize(EnumMap<RuleUnit, GaussianMixture> counts) {
		if (counts == null) { return Double.NEGATIVE_INFINITY; }
		double values = Double.NEGATIVE_INFINITY;
		for (Component comp : components) {
			double value = 0.0, vtmp = 0.0;
			for (Entry<RuleUnit, Set<GaussianDistribution>> node : comp.getMultivnd().entrySet()) {
				vtmp = 0.0;
				GaussianMixture gm = counts.get(node.getKey()); 
				for (GaussianDistribution gd : node.getValue()) {
					vtmp = mulAndMarginalize(gm, gd); // head (tail) variable & outside (inside) score 
					break;
				}
				value += vtmp;
			}
			value += comp.getWeight();
			values = FunUtil.logAdd(values, value);
		}
		return values;
	}
	
	
	public static double mulAndMarginalize(GaussianMixture gm, GaussianDistribution gd) {
		double value = Double.NEGATIVE_INFINITY, vtmp;
		for (Component comp : gm.components()) {
			GaussianDistribution ios = comp.squeeze(null);
			vtmp = gd.mulAndMarginalize(ios);
			vtmp += comp.getWeight();
			value = FunUtil.logAdd(value, vtmp);
		}
		return value;
	}
	
}
