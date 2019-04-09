package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
	
	@Override
	protected void initialize() {
		// pass
	}
	
	@Override
	public DiagonalGaussianMixture copy(boolean deep) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		copy(gm, deep);
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
		
		List<GaussianDistribution> gaussians = gausses.get(key);
		if (gaussians != null) { // w(ROOT->X) does not contain P portion when computing the outside score given the tree
			List<Double> integrals = new ArrayList<>(gaussians.size());
			for (GaussianDistribution gaussian : gaussians) {
				double logsum = Double.NEGATIVE_INFINITY, logcomp;
				for (SimpleView view : gm.spviews()) {
					logcomp = view.weight;
					if (view.gaussian != null) { 
						logcomp += gaussian.mulAndMarginalize(view.gaussian);
					} // treated as a constant when Gaussian is null
					logsum = FunUtil.logAdd(logsum, logcomp);
				}
				integrals.add(logsum);
			}
			des.marginalize(key, integrals);
		}
		return des;
	}
	
	@Override
	public GaussianMixture mul(GaussianMixture gm, GaussianMixture des, RuleUnit key) {
		if (des != null) { 
			des.clear(false); 
		} else {
			des = new DiagonalGaussianMixture();
		}
/*		
		for (Component comp : components) {
			double mw = comp.getWeight();
			GaussianDistribution gd = comp.squeeze(key);
			
			for (Component comp1 : gm.components()) {
				GaussianDistribution gd1 = comp1.squeeze(null);
				Map<Double, GaussianDistribution> prod = gd.mul(gd1);
				
				if (prod == null) { return null; } // do not throw exception here
				
				for (Map.Entry<Double, GaussianDistribution> entry : prod.entrySet()) {
					double mw0 = mw + comp1.getWeight() + entry.getKey();
					EnumMap<RuleUnit, Set<GaussianDistribution>> multivnd = copyExceptKey(comp.getMultivnd(), key);
					Set<GaussianDistribution> value = new HashSet<>();
					value.add(entry.getValue());
					multivnd.put(key, value);
					des.add(mw0, multivnd);
					break;
				}
			}
		}
*/		
		return des;
	}

	@Override
	public double mulAndMarginalize(EnumMap<RuleUnit, GaussianMixture> counts) {
		if (counts == null) { return Double.NEGATIVE_INFINITY; }
		double values = Double.NEGATIVE_INFINITY, value, vtmp;
		EnumMap<RuleUnit, List<Double>> caches = new EnumMap<>(RuleUnit.class);
		for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
			List<GaussianDistribution> gaussians = unit.getValue();
			List<Double> integrals = new ArrayList<>(gaussians.size());
			for (GaussianDistribution gaussian : gaussians) {
				value = Double.NEGATIVE_INFINITY;
				GaussianMixture gm = counts.get(unit.getKey());
				for (SimpleView view : gm.spviews()) {
					vtmp = gaussian.mulAndMarginalize(view.gaussian);
					vtmp += view.weight;
					value = FunUtil.logAdd(value, vtmp);
				}
				integrals.add(value);
			}
			caches.put(unit.getKey(), integrals);
		}
		values = eval(caches);
		return values;
	}
	
}
