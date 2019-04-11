package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.ArrayList;
import java.util.EnumMap;
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
		if (des == null) {
			des = new DiagonalGaussianMixture();
		}
		// assuming multiplication of two single-variated Gaussians, thus spviews of both which exist.
		if (gausses.size() != 1) { return null; }
		
		List<Double> aweights = new ArrayList<>();
		List<GaussianDistribution> agausses = new ArrayList<>();
		for (SimpleView view : spviews) {
			double mw = view.weight, aweight = 0;
			GaussianDistribution gd = view.gaussian;
			for (SimpleView aview : gm.spviews()) {
				Map<Double, GaussianDistribution> prod = gd.mul(aview.gaussian);
				
				if (prod == null) { return null; } // do not throw exception here
				
				for (Map.Entry<Double, GaussianDistribution> entry : prod.entrySet()) {
					agausses.add(entry.getValue());
					aweight = mw + aview.weight + entry.getKey();
					aweights.add(aweight);
					break;
				}
			}
		}
		if (des.reset(aweights, agausses, key, null)) {
			return des;
		}
		return null;
	}

	@Override
	public double mulAndMarginalize(EnumMap<RuleUnit, GaussianMixture> counts) {
		if (counts == null) { return Double.NEGATIVE_INFINITY; }
		double values = Double.NEGATIVE_INFINITY, value, vtmp;
		// special case where 'this' may come from in-/out-side scores and thus not a valid rule weight
		if (binding == null) { // have to be set manually before entering here
			GaussianMixture gm = counts.get(RuleUnit.P); 
			for (SimpleView view : spviews) {
				GaussianDistribution gd = view.gaussian;
				for (SimpleView aview : gm.spviews()) {
					vtmp = gd.mulAndMarginalize(view.gaussian);
					vtmp = vtmp + view.weight + aview.weight; // in the same component
					values = FunUtil.logAdd(values, vtmp); // among different components
				}
			}
			return values;
		}
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
