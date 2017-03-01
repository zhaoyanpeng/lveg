package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
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
			short ncomponent, List<Double> weights, List<Map<String, Set<GaussianDistribution>>> mixture) {
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
				LVeGLearner.mogPool.invalidateObject(ncomponent, obj);
			} catch (Exception e1) {
				logger.error("---------Borrow GM(invalidate) " + e + "\n");
			}
			ncomponent = ncomponent == -1 ? defNcomponent : ncomponent;
			obj = new DiagonalGaussianMixture(ncomponent);
		}
		return (DiagonalGaussianMixture) obj;
	}
	
	@Override
	public DiagonalGaussianMixture copy(boolean deep) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
//		DiagonalGaussianMixture gm = DiagonalGaussianMixture.borrowObject((short) 0); // POOL
		copy(gm, deep);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture replaceKeys(Map<String, String> keys) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
//		DiagonalGaussianMixture gm = DiagonalGaussianMixture.borrowObject((short) 0); // POOL
		replaceKeys(gm, keys);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture replaceAllKeys(String newkey) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
//		DiagonalGaussianMixture gm = DiagonalGaussianMixture.borrowObject((short) 0); // POOL
		replaceAllKeys(gm, newkey);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture multiply(GaussianMixture multiplier) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
//		DiagonalGaussianMixture gm = DiagonalGaussianMixture.borrowObject((short) 0); // POOL
		multiply(gm, multiplier);
		return gm;
	}

	@Override
	public double mulAndMarginalize(Map<String, GaussianMixture> counts) {
		if (counts == null) { return Double.NEGATIVE_INFINITY; }
		double values = Double.NEGATIVE_INFINITY;
		for (Component comp : components) {
			double value = 0.0, vtmp = 0.0;
			for (Entry<String, Set<GaussianDistribution>> node : comp.getMultivnd().entrySet()) {
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
