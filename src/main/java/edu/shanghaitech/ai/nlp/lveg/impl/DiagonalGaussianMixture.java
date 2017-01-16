package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;

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
	
	
	public DiagonalGaussianMixture(
			short ncomponent, List<Double> weights, List<Map<String, Set<GaussianDistribution>>> mixture) {
		this();
		this.ncomponent = ncomponent;
		for (int i = 0; i < weights.size(); i++) {
			this.components.add(new Component((short) i, weights.get(i), mixture.get(i)));
		}
	}
	
	
	public static DiagonalGaussianMixture instance(short ncomponent) {
		GaussianMixture obj = null;
		try {
			obj = LVeGLearner.mogPool.borrowObject(ncomponent);
		} catch (Exception e) {
			// CHECK pool.invalidateObject(key, obj);
			logger.error("---------Borrow GM " + e + "\n");
			try {
				LVeGLearner.mogPool.invalidateObject(ncomponent, obj);
			} catch (Exception e1) {
				logger.error("---------Borrow GM(invalidate) " + e + "\n");
			}
//			obj = new DiagonalGaussianMixture(ncomponent);
		}
		return (DiagonalGaussianMixture) obj;
	}

	
	@Override
	public DiagonalGaussianMixture copy(boolean deep) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		copy(gm, deep);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture replaceKeys(Map<String, String> keys) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		replaceKeys(gm, keys);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixture replaceAllKeys(String newkey) {
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
	
}
