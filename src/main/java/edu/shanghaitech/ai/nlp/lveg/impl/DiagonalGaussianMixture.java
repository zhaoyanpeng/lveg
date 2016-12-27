package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Map;
import java.util.Set;

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
		super();
	}
	
	
	public DiagonalGaussianMixture(short ncomponent) {
		super();
		this.ncomponent = ncomponent;
		initialize();
	}
	
	
	public DiagonalGaussianMixture(
			short ncomponent, List<Double> weights, List<Map<String, Set<GaussianDistribution>>> mixture) {
		super();
		this.ncomponent = ncomponent;
		this.weights = weights;
		this.mixture = mixture;
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
