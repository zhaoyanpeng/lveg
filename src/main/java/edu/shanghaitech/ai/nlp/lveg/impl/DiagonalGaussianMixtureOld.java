package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixtureOld;

/**
 * This one only differs from the GaussianMixture when in the need of creating the new specific instance.
 * 
 * @author Yanpeng Zhao
 *
 */
public class DiagonalGaussianMixtureOld extends GaussianMixtureOld {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1083077972374093199L;


	public DiagonalGaussianMixtureOld() {
		super();
	}
	
	
	public DiagonalGaussianMixtureOld(short ncomponent) {
		super();
		this.ncomponent = ncomponent;
		initialize();
	}
	
	
	public DiagonalGaussianMixtureOld(
			short ncomponent, List<Double> weights, List<Map<String, Set<GaussianDistribution>>> mixture) {
		super();
		this.ncomponent = ncomponent;
		this.weights = weights;
		this.mixture = mixture;
	}

	
	@Override
	public DiagonalGaussianMixtureOld copy(boolean deep) {
		DiagonalGaussianMixtureOld gm = new DiagonalGaussianMixtureOld();
		copy(gm, deep);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixtureOld replaceKeys(Map<String, String> keys) {
		DiagonalGaussianMixtureOld gm = new DiagonalGaussianMixtureOld();
		replaceKeys(gm, keys);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixtureOld replaceAllKeys(String newkey) {
		DiagonalGaussianMixtureOld gm = new DiagonalGaussianMixtureOld();
		replaceAllKeys(gm, newkey);
		return gm;
	}
	
	
	@Override
	public DiagonalGaussianMixtureOld multiply(GaussianMixtureOld multiplier) {
		DiagonalGaussianMixtureOld gm = new DiagonalGaussianMixtureOld();
		multiply(gm, multiplier);
		return gm;
	}
	
}
