package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.Map;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;

public class GeneralGaussianMixture extends GaussianMixture {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6052750581733945498L;
	
	public GeneralGaussianMixture() {
		super((short) 0);
	}
	
	public GeneralGaussianMixture(short ncomponent) {
		super(ncomponent);
		initialize();
	}
	
	public GeneralGaussianMixture(short ncomponent, boolean init) {
		super(ncomponent);
		if (init) { initialize(); }
	}

	@Override
	public GaussianMixture instance(short ncomponent, boolean init) {
		return new GeneralGaussianMixture(ncomponent, init);
	}

	@Override
	public double mulAndMarginalize(Map<String, GaussianMixture> counts) {
		// TODO Auto-generated method stub
		return 0;
	}

}
