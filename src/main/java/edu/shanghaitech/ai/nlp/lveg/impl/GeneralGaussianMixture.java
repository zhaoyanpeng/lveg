package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.EnumMap;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;

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
	public double mulAndMarginalize(EnumMap<RuleUnit, GaussianMixture> counts) {
		return 0;
	}

	@Override
	protected void initialize() {
	}

	@Override
	public GaussianMixture mulAndMarginalize(GaussianMixture gm, GaussianMixture des, RuleUnit key, boolean deep) {
		return null;
	}

	@Override
	public GaussianMixture mul(GaussianMixture gm, GaussianMixture des, RuleUnit key) {
		// TODO Auto-generated method stub
		return null;
	}

}
