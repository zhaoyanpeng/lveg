package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Map;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;

public class GeneralGaussianDistribution extends GaussianDistribution {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3855732607147998162L;
	
	public GeneralGaussianDistribution() {
		super((short) 0);
	}
	
	public GeneralGaussianDistribution(short ndimension) {
		super(ndimension);
		initialize();
	}
	
	public GeneralGaussianDistribution(short ndimension, boolean init) {
		super(ndimension);
		if (init) { initialize(); }
	}
	
	@Override
	public GaussianDistribution instance(short ndimension, boolean init) {
		return new GeneralGaussianDistribution(ndimension, init);
	}
	
	protected double eval(List<Double> sample, boolean normal) {
		 return 0.0;
	}

	@Override
	public double mulAndMarginalize(GaussianDistribution gd) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	protected void initialize() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Map<Double, GaussianDistribution> mul(GaussianDistribution gd) {
		// TODO Auto-generated method stub
		return null;
	}

}
