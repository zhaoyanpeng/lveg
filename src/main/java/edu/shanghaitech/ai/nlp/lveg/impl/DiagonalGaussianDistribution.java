package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;

/**
 * Diagonal Gaussian distribution.
 * 
 * @author Yanpeng Zhao
 *
 */
public class DiagonalGaussianDistribution extends GaussianDistribution {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4484028334969182856L;


	public DiagonalGaussianDistribution() {
		super((short) 0);
	}
	
	
	public DiagonalGaussianDistribution(short ndimension) {
		super(ndimension);
		initialize();
	}
	
	
	public DiagonalGaussianDistribution(short ndimension, boolean init) {
		super(ndimension);
		if (init) { initialize(); }
	}
	
	
	public static DiagonalGaussianDistribution borrowObject(short ndimension) {
		GaussianDistribution obj = null;
		try {
			obj = defObjectPool.borrowObject(ndimension);
		} catch (Exception e) {
			logger.error("---------Borrow GD " + e + "\n");
			try {
				LVeGLearner.gaussPool.invalidateObject(ndimension, obj);
			} catch (Exception e1) {
				logger.error("---------Borrow GD(invalidate) " + e + "\n");
			}
			ndimension = ndimension == -1 ? defNdimension : ndimension;
			obj = new DiagonalGaussianDistribution(ndimension);
		}
		return (DiagonalGaussianDistribution) obj;
	}
	
	
	@Override
	public DiagonalGaussianDistribution copy() {
		DiagonalGaussianDistribution gd = new DiagonalGaussianDistribution();
//		DiagonalGaussianDistribution gd = borrowObject((short) 0); // POOL
		copy(gd);
		return gd;
	}
	
	
	@Override
	protected double eval(List<Double> sample, boolean normal) { 
		if (sample != null && sample.size() == dim) {
			// sample = normalize(sample);
			double astd, norm, exps = 0.0, sinv = 1.0;
			for (int i = 0; i < dim; i++) {
				astd = Math.exp(vars.get(i));
				norm = normal ? sample.get(i) : (sample.get(i) - mus.get(i)) / astd;
				exps -= Math.pow(norm, 2) / 2;
				sinv /= astd;
			}
			double value = Math.pow(2 * Math.PI, -dim / 2) * sinv * Math.exp(exps);
			return value;
		}
		logger.error("Invalid input sample for evaling the gaussian. sample: " + sample + ", this: " + this + "\n");
		return -1.0;
	}
	
	
	@Override
	protected void derivative(double factor, List<Double> sample, List<Double> grads, boolean cumulative, boolean normal) {
		if (sample != null && sample.size() == dim) {
			if (!cumulative) {
				grads.clear();
				for (int i = 0; i < dim * 2; i++) {
					grads.add(0.0);
				}
			}
			double sigma, point, mgrad, vgrad;
			for (int i = 0; i < dim; i++) {
				sigma = Math.exp(vars.get(i));
				point = normal ? sample.get(i) : (sample.get(i) - mus.get(i)) / sigma;
				mgrad = factor * point / sigma;
				// CHECK dw / dx = (dw / ds) * (ds / dx) = (factor * (point^2 - 1) / s) * (s), 
				// where s = sigma = std = exp(x)
				vgrad = factor * (Math.pow(point, 2) - 1);
				grads.set(i * 2, grads.get(i * 2) + mgrad);
				grads.set(i * 2 + 1, grads.get(i * 2 + 1) + vgrad);
			}
			return;
		}
		logger.error("Invalid input sample for taking derivative of the gaussian w.r.t mu & sigma.\n");
	}
	
}
