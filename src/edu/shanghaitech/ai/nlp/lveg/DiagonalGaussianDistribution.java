package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

/**
 * Diagonal Gaussian distribution.
 * 
 * @author Yanpeng Zhao
 *
 */
public class DiagonalGaussianDistribution extends GaussianDistribution {
	
	
	public DiagonalGaussianDistribution() {
		super();
	}
	
	
	public DiagonalGaussianDistribution(short dim) {
		super();
		this.dim = dim;
		initialize();
	}
	
	
	@Override
	public DiagonalGaussianDistribution copy() {
		DiagonalGaussianDistribution gd = new DiagonalGaussianDistribution();
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
