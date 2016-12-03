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
	protected double eval(List<Double> sample) {
		if (sample != null && sample.size() == dim) {
			double exps = 0.0, sinv = 1.0;
			for (int i = 0; i < dim; i++) {
				exps -= Math.pow(sample.get(i), 2) / 2;
				// CHECK Math.sqrt(Math.exp(x))
				sinv /= Math.exp(vars.get(i) / 2);
			}
			// TODO shall we truncate it to zero if it is very small?
			double value = Math.pow(2 * Math.PI, -dim / 2) * sinv * Math.exp(exps);
			return value;
		}
		logger.error("The sample is not valid.");
		return -1.0;
	}
	
	
	@Override
	protected double eval(List<Double> sample, boolean normal) { 
		if (!normal) {
			// sample = normalize(sample);
			double astd, norm, exps = 0.0, sinv = 1.0;
			for (int i = 0; i < dim; i++) {
				astd = Math.exp(vars.get(i) / 2);
				norm = (sample.get(i) - mus.get(i)) / astd;
				exps -= Math.pow(norm, 2) / 2;
				sinv /= astd;
			}
			double value = Math.pow(2 * Math.PI, -dim / 2) * sinv * Math.exp(exps);
			return value;
		}
		return eval(sample);
	}
	
	
	@Override
	protected void derivative(double factor, List<Double> sample, List<Double> grads, short isample) {
		if (sample != null && sample.size() == dim) {
			if (isample == 0) {
				grads.clear();
				for (int i = 0; i < dim * 2; i++) {
					grads.add(0.0);
				}
			}
			double sigma, mgrad, vgrad;
			for (int i = 0; i < dim; i++) {
				sigma = Math.exp(vars.get(i) / 2);
				mgrad = factor * sample.get(i) / sigma;
				// CHECK dw / dx = (dw / ds) * (ds / dx) = (factor * (point^2 - 1) / s) * ((1 / 2) * s), 
				// where s = sigma = std = exp(x / 2)
				vgrad = factor * (Math.pow(sample.get(i), 2) - 1) / 2;
				grads.set(i * 2, mgrad);
				grads.set(i * 2 + 1, vgrad);
			}
		}
	}
	
	
	@Override
	protected void update(double lr, List<Double> grads) {
		assert(grads.size() == 2 * dim);
		double mu, sigma;
		for (int i = 0; i < dim; i++) {
			mu = grads.get(i * 2) - lr * mus.get(i);
			sigma = grads.get(i * 2 + 1) - lr * vars.get(i);
			mus.set(i, mu);
			vars.set(i, sigma);
		}
	}
	
}
