package edu.shanghaitech.ai.nlp.lveg;

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
	protected double eval() {
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
	
	
	@Override
	protected void derivative(double factor, int nsample) {
		if (nsample == 0) {
			mgrads.clear();
			vgrads.clear();
			for (int i = 0; i < dim; i++) {
				mgrads.add(0.0);
				vgrads.add(0.0);
			}
		}
		for (int i = 0; i < dim; i++) {
			double sigma = Math.exp(vars.get(i) / 2);
			double mgrad = factor * sample.get(i) / sigma;
			// CHECK dw / dx = (dw / ds) * (ds / dx) = (factor * (point^2 - 1) / s) * ((1 / 2) * s) 
			double vgrad = factor * (Math.pow(sample.get(i), 2) - 1) / 2;
			mgrads.set(i, mgrads.get(i) + mgrad);
			vgrads.set(i, vgrads.get(i) + vgrad);
		}
	}

}
