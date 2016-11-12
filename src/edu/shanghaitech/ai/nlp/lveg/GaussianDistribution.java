package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.shanghaitech.ai.nlp.util.MethodUtil;

/**
 * Gaussian distribution is defined as the diagonal one.
 * 
 * @author Yanpeng Zhao
 *
 */
public class GaussianDistribution implements Comparable<Object> {
	
	protected short dim;
	
	protected List<Double> mus;
	protected List<Double> sigmas;
	
	protected List<Double> mgrads;
	protected List<Double> sgrads;
	// points ~ N(0, 1)
	protected List<Double> sample;
	
	
	public GaussianDistribution() {
		this.dim = 0;
		this.mus = new ArrayList<Double>();
		this.sigmas = new ArrayList<Double>();
		this.mgrads = new ArrayList<Double>();
		this.sgrads = new ArrayList<Double>();
	}
	
	
	public GaussianDistribution(short dim) {
		this();
		this.dim = dim;
		initialize();
	}
	
	
	/**
	 * Memory allocation and initialization.
	 */
	private void initialize() {
		MethodUtil.randomInitList(mus, Double.class, dim, LVeGLearner.maxrandom, false);
		MethodUtil.randomInitList(sigmas, Double.class, dim, LVeGLearner.maxrandom, true);
	}
	
	
	/**
	 * Sample from the Gaussian distribution.
	 * 
	 * @param random random number generator
	 */
	protected void sample(Random random) {
		if (sample == null) {
			sample = new ArrayList<Double>(); 
		} else {
			sample.clear();
		}
		for (int i = 0; i < dim; i++) {
			sample.add(random.nextGaussian());
		}
	}
	
	
	/**
	 * Eval according to the sample and parameters.
	 * 
	 * @return
	 */
	protected double eval() {
		double exps = 0.0, sinv = 1.0;
		for (int i = 0; i < dim; i++) {
			exps -= Math.pow(sample.get(i), 2) / 2;
			sinv /= sigmas.get(i);
		}
		double value = Math.pow(2 * Math.PI, -dim / 2) * sinv * Math.exp(exps);
		return value;
	}
	
	
	/**
	 * Take the derivative of MoG with respect to the parameters (mu & sigma) of the component.
	 * 
	 * @param wgrad   derivative of MoG with respect to the weight of the component 
	 * @param weight  weight of the component of MoG
	 * @param nsample accumulate gradients (>0) or not (0)
	 */
	protected void derivative(double wgrad, double weight, int nsample) {
		if (nsample == 0) {
			mgrads.clear();
			sgrads.clear();
			for (int i = 0; i < dim; i++) {
				mgrads.add(0.0);
				sgrads.add(0.0);
			}
		}
		for (int i = 0; i < dim; i++) {
			double mgrad = weight * wgrad * sample.get(i) / sigmas.get(i);
			double sgrad = weight * wgrad * (Math.pow(sample.get(i), 2) - 1) / sigmas.get(i);
			mgrads.set(i, mgrads.get(i) + mgrad);
			sgrads.set(i, sgrads.get(i) + sgrad);
		}
	}
	
	
	/**
	 * Update parameters using the gradient.
	 * 
	 * @param learningRate learning rate
	 */
	protected void update(double learningRate, short nsample) {
		double mu, sigma;
		for (int i = 0; i < dim; i++) {
			mu = mus.get(i) - learningRate * mgrads.get(i) / nsample;
			sigma = sigmas.get(i) - learningRate * sgrads.get(i) / nsample;
			mus.set(i, mu);
			sigmas.set(i, sigma);
		}
	}
	
	
	/**
	 * Make a copy of the instance.
	 * 
	 * @return
	 */
	public GaussianDistribution copy() {
		GaussianDistribution gd = new GaussianDistribution();
		gd.dim = dim;
		for (int i = 0; i < dim; i++) {
			gd.mus.add(mus.get(i));
			gd.sigmas.add(sigmas.get(i));
		}
		return gd;
	}
	
	
	/**
	 * Memory clean.
	 */
	public void clear() {
		this.dim = 0;
		this.mus.clear();
		this.sigmas.clear();
	}
	
	
	@Override
	public int hashCode() {
		return dim ^ mus.hashCode() ^ sigmas.hashCode();
	}
	
	
	@Override
	public boolean equals(Object o) {
		if (this == o) { return true; }
		
		if (o instanceof GaussianDistribution) {
			GaussianDistribution gd = (GaussianDistribution) o;
			if (dim == gd.dim && mus.equals(gd.mus) && sigmas.equals(gd.sigmas)) {
				return true;
			}
		}
		return false;
	}
	
	
	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		GaussianDistribution gd = (GaussianDistribution) o;
		if (dim < gd.dim) { return -1; }
		if (dim > gd.dim) { return 1;  }
		/*
		if (dim > 0 && mus.get(0) < gd.mus.get(0)) { return -1; }
		if (dim > 0 && mus.get(0) > gd.mus.get(0)) { return  1; }
		if (dim > 0 && sigmas.get(0) < gd.sigmas.get(0)) { return -1; }
		if (dim > 0 && sigmas.get(0) > gd.sigmas.get(0)) { return  1; }
		*/
		if (mus.equals(gd.mus) && sigmas.equals(gd.sigmas)) { return 0; }
		return -1;
	}


	@Override
	public String toString() {
		return "GD [dim=" + dim + ", mus=" + MethodUtil.double2str(mus, LVeGLearner.precision, -1) + 
				", sigmas=" + MethodUtil.double2str(sigmas, LVeGLearner.precision, -1) + "]";
	}

}
