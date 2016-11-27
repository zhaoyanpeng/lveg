package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.shanghaitech.ai.nlp.util.MethodUtil;

/**
 * Gaussian distribution. We may define diagonal or general Gaussian distribution. Different 
 * Gaussian distributions only differ when we eval their values or take directives.
 * 
 * @author Yanpeng Zhao
 *
 */
public class GaussianDistribution implements Comparable<Object> {
	/**
	 * We may need the hash set be able to hold the gaussians that 
	 * are the same but the ids, which is just the future feature.
	 */
	protected char id;
	protected short dim;
	
	/**
	 * Covariances must be positive. We represent them in exponential 
	 * form, and the real variances (for the diagonal) should be read 
	 * as Math.exp(variances).
	 */
	protected List<Double> vars;
	protected List<Double> mus;
	
	protected List<Double> mgrads;
	protected List<Double> vgrads;
	// points ~ N(0, 1)
	protected List<Double> sample;
	
	
	public GaussianDistribution() {
		this.id = 0;
		this.dim = 0;
		this.mus = new ArrayList<Double>();
		this.vars = new ArrayList<Double>();
		this.mgrads = new ArrayList<Double>(0);
		this.vgrads = new ArrayList<Double>(0);
	}
	
	
	/**
	 * Memory allocation and initialization.
	 */
	protected void initialize() {
		MethodUtil.randomInitList(mus, Double.class, dim, LVeGLearner.maxrandom, false);
		MethodUtil.randomInitList(vars, Double.class, dim, LVeGLearner.maxrandom, true);
	}
	
	
	/**
	 * Eval according to the sample and parameters (mu & sigma).
	 * 
	 * @return
	 */
	protected double eval() { return -0.0; }
	
	
	/**
	 * Take the derivative of MoG with respect to the parameters (mu & sigma) of the component.
	 * 
	 * @param factor  
	 * @param nsample accumulate gradients (>0) or not (0)
	 */
	protected void derivative(double factor, int nsample) {}
	
	
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
	 * Update parameters using the gradient.
	 * 
	 * @param learningRate learning rate
	 */
	protected void update(double learningRate, short nsample) {
		double mu, sigma;
		for (int i = 0; i < dim; i++) {
			mu = mus.get(i) - learningRate * mgrads.get(i) / nsample;
			sigma = vars.get(i) - learningRate * vgrads.get(i) / nsample;
			mus.set(i, mu);
			vars.set(i, sigma);
		}
	}
	
	
	/**
	 * Make a copy of the instance.
	 * 
	 * @return
	 */
	public GaussianDistribution copy() {
		GaussianDistribution gd = new GaussianDistribution();
		gd.id = id;
		gd.dim = dim;
		for (int i = 0; i < dim; i++) {
			gd.mus.add(mus.get(i));
			gd.vars.add(vars.get(i));
		}
		return gd;
	}
	
	
	/**
	 * Memory clean.
	 */
	public void clear() {
		this.dim = 0;
		this.mus.clear();
		this.vars.clear();
	}
	
	
	@Override
	public int hashCode() {
		return dim ^ mus.hashCode() ^ vars.hashCode();
	}
	
	
	@Override
	public boolean equals(Object o) {
		if (this == o) { return true; }
		
		if (o instanceof GaussianDistribution) {
			GaussianDistribution gd = (GaussianDistribution) o;
			if (id == gd.id && dim == gd.dim && mus.equals(gd.mus) && vars.equals(gd.vars)) {
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
		if (dim > 0 && vars.get(0) < gd.vars.get(0)) { return -1; }
		if (dim > 0 && vars.get(0) > gd.vars.get(0)) { return  1; }
		*/
		if (mus.equals(gd.mus) && vars.equals(gd.vars)) { return 0; }
		return -1;
	}


	@Override
	public String toString() {
		return "GD [dim=" + dim + ", mus=" + MethodUtil.double2str(mus, LVeGLearner.precision, -1, false) + 
				", vars=" + MethodUtil.double2str(vars, LVeGLearner.precision, -1, true) + "]";
	}

}
