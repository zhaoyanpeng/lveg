package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * Gaussian distribution. We may define diagonal or general Gaussian distribution. Different 
 * Gaussian distributions only differ when we eval their values or take directives.
 * 
 * @author Yanpeng Zhao
 *
 */
public class GaussianDistribution extends Recorder implements Comparable<Object>, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5123783591853232548L;
	/**
	 * We may need the hash set be able to hold the gaussians that 
	 * are the same but the ids, which is just the future feature.
	 */
	protected char id;
	protected short dim;
	
	/**
	 * Covariances must be positive. We represent them in exponential 
	 * form, and the real variances (for the diagonal) should be read 
	 * as Math.exp(2 * variances), and standard variance is Math.exp(
	 * variances).
	 */
	protected List<Double> vars;
	protected List<Double> mus;
	
	
	public GaussianDistribution() {
		this.id = 0;
		this.dim = 0;
		this.mus = new ArrayList<Double>();
		this.vars = new ArrayList<Double>();
	}
	
	
	/**
	 * Memory allocation and initialization.
	 */
	protected void initialize() {
		MethodUtil.randomInitList(LVeGLearner.random, mus, Double.class, dim, LVeGLearner.maxrandom, LVeGLearner.nratio, false, true);
		MethodUtil.randomInitList(LVeGLearner.random, vars, Double.class, dim, LVeGLearner.maxrandom, LVeGLearner.nratio, false, true);
	}
	
	
	/**
	 * Make a copy of the instance.
	 * 
	 * @return
	 */
	public GaussianDistribution copy() { return null; }
	
	
	/**
	 * Make a copy of the instance.
	 * 
	 * @param des a placeholder of GaussianDistribution
	 */
	public void copy(GaussianDistribution des) {
		des.id = id;
		des.dim = dim;
		for (int i = 0; i < dim; i++) {
			des.mus.add(mus.get(i));
			des.vars.add(vars.get(i));
		}
	}
	
	
	/**
	 * Eval the gaussian distribution using the given sample
	 * 
	 * @param sample the sample 
	 * @param normal whether the sample is from N(0, 1) (true) or from this gaussian (false).
	 * @return
	 */
	protected double eval(List<Double> sample, boolean normal) { return -0.0; } 
	
	
	/**
	 * @param sample normalize the non-normal sample 
	 * @return
	 */
	protected List<Double> normalize(List<Double> sample) {
		List<Double> list = new ArrayList<Double>();
		for (int i = 0; i < dim; i++) {
			list.add((sample.get(i) - mus.get(i)) / Math.exp(vars.get(i)));
		}
		return list;
	}
	
	
	/**
	 * Take the derivative of gaussian distribution with respect to the parameters (mu & sigma).
	 * 
	 * @param factor  dRuleWeight * weight * dMixingWeight
	 * @param sample  the sample from this gaussian distribution
	 * @param cumulative accumulate the gradients (true) or not (false)
	 * @param grads   gradients container
	 * @param isample index of the sample
	 */
	protected void derivative(double factor, List<Double> sample, List<Double> grads, boolean cumulative, boolean normal) {}
	
	
	/**
	 * Sample from N(0, 1), and then restore the real sample according to the parameters of the gaussian distribution. 
	 * 
	 * @param slice placeholder, the sample sampled from N(0, 1)
	 * @param truth placeholder, the sample from this gaussian distribution
	 * @param rnd   random
	 */
	protected void sample(List<Double> slice, List<Double> truth, Random rnd) {
		double real, norm;
		slice.clear();
		truth.clear();
//		synchronized (rnd) {
			for (int i = 0; i < dim; i++) {
//				norm = rnd.nextGaussian();
				norm = ThreadLocalRandom.current().nextGaussian();
				real = norm * Math.exp(vars.get(i)) + mus.get(i);
				slice.add(norm);
				truth.add(real);
			}
//		}
	}
	
	
	/**
	 * Restore the real sample from the sample that is sampled from the standard normal distribution.
	 * 
	 * @param sample the sample sampled from N(0, 1)
	 * @param truth  the sample from this gaussian distribution
	 */
	protected void restoreSample(List<Double> sample, List<Double> truth) {
		assert(sample.size() == dim);
		double real;
		truth.clear();
		for (int i = 0; i < dim; i++) {
			// CHECK std = Math.exp(var)
			real = sample.get(i) * Math.exp(vars.get(i)) + mus.get(i);
			truth.add(real);
		}
	}
	
	
	/**
	 * Update parameters using the gradients.
	 * 
	 * @param grads gradients
	 */
	protected void update(List<Double> grads, int nsample) {
		if (grads.size() == 0) { 
//			logger.warn("No need to update because no gradients could be applied.");
			return; 
		}
		double mgrad, vgrad, mu, var;
		for (int i = 0; i < dim; i++) {
			mgrad = grads.get(i * 2) / nsample;
			mgrad = Params.clip ? (Math.abs(mgrad) > Params.absmax ? Params.absmax * Math.signum(mgrad) : mgrad) : mgrad;
			vgrad = grads.get(i * 2 + 1) / nsample;
			vgrad = Params.clip ? (Math.abs(vgrad) > Params.absmax ? Params.absmax * Math.signum(vgrad) : vgrad) : vgrad;
			mu = mus.get(i) - Params.lr * mgrad;
			var = vars.get(i) - Params.lr * vgrad;
			mus.set(i, mu);
			vars.set(i, var);
		}
	}
	
	
	public short getDim() {
		return dim;
	}


	public void setDim(short dim) {
		this.dim = dim;
	}


	public List<Double> getVars() {
		return vars;
	}


	public void setVars(List<Double> vars) {
		this.vars = vars;
	}


	public List<Double> getMus() {
		return mus;
	}


	public void setMus(List<Double> mus) {
		this.mus = mus;
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
		return "GD [dim=" + dim + ", mus=" + MethodUtil.double2str(mus, LVeGLearner.precision, -1, false, true) + 
				", stds=" + MethodUtil.double2str(vars, LVeGLearner.precision, -1, true, true) + "]";
	}

}
