package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.ObjectPool;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * Gaussian distribution. We may define diagonal or general Gaussian distribution. Different 
 * Gaussian distributions only differ when we eval their values or take directives.
 * 
 * @author Yanpeng Zhao
 *
 */
public abstract class GaussianDistribution extends Recorder implements Comparable<Object>, Serializable {
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
	protected short key; // from the object pool (>=-1) or not (<-1)
	
	/**
	 * Covariances must be positive. We represent them in exponential 
	 * form, and the real variances (for the diagonal) should be read 
	 * as Math.exp(2 * variances), and standard variance is Math.exp(
	 * variances).
	 */
	protected List<Double> vars;
	protected List<Double> mus;
	
	protected static double defMaxMu;
	protected static double defMaxVar;
	protected static double defNegMRatio;
	protected static double defNegVRatio;
	protected static short defNdimension;
	protected static ObjectPool<Short, GaussianDistribution> defObjectPool;
	protected static Random defRnd;
	
	
	protected GaussianDistribution(short ndimension) {
		this.id = 0;
		this.key = -2;
		this.dim = ndimension;
		this.mus = new ArrayList<>(dim);
		this.vars = new ArrayList<>(1);
	}
	
	
	/**
	 * Memory allocation and initialization.
	 */
	protected abstract void initialize();
	
	
	/**
	 * To facilitate the parameter tuning.
	 * 
	 */
	public static void config(double maxmu , double maxvar, short ndimension, double negmratio, 
			double negvratio, Random rnd, ObjectPool<Short, GaussianDistribution> pool) {
		defRnd = rnd;
		defMaxMu = maxmu;
		defMaxVar = maxvar;
		defNegMRatio = negmratio;
		defNegVRatio = negvratio;
		defNdimension = ndimension;
		defObjectPool = pool;
	}
	
	
	public static void returnObject(GaussianDistribution obj) {
		if (obj.key >= -1) {
			defObjectPool.returnObject(obj.key, obj);
		} else {
			obj.clear();
		}
	}
	
	
	/**
	 * Make a copy of the instance.
	 * 
	 * @return
	 */
	public GaussianDistribution copy() { return null; }
	public abstract GaussianDistribution instance(short ndimension, boolean init);
	
	
	/**
	 * Make a copy of the instance.
	 * 
	 * @param des a placeholder of GaussianDistribution
	 */
	public void copy(GaussianDistribution des) {
		des.id = id;
		des.dim = dim;
		des.vars.add(vars.get(0));
		for (int i = 0; i < dim; i++) {
			des.mus.add(mus.get(i));
		}
	}
	
	
	/**
	 * Eval the gaussian distribution using the given sample
	 * 
	 * @param sample the sample 
	 * @param normal whether the sample is from N(0, 1) (true) or from this gaussian (false).
	 * @return       in logarithm
	 */
	protected double eval(List<Double> sample, boolean normal) { return -0.0; } 
	
	
	/**
	 * @param sample normalize the non-normal sample 
	 * @return
	 */
	protected List<Double> normalize(List<Double> sample) {
		List<Double> list = new ArrayList<>();
		double var = vars.get(0);
		for (int i = 0; i < dim; i++) {
			list.add((sample.get(i) - mus.get(i)) / Math.exp(var));
		}
		return list;
	}
	
	
	public void derivative(boolean cumulative, double factor, List<Double> grads, List<List<Double>> caches) {}
	public void derivative(boolean cumulative, List<Double> grads, List<Double> gradst, List<Double> gradss, double scoreT, double scoreS) {}
	
	
	/**
	 * @param gd    a gaussian as a multiplier
	 * @param cache integrals
	 */
	public double integral(GaussianDistribution gd, List<List<Double>> cache) { return Double.NEGATIVE_INFINITY; }
	
	
	/**
	 * Take the product of two gaussians and marginalize it.
	 * 
	 * @param gd a gaussian as a multiplier
	 * @return   marginalization of the production
	 */
	public abstract double mulAndMarginalize(GaussianDistribution gd);
	
	
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
		double real, norm, var = vars.get(0);
		slice.clear();
		truth.clear();
		for (int i = 0; i < dim; i++) {
			norm = defRnd.nextGaussian();
//			norm = ThreadLocalRandom.current().nextGaussian();
			real = norm * Math.exp(var) + mus.get(i);
			slice.add(norm);
			truth.add(real);
		}
	}
	
	
	/**
	 * Restore the real sample from the sample that is sampled from the standard normal distribution.
	 * 
	 * @param sample the sample sampled from N(0, 1)
	 * @param truth  the sample from this gaussian distribution
	 */
	protected void restoreSample(List<Double> sample, List<Double> truth) {
		assert(sample.size() == dim);
		double real, var = vars.get(0);
		truth.clear();
		for (int i = 0; i < dim; i++) {
			// CHECK std = Math.exp(var)
			real = sample.get(i) * Math.exp(var) + mus.get(i);
			truth.add(real);
		}
	}
	
	
	/**
	 * Update parameters using the gradients.
	 * 
	 * @param grads gradients
	 */
	protected void update(List<Double> grads) {
		if (grads.size() == 0) { 
//			logger.warn("No need to update because no gradients could be applied.");
			return; 
		}
		double mu, var = vars.get(0);
		for (int i = 0; i < dim; i++) {
			mu = mus.get(i) + grads.get(i * 2);
			mus.set(i, mu);
		}
		var = vars.get(0) + grads.get(dim * 2); // gradient is stored as the last item
		vars.set(0, var);
	}
	
	
	protected void disturbParams(double delta) {
		for (int i = 0; i < dim; i++) {
			mus.set(i, mus.get(i) + delta);
		}
		vars.set(0, vars.get(0) + delta);
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
	
	
	public short getKey() {
		return key;
	}
	
	
	public void setKey(short key) {
		this.key = key;
	}
	
	
	public void destroy(short ndim) {
		clear();
		mus = null;
		vars = null;
	}
	
	
	public boolean isValid(short ndim) {
		return (vars != null && mus != null && mus.size() == dim && vars.size() == 1);
	}
	
	
	public void clear(short ndim) {
		clear();
	}
	

	/**
	 * Memory clean.
	 */
	public void clear() {
		this.dim = 0;
		if (mus != null) { mus.clear(); }
		if (vars != null) { vars.clear(); }
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
		return "GD [dim=" + dim + ", mus=" + FunUtil.double2str(mus, LVeGTrainer.precision, -1, false, true) + 
				", stds=" + FunUtil.double2str(vars, LVeGTrainer.precision, -1, true, false) + "]";
	}

}
