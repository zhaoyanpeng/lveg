package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * Variable declaration rules: mixture (mixture of gaussians), component (component of the GM), 
 * gaussians (details of the component), gaussian (portion of the component), gausses (list of  
 * GD of the portion), gauss (GD).</p>
 * 
 * TODO Implement the comparison operation between GMs.
 * 
 * @author Yanpeng Zhao
 *
 */
public class GaussianMixtureOld extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -822680841484765529L;
	/**
	 * Not sure if it is necessary.
	 * 
	 * @deprecated 
	 */
	protected double bias;
	protected int ncomponent;
	
	/**
	 * The weights need to be positive. We use the exponential function to ensure 
	 * the positiveness, thus the real weight should be read as Math.exp(weight).
	 */
	protected List<Double> weights;
	
	/**
	 * The set of Gaussian components. Each component consists of one or more 
	 * independent Gaussian distributions mapped by keys. Keys: P (parent); C 
	 * (child); LC (left child); RC (right child); UC (unary child)
	 * 
	 * TODO but why did I use the Set for the portion of MoG? To ease the comparison?
	 */
	protected List<Map<String, Set<GaussianDistribution>>> mixture;
	
	
	public GaussianMixtureOld() {
		this.bias = 0;
		this.ncomponent = 0;
		this.weights = new ArrayList<Double>();
		this.mixture = new ArrayList<Map<String, Set<GaussianDistribution>>>();
	}
	
	
	/**
	 * Initialize the fields by default.
	 */
	protected void initialize() {
		MethodUtil.randomInitList(LVeGLearner.random, weights, Double.class, ncomponent, LVeGLearner.maxrandom, LVeGLearner.nratio, false, true);
		for (int i = 0; i < ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> component = 
					new HashMap<String, Set<GaussianDistribution>>();
			mixture.add(component);
		}
	}


	/**
	 * Add the bias.
	 * 
	 * @param bias constant bias
	 */
	public void add(double bias) {
		this.bias += bias;
	}
	
	
	/**
	 * Add the mixture of gaussians.
	 * 
	 * @param gm mixture of gaussians
	 */
	public void add(GaussianMixtureOld gm) {
		// TODO bias += gm.bias;
		ncomponent += gm.ncomponent;
		weights.addAll(gm.weights);
		mixture.addAll(gm.mixture);
	}
	
	
	/**
	 * Add a component.
	 * 
	 * @param weight    weight of the component
	 * @param component a component of the mixture of gaussians
	 */
	public void add(double weight, Map<String, Set<GaussianDistribution>> component) {
		ncomponent++;
		weights.add(weight);
		mixture.add(component);
	}
	
	
	/**
	 * Add gaussian distributions to a component.
	 * 
	 * @param iComponent index of the component
	 * @param gaussians  a set of gaussian distributions
	 */
	public void add(int iComponent, Map<String, Set<GaussianDistribution>> gaussians) {
		Map<String, Set<GaussianDistribution>> component = mixture.get(iComponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : gaussians.entrySet()) {
			add(component, gaussian.getKey(), gaussian.getValue());
		}
	}
	
	
	/**
	 * Add gaussian distributions to the specific portion of a component.
	 * 
	 * @param iComponent index of the component
	 * @param key        which denotes the specific portion of a component
	 * @param value      which is added to the specific portion of a component
	 */
	public void add(int iComponent, String key, Set<GaussianDistribution> gausses) {
		Map<String, Set<GaussianDistribution>> component = mixture.get(iComponent);
		add(component, key, gausses);	
	}
	
	
	/**
	 * Add gaussian distributions to a component.
	 * 
	 * @param component the component of the mixture of gaussians
	 * @param gaussians a set of gaussian distributions
	 */
	public static void add(Map<String, Set<GaussianDistribution>> component, 
			Map<String, Set<GaussianDistribution>> gaussians) {
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : gaussians.entrySet()) {
			add(component, gaussian.getKey(), gaussian.getValue());
		}
	}
	
	
	/**
	 * Add gaussian distributions to the specific portion of a component.
	 * TODO can be further simplified, see {@code Inferencer.Cell.addScore}
	 * 
	 * @param component the component of the mixture of gaussians
	 * @param key       which denotes the specific portion of a component
	 * @param value     which is added to the specific portion of a component
	 */
	public static void add(
			Map<String, Set<GaussianDistribution>> component, 
			String key, Set<GaussianDistribution> value) {
		Set<GaussianDistribution> gausses = component.get(key);
		if (!component.containsKey(key)) {
			/*
			gausses = new HashSet<GaussianDistribution>();
			gausses.addAll(value);
			component.put(key, gausses);
			*/
			component.put(key, value);
		} else {
			if (gausses == null) {
				gausses = new HashSet<GaussianDistribution>();
			}
			gausses.addAll(value);
		}
	}

	
	/**
	 * Make a copy of this MoG. This will create a new instance of MoG.
	 * 
	 * @param deep boolean value, indicating deep (true) or shallow (false) copy
	 * @return
	 */
	public GaussianMixtureOld copy(boolean deep) { return null; }
	
	
	/**
	 * Make a copy of this MoG.
	 * 
	 * @param des  a placeholder of MoG
	 * @param deep boolean value, indicating deep (true) or shallow (false) copy
	 */
	protected void copy(GaussianMixtureOld des, boolean deep) {
		des.ncomponent = ncomponent;
		des.weights.addAll(weights);
		for (Map<String, Set<GaussianDistribution>> component : mixture) {
			if (deep) {
				des.mixture.add(copy(component));
			} else {
				des.mixture.add(component);
			}
		}
	}
	
	
	/**
	 * Make a copy of the component of the mixture of gaussians.
	 * 
	 * @param component a component of the mixture of gaussians
	 * @return
	 */
	public static Map<String, Set<GaussianDistribution>> copy(Map<String, Set<GaussianDistribution>> component) {
		Map<String, Set<GaussianDistribution>> replica = new HashMap<String, Set<GaussianDistribution>>();
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
			replica.put(gaussian.getKey(), copy(gaussian.getValue()));
		}
		return replica;
	}
	
	
	/**
	 * Make a copy of a set of gaussian distributions.
	 * 
	 * @param gausses a set of gaussian distributions
	 * @return
	 */
	public static Set<GaussianDistribution> copy(Set<GaussianDistribution> gausses) {
		Set<GaussianDistribution> replica = new HashSet<GaussianDistribution>();
		for (GaussianDistribution gd : gausses) {
			replica.add(gd.copy());
		}
		return replica;
	}
	
	
	/**
	 * Replace the key of the specific portion of the mixture of gaussians with the new key (by reference).
	 * This will create a new copy of this MoG, but with the new keys.
	 * 
	 * @param keys pairs of (old-key, new-key)
	 * @return
	 */
	public GaussianMixtureOld replaceKeys(Map<String, String> keys) { return null; }
	
	
	/**
	 * Replace the key of the specific portion of the mixture of gaussians with the new key (by reference).
	 * 
	 * @param des  a placeholder of MoG
	 * @param keys pairs of (old-key, new-key)
	 */
	protected void replaceKeys(GaussianMixtureOld des, Map<String, String> keys) {
		des.weights.addAll(weights);
		for (Map<String, Set<GaussianDistribution>> component : mixture) {
			Map<String, Set<GaussianDistribution>> acomponent = 
					new HashMap<String, Set<GaussianDistribution>>();
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				String key = gaussian.getKey();
				if (keys.containsKey(key)) {
					add(acomponent, keys.get(key), gaussian.getValue());
				} else {
					add(acomponent, key, gaussian.getValue());
				}
			}
			des.mixture.add(acomponent);
			des.ncomponent++;
		}
	}
	
	
	/**
	 * Replace all of the existing keys with a new key (by reference).
	 * This will create a new copy of this MoG, but with the new keys.
	 * 
	 * @param gm     mixture of gaussians
	 * @param newkey the new key
	 * @return
	 */
	public GaussianMixtureOld replaceAllKeys(String newkey) { return null; }
	
	
	/**
	 * Replace all of the existing keys with a new key (by reference).
	 * 
	 * @param des    a placeholder of MoG
	 * @param newkey the new key
	 */
	protected void replaceAllKeys(GaussianMixtureOld des, String newkey) {
		des.weights.addAll(weights);
		for (Map<String, Set<GaussianDistribution>> component : mixture) {
			Map<String, Set<GaussianDistribution>> acomponent = 
					new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> gausses = new HashSet<GaussianDistribution>();
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				gausses.addAll(gaussian.getValue());
			}
			acomponent.put(newkey, gausses);
			des.mixture.add(acomponent);
			des.ncomponent++;
		}
	}
	
	
	/**
	 * Problem-specific multiplication (calculation of the inside score). Inside score 
	 * of the current non-terminal relates to the inside scores (mixture of gaussians) 
	 * of the children only partially, since the gaussians are afterwards marginalized
	 * and thus only weights matter. The same rules for the outside score.
	 * 
	 * @param gm   mixture of gaussians that needs to be marginalized
	 * @param key  which denotes the portion, to be marginalized, of the component
	 * @param deep deep (true) or shallow (false) copy of the instance
	 * @return 
	 * 
	 */
	public GaussianMixtureOld mulForInsideOutside(GaussianMixtureOld gm, String key, boolean deep) {
		GaussianMixtureOld amixture = this.copy(deep);
		double logsum = gm.marginalize(true);
		for (int i = 0; i < ncomponent; i++) {
			// CHECK Math.log(Math.exp(a) * b)
			double weight = amixture.weights.get(i) + logsum;
			amixture.weights.set(i, weight);
			amixture.mixture.get(i).remove(key);
		}
		return amixture;
	}
	
	
	/**
	 * Multiply two gaussian mixtures and marginlize the specific portions of the result.
	 * 
	 * @param gm0   one mixture of gaussians
	 * @param gm1   the other mixture of gaussians
	 * @param keys0 pairs of (key, value), key denotes the specific portions of gm0, value is the new key
	 * @param keys1 pairs of (key, value), key denotes the specific portions of gm1, value is the new key
	 * @param copy  using copy or reference
	 * @return
	 */	
	public static GaussianMixtureOld mulAndMarginalize(GaussianMixtureOld gm0, GaussianMixtureOld gm1, 
			Map<String, String> keys0, Map<String, String> keys1) {
		if (gm0 == null || gm1 == null) { return null; }
		gm0 = gm0.replaceKeys(keys0);
		gm1 = gm1.replaceKeys(keys1);
		GaussianMixtureOld gm = gm0.multiply(gm1);
		
		Set<String> keys = new HashSet<String>();
		for (Map.Entry<String, String> map : keys0.entrySet()) {
			keys.add(map.getValue());
		}
		for (Map.Entry<String, String> map : keys1.entrySet()) {
			keys.add(map.getValue());
		}
		gm.marginalize(keys);
		return gm;
	}
	
	
	/**
	 * Multiply this MoG by the given mixtures of gaussians.
	 * This will create a new instance of MoG.
	 * 
	 * @param multiplier the other mixture of gaussians
	 * @return    
	 */
	public GaussianMixtureOld multiply(GaussianMixtureOld multiplier) { return null; }
	
	
	/**
	 * Multiply this MoG by the given mixtures of gaussians.
	 * 
	 * @param des        a placeholder of MoG
	 * @param multiplier the other mixture of gaussians
	 */
	protected void multiply(GaussianMixtureOld des, GaussianMixtureOld multiplier) {
		for (int i = 0; i < ncomponent; i++) {
			for (int j = 0; j < multiplier.ncomponent; j++) {
				Map<String, Set<GaussianDistribution>> component = 
						multiply(mixture.get(i), multiplier.mixture.get(j));
				des.mixture.add(component);
				// CHECK Math.log(Math.exp(a) * Math.exp(b))
				double weight = weights.get(i) + multiplier.weights.get(j);
				des.weights.add(weight);
				
				des.ncomponent++;
			}
		}
	}
	
	
	/**
	 * Multiply two components of the mixture of gaussians.
	 * 
	 * @param component0 one component of the mixture of the gaussians
	 * @param component1 the other component of the mixture of the gaussians
	 * @param copy 		  using copy or reference
	 * @return
	 */
	protected Map<String, Set<GaussianDistribution>> multiply(
			Map<String, Set<GaussianDistribution>> component0, 
			Map<String, Set<GaussianDistribution>> component1) {
		Map<String, Set<GaussianDistribution>> component = copy(component0);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component1.entrySet()) {
			add(component, gaussian.getKey(), copy(gaussian.getValue()));
		}
		return component;
	}
	
	
	
	/**
	 * Marginalize the mixture of gaussians.
	 * 
	 * @param logarithm whether the return value is the logarithm form or not
	 * @return
	 */
	public double marginalize(boolean logarithm) {
		if (logarithm) {
			double logval = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < weights.size(); i++) {
				logval = MethodUtil.logAdd(logval, weights.get(i));
			}
			return logval;
		}
		return MethodUtil.sum(weights, true);
	}
	
	
	/**
	 * Marginalize the specific portions of the mixture of gaussians.
	 * 
	 * @param keys which map to the portions, to be marginalized, of the mixture of gaussians
	 */
	public void marginalize(Set<String> keys) {
		for (Map<String, Set<GaussianDistribution>> gaussian : mixture) {
			for (String key : keys) {
				gaussian.remove(key);
			}
		}
	}
	
	
	/**
	 * Set the outside score of the root node to 1.
	 */
	public void marginalizeToOne() {
		if (ncomponent <= 0) {
			System.err.println("Fatal error when marginalize the MoG to one.\n" + toString());
			System.exit(0);
		}
		/*
		double weight = 1.0 / ncomponent;
		for (int i = 0; i < ncomponent; i++) {
			weights.set(i, weight);
			mixture.get(i).clear();
		}
		*/
		for (int i = 1; i < ncomponent; i++) {
			mixture.remove(i);
			weights.remove(i);
		}
		ncomponent = 1;
		// CHECK Math.exp(0)
		weights.set(0, 0.0);
		mixture.get(0).clear();
	}
	
	
	/**
	 * Merge the same components, which could be resulted in by the marginalization, 
	 * of the mixture of gaussians (implemented by reference).
	 * 
	 * @param gm mixture of gaussians
	 * @return
	 */
	public static GaussianMixtureOld merge(GaussianMixtureOld gm) {
		GaussianMixtureOld amixture = new GaussianMixtureOld();
		Map<String, Set<GaussianDistribution>> component;
		for (int i = 0; i < gm.ncomponent; i++) {
			component = gm.mixture.get(i);
			int idx = amixture.contains(component);
			if (idx < 0) {
				amixture.weights.add(gm.weights.get(i));
				amixture.mixture.add(gm.mixture.get(i));
				amixture.ncomponent++;
			} else {
				// CHECK Math.log(Math.exp(a) + Math.exp(b))
				double weight = MethodUtil.logAdd(amixture.weights.get(idx), gm.weights.get(i));
				amixture.weights.set(idx, weight);
			}
		}
		return amixture;
	}
	
	
	/**
	 * Determine if the component of the mixture of gaussians is contained in the given 
	 * mixture of gaussians.
	 * 
	 * @param component the component of the mixture of gaussians
	 * @return
	 */
	public int contains(Map<String, Set<GaussianDistribution>> component) {
		for (int i = 0; i < ncomponent; i++) {
			if (isEqual(component, mixture.get(i))) {
				return i;
			}
		}
		return -1;
	}
	
	
	/**
	 * Compare whether two components of the mixture of gaussians are the same.
	 * </p>
	 * CHECK to double check. It should be symmetrically the same.
	 * </p>
	 * @param component0 one component of the mixture of gaussians
	 * @param component1 the other component of the mixture of gaussians
	 * @return
	 */
	private boolean isEqual(
			Map<String, Set<GaussianDistribution>> component0,
			Map<String, Set<GaussianDistribution>> component1) {
		if (component0.size() != component1.size()) { return false; }
		// the uniqueness of the keys ensures that one for loop is enough
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component0.entrySet()) {
			if (!component1.containsKey(gaussian.getKey())) { return false; }
			Set<GaussianDistribution> gausses0 = gaussian.getValue();
			Set<GaussianDistribution> gausses1 = component1.get(gaussian.getKey());
			if (!isEqual(gausses0, gausses1)) { return false; }
		}
		return true;
	}
	
	
	/**
	 * Compare whether two set of gaussians are identical. We only cares about the value not the hash code for the
	 * item, thus A.containsAll(B) && B.containsAll(A) or A.equals(B) is not appropriate for our application since
	 * Both of which will invoke hashCode().
	 * 
	 * @param gausses0 a set of gaussians
	 * @param gausses1 a set of gaussians
	 * @return
	 */
	private boolean isEqual(Set<GaussianDistribution> gausses0, Set<GaussianDistribution> gausses1) {
		if (gausses0 == null || gausses1 == null) {
			if (gausses0 != gausses1) {
				return false;
			} else {
				return true;
			}
		}
		if (gausses0.size() != gausses1.size()) { return false; }
		/**
		 * CHECK The following code has a hidden bug. If the item in the gaussian set is modified somewhere else 
		 * and effects the return value of hashcode(), equals() will fail since Set.add() evals the hash-code of 
		 * the item when it is added and won't automatically change depending on the newest status of the item.
		 * 
		 * if (!gausses0.equals(gausses1)) { return false; }
		 * 
		 * http://stackoverflow.com/questions/32963070/hashset-containsobject-returns-false-for-instance-modified-after-insertion
		 */
		// the uniqueness of the gaussians ensures that one for loop is enough
		for (GaussianDistribution g0 : gausses0) {
			boolean found1 = false;
			for (GaussianDistribution g1 : gausses1) {
				if (g1.equals(g0)) { found1 = true; break; }
			}
			if (!found1) { return false; } 
		}
		return true;
	}
	
	
	/**
	 * Eval the values of inside and outside score functions given the sample. Normal is supposed to be false.
	 * 
	 * @param sample the sample
	 * @param normal whether the sample is from N(0, 1) (true) or not (false)
	 * @return
	 */
	public double evalInsideOutside(List<Double> sample, boolean normal) {
		double ret = 0.0, value;
		for (int i = 0; i < ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> component = mixture.get(i);
			value = 1.0;
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				for (GaussianDistribution gd : gaussian.getValue()) {
					value *= gd.eval(sample, normal);
					if (value < 0) {
						logger.trace("\n---eval inside outside---\n" + this + "\n");
					}
				}
			}
			ret += Math.exp(weights.get(i)) * value;
		}
		return ret;
	}
	
	
	/**
	 * Eval the MoG using the given sample. Pay attention to the second parameter.
	 * 
	 * @param sample the sample
	 * @param normal whether the sample is from N(0, 1) (true) or not (false); 
	 * 		   when the sample is null, normal is used to indicate the return value in logarithm (true) or decimal (false)
	 * @return
	 */
	public double eval(Map<String, List<Double>> sample, boolean normal) {
		if (sample == null) { return eval(normal); }
		
		double ret = 0.0, value;
		for (int i = 0; i < ncomponent; i++) {
			// TODO This is not correct. Because every component contains the same variables but with different parameters.
			Map<String, Set<GaussianDistribution>> component = mixture.get(i);
			value = 1.0;
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				List<Double> slice = sample.get(gaussian.getKey());
				for (GaussianDistribution gd : gaussian.getValue()) {
					value *= gd.eval(slice, normal);
				}
			}
			ret += Math.exp(weights.get(i)) * value;
		}
		return ret;
	}
	
	
	/**
	 * This method assumes the MoG is equivalent to a constant, e.g., no gaussians contained in any component.
	 * 
	 * @param normal whether the sample is from N(0, 1) (true) or not (false)
	 * @return
	 */
	private double eval(boolean logarithm) {
		for (int i = 0; i < ncomponent; i++) {
			if (mixture.get(i).size() > 0) {
				logger.error("You are not supposed to call this method if the MoG contains multivnd.\n");
				return -0.0;
			}
		}
		return marginalize(logarithm);
	}
	
	
	/**
	 * Take the derivative of the MoG w.r.t the mixing weight.
	 * 
	 * @param sample     the sample from N(0, 1)
	 * @param icomponent index of the component
	 * @return
	 */
	public double derivateMixingWeight(Map<String, List<Double>> sample, int icomponent, boolean normal) {
		double value = 1.0;
		Map<String, Set<GaussianDistribution>> component = mixture.get(icomponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
			List<Double> slice = sample.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				value *= gd.eval(slice, normal);
				if (value < 0) {
					logger.trace("\n---derivate the mixting weight---\n" + this + "\n");
				}
			}
		}
		return value;
	}
	
	
	/**
	 * Take the derivative of MoG w.r.t parameters (mu & sigma) of the component.
	 * 
	 * @param cumulative accumulate the gradients (true) or not (false)
	 * @param icomponent index of the component of MoG
	 * @param factor     derivative with respect to rule weight: (E[c(r, t) | T_x] - E[c(r, t) | x]) / w(r)
	 * @param sample     the sample from the current component
	 * @param ggrads     gradients of the parameters of gaussian distributions
	 * @param wgrads     gradients of the mixing weights of MoG
	 * @param normal     whether the sample is from N(0, 1) (true) or not (false)
	 */
	public void derivative(
			boolean cumulative, int icomponent, double factor, 
			Map<String, List<Double>> sample, Map<String, List<Double>> ggrads, List<Double> wgrads, boolean normal) {
		if (!cumulative && icomponent == 0) { // CHECK stupid if...else...
			wgrads.clear();
			for (int i = 0; i < ncomponent; i++) {
				wgrads.add(0.0);
			}
		}
		double weight = Math.exp(weights.get(icomponent));
		double penalty = Params.reg ? (Params.l1 ? Params.wdecay * weight : Params.wdecay * Math.pow(weight, 2)) : 0.0;
		double dMixingW = factor * weight * derivateMixingWeight(sample, icomponent, normal);
		Map<String, Set<GaussianDistribution>> component = mixture.get(icomponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
			List<Double> slice = sample.get(gaussian.getKey());
			List<Double> grads = ggrads.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				gd.derivative(dMixingW, slice, grads, cumulative, normal);
				// break; // CHECK only one gaussian is allowed
			}
		}
		dMixingW += penalty;
		wgrads.set(icomponent, wgrads.get(icomponent) + dMixingW);
	}
	
	
	/**
	 * Update parameters using the gradient.
	 * 
	 * @param icomponent index of the component of MoG
	 * @param ggrads     gradients of the parameters of gaussians
	 * @param wgrads     gradients of the mixing weights of MoG
	 */
	public void update(int icomponent, Map<String, List<Double>> ggrads, List<Double> wgrads, int nsample) {
		Map<String, Set<GaussianDistribution>> component = mixture.get(icomponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
			List<Double> grads = ggrads.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				gd.update(grads, nsample);
			}
		}
		double wgrad = wgrads.get(icomponent) / nsample;
		wgrad = Params.clip ? (Math.abs(wgrad) > Params.absmax ? Params.absmax * Math.signum(wgrad) : wgrad) : wgrad;
		double weight = weights.get(icomponent) - Params.lr * wgrad;
		weights.set(icomponent, weight);
	}
	
	
	/**
	 * Allocate memory space for gradients.
	 * 
	 * @return gradients holder
	 */
	public List<Map<String, List<Double>>> zeroslike() {
		List<Map<String, List<Double>>> grads = new ArrayList<Map<String, List<Double>>>(ncomponent);
		for (int i = 0; i < ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> component = mixture.get(i);
			Map<String, List<Double>> gcomp = new HashMap<String, List<Double>>(component.size(), 1);
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				if (gaussian.getValue().size() > 1) { logger.error("Invalid rule weight.\n"); }
				for (GaussianDistribution gd : gaussian.getValue()) {
					List<Double> grad = new ArrayList<Double>(gd.dim * 2);
					gcomp.put(gaussian.getKey(), grad);
				}
			}
			grads.add(gcomp);
		}
		return grads;
	}
	
	
	/**
	 * Allocate memory space for samples.
	 * 
	 * @param icomponent 0 by default, since all components are of the same dimension
	 * @return
	 */
	public List<HashMap<String, List<Double>>> zeroslike(int icomponent) {
		Map<String, Set<GaussianDistribution>> component = mixture.get(icomponent);
		List<HashMap<String, List<Double>>> holder = new ArrayList<HashMap<String, List<Double>>>(2);
		HashMap<String, List<Double>> sample = new HashMap<String, List<Double>>(component.size(), 1);
		HashMap<String, List<Double>> truths = new HashMap<String, List<Double>>(component.size(), 1);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
			if (gaussian.getValue().size() > 1) { logger.error("Invalid rule weight.\n"); }
			for (GaussianDistribution gd : gaussian.getValue()) {
				sample.put(gaussian.getKey(), new ArrayList<Double>(gd.dim));
				truths.put(gaussian.getKey(), new ArrayList<Double>(gd.dim));
			}
		}
		holder.add(sample);
		holder.add(truths);
		return holder;
	}
	
	
	/**
	 * Sample from N(0, 1), and then restore the real sample according to the parameters of the gaussian distribution. 
	 * 
	 * @param icomponent index of the component
	 * @param sample     the sample from N(0, 1)
	 * @param truths     the placeholder
	 * @param rnd        random
	 */
	public void sample(int icomponent, Map<String, List<Double>> sample, Map<String, List<Double>> truths, Random rnd) {
		Map<String, Set<GaussianDistribution>> component = mixture.get(icomponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
			 List<Double> slice = sample.get(gaussian.getKey());
			 List<Double> truth = truths.get(gaussian.getKey());
			 if (gaussian.getValue().size() > 1) { logger.error("Invalid rule weight.\n"); }
			 for (GaussianDistribution gd : gaussian.getValue()) {
				 gd.sample(slice, truth, rnd);
			 }
		}
	}
	
	
	/**
	 * Restore the real sample from the sample that is sampled from the standard normal distribution.
	 * 
	 * @param icomponent index of the component
	 * @param sample     the sample from N(0, 1)
	 * @param truths     the placeholder
	 */
	public void restoreSample(int icomponent, Map<String, List<Double>> sample, Map<String, List<Double>> truths) {
		Map<String, Set<GaussianDistribution>> component = mixture.get(icomponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
			List<Double> slice = sample.get(gaussian.getKey());
			List<Double> truth = truths.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				gd.restoreSample(slice, truth);
				// break; // CHECK only one gaussian is allowed
			}
		}
	}
	
	
	public int dim(int icomponent, String key) {
		Set<GaussianDistribution> gausses = mixture.get(icomponent).get(key);
		for (GaussianDistribution gd : gausses) {
			return gd.dim;
		}
		return -1;
	}
	
	
	public int size(int icomponent) {
		return mixture.get(icomponent).size();
	}
	
	
	public int ncomponent() {
		return ncomponent;
	}
	
	
	public double getWeight(int icomponent) {
		return Math.exp(weights.get(icomponent));
	}
	
	
	/**
	 * Memory clean.
	 */
	public void clear() {
		this.bias = 0.0;
		this.ncomponent = 0;
		this.weights.clear();
		for (Map<String, Set<GaussianDistribution>> component : mixture) {
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				for (GaussianDistribution gd : gaussian.getValue()) {
					gd.clear();
				}
				gaussian.getValue().clear();
			}
			component.clear();
		}
		this.mixture.clear();
	}
	
	
	public double getBias() {
		return bias;
	}
	

	public void setBias(double bias) {
		this.bias = bias;
	}
	
	
	public int getNcomponent() {
		return ncomponent;
	}


	public void setNcomponent(int ncomponent) {
		this.ncomponent = ncomponent;
	}
	
	
	public List<Double> getWeights() {
		return weights;
	}


	public void setWeights(List<Double> weights) {
		this.weights = weights;
	}
	

	public List<Map<String, Set<GaussianDistribution>>> getMixture() {
		return mixture;
	}


	public void setMixture(List<Map<String, Set<GaussianDistribution>>> mixture) {
		this.mixture = mixture;
	}


	public String toString(boolean simple, int nfirst) {
		if (simple) {
			StringBuffer sb = new StringBuffer();
			sb.append("GM [ncomponent=" + ncomponent + ", weights=" + 
					MethodUtil.double2str(weights, LVeGLearner.precision, nfirst, true, true));
			sb.append("]");
			return sb.toString();
		} else {
			return toString();
		}
	}
	
	
	@Override
	public String toString() {
		return "GM [bias=" + bias + ", ncomponent=" + ncomponent + ", weights=" + 
				MethodUtil.double2str(weights, LVeGLearner.precision, -1, true, true) + ", mixture=" + mixture + "]";
	}
	
}
