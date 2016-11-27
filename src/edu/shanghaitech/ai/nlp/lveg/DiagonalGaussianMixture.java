package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.shanghaitech.ai.nlp.util.MethodUtil;

/**
 * Variable declaration rules: mixture (mixture of gaussians), component (component of the GM), 
 * gaussians (details of the component), gaussian (portion of the component), gausses (list of  
 * GD of the portion), gauss (GD). 
 * </p>
 * TODO Method {@code mulAndMarginlize()} can be implemented efficiently by hacks. But when we 
 * take consideration into the further parallel optimization, explicitly implementing each step 
 * in order may be a better choice, which is exactly what I am doing now (DONE I have implement
 * -ed specifically for the multiplication in the inside score). 
 * </p>
 * TODO Implement the comparison between GMs.
 * </p>
 * @author Yanpeng Zhao
 *
 */
public class DiagonalGaussianMixture extends GaussianMixture {
	/**
	 * Not sure if it is necessary.
	 * 
	 * @deprecated 
	 */
	protected double bias;
	
	protected short nsample;
	protected short ncomponent;
	
	
	/**
	 * The weights need to be positive. We use the exponential function to ensure 
	 * the positiveness, thus the real weight should be read as Math.exp(weight).
	 */
	protected List<Double> weights;
	
	protected List<Double> values;
	protected List<Double> wgrads;
	

	/**
	 * The set of Gaussian components. Each component consists of one or more 
	 * independent Gaussian distributions mapped by keys. Keys: P (parent); C 
	 * (child); LC (left child); RC (right child); UC (unary child)
	 * 
	 * TODO but why did I use the Set for the portion of Mog? To ease the comparison?
	 */
	protected List<Map<String, Set<GaussianDistribution>>> mixture;
	
	
	public DiagonalGaussianMixture() {
		this.bias = 0;
		this.nsample = 0;
		this.ncomponent = 0;
		this.weights = new ArrayList<Double>();
		this.mixture = new ArrayList<Map<String, Set<GaussianDistribution>>>();
		this.values = new ArrayList<Double>();
		this.wgrads = new ArrayList<Double>();
	}
	
	
	public DiagonalGaussianMixture(short ncomponent) {
		this();
		this.ncomponent = ncomponent;
		initialize();
	}
	
	
	public DiagonalGaussianMixture(
			short ncomponent, List<Double> weights, List<Map<String, Set<GaussianDistribution>>> mixture) {
		this();
		this.ncomponent = ncomponent;
		this.weights = weights;
		this.mixture = mixture;
	}
	
	
	/**
	 * Initialize the fields by default.
	 */
	private void initialize() {
		MethodUtil.randomInitList(weights, Double.class, ncomponent, LVeGLearner.maxrandom, false);
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
	public void add(DiagonalGaussianMixture gm) {
		bias += gm.bias;
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
	 * Make a copy of the instance.
	 * 
	 * @param deep boolean value, indicating deep (true) or shallow (false) copy
	 * @return
	 */
	public DiagonalGaussianMixture copy(boolean deep) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		gm.ncomponent = ncomponent;
		gm.weights.addAll(weights);
		for (Map<String, Set<GaussianDistribution>> component : mixture) {
			if (deep) {
				gm.mixture.add(copy(component));
			} else {
				gm.mixture.add(component);
			}
		}
		return gm;
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
	 * Replace the key of the specific portion of the mixture of gaussians with the new key.  
	 * 
	 * @param gm   mixture of gaussians
	 * @param keys pairs of (old-key, new-key)
	 * @return
	 */
	public static DiagonalGaussianMixture replaceKeys(DiagonalGaussianMixture gm, Map<String, String> keys) {
		DiagonalGaussianMixture agm = new DiagonalGaussianMixture();
		agm.weights.addAll(gm.weights);
		for (Map<String, Set<GaussianDistribution>> component : gm.mixture) {
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
			agm.mixture.add(acomponent);
			agm.ncomponent++;
		}
		return agm;
	}
	
	
	/**
	 * Replace all the existing keys with a new key (by reference).
	 * 
	 * @param gm     mixture of gaussians
	 * @param newkey the new key
	 * @return
	 */
	public static DiagonalGaussianMixture replaceAllKeys(DiagonalGaussianMixture gm, String newkey) {
		DiagonalGaussianMixture agm = new DiagonalGaussianMixture();
		agm.weights.addAll(gm.weights);
		for (Map<String, Set<GaussianDistribution>> component : gm.mixture) {
			Map<String, Set<GaussianDistribution>> acomponent = 
					new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> gausses = new HashSet<GaussianDistribution>();
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				gausses.addAll(gaussian.getValue());
			}
			acomponent.put(newkey, gausses);
			agm.mixture.add(acomponent);
			agm.ncomponent++;
		}
		return agm;
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
	public DiagonalGaussianMixture mulForInsideOutside(DiagonalGaussianMixture gm, String key, boolean deep) {
		DiagonalGaussianMixture amixture = this.copy(deep);
		double sum = MethodUtil.sum(gm.weights, true);
		for (int i = 0; i < ncomponent; i++) {
			// CHECK Math.log(Math.exp(a) * b)
			double weight = amixture.weights.get(i) + Math.log(sum);
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
	 * @return
	 */	
	public static DiagonalGaussianMixture mulAndMarginalize(DiagonalGaussianMixture gm0, DiagonalGaussianMixture gm1, 
			Map<String, String> keys0, Map<String, String> keys1) {
		if (gm0 == null || gm1 == null) { return null; }
		
		gm0 = replaceKeys(gm0, keys0);
		gm1 = replaceKeys(gm1, keys1);
		DiagonalGaussianMixture gm = multiply(gm0, gm1);
		
		Set<String> keys = new HashSet<String>();
		for (Map.Entry<String, String> map : keys0.entrySet()) {
			keys.add(map.getValue());
		}
		for (Map.Entry<String, String> map : keys1.entrySet()) {
			keys.add(map.getValue());
		}
		
		marginalize(gm, keys);
		return gm;
	}
	
	
	/**
	 * Multiply two mixtures of gaussians.
	 * 
	 * @param gm0 one mixture of gaussians
	 * @param gm1 the other mixture of gaussians
	 * @return    
	 */
	public static DiagonalGaussianMixture multiply(DiagonalGaussianMixture gm0, DiagonalGaussianMixture gm1) {
		DiagonalGaussianMixture gm = new DiagonalGaussianMixture();
		for (int i = 0; i < gm0.ncomponent; i++) {
			for (int j = 0; j < gm1.ncomponent; j++) {
				Map<String, Set<GaussianDistribution>> component = 
						multiply(gm0.mixture.get(i), gm1.mixture.get(j));
				gm.mixture.add(component);
				// CHECK Math.log(Math.exp(a) * Math.exp(b))
				double weight = gm0.weights.get(i) + gm1.weights.get(j);
				gm.weights.add(weight);
				
				gm.ncomponent++;
			}
		}
		return gm;
	}
	
	
	/**
	 * Multiply two components of the mixture of gaussians.
	 * 
	 * @param component0 one component of the mixture of the gaussians
	 * @param component1 the other component of the mixture of the gaussians
	 * @return
	 */
	private static Map<String, Set<GaussianDistribution>> multiply(
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
	 * @return
	 */
	public double marginalize() {
		return MethodUtil.sum(weights, true);
	}
	
	
	/**
	 * Marginalize the specific portions of the mixture of gaussians.
	 * 
	 * @param gm   mixture of gaussians
	 * @param keys which map to the portions, to be marginalized, of the mixture of gaussians
	 */
	public static void marginalize(DiagonalGaussianMixture gm, Set<String> keys) {
		if (gm.mixture.size() != gm.weights.size()) { gm = null; }
		for (Map<String, Set<GaussianDistribution>> gaussian : gm.mixture) {
			for (String key : keys) {
				gaussian.remove(key);
			}
		}
	}
	
	
	/**
	 * Set the outside score of the root node to 1.
	 */
	protected void marginalizeToOne() {
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
	public static DiagonalGaussianMixture merge(DiagonalGaussianMixture gm) {
		DiagonalGaussianMixture amixture = new DiagonalGaussianMixture();
		Map<String, Set<GaussianDistribution>> component;
		for (int i = 0; i < gm.ncomponent; i++) {
			component = gm.mixture.get(i);
			int idx = isContained(component, amixture);
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
	 * @param gm        mixture of gaussians
	 * @return
	 */
	public static int isContained(Map<String, Set<GaussianDistribution>> component, 
			DiagonalGaussianMixture gm) {
		for (int i = 0; i < gm.ncomponent; i++) {
			if (isEqual(component, gm.mixture.get(i))) {
				return i;
			}
		}
		return -1;
	}
	
	
	/**
	 * Compare whether two components of the mixture of gaussians are equal.
	 * </p>
	 * TODO to double check.
	 * </p>
	 * @param component0 one component of the mixture of gaussians
	 * @param component1 the other component of the mixture of gaussians
	 * @return
	 */
	public static boolean isEqual(
			Map<String, Set<GaussianDistribution>> component0,
			Map<String, Set<GaussianDistribution>> component1) {
		if (component0.size() != component1.size()) { return false; }
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component0.entrySet()) {
			if (!component1.containsKey(gaussian.getKey())) { return false; }
			
			Set<GaussianDistribution> gausses0 = gaussian.getValue();
			Set<GaussianDistribution> gausses1 = component1.get(gaussian.getKey());
			if (gausses0 == null || gausses1 == null) {
				if (gausses0 != gausses1) {
					return false;
				} else {
					continue;
				}
			}
			if (!gausses0.equals(gausses1)) { return false; }
		}
		return true;
	}
	
	
	/**
	 * Sample from MoG.
	 * 
	 * @param random random number generator
	 */
	public void sample(Random random) {
		for (Map<String, Set<GaussianDistribution>> component : mixture) {
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				for (GaussianDistribution gd : gaussian.getValue()) {
					gd.sample(random);
				}
			}
		}
	}
	
	
	/**
	 * Eval the mixture of gaussians. Meanwhile, take the derivative of MoG w.r.t 
	 * the weight of the component is computed.
	 * 
	 * @return
	 */
	public double eval() {
		values.clear(); // clear
		double ret = 0.0, value;
		for (int i = 0; i < ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> component = mixture.get(i);
			value = 1.0;
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				for (GaussianDistribution gd : gaussian.getValue()) {
					value *= gd.eval();
				}
			}
			values.add(value);
			ret += Math.exp(weights.get(i)) * value;
		}
		return ret;
	}
	
	
	/**
	 * Take the derivative of MoG w.r.t parameters (mu & sigma) of the component.
	 * 
	 * @param factor     (E[c(r, t) | T_x] - E[c(r, t) | x]) / w(r)
	 * @param cumulative accumulate gradients or not
	 */
	public void derivative(double factor, boolean cumulative) {
		if (!cumulative) { nsample = 0; }
		if (nsample == 0) {
			wgrads.clear();
			for (int i = 0; i < ncomponent; i++) {
				wgrads.add(0.0);
			}
		}
		for (int i = 0; i < ncomponent; i++) {
			double weight = Math.exp(weights.get(i));
			double wgrad = values.get(i) * factor * weight;
			Map<String, Set<GaussianDistribution>> component = mixture.get(i);
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				for (GaussianDistribution gd : gaussian.getValue()) {
					gd.derivative(weight * wgrad, nsample);
				}
			}
			wgrads.set(i, wgrads.get(i) + wgrad);
		}
		nsample++; // accumulation counter 
	}
	
	
	/**
	 * Update parameters using the gradient.
	 * 
	 * @param learningRate learning rate
	 */
	public void update(double learningRate) {
		if (nsample <= 0) { 
			System.err.println("Nothing to update.");
			return; 
		}
		for (int i = 0; i < ncomponent; i++) {
			double weight = weights.get(i);
			Map<String, Set<GaussianDistribution>> component = mixture.get(i);
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : component.entrySet()) {
				for (GaussianDistribution gd : gaussian.getValue()) {
					gd.update(learningRate, (short) 1);
				}
			}
			weight -= learningRate * wgrads.get(i) / 1;
			weights.set(i, weight);
		}
		nsample = 0; // reset accumulation counter 
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
	
	
	/**
	 * Get the bias term of the mixture of gaussians.
	 * 
	 * @return
	 */
	public double getBias() {
		return bias;
	}


	/**
	 * Set the bias term of the mixture of gaussians.
	 * 
	 * @param bias constant in double
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}
	
	
	public String toString(boolean simple, int nfirst) {
		if (simple) {
			StringBuffer sb = new StringBuffer();
			sb.append("GM [ncomponent=" + ncomponent + ", weights=" + 
					MethodUtil.double2str(weights, LVeGLearner.precision, nfirst, true));
			sb.append("]");
			return sb.toString();
		} else {
			return toString();
		}
	}
	
	
	@Override
	public String toString() {
		return "GM [bias=" + bias + ", ncomponent=" + ncomponent + ", weights=" + 
				MethodUtil.double2str(weights, LVeGLearner.precision, -1, true) + ", mixture=" + mixture + "]";
	}
}