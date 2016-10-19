package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.util.MethodUtil;

/**
 * TODO Method {@code add()} is implemented by reference when the data type of the added items 
 * is not the basic one. Implementing them by deep copy is simple, but not necessary for now.
 * </p>
 * Variable declaration rules: mixture (mixture of gaussians), component (component of the GM), 
 * gaussians (details of the component), gaussian (portion of the component), gausses (list of  
 * GD of the portion), gauss (GD). 
 * </p>
 * TODO Method {@code mulAndMarginlize()} can be implemented efficiently by hacks. But when we 
 * take consideration into the further parallel optimization, explicitly implementing each step 
 * in order may be a better choice, which is exactly what we did.
 * </p>
 * TODO Implement the comparison between GMs.
 * </p>
 * @author Yanpeng Zhao
 *
 */
public class GaussianMixture {
	/**
	 * Not sure if it is necessary.
	 * 
	 * @deprecated 
	 */
	protected double bias;
	
	protected short ncomponent;
	protected List<Double> weights;
	

	/**
	 * The set of Gaussian components. Each component consists of one or more 
	 * independent Gaussian distributions mapped by keys. Keys: P (parent); C 
	 * (child); LC (left child); RC (right child); UC (unary child)
	 */
	protected List<Map<String, Set<GaussianDistribution>>> mixture;
	
	
	public GaussianMixture() {
		this.bias = 0;
		this.ncomponent = 0;
		this.weights = new ArrayList<Double>();
		this.mixture = new ArrayList<Map<String, Set<GaussianDistribution>>>();
	}
	
	
	public GaussianMixture(short ncomponent) {
		this.bias = 0;
		this.ncomponent = ncomponent;
		this.weights = new ArrayList<Double>();
		this.mixture = new ArrayList<Map<String, Set<GaussianDistribution>>>();
		initialize();
	}
	
	
	public GaussianMixture(
			short ncomponent, List<Double> weights, 
			List<Map<String, Set<GaussianDistribution>>> mixture) {
		this.bias = 0;
		this.ncomponent = ncomponent;
		this.weights = weights;
		this.mixture = mixture;
	}
	
	
	/**
	 * Initialize the fields by default.
	 */
	private void initialize() {
		MethodUtil.randomInitList(weights, Double.class, ncomponent, LVeGLearner.maxrandom);
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
	public void add(GaussianMixture gm) {
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
	 * @param gaussians  a set of gaussian distributions.
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
			gausses = new HashSet<GaussianDistribution>();
			gausses.addAll(value);
			component.put(key, gausses);
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
	 * @return
	 */
	public GaussianMixture copy() {
		GaussianMixture gm = new GaussianMixture();
		gm.ncomponent = ncomponent;
		gm.weights.addAll(weights);
		for (Map<String, Set<GaussianDistribution>> component : mixture) {
			gm.mixture.add(copy(component));
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
	 * Multiply two gaussian mixtures and marginlize the specific portions of the result.
	 * 
	 * @param gm0  one mixture of gaussians
	 * @param gm1  the other mixture of gaussians
	 * @param keys which denotes the specific portions of the mixture of gaussians
	 * @return
	 */
	public static GaussianMixture mulAndMargin(GaussianMixture gm0, GaussianMixture gm1, List<String> keys) {
		if (gm0 == null && gm1 == null) { return null; }
		if (gm0 == null) { return gm1.copy(); }
		if (gm1 == null) { return gm0.copy(); }
		
		GaussianMixture gm = multiply(gm0, gm1);
		
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
	public static GaussianMixture multiply(GaussianMixture gm0, GaussianMixture gm1) {
		GaussianMixture gm = new GaussianMixture();
		for (int i = 0; i < gm0.ncomponent; i++) {
			for (int j = 0; j < gm1.ncomponent; j++) {
				Map<String, Set<GaussianDistribution>> component = 
						multiply(gm0.mixture.get(i), gm1.mixture.get(j));
				gm.mixture.add(component);
				
				double weight = gm0.weights.get(i) * gm1.weights.get(j);
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
	 * Marginalize the specific portions of the mixture of gaussians.
	 * 
	 * @param gm   mixture of gaussians
	 * @param keys which map to the portions, to be marginalized, of the mixture of gaussians
	 */
	public static void marginalize(GaussianMixture gm, List<String> keys) {
		if (gm.mixture.size() != gm.weights.size()) { gm = null; }
		for (Map<String, Set<GaussianDistribution>> gaussian : gm.mixture) {
			for (String key : keys) {
				gaussian.remove(key);
			}
		}
	}
	
	
	/**
	 * Merge the same components, which could be resulted in by the marginalization, 
	 * of the mixture of gaussians (implemented by reference).
	 * 
	 * @param gm mixture of gaussians
	 * @return
	 */
	public static GaussianMixture merge(GaussianMixture gm) {
		GaussianMixture amixture = new GaussianMixture();
		Map<String, Set<GaussianDistribution>> component;
		for (int i = 0; i < gm.ncomponent; i++) {
			component = gm.mixture.get(i);
			int idx = isContained(component, amixture);
			if (idx < 0) {
				amixture.weights.add(gm.weights.get(i));
				amixture.mixture.add(gm.mixture.get(i));
				amixture.ncomponent++;
			} else {
				double weight = amixture.weights.get(idx) + gm.weights.get(i);
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
			GaussianMixture gm) {
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
	 * Multiply two Gaussian mixtures, whose variables (node IDs) may or may not overlap.
	 * 
	 * @param m
	 * @return the product
	 * 
	 */
	public GaussianMixture multiply(GaussianMixture gm) {
		// TODO
		return null;
	}
	
	
	public double getBias() {
		return bias;
	}


	public void setBias(double bias) {
		this.bias = bias;
	}
	
	
	@Override
	public String toString() {
		return "GM [bias=" + bias + ", ncomponent=" + ncomponent + ", weights=" + 
				MethodUtil.double2str(weights, LVeGLearner.precision) + ", mixture=" + mixture + "]";
	}
}
