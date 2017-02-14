package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.ObjectPool;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * 
 * @author Yanpeng Zhao
 *
 */
public class GaussianMixture extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -822680841484765529L;
	private static final double LOG_ZERO = -1.0e10;
	private static double EXP_ZERO = /*-Math.log(-LOG_ZERO)*/Math.log(1e-6);
	protected PriorityQueue<Component> components;
	protected short key; // from the object pool (>=-1) or not (<-1)
	
	protected static short defNcomponent;
	protected static double defMaxmw;
	protected static double defNegWRatio;
	protected static ObjectPool<Short, GaussianMixture> defObjectPool;
	protected static Random defRnd;
	
	/**
	 * Not sure if it is necessary.
	 * 
	 * @deprecated 
	 */
	protected double bias;
	protected int ncomponent;
	
	public GaussianMixture(short ncomponent) {
		this.bias = 0;
		this.key = -2;
		this.ncomponent = ncomponent;
		this.components = new PriorityQueue<Component>(defNcomponent + 1, wcomparator);
	}
	
	
	/**
	 * Initialize the fields by default.
	 */
	protected void initialize() {
		for (int i = 0; i < ncomponent; i++) {
			double weight = (defRnd.nextDouble() - defNegWRatio) * defMaxmw;
			Map<String, Set<GaussianDistribution>> multivnd = new HashMap<String, Set<GaussianDistribution>>();
			 weight = /*0.5*/ /*-0.69314718056*/ 0; // mixing weight 0.5, 1, 2
			components.add(new Component((short) i, weight, multivnd));
		}
	}
	
	
	/**
	 * To facilitate the parameter tuning.
	 * 
	 */
	public static void config(double expzero, double maxmw, short ncomponent, 
			double negwratio, Random rnd, ObjectPool<Short, GaussianMixture> pool) {
		EXP_ZERO = Math.log(expzero);
		defRnd = rnd;
		defMaxmw = maxmw;
		defNegWRatio = negwratio;
		defNcomponent = ncomponent;
		defObjectPool = pool;
	}
	
	
	public static void returnObject(GaussianMixture obj) {
		if (obj.key >= -1) {
			defObjectPool.returnObject(obj.key, obj);
		} else {
			obj.clear();
		}
	}
	
	
	/**
	 * Remove the trivial components.
	 */
	public void delTrivia() {
		if (ncomponent <= 1) { return; }
		PriorityQueue<Component> sorted = sort();
		double maxw = sorted.peek().weight;
		for (Component comp : components) {
			if (comp.weight > LOG_ZERO && (comp.weight - maxw) > EXP_ZERO) { 
				continue; 
			}
			sorted.remove(comp);
		}
		components = sorted;
		ncomponent = sorted.size();
	}
	
	
	/**
	 * @return the sorted components by the mixing weight in case when you have modified the mixing weights of some components.
	 */
	public PriorityQueue<Component> sort() {
		PriorityQueue<Component> sorted = new PriorityQueue<Component>(ncomponent + 1, wcomparator);
		sorted.addAll(components);
		return sorted;
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
	public void add(GaussianMixture gm, boolean prune) {
		// TODO bias += gm.bias;
		ncomponent += gm.ncomponent;
		components.addAll(gm.components);
		if (prune) { delTrivia(); }
		/*
		if (prune && components.size() > 0) {
			double maxw = components.peek().weight; // may not be the real maximum weight
			for (Component comp : gm.components) {
				if (comp.weight > LOG_ZERO && (comp.weight - maxw) > EXP_ZERO) {
					ncomponent++;
					components.add(comp);
					maxw = components.peek().weight;
				} else {
					comp.clear(); 
					// CHECK find what influence this line can cause on parsers.
					// DONE  see comments in Inferencer.Cell.addScore(...).
				}
			}
		} else {
			ncomponent += gm.ncomponent;
			components.addAll(gm.components);
		}
		*/
	}
	
	
	/**
	 * Add a component.
	 * 
	 * @param weight    weight of the component
	 * @param component a component of the mixture of gaussians
	 */
	public void add(double weight, Map<String, Set<GaussianDistribution>> component) {
		components.add(new Component((short) ncomponent, weight, component));
		ncomponent++;
	}
	
	
	/**
	 * Add gaussian distributions to a component.
	 * 
	 * @param iComponent index of the component
	 * @param gaussians  a set of gaussian distributions
	 */
	public void add(int iComponent, Map<String, Set<GaussianDistribution>> gaussians) {
		Component comp = null;
		if ((comp = getComponent((short) iComponent)) != null) {
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : gaussians.entrySet()) {
				add(comp.multivnd, gaussian.getKey(), gaussian.getValue());
			}
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
		Component comp = null;
		if ((comp = getComponent((short) iComponent)) != null) {
			add(comp.multivnd, key, gausses);
		}
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
	 * @param iComponent index of the component
	 * @return
	 */
	public Component getComponent(short iComponent) {
		for (Component comp : components) {
			if (comp.id == iComponent) {
				return comp;
			}
		}
		return null;
	}

	
	/**
	 * Make a copy of this MoG. This will create a new instance of MoG.
	 * 
	 * @param deep boolean value, indicating deep (true) or shallow (false) copy
	 * @return
	 */
	public GaussianMixture copy(boolean deep) { return null; }
	
	
	/**
	 * Make a copy of this MoG.
	 * 
	 * @param des  a placeholder of MoG
	 * @param deep boolean value, indicating deep (true) or shallow (false) copy
	 */
	protected void copy(GaussianMixture des, boolean deep) {
		des.ncomponent = ncomponent;
		for (Component comp : components) {
			if (deep) {
				des.components.add(new Component(comp.id, comp.weight, copy(comp.multivnd)));
			} else {
				des.components.add(comp);
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
	public GaussianMixture replaceKeys(Map<String, String> keys) { return null; }
	
	
	/**
	 * Replace the key of the specific portion of the mixture of gaussians with the new key (by reference).
	 * 
	 * @param des  a placeholder of MoG
	 * @param keys pairs of (old-key, new-key)
	 */
	protected void replaceKeys(GaussianMixture des, Map<String, String> keys) {
		for (Component comp : components) {
			Map<String, Set<GaussianDistribution>> multivnd = 
					new HashMap<String, Set<GaussianDistribution>>();
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
				String key = gaussian.getKey();
				if (keys.containsKey(key)) {
					add(multivnd, keys.get(key), gaussian.getValue());
				} else {
					add(multivnd, key, gaussian.getValue());
				}
			}
			des.components.add(new Component((short) des.ncomponent, comp.weight, multivnd));
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
	public GaussianMixture replaceAllKeys(String newkey) { return null; }
	
	
	/**
	 * Replace all of the existing keys with a new key (by reference).
	 * 
	 * @param des    a placeholder of MoG
	 * @param newkey the new key
	 */
	protected void replaceAllKeys(GaussianMixture des, String newkey) {
		for (Component comp : components) {
			Map<String, Set<GaussianDistribution>> multivnd = 
					new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> gausses = new HashSet<GaussianDistribution>();
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
				gausses.addAll(gaussian.getValue());
			}
			multivnd.put(newkey, gausses);
			des.components.add(new Component((short) des.ncomponent, comp.weight, multivnd));
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
	public GaussianMixture mulForInsideOutside(GaussianMixture gm, String key, boolean deep) {
		GaussianMixture amixture = this.copy(deep);
		// calculating inside score can always remove some portions, but calculating outside score
		// can not, because the rule ROOT->N has the dummy outside score for ROOT (one component but
		// without gaussians) and the rule weight does not contain "P" portion. Here is hardcoding
		if (gm.components.size() == 1 && gm.size(0) == 0) {
			return amixture;
		}
		// the following is the general case
		for (Component comp : amixture.components) {
			double logsum = Double.NEGATIVE_INFINITY;
			GaussianDistribution gd = comp.squeeze(key);
			for (Component comp1 : gm.components) {
				GaussianDistribution gd1 = comp1.squeeze(null);
				gm.size(0);
				double logcomp = comp1.weight + gd.mulAndMarginalize(gd1);
				logsum = FunUtil.logAdd(logsum, logcomp);
			}
			// CHECK Math.log(Math.exp(a) * b)
			comp.weight += logsum;
			comp.multivnd.remove(key);
		}
		return amixture;
	}
	/*
	public GaussianMixture mulForInsideOutside(GaussianMixture gm, String key, boolean deep) {
		GaussianMixture amixture = this.copy(deep);
		double logsum = gm.marginalize(true);
		for (Component comp : amixture.components) {
			// CHECK Math.log(Math.exp(a) * b)
			comp.weight += logsum;
			comp.multivnd.remove(key);
		}
		return amixture;
	}
	*/
	
	
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
	public static GaussianMixture mulAndMarginalize(GaussianMixture gm0, GaussianMixture gm1, 
			Map<String, String> keys0, Map<String, String> keys1) {
		if (gm0 == null || gm1 == null) { return null; }
		gm0 = gm0.replaceKeys(keys0);
		gm1 = gm1.replaceKeys(keys1);
		GaussianMixture gm = gm0.multiply(gm1);
		
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
	public GaussianMixture multiply(GaussianMixture multiplier) { return null; }
	
	
	/**
	 * Multiply this MoG by the given mixtures of gaussians.
	 * 
	 * @param des        a placeholder of MoG
	 * @param multiplier the other mixture of gaussians
	 */
	protected void multiply(GaussianMixture des, GaussianMixture multiplier) {
		for (Component comp0 : components) {
			for (Component comp1 : multiplier.components) {
				Map<String, Set<GaussianDistribution>> multivnd = 
						multiply(comp0.multivnd, comp1.multivnd);
				// CHECK Math.log(Math.exp(a) * Math.exp(b))
				double weight = comp0.weight + comp1.weight;
				des.components.add(new Component((short) des.ncomponent, weight, multivnd));
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
			for (Component comp : components) {
				logval = FunUtil.logAdd(logval, comp.weight);
			}
			return logval;
		} else {
			double value = 0;
			for (Component comp : components) {
				value += Math.exp(comp.weight);
			}
			return value;
		}
	}
	
	
	/**
	 * Marginalize the specific portions of the mixture of gaussians.
	 * 
	 * @param keys which map to the portions, to be marginalized, of the mixture of gaussians
	 */
	public void marginalize(Set<String> keys) {
		for (Component comp : components) {
			for (String key : keys) {
				comp.multivnd.remove(key);
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
			Component comp = components.poll();
			comp.clear();
		}
		ncomponent = 1;
		// CHECK Math.exp(0)
		Component comp = components.peek();
		comp.clear();
		comp.weight = 0;
		comp.id = 0;
	}
	
	
	/**
	 * Merge the same components, which could be resulted in by the marginalization, 
	 * of the mixture of gaussians (implemented by reference).
	 * 
	 * @param gm mixture of gaussians
	 * @return
	 */
	public static GaussianMixture merge(GaussianMixture gm) {
		GaussianMixture amixture = new GaussianMixture((short) 0); // POOL
		for (Component comp : gm.components) {
			int idx = amixture.contains(comp.multivnd);
			if (idx < 0) {
				amixture.components.add(new Component((short) amixture.ncomponent, comp.weight, comp.multivnd));
				amixture.ncomponent++;
			} else {
				Component acomp = amixture.getComponent((short) idx);
				// CHECK Math.log(Math.exp(a) + Math.exp(b))
				acomp.weight = FunUtil.logAdd(acomp.weight, comp.weight);
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
		for (Component comp : components) {
			if (isEqual(component, comp.multivnd)) {
				return comp.id;
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
		for (Component comp : components) {
			value = 1.0;
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
				for (GaussianDistribution gd : gaussian.getValue()) {
					value *= gd.eval(sample, normal);
				}
			}
			ret += Math.exp(comp.weight) * value;
		}
		return ret;
	}
	
	
	/**
	 * Eval the MoG using the given sample. Pay attention to the second parameter.
	 * 
	 * @param sample the sample, should be null
	 * @param normal whether the sample is from N(0, 1) (true) or not (false); 
	 * 		   when the sample is null, normal is used to indicate the return value in logarithm (true) or decimal (false)
	 * @return
	 */
	public double eval(Map<String, List<Double>> sample, boolean normal) {
		if (sample == null) { return eval(normal); }
		
		double ret = 0.0, value;
		for (Component comp : components) {
			value = 1.0;
			// TODO This is not correct because every component contains the same variables but with different parameters.
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
				List<Double> slice = sample.get(gaussian.getKey());
				for (GaussianDistribution gd : gaussian.getValue()) {
					value *= gd.eval(slice, normal);
					if (value < 0) {
						logger.error("\n---derivate the mixting weight---\n" + this + "\n");
					}
				}
			}
			ret += Math.exp(comp.weight) * value;
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
		for (Component comp : components) {
			if (comp.multivnd.size() > 0) {
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
	public double derivateMixingWeight(Map<String, List<Double>> sample, int iComponent, boolean normal) {
		double value = 1.0;
		Component comp = getComponent((short) iComponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
			List<Double> slice = sample.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				value *= gd.eval(slice, normal);
				if (value < 0) {
					logger.error("\n---derivate the mixting weight---\n" + this + "\n");
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
	public void derivative(boolean cumulative, int iComponent, double factor, 
			Map<String, List<Double>> sample, Map<String, List<Double>> ggrads, List<Double> wgrads, boolean normal) {
		if (!cumulative && iComponent == 0) { // CHECK stupid if...else...
			wgrads.clear();
			for (int i = 0; i < ncomponent; i++) {
				wgrads.add(0.0);
			}
		}
		Component comp = getComponent((short) iComponent);
		double weight = Math.exp(comp.weight);
		double penalty = Params.reg ? (Params.l1 ? Params.wdecay * weight : Params.wdecay * Math.pow(weight, 2)) : 0.0;
		double dMixingW = factor * weight * derivateMixingWeight(sample, iComponent, normal);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
			List<Double> slice = sample.get(gaussian.getKey());
			List<Double> grads = ggrads.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				gd.derivative(dMixingW, slice, grads, cumulative, normal);
				// break; // CHECK only one gaussian is allowed
			}
		}
		dMixingW += penalty;
		wgrads.set(iComponent, wgrads.get(iComponent) + dMixingW);
	}
	
	
	/**
	 * Update parameters using the gradient.
	 * 
	 * @param icomponent index of the component of MoG
	 * @param ggrads     gradients of the parameters of gaussians
	 * @param wgrads     gradients of the mixing weights of MoG
	 * @param minexp     minimum exponent representing the exponential mixing weight
	 */
	public void update(int iComponent, Map<String, List<Double>> ggrads, List<Double> wgrads, double minexp) {
		Component comp = getComponent((short) iComponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
			List<Double> grads = ggrads.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				gd.update(grads);
			}
		}
		comp.weight += wgrads.get(iComponent);
		comp.weight = comp.weight < minexp ? minexp : comp.weight;
	}
	
	/**
	 * Allocate memory space for gradients.
	 * 
	 * @return gradients holder
	 */
	public List<Map<String, List<Double>>> zeroslike() {
		List<Map<String, List<Double>>> grads = new ArrayList<Map<String, List<Double>>>(ncomponent);
		for (Component comp : components) {
			Map<String, List<Double>> gcomp = new HashMap<String, List<Double>>(comp.multivnd.size(), 1);
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
				if (gaussian.getValue().size() > 1) { logger.error("Invalid rule weight.\n"); }
				for (GaussianDistribution gd : gaussian.getValue()) {
					List<Double> grad = new ArrayList<Double>(gd.dim * 2);
					for (int i = 0; i < gd.dim * 2; i++) {
						grad.add(0.0); // preallocate memo
					}
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
	public List<HashMap<String, List<Double>>> zeroslike(int iComponent) {
		Component comp = getComponent((short) iComponent);
		List<HashMap<String, List<Double>>> holder = new ArrayList<HashMap<String, List<Double>>>(2);
		HashMap<String, List<Double>> sample = new HashMap<String, List<Double>>(comp.multivnd.size(), 1);
		HashMap<String, List<Double>> truths = new HashMap<String, List<Double>>(comp.multivnd.size(), 1);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
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
	public void sample(int iComponent, Map<String, List<Double>> sample, Map<String, List<Double>> truths, Random rnd) {
		Component comp = getComponent((short) iComponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
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
	public void restoreSample(int iComponent, Map<String, List<Double>> sample, Map<String, List<Double>> truths) {
		Component comp = getComponent((short) iComponent);
		for (Map.Entry<String, Set<GaussianDistribution>> gaussian : comp.multivnd.entrySet()) {
			List<Double> slice = sample.get(gaussian.getKey());
			List<Double> truth = truths.get(gaussian.getKey());
			for (GaussianDistribution gd : gaussian.getValue()) {
				gd.restoreSample(slice, truth);
				// break; // CHECK only one gaussian is allowed
			}
		}
	}
	
	
	public int dim(int iComponent, String key) {
		Component comp = null;
		if ((comp = getComponent((short) iComponent)) != null) {
			Set<GaussianDistribution> gausses = comp.multivnd.get(key);
			for (GaussianDistribution gd : gausses) {
				return gd.dim;
			}
		}
		return -1;
	}
	
	
	public int ncomponent() {
		return ncomponent;
	}
	
	
	public PriorityQueue<Component> components() {
		return components;
	}
	
	public int size(int iComponent) {
		Component comp = null;
		if ((comp = getComponent((short) iComponent)) != null) {
			return comp.multivnd.size();
		}
		return -1;
	}
	
	
	public void setWeight(int iComponent, double weight) {
		Component comp = null;
		if ((comp = getComponent((short) iComponent)) != null) {
			comp.setWeight(weight);
		}
	}
	
	
	public double getWeight(int iComponent) {
		Component comp = null;
		if ((comp = getComponent((short) iComponent)) != null) {
			return comp.weight;
		}
		return 0.0;
	}
	
	
	public void setWeights(double weight) {
		for (Component comp : components) {
			comp.setWeight(weight);
		}
	}
	
	
	public List<Double> getWeights() {
		List<Double> weights = new ArrayList<Double>(ncomponent);
		for (Component comp : components) {
			weights.add(comp.weight);
		}
		return weights;
	}
	
	
	public List<Map<String, Set<GaussianDistribution>>> getMixture() {
		List<Map<String, Set<GaussianDistribution>>> mixture = 
				new ArrayList<Map<String, Set<GaussianDistribution>>>(ncomponent);
		for (Component comp : components) {
			mixture.add(comp.multivnd);
		}
		return mixture;
	}
	
	
	public boolean isValid(short ncomp) {
		return (components != null && components.size() == ncomponent);
	}
	
	
	public void destroy(short ncomp) {
		clear();
		components = null;
	}
	
	
	public void clear(short ncomp) {
		clear();
	}
	
	
	/**
	 * Memory clean.
	 */
	public void clear() {
		this.bias = 0.0;
		this.ncomponent = 0;
		if (components != null) {
			for (Component comp : components) {
				comp.clear();
			}
			this.components.clear();
		}
	}
	
	
	public short getKey() {
		return key;
	}
	
	
	public void setKey(short key) {
		this.key = key;
	}
	
	
	public double getBias() {
		return bias;
	}
	

	public void setBias(double bias) {
		this.bias = bias;
	}
	

	public String toString(boolean simple, int nfirst) {
		if (simple) {
			StringBuffer sb = new StringBuffer();
			sb.append("GM [ncomponent=" + ncomponent + ", weights=" + 
					FunUtil.double2str(getWeights(), LVeGLearner.precision, nfirst, true, true));
			sb.append("]");
			return sb.toString();
		} else {
			return toString();
		}
	}
	
	
	@Override
	public String toString() {
		return "GM [bias=" + bias + ", ncomponent=" + ncomponent + ", weights=" + 
				FunUtil.double2str(getWeights(), LVeGLearner.precision, -1, true, true) + ", mixture=" + getMixture() + "]";
	}
	
	/*
	// http://stackoverflow.com/questions/17804704/notserializableexception-on-anonymous-class
	protected static Comparator<Component> wcomparator = new Comparator<Component>() {
		@Override
		public int compare(Component o1, Component o2) {
			double diff = o1.weight - o2.weight;
			return diff > 0 ? -1 : (diff < 0 ? 1 : 0);
		}
	};
	*/
	protected static Comparator<Component> wcomparator = new PriorityComparator<Component>();
	protected static class PriorityComparator<T> implements Comparator<T>, Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = -8813462045781155697L;
		@Override
		public int compare(T o1, T o2) {
			double diff = ((Component) o1).weight - ((Component) o2).weight;
			return diff > 0 ? -1 : (diff < 0 ? 1 : 0);
		}
	}
	public static class Component implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = -1211997783170967017L;
		protected short id;
		protected double weight;
		Map<String, Set<GaussianDistribution>> multivnd; // Multivariate Normal Distribution
		
		public Component(short id, double weight, Map<String, Set<GaussianDistribution>> multivnd) {
			this.id = id;
			this.weight = weight;
			this.multivnd = multivnd;
		}
		
		public GaussianDistribution squeeze(String key) {
			Set<GaussianDistribution> gaussians = null;
			if (key == null) {
				for (Map.Entry<String, Set<GaussianDistribution>> gaussian : multivnd.entrySet()) {
					gaussians = gaussian.getValue();
					break;
				}
			} else {
				gaussians = multivnd.get(key);
			}
			if (gaussians != null) {
				for (GaussianDistribution gd : gaussians) {
					return gd;
				}
			}
			return null;
		}
		
		public void setWeight(double weight) {
			this.weight = weight;
		}
		
		public double getWeight() {
			return weight;
		}
		
		public void clear() {
			id = -1;
			weight = Double.NEGATIVE_INFINITY;
			for (Map.Entry<String, Set<GaussianDistribution>> gaussian : multivnd.entrySet()) {
				Set<GaussianDistribution> value = null;
				if ((value = gaussian.getValue()) != null) {
					for (GaussianDistribution gd : value) {
						if (gd != null) { 
							gd.clear(); 
//							GaussianDistribution.returnObject(gd); // POOL
						}
					}
					value.clear();
				}
			}
			multivnd.clear();
		}
		
		@Override
		public String toString() {
			return "GM [id=" + id + ", weight=" + String.format( "%." + LVeGLearner.precision + "f", weight) + ", multivnd=" + multivnd + "]";
		}
	}
	
}
