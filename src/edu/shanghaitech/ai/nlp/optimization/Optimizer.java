package edu.shanghaitech.ai.nlp.optimization;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.UnaryGrammarRule;

/**
 * @author Yanpeng Zhao
 *
 */
public class Optimizer {
	/**
	 * count0 stores rule counts that are evaluated given the parse tree
	 * count1 stores rule counts that are evaluated without the parse tree 
	 */
	private Map<GrammarRule, Double> count0;
	private Map<GrammarRule, Double> count1;
	
	/**
	 * ruleset contains all the rules that are need to be optimized, it is 
	 * used to quickly index the rules.
	 */
	private Set<GrammarRule> ruleSet;
	
	/**
	 * All the objects share the same instance of Random, here the random 
	 * is used to sample the gradients, we provide the optional sampling  
	 * times and the choice that whether accumulating the gradients or not.
	 */
	protected Random random;
	private short nsample = 3;
	private boolean cumulative = true;
	
	// global learning rate
	protected double globalLR = 0.002;
	
	
	private Optimizer() {
		this.count0 = new HashMap<GrammarRule, Double>();
		this.count1 = new HashMap<GrammarRule, Double>();
		this.ruleSet = new HashSet<GrammarRule>();
	}
	
	
	public Optimizer(Random random) {
		this();
		this.random = random;
	}
	
	
	public Optimizer(Random random, short nsample, double lr) {
		this();
		this.random = random;
		this.nsample = nsample;
		this.globalLR = lr;
	}
	
	
	/**
	 * Stochastic gradient descent.
	 */
	public void applyGradientDescent() {
		double cnt0, cnt1;
		for (GrammarRule rule : ruleSet) {
			cnt0 = count0.get(rule);
			cnt1 = count1.get(rule);
			if (cnt0 == cnt1) { continue; }
			
			UnaryGrammarRule urule;
			if (rule instanceof  UnaryGrammarRule) {
				urule = (UnaryGrammarRule) rule;
			} else {
				urule = new UnaryGrammarRule((short) -1, (short) -1, (char) 0, null);
			}
			
			applyGradientDescent(rule.getWeight(), cnt0, cnt1);
		}
		resetRuleCount();
	}
	
	
	/**
	 * Stochastic gradient descent.
	 * 
	 * @param gm   the rule weight
	 * @param cnt0 conditional pseudo count
	 * @param cnt1 pseudo count
	 */
	private void applyGradientDescent(GaussianMixture gm, double cnt0, double cnt1) {
		for (short i = 0; i < nsample; i++) {
			double factor = 0.0;
			while (factor == 0.0) {
				gm.sample(random);
				factor = gm.eval();
			}
 			factor = (cnt1 - cnt0) / factor;
			gm.derivative(factor, cumulative);
		}
		gm.update(globalLR);
	}
	
	
	/**
	 * @param rule the rule need to be optimized.
	 */
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		count0.put(rule, 0.0);
		count1.put(rule, 0.0);
	}
	
	
	/**
	 * @param rule      the grammar rule
	 * @param increment which is added to the pseudo count
	 * @param withTree  type of the expected count
	 */
	public void addCount(GrammarRule rule, double increment, boolean withTree) {
		Map<GrammarRule, Double> count = withTree ? count0 : count1;
		if (rule != null && count.get(rule) != null) {
			count.put(rule, count.get(rule) + increment);
			return;
		}
		if (rule == null) {
			System.err.println("The Given Rule is NULL.");
		} else {
			System.err.println("Grammar Rule NOT Found: " + rule);
		}
	}
	
	
	/**
	 * A method for debugging.
	 * 
	 * @param rule     the grammar rule
	 * @param withTree type of the expected count
	 * @return
	 */
	public double getCount(GrammarRule rule, boolean withTree) {
		Map<GrammarRule, Double> count = withTree ? count0 : count1;
		if (rule != null && count.get(rule) != null) {
			return count.get(rule);
		}
		if (rule == null) {
			System.err.println("The Given Rule is NULL.");
		} else {
			System.err.println("Grammar Rule NOT Found: " + rule);
		}
		return -1.0;
	}
	
	
	/**
	 * Counts are only used once for a batch.
	 */
	public void resetRuleCount() {
		for (GrammarRule rule : ruleSet) {
			count0.put(rule, 0.0);
			count1.put(rule, 0.0);
		}
	}
	
	
	/**
	 * Get set of the rules.
	 * 
	 * @return 
	 */
	public Set<GrammarRule> getRuleSet() {
		return ruleSet;
	}
	
}
