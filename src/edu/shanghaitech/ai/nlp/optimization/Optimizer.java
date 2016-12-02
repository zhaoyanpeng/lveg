package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public class Optimizer extends Recorder {
	/**
	 * Pseudo counts of grammar rules given the parse tree (countWithT) or the sentence (countWithS).
	 */
	private Map<GrammarRule, List<Map<String, GaussianMixture>>> countsWithT;
	private Map<GrammarRule, List<Map<String, GaussianMixture>>> countsWithS;
	
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
	
	private SGDForMoG minimizer;
	
	
	private Optimizer() {
		this.countsWithT = new HashMap<GrammarRule, List<Map<String, GaussianMixture>>>();
		this.countsWithS = new HashMap<GrammarRule, List<Map<String, GaussianMixture>>>();
		this.ruleSet = new HashSet<GrammarRule>();
	}
	
	
	public Optimizer(Random random) {
		this();
		this.random = random;
		this.minimizer = new SGDForMoG(random);
	}
	
	
	public Optimizer(Random random, short nsample, double lr) {
		this();
		this.random = random;
		this.nsample = nsample;
		this.globalLR = lr;
		this.minimizer = new SGDForMoG(random, nsample, lr);
	}
	
	
	/**
	 * @param scoresOfST the parse tree score (odd index) and the sentence score (even index).
	 */
	public void applyGradientDescent(List<Double> scoresOfST) {
		List<Map<String, GaussianMixture>> countWithT, countWithS;
		for (GrammarRule rule : ruleSet) {
			countWithT = countsWithT.get(rule);
			countWithS = countsWithS.get(rule);
			minimizer.optimize(rule, countWithT, countWithS, scoresOfST);
		}
		reset();
	}
	
	
	/**
	 * Stochastic gradient descent.
	 */
	public void applyGradientDescent() {
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
	 * @param rule the rule that needs optimizing.
	 */
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		List<Map<String, GaussianMixture>> batchWithT = new ArrayList<Map<String, GaussianMixture>>();
		List<Map<String, GaussianMixture>> batchWithS = new ArrayList<Map<String, GaussianMixture>>();
		countsWithT.put(rule, batchWithT);
		countsWithS.put(rule, batchWithS);
	}
	
	
	/**
	 * @param rule     the grammar rule
	 * @param scores   which contains 1) key GrammarRule.Unit.P maps to the outside score of the parent node
	 * 					2) key GrammarRule.Unit.UC/C (LC) maps to the inside score (of the left node) if the rule is unary (binary)
	 * 					3) key GrammarRule.Unit.RC maps to the inside score of the right node if the rule is binary, otherwise null
	 * @param withTree type of the expected pseudo count
	 */
	public void addCount(GrammarRule rule, Map<String, GaussianMixture> scores, boolean withTree) {
		List<Map<String, GaussianMixture>> batch = null;
		Map<GrammarRule, List<Map<String, GaussianMixture>>> count = withTree ? countsWithT : countsWithS;
		if (rule != null && (batch = count.get(rule)) != null) {
			batch.add(scores);
			return;
		}
		logger.error("Not a valid grammar rule.");
	}
	
	
	/**
	 * The method for debugging.
	 * 
	 * @param rule     the grammar rule
	 * @param withTree type of the expected count
	 * @return
	 */
	public List<Map<String, GaussianMixture>> getCount(GrammarRule rule, boolean withTree) {
		List<Map<String, GaussianMixture>> batch = null;
		Map<GrammarRule, List<Map<String, GaussianMixture>>> count = withTree ? countsWithT : countsWithS;
		if (rule != null && (batch = count.get(rule)) != null) {
			return batch;
		}
		logger.error("Not a valid grammar rule or the rule was not found.");
		return null;
	}
	
	
	public void reset() {
	}
	
	
	/**
	 * Counts are only used once for a batch.
	 */
	public void resetRuleCount() {
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
