package edu.shanghaitech.ai.nlp.optimization;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.GrammarRule;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public abstract class Optimizer extends Recorder {
	/**
	 * Pseudo counts of grammar rules given the parse tree (countWithT) or the sentence (countWithS).
	 */
	protected Map<GrammarRule, Batch> cntsWithT;
	protected Map<GrammarRule, Batch> cntsWithS;
	
	/**
	 * ruleset contains all the rules that are need to be optimized, it is 
	 * used to quickly index the rules.
	 */
	protected Set<GrammarRule> ruleSet;
	protected static Random rnd;
	protected static short maxsample = 2;
	
	/**
	 * Stochastic gradient descent.
	 * 
	 * @param scoresOfST the parse tree score (odd index) and the sentence score (even index)
	 */
	protected abstract void applyGradientDescent(List<Double> scoresST);
	
	/**
	 * @param scoreSandT the parse tree score (odd index) and the sentence score (even index)
	 * @param parallel   parallel (true) or serialize (false)
	 */
	protected abstract void evalGradients(List<Double> scoresST, boolean parallel);
	
	/**
	 * @param rule the rule that needs optimizing.
	 */
	protected abstract void addRule(GrammarRule rule);
	
	protected abstract void reset();
	
	
	/**
	 * @param rule  the grammar rule
	 * @param cnt   which contains 1) key GrammarRule.Unit.P maps to the outside score of the parent node
	 * 				 2) key GrammarRule.Unit.UC/C (LC) maps to the inside score (of the left node) if the rule is unary (binary)
	 * 				 3) key GrammarRule.Unit.RC maps to the inside score of the right node if the rule is binary, otherwise null
	 * @param idx   index of the sample in this batch
	 * @param withT type of the expected pseudo count
	 */
	public void addCount(GrammarRule rule, Map<String, GaussianMixture> cnt, short idx, boolean withT) {
		Batch batch = null;
		Map<GrammarRule, Batch> cnts = withT ? cntsWithT : cntsWithS;
		if (rule != null && (batch = cnts.get(rule)) != null) {
			batch.add(idx, cnt);
			return;
		}
		logger.error("Not a valid grammar rule.\n");
	}
	
	
	/**
	 * The method for debugging.
	 * 
	 * @param rule     the grammar rule
	 * @param withT type of the expected count
	 * @return
	 */
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(GrammarRule rule, boolean withT) {
		Batch batch = null;
		Map<GrammarRule, Batch> cnts = withT ? cntsWithT : cntsWithS;
		if (rule != null && (batch = cnts.get(rule)) != null) {
			return batch.batch;
		}
		logger.error("Not a valid grammar rule or the rule was not found.\n");
		return null;
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
