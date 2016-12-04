package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
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
public class Optimizer extends Recorder {
	/**
	 * Pseudo counts of grammar rules given the parse tree (countWithT) or the sentence (countWithS).
	 */
	private Map<GrammarRule, Batch> cntsWithT;
	private Map<GrammarRule, Batch> cntsWithS;
	
	private Map<GrammarRule, List<Map<String, GaussianMixture>>> countsWithT;
	private Map<GrammarRule, List<Map<String, GaussianMixture>>> countsWithS;
	
	/**
	 * ruleset contains all the rules that are need to be optimized, it is 
	 * used to quickly index the rules.
	 */
	private Set<GrammarRule> ruleSet;
	
	private SGDForMoG minimizer;
	private double lr;
	
	
	private Optimizer() {
		this.cntsWithS = new HashMap<GrammarRule, Batch>();
		this.cntsWithT = new HashMap<GrammarRule, Batch>();
//		this.countsWithT = new HashMap<GrammarRule, List<Map<String, GaussianMixture>>>();
//		this.countsWithS = new HashMap<GrammarRule, List<Map<String, GaussianMixture>>>();
		this.ruleSet = new HashSet<GrammarRule>();
	}
	
	
	public Optimizer(Random random) {
		this();
		lr = 0.02;
		this.minimizer = new SGDForMoG(random);
	}
	
	
	public Optimizer(Random random, short nsample, double lr) {
		this();
		this.lr = lr;
		this.minimizer = new SGDForMoG(random, nsample, lr);
	}
	
	
	/**
	 * Stochastic gradient descent.
	 * 
	 * @param scoresOfST the parse tree score (odd index) and the sentence score (even index).
	 */
	public void applyGradientDescent(List<Double> scoresOfST) {
		List<Map<String, GaussianMixture>> countWithT, countWithS;
		for (GrammarRule rule : ruleSet) {
			countWithT = countsWithT.get(rule);
			countWithS = countsWithS.get(rule);
			minimizer.optimize(rule, countWithT, countWithS, scoresOfST);
		}
	}
	
	
	/**
	 * @param rule the rule that needs optimizing.
	 */
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		Batch batchWithT = new Batch();
		Batch batchWithS = new Batch();
		cntsWithT.put(rule, batchWithT);
		cntsWithS.put(rule, batchWithS);
		
//		List<Map<String, GaussianMixture>> batchWithT = new ArrayList<Map<String, GaussianMixture>>();
//		List<Map<String, GaussianMixture>> batchWithS = new ArrayList<Map<String, GaussianMixture>>();
//		countsWithT.put(rule, batchWithT);
//		countsWithS.put(rule, batchWithS);
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
		logger.error("Not a valid grammar rule.\n");
	}
	
	
	/**
	 * @param rule     the grammar rule
	 * @param cnt      which contains 1) key GrammarRule.Unit.P maps to the outside score of the parent node
	 * 					2) key GrammarRule.Unit.UC/C (LC) maps to the inside score (of the left node) if the rule is unary (binary)
	 * 					3) key GrammarRule.Unit.RC maps to the inside score of the right node if the rule is binary, otherwise null
	 * @param idx      index of the sample in this batch
	 * @param withTree type of the expected pseudo count
	 */
	public void addCount(GrammarRule rule, Map<String, GaussianMixture> cnt, short idx, boolean withTree) {
		Batch batch = null;
		Map<GrammarRule, Batch> cnts = withTree ? cntsWithT : cntsWithS;
		if (rule != null && (batch = cnts.get(rule)) != null) {
			batch.add(idx, cnt);
			return;
		}
		logger.error("Not a valid grammar rule.\n");
	}
	
	
//	public Batch getCount(GrammarRule rule, boolean withT) {
//		Batch batch = null;
//		Map<GrammarRule, Batch> cnts = withT ? countsWithT : countsWithS;
//		if (rule != null && (batch = cnts.get(rule)) != null) {
//			return batch;
//		}
//		logger.error("Not a valid grammar rule or the rule was not found.\n");
//		return null;
//	}
	
	
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
		logger.error("Not a valid grammar rule or the rule was not found.\n");
		return null;
	}
	
	
	public void reset() {
	}
	
	
	/**
	 * Get set of the rules.
	 * 
	 * @return 
	 */
	public Set<GrammarRule> getRuleSet() {
		return ruleSet;
	}
	
	
	/**
	 * @author Yanpeng Zhao
	 *
	 */
	protected class Batch {
		protected Map<Short, List<Map<String, GaussianMixture>>> batch;
		
		public Batch() {
			batch = new HashMap<Short, List<Map<String, GaussianMixture>>>();
		}
		
		protected void add(short idx, Map<String, GaussianMixture> cnt) {
			List<Map<String, GaussianMixture>> cnts = null;
			if ((cnts = batch.get(idx)) != null) {
				cnts.add(cnt);
			} else {
				cnts = new ArrayList<Map<String, GaussianMixture>>();
				batch.put(idx, cnts);
			}
		}
	}
}
