package edu.shanghaitech.ai.nlp.optimization;

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
public class ParallelOptimizer extends Recorder {
	private static Random rnd;
	private static short maxsample;
	// which contains all the rules that need optimizing, it is used to quickly index the grammar rule.
	private Set<GrammarRule> ruleSet;
	// pseudo counts of grammar rules given the parse tree (countWithT) or the sentence (countWithS).
	private Map<GrammarRule, Batch> cntsWithT;
	private Map<GrammarRule, Batch> cntsWithS;
	// gradients
	private Map<GrammarRule, Gradient> gradients;
	
	private ParallelOptimizer() {
		this.cntsWithS = new HashMap<GrammarRule, Batch>();
		this.cntsWithT = new HashMap<GrammarRule, Batch>();
		this.ruleSet = new HashSet<GrammarRule>();
		this.gradients = new HashMap<GrammarRule, Gradient>();
	}
	
	
	public ParallelOptimizer(Random random) {
		this();
		rnd = random;
		maxsample = 1;
	}
	
	
	public ParallelOptimizer(Random random, short nsample) {
		this();
		rnd = random;
		maxsample = nsample;
	}
	
	
	/**
	 * Stochastic gradient descent.
	 * 
	 * @param scoresOfST the parse tree score (odd index) and the sentence score (even index)
	 */
	public void evalGradients(List<Double> scoreSandT) {
		if (scoreSandT.size() == 0) { return; }
		Gradient gradient;
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			boolean updated = false;
			cntWithT = cntsWithT.get(rule);
			cntWithS = cntsWithS.get(rule);
			for (short i = 0; i < Gradient.MAX_BATCH_SIZE; i++) {
				if (cntWithT.get(i) != null || cntWithS.get(i) != null) { 
					updated = true; 
					break; 
				} 
			}
			if (!updated) { continue; }
			gradient = gradients.get(rule);
			gradient.eval(rule, cntWithT, cntWithS, scoreSandT);
			// clear
			cntWithT.clear();
			cntWithS.clear();
		}
	}
	
	
	
	public void applyGradientDescent(Object placeholder) {
		Gradient gradient;
		for (GrammarRule rule : ruleSet) {
			gradient = gradients.get(rule);
			gradient.applyGradientDescent(rule);
		}
		reset();
	}
	
	
	/**
	 * @param rule the rule that needs optimizing.
	 */
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		Batch batchWithT = new Batch(Gradient.MAX_BATCH_SIZE);
		Batch batchWithS = new Batch(Gradient.MAX_BATCH_SIZE);
		cntsWithT.put(rule, batchWithT);
		cntsWithS.put(rule, batchWithS);
		Gradient gradient = new Gradient(rule, rnd, maxsample);
		gradients.put(rule, gradient);
	}
	
	
	/**
	 * @param rule     the grammar rule
	 * @param cnt      which contains 1) key GrammarRule.Unit.P maps to the outside score of the parent node
	 * 					2) key GrammarRule.Unit.UC/C (LC) maps to the inside score (of the left node) if the rule is unary (binary)
	 * 					3) key GrammarRule.Unit.RC maps to the inside score of the right node if the rule is binary, otherwise null
	 * @param idx      index of the sample in this batch
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
	 * The method is used for debugging purpose.
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
	
	
	public void reset() { 
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			if ((cntWithT = cntsWithT.get(rule)) != null) { cntWithT.clear(); }
			if ((cntWithS = cntsWithS.get(rule)) != null) { cntWithS.clear(); }
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
