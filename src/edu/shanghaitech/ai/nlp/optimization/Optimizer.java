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
public class Optimizer extends Recorder {
	/**
	 * Pseudo counts of grammar rules given the parse tree (countWithT) or the sentence (countWithS).
	 */
	private Map<GrammarRule, Batch> cntsWithT;
	private Map<GrammarRule, Batch> cntsWithS;
	
	/**
	 * ruleset contains all the rules that are need to be optimized, it is 
	 * used to quickly index the rules.
	 */
	private Set<GrammarRule> ruleSet;
	
	private SGDForMoG minimizer;
	
	
	private Optimizer() {
		this.cntsWithS = new HashMap<GrammarRule, Batch>();
		this.cntsWithT = new HashMap<GrammarRule, Batch>();
		this.ruleSet = new HashSet<GrammarRule>();
	}
	
	
	public Optimizer(Random random) {
		this();
		this.minimizer = new SGDForMoG(random);
	}
	
	
	public Optimizer(Random random, short nsample) {
		this();
		this.minimizer = new SGDForMoG(random, nsample);
	}
	
	
	/**
	 * Stochastic gradient descent.
	 * 
	 * @param scoresOfST the parse tree score (odd index) and the sentence score (even index)
	 */
	@SuppressWarnings("unused")
	public void applyGradientDescent(List<Double> scoresST) {
		if (scoresST.size() == 0) { return; }
		long start, ttime;
		int count = 0, total = ruleSet.size();
		
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			cntWithT = cntsWithT.get(rule);
			cntWithS = cntsWithS.get(rule);
			if (cntWithT.size() == 0 && cntWithS.size() == 0) { continue; }
//			logger.trace(rule + "\t" + count + "\tof " + total + "..."); // DEBUG
//			start = System.currentTimeMillis();
			minimizer.optimize(rule, cntWithT, cntWithS, scoresST);
//			ttime = System.currentTimeMillis() - start;
//			logger.trace("gd consumed " + (ttime / 1000) + "s\n"); // DEBUG
			count++;
		}
		reset();
	}
	
	
	/**
	 * @param rule the rule that needs optimizing.
	 */
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		Batch batchWithT = new Batch(-1);
		Batch batchWithS = new Batch(-1);
		cntsWithT.put(rule, batchWithT);
		cntsWithS.put(rule, batchWithS);
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
