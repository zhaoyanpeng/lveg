package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.impl.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.impl.RuleTable;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class LVeGGrammar extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5874243553526905936L;
	protected RuleTable<?> uRuleTable;
	protected RuleTable<?> bRuleTable;
	
	protected List<GrammarRule>[] uRulesWithP;
	protected List<GrammarRule>[] uRulesWithC;
	
	protected List<GrammarRule>[] bRulesWithP;
	protected List<GrammarRule>[] bRulesWithLC;
	protected List<GrammarRule>[] bRulesWithRC;
	
	protected Optimizer optimizer;
	public Numberer numberer;
	public int ntag;
	
	/**
	 * Needed when we want to find a rule and access its statistics.
	 * we first construct a rule, which is used as the key, and use 
	 * the key to find the real rule that contains more information.
	 */
	protected Map<GrammarRule, GrammarRule> uRuleMap;
	protected Map<GrammarRule, GrammarRule> bRuleMap;
	
	/**
	 * For any nonterminals A \neq B \neq C, p(A->B) is computed as 
	 * p(A->B) + \sum_{C} p(A->C) \times p(C->B), in which p(A->B) 
	 * is zero if A->B does not exist, and the resulting new rules 
	 * are added to the unary rule set. Fields containing 'sum' are
	 * dedicated to the general CYK algorithm, and are dedicated to
	 * to the Viterbi algorithm if they contain 'Max'. However, the
	 * point is how to define the maximum between two MoGs.
	 */
	protected Set<GrammarRule> chainSumUnaryRules;
	protected List<GrammarRule>[] chainSumUnaryRulesWithP;
	protected List<GrammarRule>[] chainSumUnaryRulesWithC;
	
	protected abstract void initialize(); 
	
	public abstract void postInitialize();
	
	public abstract void tallyStateTree(Tree<State> tree);
	
	public abstract void addURule(UnaryGrammarRule rule);	
	
	public void addBRule(BinaryGrammarRule rule) {}
	
	public GaussianMixture getBRuleWeight(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = getBRule(idParent, idlChild, idrChild);
		if (rule != null) {
			return rule.getWeight();
		}
//		logger.warn("Binary Rule NOT Found: [P: " + idParent + ", LC: " + idlChild + ", RC: " + idrChild + "]\n");
		return null;
	}
	
	public GaussianMixture getURuleWeight(short idParent, short idChild, byte type) {
		GrammarRule rule = getURule(idParent, idChild, type);
		if (rule != null) {
			return rule.getWeight();
		}
//		logger.warn("Unary Rule NOT Found: [P: " + idParent + ", C: " + idChild + ", TYPE: " + type + "]\n");
		return null;
	}

	public GrammarRule getBRule(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = new BinaryGrammarRule(idParent, idlChild, idrChild);
		return bRuleMap.get(rule);
	}
	
	public GrammarRule getURule(short idParent, int idChild, byte type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return uRuleMap.get(rule);
	}
	
	public Map<GrammarRule, GrammarRule> getBRuleMap() {
		return bRuleMap;
	}
	
	public Map<GrammarRule, GrammarRule> getURuleMap() {
		return uRuleMap;
	}
	
	public List<GrammarRule> getChainSumUnaryRulesWithC(int itag) {
		return chainSumUnaryRulesWithC[itag];
	}
	
	public List<GrammarRule> getChainSumUnaryRulesWithP(int itag) {
		return chainSumUnaryRulesWithP[itag];
	}
	
	public List<GrammarRule> getBRuleWithRC(int itag) {
		return bRulesWithRC[itag];
	}
	
	public List<GrammarRule> getBRuleWithLC(int itag) {
		return bRulesWithLC[itag];
	}
	
	public List<GrammarRule> getBRuleWithP(int itag) {
		return bRulesWithP[itag];
	}
	
	public List<GrammarRule> getURuleWithP(int itag) {
		return uRulesWithP[itag];
	}
	
	public List<GrammarRule> getURuleWithC(int itag) {
		return uRulesWithC[itag];
	}
	
	public void addCount(short idParent, short idlChild, short idrChild, Map<String, GaussianMixture> count, short isample, boolean withTree) {
		GrammarRule rule = getBRule(idParent, idlChild, idrChild);
		addCount(rule, count, isample, withTree);
	}
	
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(short idParent, short idlChild, short idrChild, boolean withTree) {
		GrammarRule rule = getBRule(idParent, idlChild, idrChild);
		return getCount(rule, withTree);
	}
	
	public void addCount(short idParent, int idChild, Map<String, GaussianMixture> count, byte type, short isample, boolean withTree) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		addCount(rule, count, isample, withTree);
	}
	
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(short idParent, int idChild, boolean withTree, byte type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return getCount(rule, withTree);
	}
	
	public void addCount(GrammarRule rule, Map<String, GaussianMixture> count, short isample, boolean withTree) {
		optimizer.addCount(rule, count, isample, withTree);
	}
	
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(GrammarRule rule, boolean withTree) {
		return optimizer.getCount(rule, withTree);
	}
	
	public void setOptimizer(Optimizer optimizer) {
		this.optimizer = optimizer;
	}
	
	public void evalGradients(List<Double> scoreOfST) {
		optimizer.evalGradients(scoreOfST);
	}
	
	/**
	 * Apply stochastic gradient descent.
	 */
	public void applyGradientDescent(List<Double> scoreOfST) {
		optimizer.applyGradientDescent(scoreOfST);
	}
	
	/**
	 * Get the set of the rules.
	 */
	public Set<GrammarRule> getRuleSet() {
		return optimizer.getRuleSet();
	}

}
