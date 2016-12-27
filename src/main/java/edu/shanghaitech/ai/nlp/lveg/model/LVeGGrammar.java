package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.RuleTable;
import edu.shanghaitech.ai.nlp.lveg.impl.BinaryGrammarRule;
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
	protected Optimizer optimizer;
	
	protected RuleTable<?> uRuleTable;
	protected RuleTable<?> bRuleTable;
	
	protected List<GrammarRule>[] uRulesWithP;
	protected List<GrammarRule>[] uRulesWithC;
	
	protected List<GrammarRule>[] bRulesWithP;
	protected List<GrammarRule>[] bRulesWithLC;
	protected List<GrammarRule>[] bRulesWithRC;
	
	public int nTag;
	public Numberer numberer;
	
	
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

	
	public GaussianMixture getBinaryRuleWeight(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		if (rule != null) {
			return rule.getWeight();
		}
//		logger.warn("Binary Rule NOT Found: [P: " + idParent + ", LC: " + idlChild + ", RC: " + idrChild + "]\n");
		return null;
	}
	
	
	public GaussianMixture getUnaryRuleWeight(short idParent, short idChild, byte type) {
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		if (rule != null) {
			return rule.getWeight();
		}
//		logger.warn("Unary Rule NOT Found: [P: " + idParent + ", C: " + idChild + ", TYPE: " + type + "]\n");
		return null;
	}

	
	public GrammarRule getBinaryRule(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = new BinaryGrammarRule(idParent, idlChild, idrChild);
		return bRuleMap.get(rule);
	}
	
	
	public GrammarRule getUnaryRule(short idParent, int idChild, byte type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return uRuleMap.get(rule);
	}
	
	
	public Map<GrammarRule, GrammarRule> getBinaryRuleMap() {
		return bRuleMap;
	}
	
	
	public Map<GrammarRule, GrammarRule> getUnaryRuleMap() {
		return uRuleMap;
	}
	
	
	public List<GrammarRule> getChainSumUnaryRulesWithC(int idTag) {
		return chainSumUnaryRulesWithC[idTag];
	}
	
	
	public List<GrammarRule> getChainSumUnaryRulesWithP(int idTag) {
		return chainSumUnaryRulesWithP[idTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithRC(int idTag) {
		return bRulesWithRC[idTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithLC(int idTag) {
		return bRulesWithLC[idTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithP(int idTag) {
		return bRulesWithP[idTag];
	}
	
	
	public List<GrammarRule> getUnaryRuleWithP(int idTag) {
		return uRulesWithP[idTag];
	}
	
	
	public List<GrammarRule> getUnaryRuleWithC(int idTag) {
		return uRulesWithC[idTag];
	}
	
	protected abstract void initialize(); 
	
	public void addBinaryRule(BinaryGrammarRule rule) {}
	
	public abstract void addUnaryRule(UnaryGrammarRule rule);	
	
	public abstract void tallyStateTree(Tree<State> tree);
	
	public abstract void postInitialize(double randomness);
	
	
	public void addCount(short idParent, short idlChild, short idrChild, Map<String, GaussianMixture> count, short isample, boolean withTree) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		addCount(rule, count, isample, withTree);
	}
	
	
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(short idParent, short idlChild, short idrChild, boolean withTree) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		return getCount(rule, withTree);
	}
	
	public void addCount(short idParent, int idChild, Map<String, GaussianMixture> count, byte type, short isample, boolean withTree) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		addCount(rule, count, isample, withTree);
	}
	
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(short idParent, int idChild, byte type, boolean withTree) {
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
