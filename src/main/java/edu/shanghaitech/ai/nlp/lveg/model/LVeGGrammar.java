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
	
	protected RuleTable<?> unaryRuleTable;
	protected RuleTable<?> binaryRuleTable;
	
	protected List<GrammarRule>[] unaryRulesWithP;
	protected List<GrammarRule>[] unaryRulesWithC;
	
	protected List<GrammarRule>[] binaryRulesWithP;
	protected List<GrammarRule>[] binaryRulesWithLC;
	protected List<GrammarRule>[] binaryRulesWithRC;
	
	public int nTag;
	public Numberer numberer;
	
	
	/**
	 * Needed when we want to find a rule and access its statistics.
	 * we first construct a rule, which is used as the key, and use 
	 * the key to find the real rule that contains more information.
	 */
	protected Map<GrammarRule, GrammarRule> unaryRuleMap;
	protected Map<GrammarRule, GrammarRule> binaryRuleMap;
	
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
		return binaryRuleMap.get(rule);
	}
	
	
	public GrammarRule getUnaryRule(short idParent, short idChild, byte type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return unaryRuleMap.get(rule);
	}
	
	
	public Map<GrammarRule, GrammarRule> getBinaryRuleMap() {
		return binaryRuleMap;
	}
	
	
	public Map<GrammarRule, GrammarRule> getUnaryRuleMap() {
		return unaryRuleMap;
	}
	
	
	public List<GrammarRule> getChainSumUnaryRulesWithC(int idTag) {
		return chainSumUnaryRulesWithC[idTag];
	}
	
	
	public List<GrammarRule> getChainSumUnaryRulesWithP(int idTag) {
		return chainSumUnaryRulesWithP[idTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithRC(int idTag) {
		return binaryRulesWithRC[idTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithLC(int idTag) {
		return binaryRulesWithLC[idTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithP(int idTag) {
		return binaryRulesWithP[idTag];
	}
	
	
	public List<GrammarRule> getUnaryRuleWithP(int idTag) {
		return unaryRulesWithP[idTag];
	}
	
	
	public List<GrammarRule> getUnaryRuleWithC(int idTag) {
		return unaryRulesWithC[idTag];
	}
	
	public void addBinaryRule(BinaryGrammarRule rule) {
		if (binaryRulesWithP[rule.lhs].contains(rule)) { return; }
		binaryRulesWithP[rule.lhs].add(rule);
		binaryRulesWithLC[rule.lchild].add(rule);
		binaryRulesWithRC[rule.rchild].add(rule);
		binaryRuleMap.put(rule, rule);
		optimizer.addRule(rule);
	}
	
	
	public void addUnaryRule(UnaryGrammarRule rule) {
		if (unaryRulesWithP[rule.lhs].contains(rule)) { return; }
		unaryRulesWithP[rule.lhs].add(rule);
		unaryRulesWithC[rule.rhs].add(rule);
		unaryRuleMap.put(rule, rule);
		optimizer.addRule(rule);
	}
	
	/**
	 * Tally (go through) the rules existing in the parse tree.
	 * 
	 * @param tree a parse tree
	 */
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
	
	
	
	
	
	public void addCount(short idParent, short idChild, byte type, Map<String, GaussianMixture> count, short isample, boolean withTree) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		addCount(rule, count, isample, withTree);
	}
	
	public Map<Short, List<Map<String, GaussianMixture>>> getCount(short idParent, short idChild, byte type, boolean withTree) {
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
