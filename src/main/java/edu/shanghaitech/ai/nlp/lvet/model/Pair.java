package edu.shanghaitech.ai.nlp.lvet.model;

import java.io.Serializable;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.impl.RuleTable;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lvet.LVeTTrainer;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class Pair extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7779331963572971480L;
	public static final String LEADING = "<s>";
	public static final String ENDING = "</s>";
	public static int LEADING_IDX, ENDING_IDX;
	
	protected Map<GrammarRule, GrammarRule> edgeMap;
	protected RuleTable<?> edgeTable;
	
	protected List<GrammarRule>[] edgesWithP;
	protected List<GrammarRule>[] edgesWithC;
	
	protected Optimizer optimizer;
	public Numberer numberer;
	public int ntag;
	
	public Pair() {
		this.edgeTable = new RuleTable<>(UnaryGrammarRule.class);
		this.edgeMap = new HashMap<>();
	}
	
	protected abstract void initialize(); 
	
	public abstract void postInitialize();
	
	public abstract void tallyTaggedWords(List<TaggedWord> words);
	
	
	public void initializeOptimizer() {
		for (GrammarRule edge : edgeTable.keySet()) {
			optimizer.addRule(edge);
		}
	}
	
	public void addEdge(UnaryGrammarRule edge) {
		if (edgesWithP[edge.lhs].contains(edge)) { return; }
		edgesWithP[edge.lhs].add(edge);
		edgesWithC[edge.rhs].add(edge);
		edgeMap.put(edge, edge);
	}
	
	
	/**
	 * @param idParent id of the left hand side of the unary rule
	 * @param idChild id of the right hand side of the unary rule
	 * @param context  null can be returned (true) or not (false), which can be used to check if the given query rule is valid or not
	 * @return
	 */
	public GaussianMixture getEdgeWeight(short idParent, int idChild, RuleType type, boolean context) {
		GrammarRule rule = getEdge(idParent, idChild, type);
		if (rule != null) {
			return rule.getWeight();
		} 
		if (!context) {
			logger.warn("\nUnary Rule NOT Found: [P: " + idParent + ", UC: " + idChild + ", TYPE: " + type + "]\n");
			GaussianMixture weight = GrammarRule.rndRuleWeight(type, (short) -1, (short) -1);
			/*weight.setWeights(Double.NEGATIVE_INFINITY);*/
			weight.setWeights(LVeTTrainer.minmw);
			return weight;
		} else {
			return null;
		}
	}
	
	public GrammarRule getEdge(short idParent, int idChild, RuleType type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return edgeMap.get(rule);
	}
	
	public Map<GrammarRule, GrammarRule> getEdgeMap() {
		return edgeMap;
	}
	
	public List<GrammarRule> getEdgeWithP(int itag) {
		return edgesWithP[itag];
	}
	
	public List<GrammarRule> getEdgeWithC(int itag) {
		return edgesWithC[itag];
	}
	
	public boolean containsURule(short idParent, int idChild, RuleType type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return edgeTable.containsKey(rule);
	}
	
	public void addCount(short idParent, int idChild, EnumMap<RuleUnit, GaussianMixture> count, RuleType type, short isample, boolean withTree) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		addCount(rule, count, isample, withTree);
	}
	
	public Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> getCount(short idParent, int idChild, boolean withTree, RuleType type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return getCount(rule, withTree);
	}
	
	public void addCount(GrammarRule rule, EnumMap<RuleUnit, GaussianMixture> count, short isample, boolean withTree) {
		optimizer.addCount(rule, count, isample, withTree);
	}
	
	public Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> getCount(GrammarRule rule, boolean withTree) {
		return optimizer.getCount(rule, withTree);
	}
	
	public void setOptimizer(Optimizer optimizer) {
		this.optimizer = optimizer;
	}
	
	public Optimizer getOptimizer() {
		return optimizer;
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
	
	public void shutdown() {
		if (optimizer != null) {
			optimizer.shutdown();
		}
	}

}
