package edu.shanghaitech.ai.nlp.lvet.model;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.impl.RuleTable;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lvet.impl.DirectedEdge;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Recorder;

public abstract class Pair implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7779331963572971480L;
	public static final String LEADING = "<s>";
	public static final String ENDING = ".";
	public static int LEADING_IDX, ENDING_IDX;
	protected Map<GrammarRule, GrammarRule> edgeMap;
	protected RuleTable<?> edgeTable;
	protected List<GrammarRule>[] edgesWithP;
	protected List<GrammarRule>[] edgesWithC;
	
	protected Optimizer optimizer;
	public Numberer numberer;
	public int ntag;
	
	public Pair() {
		this.edgeTable = new RuleTable<DirectedEdge>(DirectedEdge.class);
		this.edgeMap = new HashMap<GrammarRule, GrammarRule>();
	}
	
	protected abstract void initialize();
	public abstract void postInitialize();
	public abstract void tallyTaggedSample(List<TaggedWord> sample);
	
	public void initializeOptimizer() {
		for (GrammarRule edge : edgeTable.keySet()) {
			optimizer.addRule(edge);
		}
	}
	
	public void addEdge(DirectedEdge edge) {
		if (edgesWithP[edge.lhs].contains(edge)) { return; }
		edgesWithP[edge.lhs].add(edge);
		edgesWithC[edge.rhs].add(edge);
		edgeMap.put(edge, edge);
	}
	
	public GaussianMixture getEdgeWeight(short idParent, int idChild, byte type, boolean context) {
		GrammarRule rule = getEdge(idParent, idChild, type);
		if (rule != null) {
			return rule.getWeight();
		} 
		if (!context) {
			Recorder.logger.warn("\nUnary Rule NOT Found: [P: " + idParent + ", UC: " + idChild + ", TYPE: " + type + "]\n");
			GaussianMixture weight = GrammarRule.rndRuleWeight(type, (short) -1, (short) -1);
			/*weight.setWeights(Double.NEGATIVE_INFINITY);*/
			weight.setWeights(LVeGTrainer.minmw);
			return weight;
		} else {
			return null;
		}
	}
	
	public GrammarRule getEdge(short idParent, int idChild, byte type) {
		GrammarRule rule = new DirectedEdge(idParent, idChild, type);
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
	
	public boolean containsEdge(short idParent, int idChild, byte type) {
		GrammarRule rule = new DirectedEdge(idParent, idChild, type);
		return edgeTable.containsKey(rule);
	}
	
	public void addCount(short idParent, int idChild, Map<String, GaussianMixture> count, byte type, short isample, boolean withTree) {
		GrammarRule rule = new DirectedEdge(idParent, idChild, type);
		addCount(rule, count, isample, withTree);
	}
	
	public void addCount(GrammarRule rule, Map<String, GaussianMixture> count, short isample, boolean withTree) {
		optimizer.addCount(rule, count, isample, withTree);
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
	
	public void applyGradientDescent(List<Double> scoreOfST) {
		optimizer.applyGradientDescent(scoreOfST);
	}
	
	public Set<GrammarRule> getEdgeSet() {
		return optimizer.getRuleSet();
	}
	
	public void shutdown() {
		if (optimizer != null) {
			optimizer.shutdown();
		}
	}
	
}
