package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGGrammar implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public int nTag;
	protected Numberer numbererTag;
	
	protected RuleTable<?> unaryRuleTable;
	protected RuleTable<?> binaryRuleTable;
	
	private List<GrammarRule>[] unaryRulesWithP;
	private List<GrammarRule>[] unaryRulesWithC;
	
	private List<GrammarRule>[] binaryRulesWithP;
	private List<GrammarRule>[] binaryRulesWithLC;
	private List<GrammarRule>[] binaryRulesWithRC;
	
	/**
	 * Needed when we want to find a rule and access its statistics.
	 * we first construct a rule, which is used as the key, and use 
	 * the key to find the real rule that contains more information.
	 */
	private Map<GrammarRule, GrammarRule> unaryRuleMap;
	private Map<GrammarRule, GrammarRule> binaryRuleMap;
	
	/**
	 * count0 stores rule counts that are evaluated given the parse tree
	 * count1 stores rule counts that are evaluated without the parse tree 
	 */
	private Map<GrammarRule, Double> count0;
	private Map<GrammarRule, Double> count1;
	
	/**
	 * Rules with probabilities below this value will be filtered.
	 */
	private double filterThreshold;
	
	
	public LVeGGrammar(LVeGGrammar oldGrammar, double filterThreshold, int nTag) {
		this.unaryRuleTable  = new RuleTable(UnaryGrammarRule.class);
		this.binaryRuleTable = new RuleTable(BinaryGrammarRule.class);
		this.unaryRuleMap  = new HashMap<GrammarRule, GrammarRule>();
		this.binaryRuleMap = new HashMap<GrammarRule, GrammarRule>();
		this.filterThreshold = filterThreshold;
		
		if (nTag < 0) {
			this.numbererTag = Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET);
			this.nTag = numbererTag.size();
		} else {
			this.numbererTag = null;
			this.nTag = nTag;
		}
		
		if (oldGrammar != null) {
			// TODO
		} else {
			// TODO
		}
		
		initialize();
	}
	
	
	private void initialize() {
		this.unaryRulesWithP = new List[nTag];
		this.unaryRulesWithC = new List[nTag];
		
		this.binaryRulesWithP  = new List[nTag];
		this.binaryRulesWithLC = new List[nTag];
		this.binaryRulesWithRC = new List[nTag];
		
		for (int i = 0; i < nTag; i++) {
			unaryRulesWithP[i] = new ArrayList<GrammarRule>();
			unaryRulesWithC[i] = new ArrayList<GrammarRule>();
			
			binaryRulesWithP[i]  = new ArrayList<GrammarRule>();
			binaryRulesWithLC[i] = new ArrayList<GrammarRule>();
			binaryRulesWithRC[i] = new ArrayList<GrammarRule>();
		}
	}
	
	
	protected void addBinaryRule(BinaryGrammarRule rule) {
		if (binaryRulesWithP[rule.lhs].contains(rule)) { return; }
		binaryRulesWithP[rule.lhs].add(rule);
		binaryRulesWithLC[rule.lchild].add(rule);
		binaryRulesWithRC[rule.rchild].add(rule);
		
		binaryRuleMap.put(rule, rule);
	}
	
	
	protected void addUnaryRule(UnaryGrammarRule rule) {
		if (unaryRulesWithP[rule.lhs].contains(rule)) { return; }
		unaryRulesWithP[rule.lhs].add(rule);
		unaryRulesWithC[rule.rhs].add(rule);
		
		unaryRuleMap.put(rule, rule);
	}
	
	
	public void postInitialize(double randomness) {
		for (GrammarRule rule : unaryRuleTable.keySet()) {
			addUnaryRule((UnaryGrammarRule) rule);
			count0.put(rule, 0.0);
			count1.put(rule, 0.0);
		}
		
		for (GrammarRule rule : binaryRuleTable.keySet()) {
			addBinaryRule((BinaryGrammarRule) rule);
			count0.put(rule, 0.0);
			count1.put(rule, 0.0);
		}
	}
	
	
	/**
	 * Tally (go through and record) the rules existing in the parse tree.
	 * 
	 * @param tree the parse tree
	 */
	public void tallyStateTree(Tree<State> tree) {
		/* LVeGLexicon will handle pre-terminal nodes */
		if (tree.isLeaf() || tree.isPreTerminal()) { return; }
		
		short idParent = tree.getLabel().getId();
		List<Tree<State>> children = tree.getChildren();
		
		switch (children.size()) {
		case 0:
			break;
		case 1: {
			UnaryGrammarRule rule;
			short idChild = children.get(0).getLabel().getId();
			if (idParent != 0) {
				rule = new UnaryGrammarRule(idParent, idChild, GrammarRule.GENERAL);
			} else { // 0 represents the root node
				rule = new UnaryGrammarRule(idParent, idChild, GrammarRule.RHSPACE);
			}
			unaryRuleTable.addCount(rule, 1.0);
			break;
		}
		case 2: {
			short idLeftChild = children.get(0).getLabel().getId();
			short idRightChild = children.get(1).getLabel().getId();
			BinaryGrammarRule rule = new BinaryGrammarRule(idParent, idLeftChild, idRightChild, true);
			binaryRuleTable.addCount(rule, 1.0);
			break;
		}
		default:
			System.err.println("Malformed tree: more than two children. Exitting...");
			System.exit(0);
		}
		
		for (Tree<State> child : children) {
			tallyStateTree(child);
		}
	}
	
	
	public void addCount(short idParent, short idChild, char type, double increment, boolean withTree) {
		Map<GrammarRule, Double> count = null;
		if (withTree) {
			count = count0;
		} else {
			count = count1;
		}
		
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		if (rule != null) {
			count.put(rule, count.get(rule) + increment);
			return;
		}
		System.err.println("Unary Rule NOT Found: [P: " + idParent + ", C: " + idChild + ", TYPE: " + type + "]");
	}
	
	
	public void addCount(short idParent, short idlChild, short idrChild, double increment, boolean withTree) {
		Map<GrammarRule, Double> count = null;
		if (withTree) {
			count = count0;
		} else {
			count = count1;
		}
		
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		if (rule != null) {
			count.put(rule, count.get(rule) + increment);
			return;
		}
		System.err.println("Binary Rule NOT Found: [P: " + idParent + ", LC: " + idlChild + ", RC: " + idrChild + "]");
		
	}

	
	public GaussianMixture getUnaryRuleScore(short idParent, short idChild, char type) {
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		if (rule != null) {
			return rule.getWeight();
		}
		System.err.println("Unary Rule NOT Found: [P: " + idParent + ", C: " + idChild + ", TYPE: " + type + "]");
		return null;
	}
	
	
	public GaussianMixture getBinaryRuleScore(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		if (rule != null) {
			return rule.getWeight();
		}
		System.err.println("Binary Rule NOT Found: [P: " + idParent + ", LC: " + idlChild + ", RC: " + idrChild + "]");
		return null;
	}
	
	
	public GrammarRule getUnaryRule(short idParent, short idChild, char type) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return this.unaryRuleMap.get(rule);
	}
	
	
	public GrammarRule getBinaryRule(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = new BinaryGrammarRule(idParent, idlChild, idrChild);
		return this.binaryRuleMap.get(rule);
	}
	
	
	public List<GrammarRule> getBinaryRuleWithRC(int iTag) {
		return this.binaryRulesWithRC[iTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithLC(int iTag) {
		return this.binaryRulesWithLC[iTag];
	}
	
	
	public List<GrammarRule> getBinaryRuleWithP(int iTag) {
		return this.binaryRulesWithP[iTag];
	}
	
	
	public List<GrammarRule> getUnaryRuleWithP(int iTag) {
		return this.unaryRulesWithP[iTag];
	}
	
	
	public List<GrammarRule> getUnaryRuleWithC(int iTag) {
		return this.unaryRulesWithC[iTag];
	}
	
	
	public Map<GrammarRule, GrammarRule> getUnaryRuleMap() {
		return this.unaryRuleMap;
	}
	
	
	public Map<GrammarRule, GrammarRule> getBinaryRuleMap() {
		return this.binaryRuleMap;
	}


	@Override
	public String toString() {
		int count = 0, ncol = 5;
		StringBuffer sb = new StringBuffer();
		sb.append("Grammar [nTag=" + nTag + "]\n");
		
		sb.append("---Unary Grammar Rules. Total: " + unaryRuleTable.size() + "\n");
		for (GrammarRule rule : unaryRuleTable.keySet()) {
			sb.append(rule + "\t");
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		
		sb.append("\n");
		sb.append("---Binary Grammar Rules. Total: " + binaryRuleTable.size() + "\n");
		for (GrammarRule rule : binaryRuleTable.keySet()) {
			sb.append(rule + "\t");
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		
		return sb.toString();
	}
}
