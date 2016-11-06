package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.ArrayList;
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
	
	protected int nTag;
	protected Numberer numbererTag;
	
	protected RuleTable<?> unaryRuleTable;
	protected RuleTable<?> binaryRuleTable;
	
	private List<UnaryGrammarRule>[] unaryRulesWithP;
	private List<UnaryGrammarRule>[] unaryRulesWithC;
	
	private List<BinaryGrammarRule>[] binaryRulesWithP;
	private List<BinaryGrammarRule>[] binaryRulesWithLC;
	private List<BinaryGrammarRule>[] binaryRulesWithRC;
	
	/**
	 * Needed when we want to find a rule and access its statistics.
	 * we first construct a rule, which is used as the key, and use 
	 * the key to find the real rule that contains more information.
	 */
	private Map<UnaryGrammarRule, UnaryGrammarRule> unaryRuleMap;
	private Map<BinaryGrammarRule, BinaryGrammarRule> binaryRuleMap;
	
	
	/**
	 * Rules with probabilities below this value will be filtered.
	 */
	private double filterThreshold;
	
	
	public LVeGGrammar(LVeGGrammar oldGrammar, double filterThreshold) {
		this.numbererTag = Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET);
		this.unaryRuleTable = new RuleTable<UnaryGrammarRule>(UnaryGrammarRule.class);
		this.binaryRuleTable = new RuleTable<BinaryGrammarRule>(BinaryGrammarRule.class);
		this.filterThreshold = filterThreshold;
		this.nTag = numbererTag.size();
		
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
			unaryRulesWithP[i] = new ArrayList<UnaryGrammarRule>();
			unaryRulesWithC[i] = new ArrayList<UnaryGrammarRule>();
			
			binaryRulesWithP[i]  = new ArrayList<BinaryGrammarRule>();
			binaryRulesWithLC[i] = new ArrayList<BinaryGrammarRule>();
			binaryRulesWithRC[i] = new ArrayList<BinaryGrammarRule>();
		}
	}
	
	
	public void addBinaryRule(BinaryGrammarRule rule) {
		if (binaryRulesWithP[rule.lhs].contains(rule)) { return; }
		binaryRulesWithP[rule.lhs].add(rule);
		binaryRulesWithLC[rule.lchild].add(rule);
		binaryRulesWithRC[rule.rchild].add(rule);
		
		binaryRuleMap.put(rule, rule);
	}
	
	
	public void addUnaryRule(UnaryGrammarRule rule) {
		if (unaryRulesWithP[rule.lhs].contains(rule)) { return; }
		unaryRulesWithP[rule.lhs].add(rule);
		unaryRulesWithC[rule.rhs].add(rule);
		
		unaryRuleMap.put(rule, rule);
	}
	
	
	public void optimize(double randomness) {
		// TODO
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
			unaryRuleTable.increaseCount(rule, 1.0);
			break;
		}
		case 2: {
			short idLeftChild = children.get(0).getLabel().getId();
			short idRightChild = children.get(1).getLabel().getId();
			BinaryGrammarRule rule = new BinaryGrammarRule(idParent, idLeftChild, idRightChild, true);
			binaryRuleTable.increaseCount(rule, 1.0);
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
	
	
	public GaussianMixture getUnaryRuleScore(short idParent, short idChild, char type) {
		UnaryGrammarRule rule = getUnaryRule(idParent, idChild, type);
		if (rule != null) {
			return rule.getWeight();
		}
		System.err.println("Unary Rule NOT Found: [P: " + idParent + ", C: " + idChild + ", TYPE: " + type);
		return null;
	}
	
	
	public GaussianMixture getBinaryRuleScore(short idParent, short idlChild, short idrChild) {
		BinaryGrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		if (rule != null) {
			return rule.getWeight();
		}
		System.err.println("Binary Rule NOT Found: [P: " + idParent + ", LC: " + idlChild + ", RC: " + idrChild);
		return null;
	}
	
	
	public UnaryGrammarRule getUnaryRule(short idParent, short idChild, char type) {
		UnaryGrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return unaryRuleMap.get(rule);
	}
	
	
	public BinaryGrammarRule getBinaryRule(short idParent, short idlChild, short idrChild) {
		BinaryGrammarRule rule = new BinaryGrammarRule(idParent, idlChild, idrChild);
		return binaryRuleMap.get(rule);
	}
	
	
	public List<BinaryGrammarRule> getBinaryRuleWithRC(int iTag) {
		return this.binaryRulesWithRC[iTag];
	}
	
	
	public List<BinaryGrammarRule> getBinaryRuleWithLC(int iTag) {
		return this.binaryRulesWithLC[iTag];
	}
	
	
	public List<BinaryGrammarRule> getBinaryRuleWithP(int iTag) {
		return this.binaryRulesWithP[iTag];
	}
	
	
	public List<UnaryGrammarRule> getUnaryRuleWithP(int iTag) {
		return this.unaryRulesWithP[iTag];
	}
	
	
	public List<UnaryGrammarRule> getUnaryRuleWithC(int iTag) {
		return this.unaryRulesWithC[iTag];
	}
	
	
	public Map<UnaryGrammarRule, UnaryGrammarRule> getUnaryRuleMap() {
		return this.unaryRuleMap;
	}
	
	
	public Map<BinaryGrammarRule, BinaryGrammarRule> getBinaryRuleMap() {
		return this.binaryRuleMap;
	}
	
}
