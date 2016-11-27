package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.optimization.SGDMinimizer;
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
	 * For any nonterminals A \neq B \neq C, p(A->B) is computed as 
	 * p(A->B) + \sum_{C} p(A->C) \times p(C->B), in which p(A->B) 
	 * is zero if A->B does not exist, and the resulting new rules 
	 * are added to the unary rule set. Fields containing 'sum' are
	 * dedicated to the general CYK algorithm, and are dedicated to
	 * to the Viterbi algorithm if they contain 'Max'. But shall we
	 * define the maximum between two MoGs.
	 */
	private List<GrammarRule> chainSumUnaryRules;
	private List<GrammarRule>[] chainSumUnaryRulesWithP;
	private List<GrammarRule>[] chainSumUnaryRulesWithC;
	
	private List<GrammarRule> chainMaxUnaryRules;
	private List<GrammarRule>[] chainMaxUnaryRulesWithP;
	private List<GrammarRule>[] chainMaxUnaryRulesWithC;
	
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
		this.count0 = new HashMap<GrammarRule, Double>();
		this.count1 = new HashMap<GrammarRule, Double>();
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
	
	
	public boolean containsRule(GrammarRule rule, boolean isUnary) {
		if (isUnary) {
			UnaryGrammarRule r = (UnaryGrammarRule) rule;
			if (unaryRulesWithP[r.lhs].contains(rule)) {
				return true;
			}
		}
		return false;
	}
	
	
	public void remove(GrammarRule rule, boolean isUnary) {
		if (isUnary) {
			UnaryGrammarRule r = (UnaryGrammarRule) rule;
			if (unaryRulesWithP[r.lhs].contains(rule)) {
				System.out.println("found the rule");
			}
			boolean signal = unaryRulesWithP[r.lhs].remove(r);
			if (!signal) {
				System.out.println("failed to remove");
			}
			
			if (unaryRulesWithP[r.lhs].contains(rule)) {
				System.out.println("failed to remove again");
			}
			signal = unaryRulesWithC[r.rhs].remove(r);
			if (!signal) {
				System.out.println("with c error remove");
			}
		} else {
			
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
			if (idParent == idChild) { 
				LVeGLearner.logger.error("Incorrect unary rule: " + rule); 
				break;
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
			LVeGLearner.logger.error("Malformed tree: more than two children. Exitting...");
			System.exit(0);
		}
		
		for (Tree<State> child : children) {
			tallyStateTree(child);
		}
	}
	
	
	/**
	 * Compute the two-order unary chain.
	 */
	protected void computeChainUnaryRule() {
		
		Map<String, String> keys0 = new HashMap<String, String>();
		Map<String, String> keys1 = new HashMap<String, String>();
		keys0.put(GrammarRule.Unit.UC, GrammarRule.Unit.RM);
		keys1.put(GrammarRule.Unit.P, GrammarRule.Unit.RM);
		
		for (short iparent = 0; iparent < nTag; iparent++) {
			for (short ichild = 0; ichild < nTag; ichild++) {
				if (iparent == ichild) { continue; }
				short bestIntermediateState = -1;
				double maxSumWeight = -1.0, total;
				
				GaussianMixture weightSum = new DiagonalGaussianMixture();
				GaussianMixture weightMax = new DiagonalGaussianMixture();
				UnaryGrammarRule uruleSum = new UnaryGrammarRule(iparent, ichild, weightSum);
				UnaryGrammarRule uruleMax = new UnaryGrammarRule(iparent, ichild, weightMax);
				GaussianMixture pruleWeight = null, cruleWeight = null, aruleWeight = null;
				
				for (GrammarRule prule : unaryRulesWithP[iparent]) {
					UnaryGrammarRule uprule = (UnaryGrammarRule) prule;
					pruleWeight = uprule.getWeight(); // one-order chain rule
					if (uprule.rhs == ichild) {
						weightSum.add(pruleWeight.copy(true));
						total = pruleWeight.marginalize();
						if (total > maxSumWeight) { 
							weightMax.clear();
							maxSumWeight = total;
							bestIntermediateState = -1;
							weightMax = pruleWeight.copy(true);
						}
					} else { // two-order chain rule
						for (GrammarRule crule : unaryRulesWithC[ichild]) {
							UnaryGrammarRule ucrule = (UnaryGrammarRule) crule;
							if (ucrule.lhs != uprule.rhs) { continue; }
							cruleWeight = ucrule.getWeight();
							aruleWeight = GaussianMixture.mulAndMarginalize(pruleWeight, cruleWeight, keys0, keys1);
							weightSum.add(aruleWeight);
							total = aruleWeight.marginalize();
							if (total > maxSumWeight) { 
								weightMax.clear();
								maxSumWeight = total;
								bestIntermediateState = ucrule.lhs;
								weightMax = aruleWeight.copy(true);
							}
						}
					}
				} 
				if (maxSumWeight > 0) {
					// why shall we add it?
					// addUnaryRule(uruleSum);
					// the resulting chain rule
					chainSumUnaryRules.add(uruleSum);
					chainSumUnaryRulesWithP[iparent].add(uruleSum);
					chainSumUnaryRulesWithC[ichild].add(uruleSum);
					// incorrect for Viterbi parser
					chainMaxUnaryRules.add(uruleMax);
					chainMaxUnaryRulesWithP[iparent].add(uruleMax);
					chainMaxUnaryRulesWithC[ichild].add(uruleMax);
					// how to compare the magnitudes of two MoG
					
				}
				
			}
		}
	}
	
	
	protected void applyGradientDescent(Random random, double learningRate) {
		double cnt0, cnt1;
		for (GrammarRule rule : unaryRuleTable.keySet()) {
			cnt0 = count0.get(rule);
			cnt1 = count1.get(rule);
			if (cnt0 == cnt1) { continue; }
			SGDMinimizer.applyGradientDescent(rule.getWeight(), random, cnt0, cnt1, learningRate);
		}
		
		for (GrammarRule rule : binaryRuleTable.keySet()) {
			cnt0 = count0.get(rule);
			cnt1 = count1.get(rule);
			if (cnt0 == cnt1) { continue; }
			SGDMinimizer.applyGradientDescent(rule.getWeight(), random, cnt0, cnt1, learningRate);
		}
		resetCount();
	}
	
	
	private void resetCount() {
		for (Map.Entry<GrammarRule, Double> count : count0.entrySet()) {
			count.setValue(0.0);
		}
		for (Map.Entry<GrammarRule, Double> count : count1.entrySet()) {
			count.setValue(0.0);
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
	
	
	public void addCount(short idParent, short idChild, char type, double increment, boolean withTree) {
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		addCount(rule, increment, withTree);
	}
	
	
	public double getCount(short idParent, short idChild, char type, boolean withTree) {
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		return getCount(rule, withTree);
	}
	
	
	public void addCount(short idParent, short idlChild, short idrChild, double increment, boolean withTree) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		addCount(rule, increment, withTree);
	}
	
	
	public double getCount(short idParent, short idlChild, short idrChild, boolean withTree) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		return getCount(rule, withTree);
	}
	
	
	public void addCount(GrammarRule rule, double increment, boolean withTree) {
		Map<GrammarRule, Double> count = withTree ? count0 : count1;
		if (rule != null && count.get(rule) != null) {
			count.put(rule, count.get(rule) + increment);
			return;
		}
		if (rule == null) {
			System.err.println("The Given Rule is NULL.");
		} else {
			System.err.println("Grammar Rule NOT Found: " + rule);
		}
	}
	
	
	public double getCount(GrammarRule rule, boolean withTree) {
		Map<GrammarRule, Double> count = withTree ? count0 : count1;
		if (rule != null && count.get(rule) != null) {
			return count.get(rule);
		}
		if (rule == null) {
			System.err.println("The Given Rule is NULL.");
		} else {
			System.err.println("Grammar Rule NOT Found: " + rule);
		}
		return -1.0;
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
		int count = 0, ncol = 1;
		StringBuffer sb = new StringBuffer();
		sb.append("Grammar [nTag=" + nTag + "]\n");
		
		sb.append("---Unary Grammar Rules. Total: " + unaryRuleTable.size() + "\n");
		for (GrammarRule rule : unaryRuleTable.keySet()) {
			sb.append(rule + "\t" + unaryRuleTable.getCount(rule).getBias());
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		
		sb.append("\n");
		sb.append("---Binary Grammar Rules. Total: " + binaryRuleTable.size() + "\n");
		for (GrammarRule rule : binaryRuleTable.keySet()) {
			sb.append(rule + "\t" + binaryRuleTable.getCount(rule).getBias());
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		
		return sb.toString();
	}
}
