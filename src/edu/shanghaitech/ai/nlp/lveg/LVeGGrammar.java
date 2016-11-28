package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public class LVeGGrammar extends Recorder implements Serializable {
	
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
	 * to the Viterbi algorithm if they contain 'Max'. However, the
	 * point is how to define the maximum between two MoGs.
	 */
	private List<GrammarRule> chainSumUnaryRules;
	private List<GrammarRule>[] chainSumUnaryRulesWithP;
	private List<GrammarRule>[] chainSumUnaryRulesWithC;
	
	private List<GrammarRule> chainMaxUnaryRules;
	private List<GrammarRule>[] chainMaxUnaryRulesWithP;
	private List<GrammarRule>[] chainMaxUnaryRulesWithC;
	
	private short[][] chainMaxUnaryPath;
	
	/**
	 * Needed when we want to find a rule and access its statistics.
	 * we first construct a rule, which is used as the key, and use 
	 * the key to find the real rule that contains more information.
	 */
	private Map<GrammarRule, GrammarRule> unaryRuleMap;
	private Map<GrammarRule, GrammarRule> binaryRuleMap;
	
	private Optimizer optimizer;
	
	/**
	 * Rules with probabilities below this value will be filtered.
	 * @deprecated
	 */
	private double filterThreshold;
	
	
	public LVeGGrammar(LVeGGrammar oldGrammar, double filterThreshold, int nTag) {
		this.unaryRuleTable  = new RuleTable<UnaryGrammarRule>(UnaryGrammarRule.class);
		this.binaryRuleTable = new RuleTable<BinaryGrammarRule>(BinaryGrammarRule.class);
		this.unaryRuleMap  = new HashMap<GrammarRule, GrammarRule>();
		this.binaryRuleMap = new HashMap<GrammarRule, GrammarRule>();
		this.optimizer = new Optimizer(LVeGLearner.random);
		
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
	
	
	@SuppressWarnings("unchecked")
	private void initialize() {
		this.unaryRulesWithP = new List[nTag];
		this.unaryRulesWithC = new List[nTag];
		
		this.binaryRulesWithP  = new List[nTag];
		this.binaryRulesWithLC = new List[nTag];
		this.binaryRulesWithRC = new List[nTag];
		
		this.chainSumUnaryRules = new ArrayList<GrammarRule>();
		this.chainSumUnaryRulesWithP = new List[nTag];
		this.chainSumUnaryRulesWithC = new List[nTag];
		
		this.chainMaxUnaryRules = new ArrayList<GrammarRule>();
		this.chainMaxUnaryRulesWithP = new List[nTag];
		this.chainMaxUnaryRulesWithC = new List[nTag];
		
		this.chainMaxUnaryPath = new short[nTag][nTag];
		
		for (int i = 0; i < nTag; i++) {
			unaryRulesWithP[i] = new ArrayList<GrammarRule>();
			unaryRulesWithC[i] = new ArrayList<GrammarRule>();
			
			binaryRulesWithP[i]  = new ArrayList<GrammarRule>();
			binaryRulesWithLC[i] = new ArrayList<GrammarRule>();
			binaryRulesWithRC[i] = new ArrayList<GrammarRule>();
			
			chainSumUnaryRulesWithP[i] = new ArrayList<GrammarRule>();
			chainSumUnaryRulesWithC[i] = new ArrayList<GrammarRule>();
			chainMaxUnaryRulesWithP[i] = new ArrayList<GrammarRule>();
			chainMaxUnaryRulesWithC[i] = new ArrayList<GrammarRule>();
		}
	}
	
	
	public void postInitialize(double randomness) {
		for (GrammarRule rule : unaryRuleTable.keySet()) {
			addUnaryRule((UnaryGrammarRule) rule);
		}
		for (GrammarRule rule : binaryRuleTable.keySet()) {
			addBinaryRule((BinaryGrammarRule) rule);
		}
		computeChainUnaryRule();
	}
	
	
	protected void addBinaryRule(BinaryGrammarRule rule) {
		if (binaryRulesWithP[rule.lhs].contains(rule)) { return; }
		binaryRulesWithP[rule.lhs].add(rule);
		binaryRulesWithLC[rule.lchild].add(rule);
		binaryRulesWithRC[rule.rchild].add(rule);
		binaryRuleMap.put(rule, rule);
		optimizer.addRule(rule);
	}
	
	
	protected void addUnaryRule(UnaryGrammarRule rule) {
		if (unaryRulesWithP[rule.lhs].contains(rule)) { return; }
		unaryRulesWithP[rule.lhs].add(rule);
		unaryRulesWithC[rule.rhs].add(rule);
		unaryRuleMap.put(rule, rule);
		optimizer.addRule(rule);
	}
	
	
	/**
	 * Apply stochastic gradient descent.
	 */
	public void applyGradientDescent() {
		optimizer.applyGradientDescent();
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
			logger.error("Malformed tree: more than two children. Exitting...");
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
		keys1.put(GrammarRule.Unit.P, GrammarRule.Unit.RM);
		
		// rules of the from X->ROOT(0) are not allowed
		for (short iparent = 0; iparent < nTag; iparent++) {
			for (short ichild = 1; ichild < nTag; ichild++) {
				if (iparent == ichild) { continue; }
				short bestIntermediateState = -1;
				double maxSumWeight = -1.0, total;
				
				char type;
				keys0.clear();
				if (iparent == 0) {
					type = GrammarRule.RHSPACE;
					keys0.put(GrammarRule.Unit.C, GrammarRule.Unit.RM);
				} else {
					type = GrammarRule.GENERAL;
					keys0.put(GrammarRule.Unit.UC, GrammarRule.Unit.RM);
				}
				
				GaussianMixture weightSum = new DiagonalGaussianMixture();
				GaussianMixture weightMax = new DiagonalGaussianMixture();
				
				UnaryGrammarRule uruleSum = new UnaryGrammarRule(iparent, ichild, type, weightSum);
				UnaryGrammarRule uruleMax = new UnaryGrammarRule(iparent, ichild, type, weightMax);
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
							if (iparent == 0) {
								aruleWeight = aruleWeight.replaceAllKeys(GrammarRule.Unit.C);
							}
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
					addUnaryRule(uruleSum);
					// the resulting chain rule
					chainSumUnaryRules.add(uruleSum);
					chainSumUnaryRulesWithP[iparent].add(uruleSum);
					chainSumUnaryRulesWithC[ichild].add(uruleSum);
					// incorrect for Viterbi parser
					chainMaxUnaryRules.add(uruleMax);
					chainMaxUnaryRulesWithP[iparent].add(uruleMax);
					chainMaxUnaryRulesWithC[ichild].add(uruleMax);
					// how to compare the magnitudes of two MoG
					chainMaxUnaryPath[iparent][ichild] = bestIntermediateState;
				}
			}
		}
	}
	
	
	public GaussianMixture getBinaryRuleScore(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		if (rule != null) {
			return rule.getWeight();
		}
		System.err.println("Binary Rule NOT Found: [P: " + idParent + ", LC: " + idlChild + ", RC: " + idrChild + "]");
		return null;
	}
	
	
	public GaussianMixture getUnaryRuleScore(short idParent, short idChild, char type) {
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		if (rule != null) {
			return rule.getWeight();
		}
		System.err.println("Unary Rule NOT Found: [P: " + idParent + ", C: " + idChild + ", TYPE: " + type + "]");
		return null;
	}
	
	
	public void addCount(short idParent, short idlChild, short idrChild, double increment, boolean withTree) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		addCount(rule, increment, withTree);
	}
	
	
	public double getCount(short idParent, short idlChild, short idrChild, boolean withTree) {
		GrammarRule rule = getBinaryRule(idParent, idlChild, idrChild);
		return getCount(rule, withTree);
	}
	
	
	public void addCount(short idParent, short idChild, char type, double increment, boolean withTree) {
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		addCount(rule, increment, withTree);
	}
	
	
	public double getCount(short idParent, short idChild, char type, boolean withTree) {
		GrammarRule rule = getUnaryRule(idParent, idChild, type);
		return getCount(rule, withTree);
	}
	
	
	public void addCount(GrammarRule rule, double increment, boolean withTree) {
		optimizer.addCount(rule, increment, withTree);
	}
	
	
	public double getCount(GrammarRule rule, boolean withTree) {
		return optimizer.getCount(rule, withTree);
	}
	
	
	public GrammarRule getBinaryRule(short idParent, short idlChild, short idrChild) {
		GrammarRule rule = new BinaryGrammarRule(idParent, idlChild, idrChild);
		return binaryRuleMap.get(rule);
	}
	
	
	public GrammarRule getUnaryRule(short idParent, short idChild, char type) {
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
