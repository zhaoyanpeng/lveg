package edu.shanghaitech.ai.nlp.lveg.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.RuleTable;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.Unit;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.optimization.Optimizer;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public class SimpleLVeGGrammar extends LVeGGrammar implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 650638115156791313L;

	
	public SimpleLVeGGrammar(Numberer numberer, int nTag) {
		this.unaryRuleTable  = new RuleTable<UnaryGrammarRule>(UnaryGrammarRule.class);
		this.binaryRuleTable = new RuleTable<BinaryGrammarRule>(BinaryGrammarRule.class);
		this.unaryRuleMap  = new HashMap<GrammarRule, GrammarRule>();
		this.binaryRuleMap = new HashMap<GrammarRule, GrammarRule>();
		if (numberer == null) {
			this.numberer = null;
			this.nTag = nTag;
		} else {
			this.numberer = numberer;
			this.nTag = numberer.size();
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
		this.chainSumUnaryRules = new HashSet<GrammarRule>();
		this.chainSumUnaryRulesWithP = new List[nTag];
		this.chainSumUnaryRulesWithC = new List[nTag];
		
		for (int i = 0; i < nTag; i++) {
			unaryRulesWithP[i] = new ArrayList<GrammarRule>();
			unaryRulesWithC[i] = new ArrayList<GrammarRule>();
			binaryRulesWithP[i]  = new ArrayList<GrammarRule>();
			binaryRulesWithLC[i] = new ArrayList<GrammarRule>();
			binaryRulesWithRC[i] = new ArrayList<GrammarRule>();
			chainSumUnaryRulesWithP[i] = new ArrayList<GrammarRule>();
			chainSumUnaryRulesWithC[i] = new ArrayList<GrammarRule>();
		}
	}
	
	
	public void postInitialize(double randomness) {
		for (GrammarRule rule : unaryRuleTable.keySet()) {
			addUnaryRule((UnaryGrammarRule) rule);
		}
		for (GrammarRule rule : binaryRuleTable.keySet()) {
			addBinaryRule((BinaryGrammarRule) rule);
		}
//		computeChainUnaryRule();
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
				rule = new UnaryGrammarRule(idParent, idChild, GrammarRule.LRURULE);
			} else { // the root node
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
			logger.error("Malformed tree: more than two children. Exitting...\n");
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
		short count = 0, total = 0;
		
		// rules of the from X->ROOT(0) are not allowed
		for (short iparent = 0; iparent < nTag; iparent++) {
			for (short ichild = 1; ichild < nTag; ichild++) {
				if (iparent == ichild) { continue; }
				boolean found = false;
				int cnt = 0;
				byte type;
				keys0.clear();
				if (iparent == 0) {
					type = GrammarRule.RHSPACE;
					keys0.put(GrammarRule.Unit.C, GrammarRule.Unit.RM);
				} else {
					type = GrammarRule.LRURULE;
					keys0.put(GrammarRule.Unit.UC, GrammarRule.Unit.RM);
				}
				
				GaussianMixture weightSum = new DiagonalGaussianMixture();
				UnaryGrammarRule uruleSum = new UnaryGrammarRule(iparent, ichild, type, weightSum);
				GaussianMixture pruleWeight = null, cruleWeight = null, aruleWeight = null;
				
				for (GrammarRule prule : unaryRulesWithP[iparent]) {
					UnaryGrammarRule uprule = (UnaryGrammarRule) prule;
					pruleWeight = uprule.getWeight(); // one-order chain rule
					if (uprule.rhs == ichild) {
						weightSum.add(pruleWeight.copy(true));
						found = true;
						cnt++;
					} else { // two-order chain rule
						for (GrammarRule crule : unaryRulesWithC[ichild]) {
							UnaryGrammarRule ucrule = (UnaryGrammarRule) crule;
							if (ucrule.lhs != uprule.rhs) { continue; }
							cruleWeight = ucrule.getWeight();
							aruleWeight = GaussianMixture.mulAndMarginalize(pruleWeight, cruleWeight, keys0, keys1);
							if (iparent == 0) { aruleWeight = aruleWeight.replaceAllKeys(GrammarRule.Unit.C); }
							weightSum.add(aruleWeight);
							found = true;
							cnt++;
						}
					}
				} 
				if (found) {
					total++;
					// why shall we add it? Adding could result in the exponential increase of the # of components.
					// addUnaryRule(uruleSum);
					// the resulting chain rule
					chainSumUnaryRules.add(uruleSum);
					chainSumUnaryRulesWithP[iparent].add(uruleSum);
					chainSumUnaryRulesWithC[ichild].add(uruleSum);
//					logger.trace("Rule: [" + iparent + ", " + ichild + "]\t# of rules combined: " + cnt + 
//							"\t# of components: " + uruleSum.getWeight().getNcomponent());
				} 
			}
		}
		// TODO a temporary implementation, it may weaken the unary rules of length 1.
		for (GrammarRule rule : chainSumUnaryRules) {
			if (!unaryRuleMap.containsKey(rule)) { count++; continue; }
			addUnaryRule((UnaryGrammarRule) rule);
		}
		logger.trace("# of new rules: " + count + " \t# of all rules: " + total + "\n");
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
