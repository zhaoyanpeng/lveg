package edu.shanghaitech.ai.nlp.lveg.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Numberer;

/**
 * @author Yanpeng Zhao
 *
 */
public class SimpleLVeGGrammar extends LVeGGrammar implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 650638115156791313L;

	
	public SimpleLVeGGrammar(Numberer numberer, int ntag) {
		this.uRuleTable = new RuleTable<UnaryGrammarRule>(UnaryGrammarRule.class);
		this.bRuleTable = new RuleTable<BinaryGrammarRule>(BinaryGrammarRule.class);
		this.uRuleMap = new HashMap<GrammarRule, GrammarRule>();
		this.bRuleMap = new HashMap<GrammarRule, GrammarRule>();
		if (numberer == null) {
			this.numberer = null;
			this.ntag = ntag;
		} else {
			this.numberer = numberer;
			this.ntag = numberer.size();
		}
		initialize();
	}
	
	
	protected void initialize() {
		this.uRulesWithP = new List[ntag];
		this.uRulesWithC = new List[ntag];
		this.bRulesWithP  = new List[ntag];
		this.bRulesWithLC = new List[ntag];
		this.bRulesWithRC = new List[ntag];
		this.chainSumUnaryRules = new HashSet<GrammarRule>();
		this.chainSumUnaryRulesWithP = new List[ntag];
		this.chainSumUnaryRulesWithC = new List[ntag];
		for (int i = 0; i < ntag; i++) {
			uRulesWithP[i] = new ArrayList<GrammarRule>();
			uRulesWithC[i] = new ArrayList<GrammarRule>();
			bRulesWithP[i]  = new ArrayList<GrammarRule>();
			bRulesWithLC[i] = new ArrayList<GrammarRule>();
			bRulesWithRC[i] = new ArrayList<GrammarRule>();
			chainSumUnaryRulesWithP[i] = new ArrayList<GrammarRule>();
			chainSumUnaryRulesWithC[i] = new ArrayList<GrammarRule>();
		}
	}
	
	
	public void postInitialize() {
		for (GrammarRule rule : uRuleTable.keySet()) {
			addURule((UnaryGrammarRule) rule);
		}
		for (GrammarRule rule : bRuleTable.keySet()) {
			addBRule((BinaryGrammarRule) rule);
		}
//		computeChainUnaryRule();
	}
	
	
	public void addBRule(BinaryGrammarRule rule) {
		if (bRulesWithP[rule.lhs].contains(rule)) { return; }
		bRulesWithP[rule.lhs].add(rule);
		bRulesWithLC[rule.lchild].add(rule);
		bRulesWithRC[rule.rchild].add(rule);
		bRuleMap.put(rule, rule);
		optimizer.addRule(rule);
	}
	
	
	public void addURule(UnaryGrammarRule rule) {
		if (uRulesWithP[rule.lhs].contains(rule)) { return; }
		uRulesWithP[rule.lhs].add(rule);
		uRulesWithC[rule.rhs].add(rule);
		uRuleMap.put(rule, rule);
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
			uRuleTable.addCount(rule, 1.0);
			break;
		}
		case 2: {
			short idLeftChild = children.get(0).getLabel().getId();
			short idRightChild = children.get(1).getLabel().getId();
			BinaryGrammarRule rule = new BinaryGrammarRule(idParent, idLeftChild, idRightChild, true);
			bRuleTable.addCount(rule, 1.0);
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
	protected void computeChainUnaryRule(boolean prune) {
		Map<String, String> keys0 = new HashMap<String, String>();
		Map<String, String> keys1 = new HashMap<String, String>();
		keys1.put(GrammarRule.Unit.P, GrammarRule.Unit.RM);
		short count = 0, total = 0;
		
		// rules of the from X->ROOT(0) are not allowed
		for (short iparent = 0; iparent < ntag; iparent++) {
			for (short ichild = 1; ichild < ntag; ichild++) {
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
				
				for (GrammarRule prule : uRulesWithP[iparent]) {
					UnaryGrammarRule uprule = (UnaryGrammarRule) prule;
					pruleWeight = uprule.getWeight(); // one-order chain rule
					if (uprule.rhs == ichild) {
						weightSum.add(pruleWeight.copy(true), prune);
						found = true;
						cnt++;
					} else { // two-order chain rule
						for (GrammarRule crule : uRulesWithC[ichild]) {
							UnaryGrammarRule ucrule = (UnaryGrammarRule) crule;
							if (ucrule.lhs != uprule.rhs) { continue; }
							cruleWeight = ucrule.getWeight();
							aruleWeight = GaussianMixture.mulAndMarginalize(pruleWeight, cruleWeight, keys0, keys1);
							if (iparent == 0) { aruleWeight = aruleWeight.replaceAllKeys(GrammarRule.Unit.C); }
							weightSum.add(aruleWeight, prune);
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
			if (!uRuleMap.containsKey(rule)) { count++; continue; }
			addURule((UnaryGrammarRule) rule);
		}
		logger.trace("# of new rules: " + count + " \t# of all rules: " + total + "\n");
	}
	

	@Override
	public String toString() {
		int count = 0, ncol = 1;
		StringBuffer sb = new StringBuffer();
		sb.append("Grammar [nTag=" + ntag + "]\n");
		
		sb.append("---Unary Grammar Rules. Total: " + uRuleTable.size() + "\n");
		for (GrammarRule rule : uRuleTable.keySet()) {
			sb.append(rule + "\t" + uRuleTable.getCount(rule).getBias());
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		
		sb.append("\n");
		sb.append("---Binary Grammar Rules. Total: " + bRuleTable.size() + "\n");
		for (GrammarRule rule : bRuleTable.keySet()) {
			sb.append(rule + "\t" + bRuleTable.getCount(rule).getBias());
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		
		return sb.toString();
	}
	
}
