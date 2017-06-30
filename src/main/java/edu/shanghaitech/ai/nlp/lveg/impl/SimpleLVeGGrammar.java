package edu.shanghaitech.ai.nlp.lveg.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
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
		this.uRuleMap = new HashMap<>();
		this.bRuleMap = new HashMap<>();
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
			uRulesWithP[i] = new ArrayList<>();
			uRulesWithC[i] = new ArrayList<>();
			bRulesWithP[i]  = new ArrayList<>();
			bRulesWithLC[i] = new ArrayList<>();
			bRulesWithRC[i] = new ArrayList<>();
			chainSumUnaryRulesWithP[i] = new ArrayList<>();
			chainSumUnaryRulesWithC[i] = new ArrayList<>();
		}
	}
	
	
	@Override
	public void postInitialize() {
		for (GrammarRule rule : uRuleTable.keySet()) {
			rule.getWeight().setBias(uRuleTable.getCount(rule).getBias());
			addURule((UnaryGrammarRule) rule);
		}
		for (GrammarRule rule : bRuleTable.keySet()) {
			rule.getWeight().setBias(bRuleTable.getCount(rule).getBias());
			addBRule((BinaryGrammarRule) rule);
		}
//		computeChainUnaryRule();
	}
	
	
	@Override
	public void initializeOptimizer() {
		for (GrammarRule rule : uRuleTable.keySet()) {
			optimizer.addRule(rule);
		}
		for (GrammarRule rule : bRuleTable.keySet()) {
			optimizer.addRule(rule);
		}
	}
	
	
	public void addBRule(BinaryGrammarRule rule) {
		if (bRulesWithP[rule.lhs].contains(rule)) { return; }
		bRulesWithP[rule.lhs].add(rule);
		bRulesWithLC[rule.lchild].add(rule);
		bRulesWithRC[rule.rchild].add(rule);
		bRuleMap.put(rule, rule);
	}
	
	
	public void addURule(UnaryGrammarRule rule) {
		if (uRulesWithP[rule.lhs].contains(rule)) { return; }
		uRulesWithP[rule.lhs].add(rule);
		uRulesWithC[rule.rhs].add(rule);
		uRuleMap.put(rule, rule);
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
			short idChild = children.get(0).getLabel().getId();
			RuleType type = idParent != 0 ? RuleType.LRURULE : RuleType.RHSPACE;
			GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
			if (!uRuleTable.containsKey(rule)) { 
				rule.initializeWeight(type, (short) -1, (short) -1); 
			}
			uRuleTable.addCount(rule, 1.0);
			break;
		}
		case 2: {
			short idlChild = children.get(0).getLabel().getId();
			short idrChild = children.get(1).getLabel().getId();
			GrammarRule rule = new BinaryGrammarRule(idParent, idlChild, idrChild);
			if (!bRuleTable.containsKey(rule)) { 
				rule.initializeWeight(RuleType.LRBRULE, (short) -1, (short) -1); 
			}
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
		EnumMap<RuleUnit, RuleUnit> keys0 = new EnumMap<>(RuleUnit.class);
		EnumMap<RuleUnit, RuleUnit> keys1 = new EnumMap<>(RuleUnit.class);
		keys1.put(RuleUnit.P, RuleUnit.RM);
		short count = 0, total = 0;
		
		// rules of the from X->ROOT(0) are not allowed
		for (short iparent = 0; iparent < ntag; iparent++) {
			for (short ichild = 1; ichild < ntag; ichild++) {
				if (iparent == ichild) { continue; }
				boolean found = false;
				int cnt = 0;
				RuleType type;
				keys0.clear();
				if (iparent == 0) {
					type = RuleType.RHSPACE;
					keys0.put(RuleUnit.C, RuleUnit.RM);
				} else {
					type = RuleType.LRURULE;
					keys0.put(RuleUnit.UC, RuleUnit.RM);
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
							if (iparent == 0) { aruleWeight = aruleWeight.replaceAllKeys(RuleUnit.C); }
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
		int count = 0, ncol = 1, ncomp = 0;
		StringBuffer sb = new StringBuffer();
		sb.append("Grammar [nTag=" + ntag + "]\n");
		for (int i = 0; i < numberer.size(); i++) {
			sb.append("Tag " + i + "\t" +  (String) numberer.object(i) + "\n");
		}
		
		int nurule = uRuleTable.size();
		sb.append("---Unary Grammar Rules. Total: " + nurule + "\n");
		for (GrammarRule rule : uRuleTable.keySet()) {
			ncomp += rule.weight.ncomponent();
			sb.append(rule + "\t\t" + uRuleTable.getCount(rule).getBias() + "\t\t" 
					+ rule.weight.ncomponent() + "\t\t" + Math.exp(rule.weight.getWeight(0)) + "\t\t" + Math.exp(rule.weight.getProb()));
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		sb.append("---Unary Grammar Rules. Total: " + nurule + ", average ncomp: " + ((double) ncomp / nurule) + "\n");
		
		sb.append("\n");
		
		ncomp = 0;
		int nbrule = bRuleTable.size();
		sb.append("---Binary Grammar Rules. Total: " + nbrule + "\n");
		for (GrammarRule rule : bRuleTable.keySet()) {
			ncomp += rule.weight.ncomponent();
			sb.append(rule + "\t\t" + bRuleTable.getCount(rule).getBias() + "\t\t"
					+ rule.weight.ncomponent() + "\t\t" + Math.exp(rule.weight.getWeight(0)) + "\t\t" + Math.exp(rule.weight.getProb()));
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		sb.append("---Binary Grammar Rules. Total: " + nbrule + ", average ncomp: " + ((double) ncomp / nbrule) + "\n");
		
		return sb.toString();
	}
	
}
