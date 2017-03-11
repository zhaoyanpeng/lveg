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

	
	public SimpleLVeGGrammar(Numberer numberer, int ntag, boolean useRef, Map<Short, Short> nSubTypes) {
		this.uRuleTable = new RuleTable<UnaryGrammarRule>(UnaryGrammarRule.class);
		this.bRuleTable = new RuleTable<BinaryGrammarRule>(BinaryGrammarRule.class);
		this.uRuleMap = new HashMap<GrammarRule, GrammarRule>();
		this.bRuleMap = new HashMap<GrammarRule, GrammarRule>();
		this.refSubTypes = nSubTypes;
		this.useRef = useRef;
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
			short idChild = children.get(0).getLabel().getId();
			byte type = idParent != 0 ? GrammarRule.LRURULE : GrammarRule.RHSPACE;
			GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
			if (!uRuleTable.containsKey(rule)) { 
				int ntype, ncomp = (short) -1;
				if (useRef) {
					ntype = refSubTypes.get(idParent) * refSubTypes.get(idChild);
					if (ntype < 600) {
						ncomp = (short) (Math.floor(ntype / 250.0));
					} else if (ntype < 1300) {
						ncomp = (short) (Math.floor(ntype / 300.0));
					} else if (ntype < 2100) {
						ncomp = (short) (Math.floor(ntype / 400.0));
					} else {
						ncomp = (short) (Math.floor(ntype / 500.0));
					}
					ncomp = ncomp == 0 ? -1 : (ncomp > 6 ? 6 : ncomp);
				}
				rule.initializeWeight(type, (short) ncomp, (short) -1); 
			}
			uRuleTable.addCount(rule, 1.0);
			break;
		}
		case 2: {
			short idlChild = children.get(0).getLabel().getId();
			short idrChild = children.get(1).getLabel().getId();
			GrammarRule rule = new BinaryGrammarRule(idParent, idlChild, idrChild);
			if (!bRuleTable.containsKey(rule)) { 
				int ntype, ncomp = (short) -1;
				if (useRef) {
					ntype = refSubTypes.get(idParent) * refSubTypes.get(idlChild) * refSubTypes.get(idrChild);
					if (ntype < 600) {
						ncomp = (short) (Math.floor(ntype / 250.0));
					} else if (ntype < 1300) {
						ncomp = (short) (Math.floor(ntype / 300.0));
					} else if (ntype < 2100) {
						ncomp = (short) (Math.floor(ntype / 400.0));
					} else if (ntype < 3100) {
						ncomp = (short) (Math.floor(ntype / 500.0));
					} else if (ntype < 4300) {
						ncomp = (short) (Math.floor(ntype / 600.0));
					} else if (ntype < 5700) {
						ncomp = (short) (Math.floor(ntype / 700.0));
					} else if (ntype < 7300) {
						ncomp = (short) (Math.floor(ntype / 800.0));
					} else {
						ncomp = (short) (Math.floor(ntype / 900.0));
					}
					ncomp = ncomp == 0 ? -1 : (ncomp > 10 ? 10 : ncomp);
				}
				rule.initializeWeight(GrammarRule.LRBRULE, (short) ncomp, (short) -1); 
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
//				GaussianMixture weightSum = DiagonalGaussianMixture.borrowObject((short) 0); // POOL
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
			sb.append(rule + "\t" + uRuleTable.getCount(rule).getBias() + "\t" + rule.weight.ncomponent());
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
			sb.append(rule + "\t" + bRuleTable.getCount(rule).getBias() + "\t" + rule.weight.ncomponent());
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		sb.append("---Binary Grammar Rules. Total: " + nbrule + ", average ncomp: " + ((double) ncomp / nbrule) + "\n");
		
		return sb.toString();
	}
	
}
