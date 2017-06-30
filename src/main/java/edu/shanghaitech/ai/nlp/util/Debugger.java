package edu.shanghaitech.ai.nlp.util;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Cell;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.FunUtil.KeyComparator;

public class Debugger extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -426101396754992389L;
	private static final String DUMMY_STR = "OOPS";

	
	public static void debugShuffle(StateTreeList trainTrees) {
		int cnt = 0;
		for (Tree<State> tree : trainTrees) {
			logger.trace(tree.getYield() + "\n\n");
			if (++cnt > 4) {
				break;
			}
		}
		trainTrees.shuffle(new Random(0));
		cnt = 0;
		for (Tree<State> tree : trainTrees) {
			logger.trace(tree.getYield() + "\n\n");
			if (++cnt > 4) {
				break;
			}
		}
	}
	
	
	public static void debugChainRule(LVeGGrammar grammar) {
		int count = 0, ncol = 1;
		StringBuffer sb = new StringBuffer();
		sb.append("\n---Chain Unary Rules---\n");
		for (int i = 0; i < grammar.ntag; i++ ) {
			logger.trace("Tag: " + i + "\n");
			List<GrammarRule> rules = grammar.getChainSumUnaryRulesWithP(i);
			for (GrammarRule rule : rules) {
				sb.append(rule + "\t" + rule.getWeight().ncomponent());
				if (++count % ncol == 0) {
					sb.append("\n");
				}
			}
		}
		logger.trace(sb.toString() + "\n");
	}
	
	
	public static void debugRuleWeightInTheTree(LVeGGrammar grammar, LVeGLexicon lexicon, Tree<State> tree) {
		logger.trace("\n---Rules's Weights Used in The Tree---\n\n");
		checkRuleWeightInTheTree(grammar, lexicon, tree);
	}
	
	
	public static void checkRuleWeightInTheTree(LVeGGrammar grammar, LVeGLexicon lexicon, Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			checkRuleWeightInTheTree(grammar, lexicon, child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture cinScore = lexicon.score(word, idParent);
			logger.trace("Word\trule: [" + idParent + ", " + word.wordIdx + "/" + word.getName() + "] " + cinScore + "\n"); // DEBUG
			parent.setInsideScore(cinScore.copy(true));
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				GaussianMixture ruleScore;
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				RuleType type = idParent == 0 ? RuleType.RHSPACE : RuleType.LRURULE;
				ruleScore = grammar.getURuleWeight(idParent, idChild, type, false);
				logger.trace("Unary\trule: [" + idParent + ", " + idChild + "] " + ruleScore + "\n"); // DEBUG
				break;
			}
			case 2: {
				GaussianMixture ruleScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();

				ruleScore = grammar.getBRuleWeight(idParent, idlChild, idrChild, false);
				logger.trace("Binary\trule: [" + idParent + ", " + idlChild + ", " + idrChild + "] " + ruleScore + "\n"); // DEBUG
				break;
			}
			default:
				throw new RuntimeException("Malformed tree: invalid # of children. # children: " + children.size());
			}
		}
	}
	
	
	public static void debugCount(LVeGGrammar grammar, LVeGLexicon lexicon, Tree<State> tree, Chart chart) {
		int niter = 20, iiter = 0;
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getURuleMap();
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		// unary grammar rules
		logger.trace("\n---Unary Grammar Rules---\n\n");
		for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
			GrammarRule rule = rmap.getValue();
			Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> count = grammar.getCount(rule, false);
			logger.trace(rule + "\tcount=" + count + "\n");
			if (++iiter >= niter) { break; }
		}
		
		iiter = 0;
		// binary grammar rules
		logger.trace("\n---Binary Grammar Rules---\n\n");
		for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
			GrammarRule rule = rmap.getValue();
			Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> count = grammar.getCount(rule, false);
			logger.trace(rule + "\tcount=" + count + "\n");
			if (++iiter > niter) { break; }
		}
		
		iiter = 0;
		// unary rules in lexicon
		Set<GrammarRule> ruleSet = lexicon.getRuleSet();
		logger.trace("\n---Unary Grammar Rules in Lexicon---\n");
		for (GrammarRule rule : ruleSet) {
			Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> count = lexicon.getCount(rule, false);
			logger.trace(rule + "\tcount=" + count + "\n");
			if (++iiter >= niter) { break; }
		}
	}
	
	
	public static void debugCount(LVeGGrammar grammar, LVeGLexicon lexicon, Tree<State> tree) {
		logger.trace("\n---Rule Counts in The Tree---\n\n");
		checkCount(grammar, lexicon, tree);
	}
	
	
	public static void checkCount(LVeGGrammar grammar, LVeGLexicon lexicon, Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			checkCount(grammar, lexicon, child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> count = lexicon.getCount(idParent, (short) word.wordIdx, true, RuleType.LHSPACE);
			logger.trace("Word\trule: [" + idParent + ", " + word.wordIdx + "/" + word.getName() + "] count=" + count + "\n"); // DEBUG
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				// root, if (idParent == 0) is true
				RuleType type = idParent == 0 ? RuleType.RHSPACE : RuleType.LRURULE;
				
				Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> count = grammar.getCount(idParent, idChild, true, type);
				logger.trace("Unary\trule: [" + idParent + ", " + idChild + "] count=" + count + "\n"); // DEBUG
				break;
			}
			case 2: {
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				Map<Short, List<EnumMap<RuleUnit, GaussianMixture>>> count = grammar.getCount(idParent, idlChild, idrChild, true);
				logger.trace("Binary\trule: [" + idParent + ", " + idlChild + ", " + idrChild + "] count=" + count + "\n"); // DEBUG
				break;
			}
			default:
				throw new RuntimeException("Malformed tree: invalid # of children. # children: " + children.size());
			}
		}
	}
	
	
	public static void debugChart(List<Cell> chart, short nfirst, int nword, Numberer numberer) {
		int size = nword * (nword + 1) / 2;
		if (chart != null) {
			for (int i = 0; i < size; i++) {
				logger.debug(i + "\t" + chart.get(i).toString(true, nfirst, true, numberer) + "\n\n");
			}
		}
	}
	
	
	public static String debugTree(Tree<State> tree, boolean simple, short nfirst, Numberer numberer, boolean onlyname) {
		StringBuilder sb = new StringBuilder();
		if (!onlyname) {
			toString(tree, simple, nfirst, sb, numberer);
		} else {
			toString(tree, sb, numberer);
		}
		return sb.toString();
	}	
	
	
	private static void toString(Tree<State> tree, StringBuilder sb, Numberer numberer) {
		if (!tree.isLeaf()) {
			sb.append('(');
		}
		State state = tree.getLabel();
		if (state != null) {
			String name = state.getName();
			name = name != null ? name : (String) numberer.object(state.getId());
			sb.append(name);
		}
		if (!tree.isLeaf()) {
			for (Tree<State> child : tree.getChildren()) {
				sb.append(' ');
				toString(child, sb, numberer);
			}
			sb.append(')');
		}
	}
	
	
	private static void toString(Tree<State> tree, boolean simple, short nfirst, StringBuilder sb, Numberer numberer) {
		if (tree.isLeaf()) { sb.append("[" + tree.getLabel().wordIdx + ", " + tree.getLabel().toString(numberer) + "]"); return; }
		sb.append('(');
		
		State state = tree.getLabel();
		if (state != null) {
			sb.append(state.toString(numberer));
			if (state.getInsideScore() != null) { 
				sb.append(" iscore=" + state.getInsideScore().toString(simple, nfirst)); 
			} else {
				sb.append(" iscore=null");
			}
			if (state.getOutsideScore() != null) {
				sb.append(" oscore=" + state.getOutsideScore().toString(simple, nfirst));
			} else {
				sb.append(" oscore=null");
			}
		}
		for (Tree<State> child : tree.getChildren()) {
			sb.append(' ');
			toString(child, simple, nfirst, sb, numberer);
		}
		sb.append(')');
	}
	
	
	/**
	 * @param stateTreeList a set of parse trees
	 * @param maxLength     find the unary chain rule with the specific length
	 * @param treeFileName  file name that is used to save the tree to the image
	 */
	public static void lenUnaryRuleChain(StateTreeList stateTreeList, short maxLength, String treeFileName, Numberer numberer) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (lenUnaryRuleChain(tree, (short) 0, maxLength)) {
				if (count == 0) {
					logger.trace("The tree contains the unary rule chain of length >= " + maxLength + ":\n");
					logger.trace(tree + "\n");
					try {
						FunUtil.saveTree2image(tree, treeFileName + maxLength + "_" + count, null, numberer);
						logger.trace("The tree has been saved to " + treeFileName + maxLength + "\n");
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
				count++;
			}
		}
		logger.trace("# of trees containing the unary rule chain of length >= " + maxLength + ": " + count + "\n");
	}
	
	
	/**
	 * @param tree      the parse tree
	 * @param length    stack variable
	 * @param maxLength find the unary chain rule with the specific length
	 * @return
	 */
	public static boolean lenUnaryRuleChain(Tree<State> tree, short length, short maxLength) {
		if (length >= maxLength) { return true; }
		if (tree.isPreTerminal()) { return false; }
		
		List<Tree<State>> children = tree.getChildren();
		short idParent = tree.getLabel().getId();
		switch (children.size()) {
		case 0:
			break;
		case 1: {
			if (idParent != 0) {
				length += 1;
			}
			break;
		}
		case 2: {
			length = 0;
			break;
		}
		default:
			throw new RuntimeException("Malformed tree: invalid # of children. # children: " + children.size());
		}
		for (Tree<State> child : children) {
			if (lenUnaryRuleChain(child, length, maxLength)) {
				return true;
			}
		}
		return false;
	}
	
	
	
	/**
	 * Check if unary grammar rules could make a circle.
	 */
	public static boolean checkUnaryRuleCircle(LVeGGrammar grammar, LVeGLexicon lexicon, boolean startWithC) {
		boolean found = false;
		for (int i = 0; i < grammar.ntag; i++) {
			int repeated = 0;
			Set<Integer> visited = new LinkedHashSet<>();
//			logger.trace("Tag " + i + "\t\n");
			if ((repeated = checkUnaryRuleCircle(grammar, lexicon, i, visited, startWithC)) > 0) {
				logger.error("Repeated item: " + repeated + "\tin the path that begins with " + i + " was found: " + visited + "\n");
				found = true;
			}
		}
//		logger.trace("No circles were found.\n");
		return found;
	}
	
	
	public static int checkUnaryRuleCircle(LVeGGrammar grammar, LVeGLexicon lexicon, int index, Set<Integer> visited, boolean startWithC) {	
		if (visited.contains(index)) { 
//			logger.trace("Repeated item: " + index + "\n");
			return index; 
		} else {
			visited.add(index);
		}
		List<GrammarRule> rules;
		if (startWithC) {
			rules = grammar.getURuleWithC(index);
		} else {
			rules = grammar.getURuleWithP(index);
		}
		
		for (GrammarRule rule : rules) {
			UnaryGrammarRule r = (UnaryGrammarRule) rule;
			int id = startWithC ? r.lhs : r.rhs, repeat;
			if ((repeat = checkUnaryRuleCircle(grammar, lexicon, id, visited, startWithC)) > 0) {
				return repeat;
			}
		}
		if (rules.isEmpty()) { /*logger.trace("Path: " + visited + "\n");*/ }
		visited.remove(index);
		return -1;
	}
	
	
	public static void isChildrenSizeZero(StateTreeList stateTreeList) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (isChildrenSizeZero(tree)) {
				logger.trace("Pre-terminal node has no children in the tree: " + count + ":\n");
				logger.trace(tree + "\n");
				return;
			}
			count++;
		}
		logger.trace(DUMMY_STR + "\n");
	}
	
	
	public static boolean isChildrenSizeZero(Tree<State> tree) {
		if (tree.isLeaf()) { return false; }
		List<Tree<State>> children = tree.getChildren();
		switch (children.size()) {
		case 0:
			if (tree.isPreTerminal()) {
				logger.trace("Pre-terminal: " + tree.getLabel() + "\n");
			}
			logger.trace("Non-terminal: " + tree.getLabel() + "\n");
			return true;
		case 1:
			break;
		case 2:
			break;
		default:
			throw new RuntimeException("Malformed tree: invalid # of children. # children: " + children.size());
		}
		
		for (Tree<State> child : children) {
			if (isChildrenSizeZero(child)) {
				return true;
			}
		}
		return false;
	}
	
	
	public static void isParentEqualToChild(StateTreeList stateTreeList) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (isParentEqualToChild(tree)) {
				logger.trace("The parent and children are the same in the tree " + count + ":\n");
				logger.trace(tree + "\n");
				return;
			}
			count++;
		}
		logger.trace(DUMMY_STR + "\n");
	}
	
	
	public static boolean isParentEqualToChild(Tree<State> tree) {
		if (tree.isLeaf() || tree.isPreTerminal()) { return false; }
		
		short idParent = tree.getLabel().getId();
		List<Tree<State>> children = tree.getChildren();
		
		switch (children.size()) {
		case 0:
			
			break;
		case 1:
			short idChild = children.get(0).getLabel().getId();
			if (idParent == idChild) {
				logger.trace("Unary rule: parent and child have the same id.\n");
				logger.trace(tree + "\n");
				return true;
			}
			break;
		case 2:
			short idLeftChild = children.get(0).getLabel().getId();
			short idRightChild = children.get(1).getLabel().getId();
			if (idParent == idLeftChild || idParent == idRightChild) {
				logger.trace("Binary rule: parent and children have the same id.\n");
				logger.trace(tree + "\n");
				return true;
			}
			break;
		default:
			throw new RuntimeException("Malformed tree: invalid # of children. # children: " + children.size());
		}
		
		for (Tree<State> child : children) {
			if (isParentEqualToChild(child)) {
				return true;
			}
		}
		return false;
	}
	
	
	public static boolean containsRule(Tree<State> tree, short idp, short idc) {
		if (tree.isLeaf()) { return false; }
		
		List<Tree<State>> children = tree.getChildren();
		short idParent = tree.getLabel().getId();
		switch (children.size()) {
		case 0:
			break;
		case 1: {
			short idChild = children.get(0).getLabel().getId();
			if (idParent == idp && idChild == idc) {
				return true;
			}
			break;
		}
		case 2:
			break;
		default:
			throw new RuntimeException("Malformed tree: invalid # of children. # children: " + children.size());
		}
		for (Tree<State> child : children) {
			if (containsRule(child, idp, idc)) {
				return true;
			}
		}
		return false;
	}
	
	
	public static void debugTreebank() {
		String[] args = null;
		String fparams = "param.in";
		try {
			args = LearnerConfig.readFile(fparams, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		Options opts = (Options) optionParser.parse(args, true);
		// configurations
		LearnerConfig.initialize(opts, true); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = LearnerConfig.loadData(wrapper, opts);
		treeSummary(trees);
	}
	
	
	public static void treeSummary(Map<String, StateTreeList> trees) {
		StateTreeList trainTrees = trees.get(LearnerConfig.ID_TRAIN);
		StateTreeList testTrees = trees.get(LearnerConfig.ID_TEST);
		StateTreeList devTrees = trees.get(LearnerConfig.ID_DEV);
		
		logger.trace("\n---training sentence length summary---\n");
		lengthSummary(trainTrees);
		logger.trace("\n---  test sentence length summary  ---\n");
		lengthSummary(testTrees);
		logger.trace("\n---   dev sentence length summary   ---\n");
		lengthSummary(devTrees);
	}
	
	
	public static void lengthSummary(StateTreeList trees) {
		Map<Integer, Integer> summary = new HashMap<>();
		for (Tree<State> tree : trees) {
			int len = tree.getTerminalYield().size();
			if (summary.containsKey(len)) {
				summary.put(len, summary.get(len) + 1);
			} else {
				summary.put(len, 1);
			}
		}
		logger.trace(summary + "\n");
		logger.trace(summary.keySet() + "\n");
		logger.trace(summary.values() + "\n");
		
		int nbin = 150;
		Map<Integer, Integer> lens = new HashMap<>();
		for (Map.Entry<Integer, Integer> entry : summary.entrySet()) {
			for (int i = 0; i < nbin ; i += 10) {
				int len = entry.getKey();
				if (len <= i) {
					if (lens.containsKey(i)) {
						lens.put(i, lens.get(i) + entry.getValue());
					} else {
						lens.put(i, entry.getValue());
					}
				}
			}
		}
		
		KeyComparator bykey = new KeyComparator(lens);
		TreeMap<Integer, Integer> sorted = new TreeMap<>(bykey);
		sorted.putAll(lens);
		logger.trace(sorted.keySet() + "\n");
		logger.trace(sorted.values() + "\n");
	}
	
}
