package edu.shanghaitech.ai.nlp.util;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Cell;
import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * Useful methods.
 * 
 * @author Yanpeng Zhao
 *
 */
public class MethodUtil {
	
	private static Random random = new Random(LVeGLearner.randomseed);
	private static LVeGGrammar grammar;
	private static LVeGLexicon lexicon;
	
	
	public static void debugCount(LVeGGrammar agrammar, LVeGLexicon alexicon, Tree<State> tree, Chart chart) {
		grammar = agrammar;
		lexicon = alexicon;
		
		double count0 = 0;
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		// unary grammar rules
		for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
			GrammarRule rule = rmap.getValue();
			double count = grammar.getCount(rule, false);
			LVeGLearner.logger.trace(rule + " count=" + count);
			
			if (rule.getLhs() == 0) {
				count0 += count;
			}
		}
		LVeGLearner.logger.trace("Count for rules starting with 0 is " + count0);
	}
	
	
	public static void debugCount(LVeGGrammar agrammar, LVeGLexicon alexicon, Tree<State> tree) {
		grammar = agrammar;
		lexicon = alexicon;
		checkCount(tree);
	}
	
	
	public static void checkCount(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			checkCount(child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			double count = lexicon.getCount(idParent, (short) word.wordIdx, GrammarRule.LHSPACE, true);
			LVeGLearner.logger.trace("Word\trule: [" + idParent + "] count=" + count); // DEBUG
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				// root, if (idParent == 0) is true
				char type = idParent == 0 ? GrammarRule.RHSPACE : GrammarRule.GENERAL;
				
				double count = grammar.getCount(idParent, idChild, type, true);
				LVeGLearner.logger.trace("Unary\trule: [" + idParent + ", " + idChild + "] count=" + count); // DEBUG
				break;
			}
			case 2: {
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				double count = grammar.getCount(idParent, idlChild, idrChild, true);
				LVeGLearner.logger.trace("Binary\trule: [" + idParent + ", " + idlChild + ", " + idrChild + "] count=" + count); // DEBUG
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	public static void debugChart(List<Cell> chart, short nfirst) {
		if (chart != null) {
			for (int i = 0; i < chart.size(); i++) {
				LVeGLearner.logger.debug(i + "\t" + chart.get(i).toString(true, nfirst) + "\n");
			}
		}
	}
	
	
	public static void debugTree(Tree<State> tree, boolean simple, short nfirst) {
		StringBuilder sb = new StringBuilder();
		toString(tree, simple, nfirst, sb);
		LVeGLearner.logger.debug(sb);
	}
	
	
	private static void toString(Tree<State> tree, boolean simple, short nfirst, StringBuilder sb) {
		if (tree.isLeaf()) { return; }
		sb.append('(');
		
		State state = tree.getLabel();
		if (state != null) {
			sb.append(state);
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
			toString(child, simple, nfirst, sb);
		}
		sb.append(')');
	}
	
	
	/**
	 * @param trees
	 * 
	 * @deprecated
	 * 
	 */
	public static void isParentEqualToChild(List<Tree<State>> trees) {
		int count = 0;
		for (Tree<State> tree : trees) {
			if (isParentEqualToChild(tree)) {
				System.out.println("The parent and children are the same in the tree " + count + ":");
				System.out.println(tree);
				return;
			}
			count++;
		}
		System.out.println("Oops!");
	}
	
	
	/**
	 * Check if the unary grammar rules could make a circle.
	 * 
	 * @param grammar the grammar
	 */
	public static boolean checkUnaryRuleCircle(LVeGGrammar agrammar, LVeGLexicon alexicon, boolean startWithC) {
		grammar = agrammar;
		lexicon = alexicon;
		boolean found = false;
		for (int i = 0; i < grammar.nTag; i++) {
			int repeated = 0;
			Set<Integer> visited = new LinkedHashSet<Integer>();
			// System.out.println("Tag " + i + "\t");
			if ((repeated = checkUnaryRuleCircle(i, visited, startWithC)) > 0) {
				LVeGLearner.logger.error("Repeated item: " + repeated + "\tin the path that begins with " + i + " was found: " + visited);
				found = true;
			}
		}
		// System.out.println("No circles were found.");
		return found;
	}
	
	
	public static int checkUnaryRuleCircle(int index, Set<Integer> visited, boolean startWithC) {	
		if (visited.contains(index)) { 
			// System.out.println("Repeated item: " + index);
			return index; 
		} else {
			visited.add(index);
		}
		List<GrammarRule> rules;
		if (startWithC) {
			rules = grammar.getUnaryRuleWithC(index);
		} else {
			rules = grammar.getUnaryRuleWithP(index);
		}
		
		for (GrammarRule rule : rules) {
			UnaryGrammarRule r = (UnaryGrammarRule) rule;
			int id = startWithC ? r.getLhs() : r.getRhs(), repeat;
			if ((repeat = checkUnaryRuleCircle(id, visited, startWithC)) > 0) {
				return repeat;
			}
		}
		if (rules.isEmpty()) { /*System.out.println("Path: " + visited);*/ }
		visited.remove(index);
		return -1;
	}
	
	
	public static void isParentEqualToChild(StateTreeList stateTreeList) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (isParentEqualToChild(tree)) {
				System.out.println("The parent and children are the same in the tree " + count + ":");
				System.out.println(tree);
				return;
			}
			count++;
		}
		System.out.println("Oops!");
	}
	
	
	public static void isChildrenSizeZero(StateTreeList stateTreeList) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (isChildrenSizeZero(tree)) {
				System.out.println("Pre-terminal node has no children in the tree: " + count + ":");
				System.out.println(tree);
				return;
			}
			count++;
		}
		System.out.println("Oops!");
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
			System.err.println("Malformed tree: more than two children. Exiting...");
			System.exit(0);
		}
		for (Tree<State> child : children) {
			if (containsRule(child, idp, idc)) {
				return true;
			}
		}
		return false;
	}
	
	
	public static boolean isChildrenSizeZero(Tree<State> tree) {
		if (tree.isLeaf()) { return false; }
		
		List<Tree<State>> children = tree.getChildren();
		
		switch (children.size()) {
		case 0:
			if (tree.isPreTerminal()) {
				System.out.println("Pre-terminal: " + tree.getLabel());
			}
			System.out.println("Non-terminal: " + tree.getLabel());
			return true;
		case 1:
			break;
		case 2:
			break;
		default:
			System.err.println("Malformed tree: more than two children. Exiting...");
			System.exit(0);
		}
		
		for (Tree<State> child : children) {
			if (isChildrenSizeZero(child)) {
				return true;
			}
		}
		return false;
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
				System.out.println("Unary rule: parent and child have the same id.");
				System.out.println(tree);
				return true;
			}
			break;
		case 2:
			short idLeftChild = children.get(0).getLabel().getId();
			short idRightChild = children.get(1).getLabel().getId();
			if (idParent == idLeftChild || idParent == idRightChild) {
				System.out.println("Binary rule: parent and children have the same id");
				System.out.println(tree);
				return true;
			}
			break;
		default:
			System.err.println("Malformed tree: more than two children. Exiting...");
			System.exit(0);
		}
		
		for (Tree<State> child : children) {
			if (isParentEqualToChild(child)) {
				return true;
			}
		}
		
		return false;
	}
	
	
	public static void randomInitArrayInt(int[] array, int maxint) {
		for (int i = 0; i < array.length; i++) {
			array[i] = (int) (random.nextDouble() * maxint);
		}
	}
	
	
	public static void randomInitArrayDouble(double[] array) {
		for (int i = 0; i < array.length; i++) {
			array[i] = random.nextDouble();
		}
	}
	
	
	/**
	 * @param list    item container
	 * @param type    Doubel.class or Integer.class
	 * @param length  number of items in the list
	 * @param maxint  maximum for integer, and 1 for double
	 * @param nonzero zero inclusive (false) or not exclusive (true)
	 */
	public static <T> void randomInitList(List<T> list, Class<T> type, int length, int maxint, boolean nonzero) {
		Double obj = new Double(0);
		for (int i = 0; i < length; i++) {
			double tmp = random.nextDouble() * maxint;
			if (nonzero) { while (tmp == 0.0) { tmp = random.nextDouble() * maxint; } }
			list.add(type.isInstance(obj) ? type.cast(tmp) : type.cast((int) tmp));
		}
	}
	
	
	/**
	 * @param array  item container
	 * @param type   Double.class or Integer.class
	 * @param maxint maximum for integer, and 1 for double
	 * 
	 */
	public static <T> void randomInitArray(T[] array, Class<T> type, int maxint) {
		Double obj = new Double(0);
		for (int i = 0; i < array.length; i++) {
			double tmp = random.nextDouble() * maxint;
			array[i] = type.isInstance(obj) ? type.cast(tmp) : type.cast((int) tmp);
		}
		
	}
	
	
	public static List<String> double2str(List<Double> list, int precision, int nfirst) {
		List<String> strs = new ArrayList<String>();
		String format = "%." + precision + "f";
		if (nfirst < 0 || nfirst > list.size()) { nfirst = list.size(); }
		for (int i = 0; i < nfirst; i++) {
			strs.add(String.format(format, list.get(i)));
		}
		/*
		for (Double d : list) {
			strs.add(String.format(format, d));
		}
		*/
		return strs;
	}
	
	
	public static double sum(List<Double> list) {
		double sum = 0.0;
		for (Double d : list) {
			sum += d;
		}
		return sum;
	}
	
	
	public static void printArrayInt(int[] array) {
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		System.out.println(string);
	}
	
	
	public static void printArrayDouble(double[] array) {
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		System.out.println(string);
	}
	
	
	public static <T> void printArray(T[] array) {
		if (isEmpty(array)) { return; }
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		System.out.println(string);
	}
	
	
	public static <T> void printList(List<T> list) {
		if (isEmpty(list)) { return; }
		String string = "[";
		for (int i = 0; i < list.size() - 1; i++) {
			string += list.get(i) + ", ";
		}
		string += list.get(list.size() - 1) + "]";
		System.out.println(string);
	}
	
	
	public static <T> boolean isEmpty(T[] array) {
		if (array == null || array.length == 0) {
			System.err.println("[null or empty]");
			return true;
		}
		return false;
	}
	
	
	public static <T> boolean isEmpty(List<T> list) {
		if (list == null || list.isEmpty()) {
			System.err.println("[null or empty]");
			return true;
		}
		return false;
	}

}
