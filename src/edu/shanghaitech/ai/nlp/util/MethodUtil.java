package edu.shanghaitech.ai.nlp.util;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
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
	public static void checkUnaryRuleCircle(LVeGGrammar agrammar, LVeGLexicon alexicon) {
		grammar = agrammar;
		lexicon = alexicon;
		int nword = 0;
		Queue<Short> children = new LinkedList<Short>();
		for (int i = 0; i < nword; i++) {
			Set<Integer> visited = new LinkedHashSet<Integer>();
			if (checkUnaryRuleCircle(i, visited)) {
				System.out.println(visited);
				break;
			}
		}
	}
	
	
	public static boolean checkUnaryRuleCircle(int index, Set<Integer> visited) {	
		if (visited.contains(index)) { 
			return true; 
		} else {
			visited.add(index);
		}
		
		List<UnaryGrammarRule> rules = grammar.getUnaryRuleWithC(index);
		for (UnaryGrammarRule rule : rules) {
			if (checkUnaryRuleCircle(rule.getLhs(), visited)) {
				return true;
			}
		}
		visited.remove(index);
		return false;
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
	 * @param list   item container
	 * @param type   Doubel.class or Integer.class
	 * @param length number of items in the list
	 * @param maxint maximum for integer, and 1 for double
	 * 
	 */
	public static <T> void randomInitList(List<T> list, Class<T> type, int length, int maxint) {
		Double obj = new Double(0);
		for (int i = 0; i < length; i++) {
			double tmp = random.nextDouble() * maxint;
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
	
	
	public static List<String> double2str(List<Double> list, int precision) {
		List<String> strs = new ArrayList<String>();
		String format = "%." + precision + "f";
		for (Double d : list) {
			strs.add(String.format(format, d));
		}
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
