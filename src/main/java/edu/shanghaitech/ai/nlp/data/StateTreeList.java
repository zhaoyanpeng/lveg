package edu.shanghaitech.ai.nlp.data;

import java.io.Serializable;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Numberer;

/**
 * @author Yanpeng Zhao
 *
 */
public class StateTreeList extends AbstractCollection<Tree<State>> implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6424710107698526446L;
	private final static short ID_WORD = -1;
	private List<Tree<State>> trees;
	
	
	public class StateTreeListIterator implements Iterator<Tree<State>> {
		
		Tree<State> currentTree;
		Iterator<Tree<State>> treeListIter;
		
		public StateTreeListIterator() {
			treeListIter = trees.iterator();
			currentTree = null;
		}
		
		@Override
		public boolean hasNext() {
			// TODO Auto-generated method stub
			if (currentTree != null) {
				// TODO 
			}
			return treeListIter.hasNext();
		}

		@Override
		public Tree<State> next() {
			// TODO Auto-generated method stub
			currentTree = treeListIter.next();
			return currentTree;
		}
		
		@Override
		public void remove() {
			treeListIter.remove();
		}
		
	}
	

	@Override
	public Iterator<Tree<State>> iterator() {
		// TODO Auto-generated method stub
		return new StateTreeListIterator();
	}

	@Override
	public int size() {
		// TODO Auto-generated method stub
		return trees.size();
	}
	
	@Override
	public boolean isEmpty() {
		return trees.isEmpty();
	}
	
	@Override
	public boolean add(Tree<State> tree) {
		return trees.add(tree);
	}
	
	public Tree<State> get(int i) {
		return trees.get(i);
	}
	
	
	public StateTreeList() {
		this.trees = new ArrayList<>();
	}
	
	
	public StateTreeList(StateTreeList stateTreeList) {
		this.trees = new ArrayList<>();
		for (Tree<State> tree: stateTreeList.trees) {
			trees.add(copyTreeButLeaf(tree));
		}
	}
	
	
	/**
	 * The leaf is copied by reference. It's the same as 
	 * {@code edu.berkeley.nlp.PCFGLA.StateSetTreeList.resizeStateSetTree}
	 * 
	 * @param tree a parse tree
	 * @return
	 * 
	 */
	private Tree<State> copyTreeButLeaf(Tree<State> tree) {
		if (tree.isLeaf()) { return tree; }
		State state = new State(tree.getLabel(), false);
		List<Tree<State>> children = new ArrayList<>();
		for (Tree<State> child : tree.getChildren()) {
			children.add(copyTreeButLeaf(child));
		}
		return new Tree<State>(state, children);
	}
	
	
	public StateTreeList copy() {
		StateTreeList stateTreeList = new StateTreeList();
		for (Tree<State> tree : trees) {
			stateTreeList.add(copyTree(tree));
		}
		return stateTreeList;
	}
	
	
	private Tree<State> copyTree(Tree<State> tree) {
		List<Tree<State>> children = new ArrayList<>(tree.getChildren().size());
		for (Tree<State> child : tree.getChildren()) {
			children.add(copyTree(child));
		}
		return new Tree<State>(tree.getLabel().copy(), children);
	}
	
	
	public StateTreeList(List<Tree<String>> trees, Numberer numberer) {
		this.trees = new ArrayList<>();
		for (Tree<String> tree : trees) {
			this.trees.add(stringTreeToStateTree(tree, numberer));
			tree = null; // clean the memory
		}
	}
	

	/**
	 * @param trees       parse trees
	 * @param numberer recording the ids of tags
	 * 
	 */
	public static void initializeNumbererTag(
			List<Tree<String>> trees, Numberer numberer) {
		for (Tree<String> tree : trees) {
			stringTreeToStateTree(tree, numberer);
		}
	}
	
	
	public static void stringTreeToStateTree(
			List<Tree<String>> trees, Numberer numberer) {
		for (Tree<String> tree : trees) {
			stringTreeToStateTree(tree, numberer);
		}
	}
	
	
	/**
	 * @param tree        a parse tree
	 * @param numberer record the ids of tags
	 * @return            parse tree represented by the state list
	 * 
	 */
	public static Tree<State> stringTreeToStateTree(Tree<String> tree, Numberer numberer) {
		Tree<State> result = stringTreeToStateTree(tree, numberer, 0, tree.getYield().size());
		List<State> words = result.getYield();
		for (short pos = 0; pos < words.size(); pos++) {
			words.get(pos).from = pos;
			words.get(pos).to = (short) (pos + 1);
		}
		return result;
	}
	
	
	public void shuffle(Random rnd) {
		Collections.shuffle(trees, rnd);
	}
	
	
	public void reset() {
		for (Tree<State> tree : trees) {
			reset(tree);
		}
	}
	
	
	public void reset(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		if (tree.getLabel() != null) {
			tree.getLabel().clear(false);
		}
		for (Tree<State> child : tree.getChildren()) {
			reset(child);
		}
	}
	
	
	/**
	 * Convert a state tree to a string tree.
	 * 
	 * @param tree        a state tree
	 * @param numberer which records the ids of tags
	 * @return
	 */
	public static Tree<String> stateTreeToStringTree(Tree<State> tree, Numberer numberer) {
		if (tree.isLeaf()) {
			String name = tree.getLabel().getName();
			return new Tree<String>(name);
		}
		
		String name = (String) numberer.object(tree.getLabel().getId());
		Tree<String> newTree = new Tree<String>(name);
		List<Tree<String>> children = new ArrayList<>();
		
		for (Tree<State> child : tree.getChildren()) {
			Tree<String> newChild = stateTreeToStringTree(child, numberer);
			children.add(newChild);
		}
		newTree.setChildren(children);
		return newTree;
	}
	
	
	/**
	 * Convert a string tree to a state tree.
	 * 
	 * @param tree        a parse tree
	 * @param numberer which records the ids of tags
	 * @param from        starting point of the span
	 * @param to          ending point of the span
	 * @return            parse tree represented by the state list
	 * 
	 */
	private static Tree<State> stringTreeToStateTree(Tree<String> tree, Numberer numberer, int from, int to) {
		if (tree.isLeaf()) {
			State state = new State(tree.getLabel().intern(), (short) ID_WORD, (short) from, (short) to);
			return new Tree<State>(state);
		}
		/* numberer is initialized here */
		short id = (short) numberer.number(tree.getLabel());
		
		// System.out.println(tree.getLabel().intern()); // tag name
		State state = new State(null, id, (short) from, (short) to);
		Tree<State> newTree = new Tree<State>(state);
		List<Tree<State>> children = new ArrayList<>();
		for (Tree<String> child : tree.getChildren()) {
			short length = (short) child.getYield().size();
			Tree<State> newChild = stringTreeToStateTree(child, numberer, from, from + length);
			from += length;
			children.add(newChild);
		}
		newTree.setChildren(children);
		return newTree;
	}

}
