package edu.shanghaitech.ai.nlp.lveg;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;

public interface LVeGLexicon {
	
	/**
	 * Tally (go over and record) the rules existing in the parse tree.
	 * 
	 * @param tree a parse tree
	 * 
	 */
	public void tallyStateTree(Tree<State> tree);
	
	public String getSignature(String word, int pos);
	
	public void tieRareWordStats(int threshold);
	
	public void optimize();
	
	public GaussianMixture score(State wordState, short idTag);
	
}
