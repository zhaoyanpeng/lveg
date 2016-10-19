package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Counter;
import edu.shanghaitech.ai.nlp.syntax.State;

public class LVeGCorpus {


	/**
	 * @param trees         the parse tree
	 * @param lexicon       temporary data holder (wordIndexer)
	 * @param rareThreshold words with frequencies lower than this value will be replaced with its signature
	 * 
	 */
	public static void replaceRareWords(
			StateTreeList trees, SimpleLVeGLexicon lexicon, int rareThreshold) {
		Counter<String> wordCounter = new Counter<String>();
		for (Tree<State> tree : trees) {
			List<State> words = tree.getYield();
			for (State word : words) {
				String name = word.getName();
				wordCounter.incrementCount(name, 1.0);
				lexicon.wordIndexer.add(name);
			}
		}
		
		for (Tree<State> tree : trees) {
			List<State> words = tree.getYield();
			int pos = 0;
			for (State word : words) {
				String name = word.getName();
				if (wordCounter.getCount(name) <= rareThreshold) {
					name = lexicon.getCachedSignature(name, pos);
					word.setName(name);
				}
				pos++;
			}
		}
	}

}
