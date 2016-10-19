package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.shanghaitech.ai.nlp.syntax.State;

public class SimpleLVeGLexicon extends LVeGLexicon implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public Indexer<String> wordIndexer;
	
	protected int nTag;
	protected int nWord;
	protected int[] wordCounter;
	
	protected GaussianMixture[][] counts;  // tag-word
	protected UnaryGrammarRule[][] urules; // tag-word

	
	/**
	 * Rules with probabilities below this value will be filtered.
	 */
	private double filterThreshold;
	
	
	public SimpleLVeGLexicon() {
		this.wordIndexer = new Indexer<String>();
		this.lastWord = LVeGLearner.TOKEN_UNKNOWN;
		this.lastPosition = -1;
		this.lastSignature = "";
		this.unknownLevel = 5; // 5 is English specific
		
		if ((Corpus.myTreebank != Corpus.TreeBankType.WSJ) ||
				Corpus.myTreebank == Corpus.TreeBankType.BROWN) {
			this.unknownLevel = 4;
		}
	}
	
	
	public SimpleLVeGLexicon(StateTreeList trees, int nTag, double filterThreshold) {
		this();
		this.nTag = nTag;
		this.filterThreshold = filterThreshold;
		initialize(trees);
	}
	
	
	/**
	 * Assume that rare words have been replaced by their signature.
	 * 
	 * @param trees a set of trees
	 */
	private void initialize(StateTreeList trees) {
		for (Tree<State> tree : trees) {
			List<State> words = tree.getYield();
			for (State word : words) {
				String name = word.getName();
				wordIndexer.add(name);
			}
		}
		
		this.nWord = wordIndexer.size();
		this.counts = new GaussianMixture[nTag][nWord];
		this.urules = new UnaryGrammarRule[nTag][nWord];
		
		for (short i = 0; i < nTag; i++) {
			for (short j = 0; j < nWord; j++) {
				urules[i][j] = new UnaryGrammarRule(i, j, GrammarRule.LHSPACE);
				counts[i][j] = new GaussianMixture();
			}
		}
		
		labelTrees(trees);
	}
	
	
	/**
	 * What is the method used for?
	 * 
	 * @param trees
	 */
	public void labelTrees(StateTreeList trees) {
		// TODO whether we need it or not
		for (Tree<State> tree : trees) {
			List<State> words = tree.getYield();
			for (State word : words) {
				word.wordIdx = wordIndexer.indexOf(word.getName());
				word.signIdx = -1;
			}
		}
	}
	

	@Override
	public void tallyStateTree(Tree<State> tree) {
		
		return; // nothing to do
		
		/*
		List<State> words = tree.getYield();
		List<State> tags = tree.getPreTerminalYield();
		
		for (int pos = 0; pos < words.size(); pos++) {
			short tag = tags.get(pos).getId();
			String word = words.get(pos).getName();
			
			int wordIdx = wordIndexer.indexOf(word);
			
			// expected counts have been initialized when instantiated
			// TODO nothing to do
		}
		*/
	}


	@Override
	public GaussianMixture score(State word, short idTag) {
		// TODO Auto-generated method stub
		int wordIdx = word.wordIdx;
		if (wordIdx == -1) {
			System.err.println("Unknown word: " + word.getName());
			return null;
		}
		return urules[idTag][wordIdx].getWeight();
	}                                           
	
	
	@Override
	protected boolean isKnown(String word) {
		return wordIndexer.indexOf(word) != -1;
	}


	@Override
	public void tieRareWordStats(int threshold) {
		// TODO Auto-generated method stub
		return;
	}


	@Override
	public void optimize() {
		// TODO Auto-generated method stub
		for (int i = 0; i < nTag; i++) {
			//
		}
		
		// TODO smooth the score
	}
}
