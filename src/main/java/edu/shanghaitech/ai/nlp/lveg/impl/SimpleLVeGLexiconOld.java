package edu.shanghaitech.ai.nlp.lveg.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Indexer;
import edu.shanghaitech.ai.nlp.util.Numberer;

/**
 * TODO find the difference from SimpleLVeGLexicon. They have different outputs with 
 * the only difference in time when the rules are created and initialized.
 * 
 * @author Yanpeng Zhao
 *
 */
public class SimpleLVeGLexiconOld extends LVeGLexicon {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1066429113106930585L;
	public IndexMap[] wordIndexMap;
	public Indexer<String> wordIndexer;
	protected int nword;
	protected int[] wordCounter;
	
	protected GaussianMixture[][] counts;  // tag-word
	protected UnaryGrammarRule[][] urules; // tag-word
	
	
	public SimpleLVeGLexiconOld(Numberer numberer, int ntag) {
		this.wordIndexer = new Indexer<String>();
		this.lastWord = TOKEN_UNKNOWN;
		this.lastPosition = -1;
		this.lastSignature = "";
		this.unknownLevel = 5; // 5 is English specific
		if ((Corpus.myTreebank != Corpus.TreeBankType.WSJ) ||
				Corpus.myTreebank == Corpus.TreeBankType.BROWN) {
			this.unknownLevel = 4;
		}
		if (numberer == null) {
			this.numberer = null;
			this.ntag = ntag;
		} else {
			this.numberer = numberer;
			this.ntag = numberer.size();
		}
	}
	
	
	@Override
	public void postInitialize(StateTreeList trees) {
		this.nword = wordIndexer.size();
		this.counts = new GaussianMixture[ntag][];
		this.urules = new UnaryGrammarRule[ntag][];
		this.wordIndexMap = new IndexMap[ntag];
		this.wordCounter = new int[nword];
		for (int i = 0; i < ntag; i++) {
			wordIndexMap[i] = new IndexMap(nword);
		}
		
		for (Tree<State> tree : trees) {	
			List<State> words = tree.getTerminalYield();
			List<State> tags = tree.getPreTerminalYield();
			
			for (int i = 0; i < words.size(); i++) {
				int wordIdx = wordIndexer.indexOf(words.get(i).getName());
				if (wordIdx < 0) { 
					System.err.println("Word \"" + words.get(i).getName() + "\" NOT Found.");
					continue; 
				}
				wordCounter[wordIdx]++;
				wordIndexMap[tags.get(i).getId()].add(wordIdx);
			}
		}
		
		for (short i = 0; i < ntag; i++) {
			int nmap = wordIndexMap[i].size();
			counts[i] = new GaussianMixture[nmap];
			urules[i] = new UnaryGrammarRule[nmap];
			for (short j = 0; j < nmap; j++) {
				int wordIdx = wordIndexMap[i].get(j);
				int frequency = wordIndexMap[i].frequency(wordIdx);
				
				counts[i][j] = new DiagonalGaussianMixture();
				urules[i][j] = new UnaryGrammarRule(i, (short) wordIdx, GrammarRule.LHSPACE);
				urules[i][j].getWeight().setBias(frequency);
				
				optimizer.addRule(urules[i][j]);
			}
		}
	}
	
	
	@Override
	public void labelTrees(StateTreeList trees) {
		labelTrees(wordIndexer, trees);
	}
	

	@Override
	public void tallyStateTree(Tree<State> tree) {
		List<State> words = tree.getYield();
		for (State word : words) {
			String name = word.getName();
			wordIndexer.add(name);
		}
	}
	
	
	@Override
	public List<GrammarRule> getRulesWithWord(State word) {
		int wordIdx = word.wordIdx;
		List<GrammarRule> list = new ArrayList<GrammarRule>();
		for (short i = 0; i < ntag; i++) {
			int ruleIdx = wordIndexMap[i].indexOf(wordIdx);
			if (ruleIdx >= 0) {
				list.add(urules[i][ruleIdx]);
			}
		}
		return list;
	}


	@Override
	public GaussianMixture score(State word, short idTag) {
		// map the word to its real index
		int wordIdx = word.wordIdx;
		if (wordIdx < 0) {
			wordIdx = wordIndexer.indexOf(word.getName());
			word.wordIdx = wordIdx;
		}
		int ruleIdx = wordIndexMap[idTag].indexOf(wordIdx); 
		if (ruleIdx == -1) {
			System.err.println("Unknown word: " + word.getName());
			return null;
		}
		return urules[idTag][ruleIdx].getWeight();
	}                                           
	
	
	@Override
	protected boolean isKnown(String word) {
		return wordIndexer.indexOf(word) != -1;
	}

	
	public String toString(Numberer numberer) {
		String word = null;
		int count = 0, ncol = 1;
		StringBuffer sb = new StringBuffer();
		sb.append("Grammar [nWord=" + nword + "]\n");
		
		sb.append("---Words. Total: " + nword + "\n");
		for (int i = 0; i < wordIndexer.size(); i++) {
			word = wordIndexer.get(i);
			sb.append(wordIndexer.indexOf(word) + " : " + word + "\t");
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}

		sb.append("\n");
		sb.append("---Unary Rules---\n");
		for (int i = 0; i < ntag; i++) {
			count = 0;
			int nmap = wordIndexMap[i].size();
			sb.append("Tag " + i + "\t[" + numberer.object(i) + "] has " + nmap + " rules\n" );
			for (int j = 0; j < nmap; j++) {
				sb.append(urules[i][j] + "\t" + urules[i][j].getWeight().getBias());
				if (++count % ncol == 0) {
					sb.append("\n");
				}
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}


	/**
	 * @author Yanpeng Zhao
	 *
	 */
	protected static class IndexMap implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 7305207097186685083L;
		private int count;
		private List<Integer> to;
		private List<Integer> from;
		private List<Integer> frequency;
		
		
		public IndexMap(int n) {
			this.count = 0;
			this.to = new ArrayList<Integer>(n);
			this.from = new ArrayList<Integer>(n);
			this.frequency = new ArrayList<Integer>(n);
			// initialization
			for (int i = 0; i < n; i++) {
				to.add(-1);
				from.add(-1);
				frequency.add(0);
			}
		}
		
		
		/**
		 * @param i word index
		 */
		public void add(int i) {
			if (i < 0 || i > to.size()) { return; }
			if (to.get(i) == -1) {
				to.set(i, count);
				from.set(count, i);
				count++;
			}
			frequency.set(i, frequency.get(i) + 1);
		}
		
		
		/**
		 * @param i word index
		 * @return  frequency of the rule with i as the child
		 */
		public int frequency(int i) {
			if (i < 0 || i > to.size()) { return -1; }
			return frequency.get(i);
		}
		
		
		/**
		 * @param i mapping index
		 * @return  word index
		 */
		public int get(int i) {
			if (i < 0 || i > to.size()) { return -1; }
			return from.get(i);
		}
		
		
		public int indexOf(int i) {
			if (i < 0 || i > to.size()) { return -1; }
			return to.get(i);
		}
		
		
		public int size() {
			return this.count;
		}
		
		
		public IndexMap copy() {
			IndexMap map = new IndexMap(to.size());
			map.to    = new ArrayList<Integer>(to.size());
			map.from  = new ArrayList<Integer>(to.size());
			map.count = count;
			map.to.addAll(to);
			map.from.addAll(from);
			return map;
		}
		
		
		public void clear() {
			this.count = 0;
			this.to.clear();
			this.from.clear();
		}
	}


	@Override
	protected void initialize() {
	}


	@Override
	public void postInitialize() {
	}


	@Override
	public void addURule(UnaryGrammarRule rule) {
	}
	
}
