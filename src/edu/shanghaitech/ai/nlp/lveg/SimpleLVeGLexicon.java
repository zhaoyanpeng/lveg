package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.optimization.SGDMinimizer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
public class SimpleLVeGLexicon extends LVeGLexicon implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public IndexMap[] wordIndexMap;
	public Indexer<String> wordIndexer;

	protected int nTag;
	protected int nWord;
	protected int[] wordCounter;
	
	protected GaussianMixture[][] counts;  // tag-word
	protected UnaryGrammarRule[][] urules; // tag-word
	
	/**
	 * count0 stores rule counts that are evaluated given the parse tree
	 * count1 stores rule counts that are evaluated without the parse tree 
	 */
	private Map<GrammarRule, Double> count0;
	private Map<GrammarRule, Double> count1;

	
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
	
	
	/**
	 * Assume that rare words have been replaced by their signature.
	 * 
	 * @param trees a set of trees
	 */
	public void postInitialize(StateTreeList trees, int nTag) {
		this.nTag = nTag;
		this.nWord = wordIndexer.size();
		this.counts = new GaussianMixture[nTag][];
		this.urules = new UnaryGrammarRule[nTag][];
		this.wordIndexMap = new IndexMap[nTag];
		this.wordCounter = new int[nWord];
		this.count0 = new HashMap<GrammarRule, Double>();
		this.count1 = new HashMap<GrammarRule, Double>();
		
		for (int i = 0; i < nTag; i++) {
			wordIndexMap[i] = new IndexMap(nWord);
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
		
		for (short i = 0; i < nTag; i++) {
			int nmap = wordIndexMap[i].size();
			counts[i] = new GaussianMixture[nmap];
			urules[i] = new UnaryGrammarRule[nmap];
			for (short j = 0; j < nmap; j++) {
				int wordIdx = wordIndexMap[i].get(j);
				int frequency = wordIndexMap[i].frequency(wordIdx);
				
				counts[i][j] = new DiagonalGaussianMixture();
				urules[i][j] = new UnaryGrammarRule(i, (short) wordIdx, GrammarRule.LHSPACE);
				urules[i][j].getWeight().setBias(frequency);
				
				count0.put(urules[i][j], 0.0);
				count1.put(urules[i][j], 0.0);
			}
		}
		
		labelTrees(trees);
	}
	

	@Override
	public void tallyStateTree(Tree<State> tree) {
		
		List<State> words = tree.getYield();
		for (State word : words) {
			String name = word.getName();
			wordIndexer.add(name);
		}
		
		/*
		List<State> words = tree.getYield();
		List<State> tags = tree.getPreTerminalYield();
		
		for (int pos = 0; pos < words.size(); pos++) {
			short tag = tags.get(pos).getId();
			String word = words.get(pos).getName();
			
			int wordIdx = wordIndexer.indexOf(word);
			
			// This method corresponds to the trainTree() in Berkeley's implementation, in 
			 * which expected counts are initialized according to some random strategies. 
			 * But we do not need it since the counts have been already initialized before. 
			// TODO nothing to do (see reasons above)
		}
		*/
	}
	
	
	@Override
	protected void applyGradientDescent(Random random, double learningRate) {
		double cnt0, cnt1;
		for (short i = 0; i < nTag; i++) {
			for (short j = 0; j < wordIndexMap[i].size(); j++) {
				cnt0 = count0.get(urules[i][j]);
				cnt1 = count1.get(urules[i][j]);
				if (cnt0 == cnt1) { continue; }
				SGDMinimizer.applyGradientDescent(urules[i][j].getWeight(), random, cnt0, cnt1, learningRate);
			}
		}
		resetCount();
	}
	
	
	/**
	 * Initialize word index.
	 * 
	 * @param trees
	 */
	public void labelTrees(StateTreeList trees) {
		for (Tree<State> tree : trees) {
			List<State> words = tree.getYield();
			for (State word : words) {
				word.wordIdx = wordIndexer.indexOf(word.getName());
				word.signIdx = -1;
			}
		}
	}
	
	
	private void resetCount() {
		for (Map.Entry<GrammarRule, Double> count : count0.entrySet()) {
			count.setValue(0.0);
		}
		for (Map.Entry<GrammarRule, Double> count : count1.entrySet()) {
			count.setValue(0.0);
		}
	}
	
	
	public void addCount(short idParent, short idChild, char type, double increment, boolean withTree) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		addCount(rule, increment, withTree);
	}
	
	
	public double getCount(short idParent, short idChild, char type, boolean withTree) {
		GrammarRule rule = new UnaryGrammarRule(idParent, idChild, type);
		return getCount(rule, withTree);
	}
	
	
	public void addCount(GrammarRule rule, double increment, boolean withTree) {
		Map<GrammarRule, Double> count = withTree ? count0 : count1;
		if (rule != null && count.get(rule) != null) {
			count.put(rule, count.get(rule) + increment);
			return;
		}
		if (rule == null) {
			System.err.println("The Given Rule is NULL.");
		} else {
			System.err.println("Grammar Rule NOT Found: " + rule);
		}
	}
	
	
	public double getCount(GrammarRule rule, boolean withTree) {
		Map<GrammarRule, Double> count = withTree ? count0 : count1;
		if (rule != null && count.get(rule) != null) {
			return count.get(rule);
		}
		if (rule == null) {
			System.err.println("The Given Rule is NULL.");
		} else {
			System.err.println("Grammar Rule NOT Found: " + rule);
		}
		return -1.0;
	}
	
	
	@Override
	protected List<GrammarRule> getRulesWithWord(int wordIdx) {
		List<GrammarRule> list = new ArrayList<GrammarRule>();
		for (short i = 0; i < nTag; i++) {
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
		int ruleIdx = wordIndexMap[idTag].indexOf(word.wordIdx); 
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


	@Override
	public void tieRareWordStats(int threshold) {
		// TODO nothing to do (the same as the Berkeley's implementation)
		// DONE nothing to do
		return;
	}


	@Override
	public void optimize() {
		// TODO smooth the score
		// In Berkeley's implementation, scores are initialized with the expected counts. 
		// The same for the unary or binary rule probability initialization, which first
		// is randomly initialized to a double value, and then is normalized.
		// 
		// How should we implement the same mechanism with MixtureGaussian counts? 
		// DONE We can start training with E-Step or M-Step in EM. And Berkeley's
		// way can be seen as starting training with M-Step.
		return;
	}
	
	
	@Override
	public String toString() {
		String word = null;
		int count = 0, ncol = 1;
		StringBuffer sb = new StringBuffer();
		sb.append("Grammar [nWord=" + nWord + "]\n");
		
		sb.append("---Words. Total: " + nWord + "\n");
		for (int i = 0; i < wordIndexer.size(); i++) {
			word = wordIndexer.get(i);
			sb.append(wordIndexer.indexOf(word) + " : " + word + "\t");
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		
		Numberer numberer = Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET);

		sb.append("\n");
		sb.append("---Unary Rules---\n");
		for (int i = 0; i < nTag; i++) {
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
		private static final long serialVersionUID = 1L;
		
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
	
}
