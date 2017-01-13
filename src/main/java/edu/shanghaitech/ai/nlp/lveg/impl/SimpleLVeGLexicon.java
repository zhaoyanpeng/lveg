package edu.shanghaitech.ai.nlp.lveg.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Indexer;
import edu.shanghaitech.ai.nlp.util.Numberer;

/**
 * @author Yanpeng Zhao
 *
 */
public class SimpleLVeGLexicon extends LVeGLexicon {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1066429113106930585L;
	protected int nword;
	
	public Indexer<String> wordIndexer;
	
	
	public SimpleLVeGLexicon() {
		this.uRuleTable = new RuleTable<UnaryGrammarRule>(UnaryGrammarRule.class);
		this.uRuleMap = new HashMap<GrammarRule, GrammarRule>();
		this.wordIndexer = new Indexer<String>();
		this.lastWord = TOKEN_UNKNOWN;
		this.lastPosition = -1;
		this.lastSignature = "";
		this.unknownLevel = 5; // 5 is English specific
	}
	
	
	public SimpleLVeGLexicon(Numberer numberer, int ntag) {
		this();
		if (numberer == null) {
			this.numberer = null;
			this.ntag = ntag;
		} else {
			this.numberer = numberer;
			this.ntag = numberer.size();
		}
		initialize();
	}
	
	
	@Override
	protected void initialize() {
		this.uRulesWithP = new List[ntag];
		for (int i = 0; i < ntag; i++) {
			uRulesWithP[i] = new ArrayList<GrammarRule>();
		}
	}
	
	
	@Override
	public void postInitialize() {
		this.nword = wordIndexer.size();
		this.uRulesWithC = new List[nword];
		for (int i = 0; i < nword; i++) {
			uRulesWithC[i] = new ArrayList<GrammarRule>(5);
		}
		for (GrammarRule rule : uRuleTable.keySet()) {
			addURule((UnaryGrammarRule) rule);
		}
	}


	@Override
	public void addURule(UnaryGrammarRule rule) {
		if (uRulesWithP[rule.lhs].contains(rule)) { return; }
		uRulesWithP[rule.lhs].add(rule);
		uRulesWithC[rule.rhs].add(rule);
		uRuleMap.put(rule, rule);
		optimizer.addRule(rule);
	}
	
	
	@Override
	public void labelTrees(StateTreeList trees) {
		labelTrees(wordIndexer, trees);
	}
	

	@Override
	public void tallyStateTree(Tree<State> tree) {
		List<State> words = tree.getTerminalYield();
		List<State> tags = tree.getPreTerminalYield();
		for (int i = 0; i < words.size(); i++) {
			State word = words.get(i);
			String name = word.getName();
			wordIndexer.add(name); // generate word index
			int wordIdx = wordIndexer.indexOf(name);
			word.wordIdx = wordIdx;
			short tagIdx = tags.get(i).getId();
			GrammarRule rule = new UnaryGrammarRule((short) tagIdx, wordIdx, GrammarRule.LHSPACE);
			uRuleTable.addCount(rule, 1.0);
		}
	}
	
	
	@Override
	public List<GrammarRule> getRulesWithWord(State word) {
		// map the word to its real index
		int wordIdx = word.wordIdx;
		if (wordIdx < 0) {
			wordIdx = wordIndexer.indexOf(word.getName());
			word.wordIdx = wordIdx;
		}
		return uRulesWithC[wordIdx];
	}
	

	@Override
	public GaussianMixture score(State word, short itag) {
		// map the word to its real index
		int wordIdx = word.wordIdx;
		String signature = "(UNK)", name = word.getName();
		if (wordIdx < 0) { // the unlabeled word
			wordIdx = wordIndexer.indexOf(name);
			word.wordIdx = wordIdx;
		}
		if (wordIdx == -1) { // the rare word
			signature = getSignature(name, word.from);
			wordIdx = wordIndexer.indexOf(signature);
			word.wordIdx = wordIdx;
		}
		if (wordIdx == -1) { // the unknown word
			System.err.println("\nunknown word signature [tag = " + itag + ", word = " + name + ", sig = " +  signature + "]");
			return rndWeight(LVeGLearner.minmw);
		}
		GrammarRule rule = getURule(itag, wordIdx, GrammarRule.LHSPACE);
		if (rule == null) {
			System.err.println("\nunknown lexicon rule [tag = " + itag + ", word = " + name + ", sig = " +  signature + "]");
			return rndWeight(Double.NEGATIVE_INFINITY);
		}
		return rule.getWeight();
	}                                           
	
	
	@Override
	protected boolean isKnown(String word) {
		return wordIndexer.indexOf(word) != -1;
	}
	
	
	@Override
	public String toString() {
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
		
		int cnt = 0;
		sb.append("\n");
		sb.append("---Unary Rules---\n");
		for (int i = 0; i < ntag; i++) {
			count = 0;
			List<GrammarRule> rules = uRulesWithP[i];
			sb.append("Tag " + i + "\t[" + numberer.object(i) + "] has " + rules.size() + " rules\n" );
			for (GrammarRule rule : rules) {
				sb.append(rule + "\t" + rule.getWeight().getBias());
				if (++count % ncol == 0) {
					sb.append("\n");
				}
			}
			cnt += rules.size();
			sb.append("\n");
		}
		sb.append("---Lexicon rules. Total: " + cnt + "\n");
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
	
}
