package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lvet.LVeTTrainer;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.lvet.model.Word;
import edu.shanghaitech.ai.nlp.util.Indexer;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class TagWPair extends Pair {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6973533800445764594L;
	public static final String TOKEN_UNKNOWN = "UNK";
	public Indexer<String> wordIndexer;
	protected int unknownLevel;
	protected int nword;
	
	protected transient String lastSignature;
	protected transient String lastWord;
	protected transient int lastPosition;
	
	public TagWPair() {
		super();
		this.wordIndexer = new Indexer<String>();
		this.lastWord = TOKEN_UNKNOWN;
		this.lastPosition = -1;
		this.lastSignature = "";
		this.unknownLevel = 5; // 5 is English specific
	}
	
	public TagWPair(Numberer numberer, int ntag) {
		this();
		if (numberer == null) {
			this.numberer = null;
			this.ntag = ntag;
		} else {
			this.numberer = numberer;
			this.ntag = numberer.size();
			LEADING_IDX = numberer.number(LEADING);
			ENDING_IDX = numberer.number(ENDING);
		}
		initialize();
	}
	
	@Override
	protected void initialize() {
		this.edgesWithP = new List[ntag];
		for (int i = 0; i < ntag; i++) {
			edgesWithP[i] = new ArrayList<GrammarRule>();
		}
	}

	@Override
	public void postInitialize() {
		this.nword = wordIndexer.size();
		this.edgesWithC = new List[nword];
		for (int i = 0; i < nword; i++) {
			edgesWithC[i] = new ArrayList<GrammarRule>(5);
		}
		for (GrammarRule edge : edgeTable.keySet()) {
			edge.getWeight().setBias(edgeTable.getCount(edge).getBias());
			addEdge((UnaryGrammarRule) edge);
		}
	}

	@Override
	public void tallyTaggedWords(List<TaggedWord> words) {
		for (TaggedWord word : words) {
			String name = word.word;
			wordIndexer.add(name);
			word.wordIdx = wordIndexer.indexof(name);
			short tagIdx = (short) word.tagIdx;
			GrammarRule edge = new UnaryGrammarRule(tagIdx, word.wordIdx, RuleType.LHSPACE);
			if (!edgeTable.containsKey(edge)) {
				edge.initializeWeight(RuleType.LHSPACE, (short) -1, (short) -1); 
			}
			edgeTable.addCount(edge, 1.0);
		}
	}	
	
	public List<GrammarRule> getEdgesWithWord(Word word) {
		int wordIdx = getWordIdx(word);
		return edgesWithC[wordIdx];
	}
	
	public GaussianMixture score(Word word, short itag) {
		int wordIdx = getWordIdx(word); // should be >= 0
		if (wordIdx == -1) { // double check
			Recorder.logger.warn("\nUnknown Word Signature [P: " + itag + ", UC: " + wordIdx + ", word = " + word.getName() + ", sig = (UNK)]\n");
			GaussianMixture weight = GrammarRule.rndRuleWeight(RuleType.LHSPACE, (short) -1, (short) -1);
			weight.setWeights(LVeTTrainer.minmw);
			return weight;
		}
		GrammarRule rule = getEdge(itag, wordIdx, RuleType.LHSPACE);
		if (rule == null) { // double check
			Recorder.logger.warn("\nUnknown Lexicon Rule [P: " + itag + ", UC: " + wordIdx + ", word = " + word.getName() + ", sig = (UNK)]\n");
			GaussianMixture weight = GrammarRule.rndRuleWeight(RuleType.LHSPACE, (short) -1, (short) -1);
			/*weight.setWeights(Double.NEGATIVE_INFINITY);*/
			weight.setWeights(LVeTTrainer.minmw);
			return weight;
		}
		return rule.getWeight();
	} 
	
	public int getWordIdx(Word word) {
		// map the word to its real index
		int wordIdx = word.wordIdx, idx;
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
		while (wordIdx == -1 && (idx = signature.lastIndexOf('-')) > 0) { // the unknown word
			// stupid special case, such as Hardest UNK-INITC-est with pos = 0
			Recorder.logger.warn("\nUnknown word signature checking [word: " + name + ", sig: " + signature + "]\n");
			signature = signature.substring(0, idx);
			wordIdx = wordIndexer.indexof(signature);
			word.wordIdx = wordIdx;
		}
		if (wordIdx == -1) { // CHECK set it to UNK
			Recorder.logger.warn("\nUnknown word signature finalizing [word: " + name + ", sig: " + "UNK (DUMY)]\n");
			wordIdx = wordIndexer.indexof("UNK");
			word.wordIdx = wordIdx;
		}
		return wordIdx;
	}
	
	public void labelSequences(List<List<TaggedWord>> sequences) {
		for (List<TaggedWord> words : sequences) {
			int pos = 0;
			for (TaggedWord word : words) {
				word.wordIdx = wordIndexer.indexof(word.word);
				word.signIdx = -1;
				word.from = pos;
				word.to = pos + 1;
				pos++;
			}
		}
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
		
		int cnt = 0, ncomp = 0;
		sb.append("\n");
		sb.append("---Unary Rules---\n");
		for (int i = 0; i < ntag; i++) {
			count = 0;
			List<GrammarRule> rules = edgesWithP[i];
			sb.append("Tag " + i + "\t[" + numberer.object(i) + "] has " + rules.size() + " rules\n" );
			for (GrammarRule rule : rules) {
				ncomp += rule.weight.ncomponent();
				sb.append(rule + "\t\t" + edgeTable.getCount(rule).getBias() + "\t\t" 
						+ rule.weight.ncomponent() + "\t\t" + Math.exp(rule.weight.getWeight(0)) + "\t\t" + Math.exp(rule.weight.getProb()));
				if (++count % ncol == 0) {
					sb.append("\n");
				}
			}
			cnt += rules.size();
			sb.append("\n");
		}
		sb.append("---Lexicon rules. Total: " + cnt + ", average ncomp: " + ((double) ncomp / cnt) + "\n");
		return sb.toString();
	}
	
	protected boolean isKnown(String word) {
		return wordIndexer.indexOf(word) != -1;
	}
	
	public String getCachedSignature(String word, int pos) {
		// TODO determine the frequency of the usage of "if" branch
		if (word == null) { return lastWord; }
		if (word.equals(lastWord) && pos == lastPosition) {
			return lastSignature;
		} else {
			String signature = getSignature(word, pos);
			lastWord = word;
			lastPosition = pos;
			lastSignature = signature;
			return signature;
		}
	}
	
	public String getSignature(String word, int pos) {
		StringBuffer sb = new StringBuffer(TOKEN_UNKNOWN);
		if (word == null || word.length() == 0) {
			return sb.toString();
		}
		switch (unknownLevel) {
		case 5: {
			char ch;
			int ncap = 0;
			int wlen = word.length();
			boolean hasDash = false;
			boolean hasDigit = false;
			boolean hasLower = false;

			for (int i = 0; i < wlen; i++) {
				ch = word.charAt(i);
				if (Character.isDigit(ch)) {
					hasDigit = true;
				} else if (ch == '-') {
					hasDash = true;
				} else if (Character.isLetter(ch)) {
					if (Character.isLowerCase(ch)) {
						hasLower = true;
					} else if (Character.isTitleCase(ch)) {
						hasLower = true;
						ncap++;
					} else {
						ncap++;
					}
				}
			}
			
			ch = word.charAt(0);
			String lowered = word.toLowerCase();
			// See http://stackoverflow.com/questions/31770995/difference-between-uppercase-and-titlecase
			if ((Character.isUpperCase(ch) || Character.isTitleCase(ch))) {
				// TODO do not understand this branch, word has been lowered already
				if (pos == 0 && ncap == 1) {
					sb.append("-INITC");
					if (isKnown(lowered)) {
						sb.append("-KNOWNLC");
					}
				} else {
					sb.append("-CAPS");
				}
			} else if(!Character.isLetter(ch) && ncap > 0) {
				sb.append("-CAPS");
			} else if (hasLower) {
				sb.append("-LC");
			}
			
			if (hasDigit) { sb.append("-NUM"); }
			if (hasDash) { sb.append("-DASH"); }
			
			if (lowered.endsWith("s") && wlen >= 3) {
				ch = lowered.charAt(wlen - 2);
				if (ch != 's' && ch != 'i' && ch != 'u') {
					sb.append("-s");
				}
			} else if (word.length() >= 5 && !hasDash && !(hasDigit && ncap > 0)) {
				if (lowered.endsWith("ed")) {
					sb.append("-ed");
				} else if (lowered.endsWith("ing")) {
					sb.append("-ing");
				} else if (lowered.endsWith("ion")) {
					sb.append("-ion");
				} else if (lowered.endsWith("er")) {
					sb.append("-er");
				} else if (lowered.endsWith("est")) {
					sb.append("-est");
				} else if (lowered.endsWith("ly")) {
					sb.append("-ly");
				} else if (lowered.endsWith("ity")) {
					sb.append("-ity");
				} else if (lowered.endsWith("y")) {
					sb.append("-y");
				} else if (lowered.endsWith("al")) {
					sb.append("-al");	
				}
			}
			break;
		}
		case 4: {
			char ch;
			boolean hasDash  = false;
			boolean hasDigit = false;
			boolean hasLower = false;
			boolean hasComma = false;
			boolean hasLetter = false;
			boolean hasPeriod = false;
			boolean hasNonDigit = false;
			
			for (int i = 0; i < word.length(); i++) {
				ch = word.charAt(i);
				if (Character.isDigit(ch)) {
					hasDigit = true;
				} else {
					hasNonDigit = true;
					if (Character.isLetter(ch)) {
						hasLetter = true;
						if (Character.isLowerCase(ch) || Character.isTitleCase(ch)) {
							hasLower = true;
						}
					} else {
						if (ch == '-') {
							hasDash = true;
						} else if (ch == '.') {
							hasPeriod = true;
						} else if (ch == ',') {
							hasComma = true;
						}
					}
				}
			}
			
			// 6 way on letters
			if (Character.isUpperCase(word.charAt(0)) || 
					Character.isTitleCase(word.charAt(0))) {
				if (!hasLower) {
					sb.append("-AC");
				} else if (pos == 0) {
					sb.append("-SC");
				} else {
					sb.append("-C");
				} 
			} else if (hasLower) {
				sb.append("-L");
			} else if (hasLetter) {
				sb.append("-U");
			} else {
				sb.append("-S");
			}
			
			// 3 way on numbers
			if (hasDigit && !hasNonDigit) {
				sb.append("-N");
			} else if (hasDigit) {
				sb.append("-n");
			}
			
			// binary on period, dash, comma
			if (hasDash) { sb.append("-H"); }
			if (hasComma) { sb.append("-C"); }
			if (hasPeriod) { sb.append("-P"); }
			
			if (word.length() > 3) {
				ch = word.charAt(word.length() - 1);
				if (Character.isLetter(ch)) {
					sb.append("-");
					sb.append(Character.toLowerCase(ch));
				}
			}
			break;
		}
		case 3: {
			sb.append("-");
			
			char ch;
			int num = 0;
			char newClass, lastClass = '-';
			
			for (int i = 0; i < word.length(); i++) {
				ch = word.charAt(i);
				if (Character.isUpperCase(ch) || Character.isTitleCase(ch)) {
					if (pos == 0) {
						newClass = 'S';
					} else {
						newClass = 'L';
					}
				} else if (Character.isLetter(ch)) {
					newClass = 'l';
				} else if (Character.isDigit(ch)) {
					newClass = 'd';
				} else if (ch == '-') {
					newClass = 'h';
				} else if (ch == '.') {
					newClass = 'p';
				} else {
					newClass = 's';
				}
				
				if (newClass != lastClass) {
					lastClass = newClass;
					sb.append(lastClass);
					num = 1;
				} else {
					if (num < 2) {
						sb.append('+');
					}
					num++;
				}
			}
			
			if (word.length() > 3) {
				ch = Character.toLowerCase(word.charAt(word.length() - 1));
				sb.append('-');
				sb.append(ch);
			}
			break;
		}
		case 2: {
			char ch;
			boolean hasDigit = false;
			boolean hasLower = false;
			boolean hasNonDigit = false;
			
			for (int i = 0; i < word.length(); i++) {
				ch = word.charAt(i);
				if (Character.isDigit(ch)) {
					hasDigit = true;
				} else {
					hasNonDigit = true;
					if (Character.isLetter(ch)) {
						if (Character.isLowerCase(ch) || Character.isTitleCase(ch)) {
							hasLower = true;
						}
					}
				}
			}
			
			if (Character.isUpperCase(word.charAt(0))
					|| Character.isTitleCase(word.charAt(0))) {
				if (!hasLower) {
					sb.append("-ALLC");
				} else if (pos == 0) {
					sb.append("-INIT");
				} else {
					sb.append("-UC");
				}
			} else if (hasLower) {
				sb.append("-LC");
			}
			
			if (word.indexOf('-') >= 0) {
				sb.append("-DASH");
			}
			if (hasDigit) {
				if (!hasNonDigit) {
					sb.append("-NUM");
				} else {
					sb.append("-DIG");
				}
			} else if (word.length() > 3) {
				ch = word.charAt(word.length() - 1);
				sb.append(Character.toLowerCase(ch));
			}
			break;
		}
		default: {
			sb.append("-");
			sb.append(word.substring(Math.max(word.length() - 2, 0), word.length()));
			sb.append("-");
			
			if (Character.isLowerCase(word.charAt(0))) {
				sb.append("LOWER");
			} else {
				if (Character.isUpperCase(word.charAt(0))) {
					if (pos == 0) {
						sb.append("INIT");
					} else {
						sb.append("UPPER");
					}
				} else {
					sb.append("OTHER");
				}
			}
		} // end of default
		} // end of switch
		return sb.toString();
	}
}
