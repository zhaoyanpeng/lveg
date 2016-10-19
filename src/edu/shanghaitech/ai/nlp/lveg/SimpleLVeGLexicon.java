package edu.shanghaitech.ai.nlp.lveg;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.shanghaitech.ai.nlp.syntax.State;

public class SimpleLVeGLexicon implements Serializable, LVeGLexicon {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public Indexer<String> wordIndexer;
	
	protected transient String lastWord;
	protected transient String lastSignature;
	protected transient int lastPosition;
	
	protected int nTag;
	protected int nWord;
	protected List<Integer> wordCounter;
	
	/**
	 * TODO do we really need it? 
	 * @deprecated
	 */
	protected IntegerIndexer[] tagWordIndexer;
	
	protected GaussianMixture[][] weights;        // tag-word
	protected GaussianMixture[][] expectedCounts; // tag-word
	
	/**
	 * Different modes for unknown words. See {@link #SimpleLVeGLexicon()}.
	 */
	private int unknownLevel;
	
	
	/**
	 * Rules with probabilities below this value will be filtered.
	 */
	private double filterThreshold;
	
	
	public SimpleLVeGLexicon() {
		this.wordIndexer = new Indexer<String>();
		this.lastWord = "";
		this.lastPosition = -1;
		this.lastSignature = "";
		this.unknownLevel = 5; // 5 is English specific
		
		if ((Corpus.myTreebank != Corpus.TreeBankType.WSJ) ||
				Corpus.myTreebank == Corpus.TreeBankType.BROWN) {
			this.unknownLevel = 4;
		}
	}
	
	
	public SimpleLVeGLexicon(StateTreeList trees, int nTag, double filterThreshold) {
		this.wordIndexer = new Indexer<String>();
		this.filterThreshold = filterThreshold;
		this.unknownLevel = 5; // 5 is English specific
		
		this.nTag = nTag;
		this.tagWordIndexer = new IntegerIndexer[nTag];
		
		if ((Corpus.myTreebank != Corpus.TreeBankType.WSJ) ||
				Corpus.myTreebank == Corpus.TreeBankType.BROWN) {
			this.unknownLevel = 4;
		}
		
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
		this.wordCounter = new ArrayList<Integer>(nWord);
		
		for (int i = 0; i < nTag; i++) {
			tagWordIndexer[i] = new IntegerIndexer(wordIndexer.size());
		}
		
		for (Tree<State> tree : trees) {
			List<State> tags = tree.getPreTerminalYield();
			List<State> words = tree.getYield();
			
			int pos = 0;
			for (State word : words) {
				String name = word.getName();
				int wordIdx = wordIndexer.indexOf(name);
				wordCounter[wordIdx]++;
				tagWordIndexer[tags.get(pos).getId()].add(wordIdx);
				pos++;
			}
		}
		
		this.weights = new GaussianMixture[nTag][];
		this.expectedCounts = new GaussianMixture[nTag][];
		for (int i = 0; i < nTag; i++) {
			weights[i] = new GaussianMixture[nWord];
			expectedCounts[i] = new GaussianMixture[nWord];
			for (int j = 0; j < nWord; j++) {
				weights[i][j] = new GaussianMixture();
				expectedCounts[i][j] = new GaussianMixture();
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
		// TODO Auto-generated method stub
		List<State> words = tree.getYield();
		List<State> tags = tree.getPreTerminalYield();
		
		for (int pos = 0; pos < words.size(); pos++) {
			short tag = tags.get(pos).getId();
			String word = words.get(pos).getName();
			
			int wordIdx = wordIndexer.indexOf(word);
			int tagWordIdx = tagWordIndexer[tag].indexOf(wordIdx);
			
			// expected counts have been initialized when instantiated
			// TODO
		}
	}


	@Override
	public GaussianMixture score(State word, short idTag) {
		// TODO Auto-generated method stub
		int wordIdx = word.wordIdx;
		if (wordIdx == -1) {
			System.err.println("Unknown word: " + word.getName());
			return null;
		}
		int tagWordIdx = tagWordIndexer[idTag].indexOf(wordIdx);
		return weights[idTag][wordIdx];
	}


	public String getCachedSignature(String word, int pos) {
		// TODO count the use frequency of the if branch
		if (word == null || (word.equals(lastWord) && pos == lastPosition)) {
			return lastSignature;
		} else {
			String signature = getSignature(word, pos);
			lastWord = word;
			lastPosition = pos;
			lastSignature = signature;
			return signature;
		}
	}
	
	
	@Override
	public String getSignature(String word, int pos) {
		StringBuffer sb = new StringBuffer("UNK");
		
		if (word == null || word.length() == 0) {
			return sb.toString();
		}
		
		switch (unknownLevel) {
		case 5: {
			char ch;
			int numCaps = 0;
			int wordLength = word.length();
			
			boolean hasDash = false;
			boolean hasDigit = false;
			boolean hasLower = false;
			
			for (int i = 0; i < wordLength; i++) {
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
						numCaps++;
					} else {
						numCaps++;
					}
				}
			}
			
			
			ch = word.charAt(0);
			String lowered = word.toLowerCase();
			// See http://stackoverflow.com/questions/31770995/difference-between-uppercase-and-titlecase
			if ((Character.isUpperCase(ch) || Character.isTitleCase(ch))) {
				// TODO do not understand this branch, word has been lowered already
				if (pos == 0 && numCaps == 1) {
					sb.append("-INITC");
					if (isKnown(lowered)) {
						sb.append("-KNOWNLC");
					}
				} else {
					sb.append("-CAPS");
				}
			} else if(!Character.isLetter(ch) && numCaps > 0) {
				sb.append("-CAPS");
			} else if (hasLower) {
				sb.append("-LC");
			}
			
			if (hasDigit) { sb.append("-NUM"); }
			if (hasDash) { sb.append("-DASH"); }
			
			if (lowered.endsWith("s") && wordLength >= 3) {
				ch = lowered.charAt(wordLength - 2);
				if (ch != 's' && ch != 'i' && ch != 'u') {
					sb.append("-s");
				}
			} else if (word.length() >= 5 && !hasDash && !(hasDigit && numCaps > 0)) {
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
			boolean hasDash = false;
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
	
	
	private boolean isKnown(String word) {
		return wordIndexer.indexOf(word) != -1;
	}
	
	
	public static class IntegerIndexer implements Serializable {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		
		private int n;
		private int[] indexTo;
		private int[] indexFrom;

		IntegerIndexer(int capacity) {
			indexTo = new int[capacity];
			indexFrom = new int[capacity];
			Arrays.fill(indexTo, -1);
			Arrays.fill(indexFrom, -1);
			n = 0;
		}

		public void add(int i) {
			if ( i < 0 || i > indexFrom.length) { return; }
			if (indexTo[i] == -1) {
				indexTo[i] = n;
				indexFrom[n] = i;
				n++;
			}
		}

		public int get(int i) {
			if (i < indexFrom.length) {
				return indexFrom[i];
			} else {
				return -1;
			}
		}

		public int indexOf(int i) {
			if (i < indexTo.length) {
				return indexTo[i];
			} else {
				return -1;
			}
		}

		public int size() {
			return n;
		}

		public IntegerIndexer copy() {
			IntegerIndexer copy = new IntegerIndexer(indexFrom.length);
			copy.n = n;
			copy.indexFrom = this.indexFrom.clone();
			copy.indexTo = this.indexTo.clone();
			return copy;
		}
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







