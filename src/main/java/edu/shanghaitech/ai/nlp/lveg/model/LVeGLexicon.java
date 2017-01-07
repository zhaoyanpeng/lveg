package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.Unit;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Indexer;

/**
 * @author Yanpeng Zhao
 *
 */
public abstract class LVeGLexicon extends LVeGGrammar implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 255983044651602165L;
	public static final String TOKEN_UNKNOWN = "UNK";
	protected transient String lastSignature;
	protected transient String lastWord;
	protected transient int lastPosition;
	/**
	 * Different modes for unknown words. See {@link #SimpleLVeGLexicon()}.
	 */
	protected int unknownLevel;
	
	/**
	 * Assume that rare words have been replaced by their signature.
	 * 
	 * @param trees a set of trees
	 */
	public void postInitialize(StateTreeList trees){}
	
	/**
	 * Get the unary rules that contain the word specified by the index (by reference).
	 * 
	 * @param wordIdx id of the word
	 * @return
	 */
	public abstract List<GrammarRule> getRulesWithWord(State word);
	
	/**
	 * @param state a leaf representing the word
	 * @param idTag the id of the tag
	 */
	public abstract GaussianMixture score(State state, short itag);
	
	/**
	 * @param trees a set of trees to be labeled
	 */
	public abstract void labelTrees(StateTreeList trees);
	
	/**
	 * @param word a leaf node
	 */
	protected abstract boolean isKnown(String word);
	
	/**
	 * Initializing word index, which is also called labeling trees.
	 * 
	 * @param trees a set of trees to be labeled
	 */
	public void labelTrees(Indexer<String> wordIndexer, StateTreeList trees) {
		for (Tree<State> tree : trees) {
			List<State> words = tree.getYield();
			for (State word : words) {
				word.wordIdx = wordIndexer.indexOf(word.getName());
				word.signIdx = -1;
			}
		}
	}
	
	/**
	 * @param mw the mixing weight in logarithm
	 * @return   randomly initialized rule weight
	 */
	protected static GaussianMixture rndWeight(double mw) {
		GaussianMixture weight = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Set<GaussianDistribution> set = new HashSet<GaussianDistribution>();
			set.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			weight.add(i, Unit.P, set);
		}
		weight.setWeights(mw);
		return weight;
	}
	
	/**
	 * Sort of different from the Berkeley's implementation. It returns the 
	 * last word (initially initialized to "UNK") if the first word is null 
	 * , and always returns "UNK" if the word is of length 0. In Berkeley's 
	 * implementation, the first null word produces empty string, which can 
	 * be divided into "UNK" since the empty string is of length 0.
	 * 
	 * @param word the word
	 * @param pos  the position of the word in the sentence
	 * @return
	 */
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
	
	/**
	 * Get the signature of the word.
	 * 
	 * @param word a terminal
	 * @param pos  position of the terminal in a sentence
	 * @return
	 */
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
