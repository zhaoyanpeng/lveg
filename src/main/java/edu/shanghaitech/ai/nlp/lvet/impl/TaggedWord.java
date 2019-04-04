package edu.shanghaitech.ai.nlp.lvet.impl;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lvet.model.Word;

public class TaggedWord extends Word {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4502063096855924969L;
	protected String tag;
	protected int tagIdx;
	
	protected GaussianMixture insideScore;
	protected GaussianMixture outsideScore;
	protected GaussianMixture insWithWord;
	protected GaussianMixture outWithWord;
	
	public TaggedWord(String tag, String word) {
		super(word);
		this.tag = tag;
	}

	public String tag() {
		return tag;
	}

	public void setTag(String tag) {
		this.tag = tag;
	}

	public int getTagIdx() {
		return tagIdx;
	}

	public void setTagIdx(int tagIdx) {
		this.tagIdx = tagIdx;
	}
	
	public GaussianMixture getInsideScore() {
		return insideScore;
	}

	public void setInsideScore(GaussianMixture insideScore) {
		this.insideScore = insideScore;
	}

	public GaussianMixture getOutsideScore() {
		return outsideScore;
	}

	public void setOutsideScore(GaussianMixture outsideScore) {
		this.outsideScore = outsideScore;
	}
	
	public GaussianMixture getInsideScore(boolean withWord) {
		return withWord ? insWithWord : insideScore;
	}
	
	public void setInsideScore(GaussianMixture insideScore, boolean withWord) {
		if (withWord) {
			this.insWithWord = insideScore;
		} else {
			this.insideScore = insideScore;
		}
	}
	
	public GaussianMixture getOutsideScore(boolean withWord) {
		return withWord ? outWithWord : outsideScore;
	}
	
	public void setOutsideScore(GaussianMixture outsideScore, boolean withWord) {
		if (withWord) {
			this.outWithWord = outsideScore;
		} else {
			this.outsideScore = outsideScore;
		}
	}
	
	private void resetScore() {
		if (insideScore != null) { 
			insideScore.clear(true); 
		}
		if (outsideScore != null) { 
			outsideScore.clear(true); 
		}
		if (insWithWord != null) {
			insWithWord.clear(true);
		}
		if (outWithWord != null) {
			outWithWord.clear(true);
		}
		this.insideScore = null;
		this.outsideScore = null;
	}
	
	public void clear(boolean deep) {
		if (deep) {
			this.tag = null;
			this.tagIdx = -1;
			this.word = null; // would be better if reset in `Word`
		} 
		resetScore();
	}
	
	public void clear() {
		clear(true);
	}

	@Override
	public String toString() {
		return "TW [tag=" + tag + ", tidx=" + tagIdx + ", word=" + word + ", widx=" + wordIdx + "]";
	}
	
}
