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

	@Override
	public String toString() {
		return "TW [tag=" + tag + ", tidx=" + tagIdx + ", word=" + word + ", widx=" + wordIdx + "]";
	}
	
}
