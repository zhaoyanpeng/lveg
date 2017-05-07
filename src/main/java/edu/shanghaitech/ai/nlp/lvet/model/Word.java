package edu.shanghaitech.ai.nlp.lvet.model;

import java.io.Serializable;

public class Word implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5252063507849484266L;
	public String word;
	public int wordIdx;
	public int signIdx;
	
	public int from;
	public int to;
	
	public Word(String word) { 
		this.word = word;
	}
	
	public String getName() {
		return word;
	}
	
}
