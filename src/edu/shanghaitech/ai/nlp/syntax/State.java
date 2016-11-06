package edu.shanghaitech.ai.nlp.syntax;

import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;

/**
 * Represent the nodes, non-terminals or terminals (words), of parse tree. 
 * Each State encapsulates a tag (word), recording the id and the name of 
 * the tag (word).
 * 
 * @author Yanpeng Zhao
 *
 */
public class State {
	
	private String name; 
	private short id;
	
	public short from;
	public short to;
	
	/* when the node is a terminal */
	public int wordIdx; // index of the word 
	public int signIdx; // signature index of the word 
	
	protected GaussianMixture insideScore;
	protected GaussianMixture outsideScore;
	
	
	/**
	 * @param id   id of the tag
	 * @param name name of the tag, null for non-terminals, and word itself for terminals
	 * @param from starting point of the span of the current state
	 * @param to   ending point of the span of the current state
	 * 
	 */
	public State(String name, short id, short from, short to) {
		this.name = name;
		this.from = from;
		this.to = to;
		this.id = id;
	}
	
	
	public State(State state, boolean copyScore) {
		this.name = state.name;
		this.from = state.from;
		this.to = state.to;
		this.id = state.id;
		
		if (copyScore) {
			this.insideScore = state.insideScore;
			this.outsideScore = state.outsideScore;
		}
	}
	
	
	public State copy() {
		return new State(this, false);
	}
	
	
	public State copy(boolean copyScore) {
		return new State(this, copyScore);
	}
	

	public String getName() {
		return name;
	}


	public void setName(String name) {
		this.name = name;
	}


	public short getId() {
		return id;
	}


	public void setId(short id) {
		this.id = id;
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
		name = name != null ? name
				: (String) Numberer.getGlobalNumberer(LVeGLearner.KEY_TAG_SET).object(id);
		return "State [name=" + name + ", id=" + id + ", from=" + from + ", to=" + to + "]";
	}
	
}
