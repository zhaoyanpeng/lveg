package edu.shanghaitech.ai.nlp.syntax;

import java.io.Serializable;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.util.Numberer;

/**
 * Represent the nodes, non-terminals or terminals (words), of parse tree. 
 * Each State encapsulates a tag (word), recording the id and the name of 
 * the tag (word).
 * 
 * @author Yanpeng Zhao
 *
 */
public class State implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7306026179545408514L;
	private String name; 
	private short id;
	
	public short from;
	public short to;
	
	/* if the node is a terminal */
	public int wordIdx; // index of the word 
	public int signIdx; // signature index of the word 
	
	protected GaussianMixture insideScore;
	protected GaussianMixture outsideScore;
	
	
	/**
	 * @param name name of the tag, null for non-terminals, and word itself for terminals
	 * @param id   id of the tag
	 * @param from starting point of the span of the current state
	 * @param to   ending point of the span of the current state
	 * 
	 */
	public State(String name, short id, short from, short to) {
		this.name = name;
		this.from = from;
		this.to = to;
		this.id = id;
		this.wordIdx = -1;
		this.signIdx = -1;
	}
	
	
	public State(State state, boolean copyScore) {
		this.name = state.name;
		this.from = state.from;
		this.to = state.to;
		this.id = state.id;
		this.wordIdx = state.wordIdx;
		this.signIdx = state.signIdx;
		
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
	
	
	private void resetScore() {
		if (insideScore != null) { insideScore.clear(); }
		if (outsideScore != null) { outsideScore.clear(); }
		this.insideScore = null;
		this.outsideScore = null;
	}
	
	
	public void clear(boolean deep) {
		if (deep) {
			this.name = null;
			this.from = -1;
			this.to = -1;
			this.id = -1;
			this.wordIdx = -1;
			this.signIdx = -1;
			resetScore();
		} else {
			resetScore();
		}
	}
	
	
	public void clear() {
		clear(true);
	}
	
	
	public String toString(boolean simple, short nfirst, Numberer numberer) {
		if (simple) {
			return toString();
		} else {
			StringBuffer sb = new StringBuffer();
			name = name != null ? name : (String) numberer.object(id);
			sb.append("State [name=" + name + ", id=" + id + ", from=" + from + ", to=" + to + "]");
			if (insideScore != null) {
				sb.append("->[iscore=");
				sb.append(insideScore.toString(!simple, nfirst));
				sb.append("]");
			} else {
				sb.append("->[iscore=null]");
			}
			if (outsideScore != null) {
				sb.append("->[oscore=");
				sb.append(outsideScore.toString(!simple, nfirst));
				sb.append("]");
			} else {
				sb.append("->[oscore=null]");
			}
			return sb.toString();
		}
	}


	public String toString(Numberer numberer) {
		name = name != null ? name : (String) (String) numberer.object(id);
		return "State [name=" + name + ", id=" + id + ", from=" + from + ", to=" + to + "]";
	}
	
}
