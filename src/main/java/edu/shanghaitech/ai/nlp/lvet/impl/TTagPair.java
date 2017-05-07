package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;

public class TTagPair extends Pair {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5243987757351187035L;

	@Override
	public void postInitialize() {
		for (GrammarRule edge : edgeTable.keySet()) {
			edge.getWeight().setBias(edgeTable.getCount(edge).getBias());
			addEdge((DirectedEdge) edge);
		}
	}

	@Override
	public void tallyTaggedSample(List<TaggedWord> tree) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected void initialize() {
		this.edgesWithP = new List[ntag];
		this.edgesWithC = new List[ntag];
		for (int i = 0; i < ntag; i++) {
			edgesWithP[i] = new ArrayList<GrammarRule>();
			edgesWithC[i] = new ArrayList<GrammarRule>();
		}
	}


}
