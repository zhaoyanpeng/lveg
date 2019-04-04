package edu.shanghaitech.ai.nlp.lvet.impl;

import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lvet.model.Pair;
import edu.shanghaitech.ai.nlp.util.Numberer;

public class TagTPair extends Pair {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5243987757351187035L;
	
	
	public TagTPair(Numberer numberer, int ntag) {
		super();
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
		this.edgesWithC = new List[ntag];
		for (int i = 0; i < ntag; i++) {
			edgesWithP[i] = new ArrayList<GrammarRule>();
			edgesWithC[i] = new ArrayList<GrammarRule>();
		}
	}
	
	@Override
	public void postInitialize() {
		for (GrammarRule edge : edgeTable.keySet()) {
			edge.getWeight().setBias(edgeTable.getCount(edge).getBias());
			addEdge((UnaryGrammarRule) edge);
		}
	}

	@Override
	public void tallyTaggedWords(List<TaggedWord> words) {
		int pretag = LEADING_IDX;
		for (int i = 0; i < words.size(); i++) {
			TaggedWord word = words.get(i);
			RuleType type = i == 0 ? RuleType.RHSPACE : RuleType.LRURULE;
			GrammarRule edge = new UnaryGrammarRule((short) pretag, word.tagIdx, type);
			if (!edgeTable.containsKey(edge)) {
				edge.initializeWeight(type, (short) -1, (short) -1);
			}
			edgeTable.addCount(edge, 1.0);
			pretag = word.tagIdx;
		}
		GrammarRule edge = new UnaryGrammarRule((short) pretag, ENDING_IDX, RuleType.LHSPACE);
		if (!edgeTable.containsKey(edge)) {
			edge.initializeWeight(RuleType.LHSPACE, (short) -1, (short) -1);
		}
		edgeTable.addCount(edge, 1.0);
	}
	
	@Override
	public String toString() {
		int count = 0, ncol = 1, ncomp = 0;
		StringBuffer sb = new StringBuffer();
		sb.append("Grammar [nTag=" + ntag + "]\n");
		for (int i = 0; i < numberer.size(); i++) {
			sb.append("Tag " + i + "\t" +  (String) numberer.object(i) + "\n");
		}
		
		int nurule = edgeTable.size();
		sb.append("---Unary Grammar Rules. Total: " + nurule + "\n");
		for (GrammarRule rule : edgeTable.keySet()) {
			ncomp += rule.weight.ncomponent();
			sb.append(rule + "\t\t" + edgeTable.getCount(rule).getBias() + "\t\t" 
					+ rule.weight.ncomponent() + "\t\t" + Math.exp(rule.weight.getWeight(0)) + "\t\t" + Math.exp(rule.weight.getProb()));
			if (++count % ncol == 0) {
				sb.append("\n");
			}
		}
		sb.append("---Unary Grammar Rules. Total: " + nurule + ", average ncomp: " + ((double) ncomp / nurule) + "\n");
		
		sb.append("\n");
		return sb.toString();
	}

}
