package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;

public class LVeGParser {
	
	private LVeGLexicon lexicon;
	private LVeGGrammar grammar;
	
	public LVeGParser(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.grammar = grammar;
		this.lexicon = lexicon;
	}
	
	
	public void calculateInsideScore(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			calculateInsideScore(child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture inScore = lexicon.score(word, idParent);
			parent.setInsideScore(inScore);
		} else {
			switch (children.size()) {
			case 0:
				break;
			case 1: {
				GaussianMixture ruleScore, inScore, childInScore;
				State child = children.get(0).getLabel();
				childInScore = child.getInsideScore();
				short idChild = child.getId();
				
				if (idParent != 0) {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.GENERAL);
				} else {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
				}
				
//				inScore = 
				
				parent.setInsideScore(ruleScore);
				break;
			}
			case 2: {
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);
					
			}
		}
	}
}
