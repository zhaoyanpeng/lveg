package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * @author Yanpeng Zhao
 *
 */
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
			
			parent.setInsideScore(inScore.copy(true));
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
					inScore = ruleScore.multiplyForInsideScore(childInScore, GrammarRule.Unit.UC, true);
				} else { // root
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
					inScore = ruleScore.multiplyForInsideScore(childInScore, GrammarRule.Unit.C, true);
				}
				
				parent.setInsideScore(inScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, inScore, lchildInScore, rchildInScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				lchildInScore = lchild.getInsideScore();
				rchildInScore = rchild.getInsideScore();
				
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				inScore = ruleScore.multiplyForInsideScore(lchildInScore, GrammarRule.Unit.LC, true);
				inScore = inScore.multiplyForInsideScore(rchildInScore, GrammarRule.Unit.RC, false);
				
				parent.setInsideScore(inScore);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);
					
			}
		}
	}
	
	
}
