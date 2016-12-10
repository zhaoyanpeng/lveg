package edu.shanghaitech.ai.nlp.lveg;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * Compute the inside and outside scores and store them in a chart. 
 * Imagine the upper right triangle chart as the pyramid. E.g., see
 * the pyramid below, which could represent a sentence of length 6.
 * <pre>
 * + + + + + +
 *   + + + + +
 *     + + + +
 *       + + +
 *         + +
 *           + 
 * </pre>
 * @author Yanpeng Zhao
 *
 */
public class LVeGInferencer extends Inferencer {
	
	protected LVeGLexicon lexicon;
	protected LVeGGrammar grammar;
	
	protected ChainUrule chainurule;
	
	
	public LVeGInferencer(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.grammar = grammar;
		this.lexicon = lexicon;
		this.chainurule = ChainUrule.DEFAULT;
	}
	
	
	/**
	 * Compute the inside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used.
	 * @param nword # of words in the sentence
	 */
	protected void insideScore(Chart chart, List<State> sentence, int nword) {
		int x0, y0, x1, y1, c0, c1, c2;
		Queue<Short> tagQueue = new LinkedList<Short>(); 
		GaussianMixture pinScore, linScore, rinScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int i = 0; i < nword; i++) {
			int iCell = Chart.idx(i, nword);
			int wordIdx = sentence.get(i).wordIdx; 
			List<GrammarRule> rules = lexicon.getRulesWithWord(wordIdx);
			// pre-terminals
			for (GrammarRule rule : rules) {
				tagQueue.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true), (short) 0);
			}
			// unary grammar rules
			// unary grammar rules
			logger.trace("Cell [" + i + ", " + (i + 0) + "]="+ iCell + "\t is being estimated. # " + tagQueue.size());
			long start = System.currentTimeMillis();
			insideScoreForUnaryRule(chart, tagQueue, iCell, chainurule);
			long ttime = System.currentTimeMillis() - start;
			logger.trace("\tafter chain unary\t" + chart.size(iCell, true) + "\ttime: " + ttime / 1000 + "\n"); // DEBUG
		}		
		
		// inside score
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				// to ensure...
				tagQueue.clear();
				
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				for (int right = left; right < left + ilayer; right++) {
					y0 = right;
					x1 = right + 1;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lchild, c0, true) && chart.containsKey(rule.rchild, c1, true)) {
							if (!tagQueue.contains(rule.lhs)) {tagQueue.offer(rule.lhs);}
							ruleScore = rule.getWeight();
							linScore = chart.getInsideScore(rule.lchild, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
							pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addInsideScore(rule.lhs, c2, pinScore, (short) 0);
						}
					}
				}
				// unary grammar rules
				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # " + tagQueue.size());
				long start = System.currentTimeMillis();
				insideScoreForUnaryRule(chart, tagQueue, c2, chainurule);
				long ttime = System.currentTimeMillis() - start;
				logger.trace("\tafter chain unary\t" + chart.size(c2, true) + "\ttime: " + ttime / 1000 + "\n"); // DEBUG
			}
		}
	}
	
	
	/**
	 * Compute the outside score given the sentence and grammar rules.
	 * 
	 * @param chart [in/out]-side score container
	 * @param tree  in which only the sentence is used.
	 * @param nword # of words in the sentence
	 */
	protected void outsideScore(Chart chart, List<State> sentence, int nword) {
		
		int x0, y0, x1, y1, c0, c1, c2;
		Queue<Short> tagQueue = new LinkedList<Short>(); 
		GaussianMixture poutScore, linScore, rinScore, loutScore, routScore, ruleScore;
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		tagQueue.offer((short) 0);
		outsideScoreForUnaryRule(chart, tagQueue, Chart.idx(0, 1), chainurule);
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				// to ensure...
				tagQueue.clear();
				
				x0 = left;
				x1 = left + ilayer + 1; 
				c2 = Chart.idx(left, nword - ilayer);
				for (int right = left + ilayer + 1; right < nword; right++) {
					y0 = right;
					y1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lhs, c0, false) && chart.containsKey(rule.rchild, c1, true)) {
							if (!tagQueue.contains(rule.lchild)) { tagQueue.offer(rule.lchild); }
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
							chart.addOutsideScore(rule.lchild, c2, loutScore, (short) 0);
						}
					}
				}
				
				y0 = left + ilayer;
				y1 = left - 1;
				// c2 = Chart.idx(left, nword - ilayer);
				for (int right = 0; right < left; right++) {
					x0 = right; 
					x1 = right;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lhs, c0, false) && chart.containsKey(rule.lchild, c1, true)) {
							if (!tagQueue.contains(rule.rchild)) { tagQueue.offer(rule.rchild); }
							ruleScore = rule.getWeight();
							poutScore = chart.getOutsideScore(rule.lhs, c0);
							linScore = chart.getInsideScore(rule.lchild, c1);
							
							routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
							routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
							chart.addOutsideScore(rule.rchild, c2, routScore, (short) 0);
						}
					}
				}
				// unary grammar rules
				logger.trace("Cell [" + left + ", " + (left + ilayer) + "]="+ c2 + "\t is being estimated. # " + tagQueue.size());
				long start = System.currentTimeMillis();
				outsideScoreForUnaryRule(chart, tagQueue, c2, chainurule);
				long ttime = System.currentTimeMillis() - start;
				logger.trace("\tafter chain unary\t" + chart.size(c2, false) + "\ttime: " + ttime / 1000 + "\n"); // DEBUG
			}
		}
	}
	
	
	/**
	 * Compute the inside score with the parse tree known.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	protected void insideScoreWithTree(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			insideScoreWithTree(child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture cinScore = lexicon.score(word, idParent);
			parent.setInsideScore(cinScore.copy(true));
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				GaussianMixture ruleScore, cinScore, pinScore;
				State child = children.get(0).getLabel();
				cinScore = child.getInsideScore();
				short idChild = child.getId();
				
				if (idParent != 0) {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.GENERAL);
					pinScore = ruleScore.mulForInsideOutside(cinScore, GrammarRule.Unit.UC, true);
				} else { // root, inside score of the root node is a constant in double
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
					pinScore = ruleScore.mulForInsideOutside(cinScore, GrammarRule.Unit.C, true);
				}
				parent.setInsideScore(pinScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, pinScore, linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
				pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
				pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
				parent.setInsideScore(pinScore);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	/**
	 * Compute the outside score with the parse tree known.
	 * 
	 * @param tree the parse tree
	 */
	protected void outsideScoreWithTree(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		List<Tree<State>> children = tree.getChildren();
		State parent = tree.getLabel();
		short idParent = parent.getId();
		
		if (tree.isPreTerminal()) {
			// nothing to do
		} else {
			GaussianMixture poutScore = parent.getOutsideScore();
			
			switch (children.size()) {
			case 0:
				break;
			case 1: {
				GaussianMixture ruleScore, coutScore;
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				if (idParent != 0) {
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.GENERAL);
				} else { // root
					ruleScore = grammar.getUnaryRuleScore(idParent, idChild, GrammarRule.RHSPACE);
				}
				// rule: p(root->nonterminal) does not contain "P" part, so no removing occurs when
				// the current parent node is the root node
				coutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				child.setOutsideScore(coutScore);
				break;
			}
			case 2: {
				GaussianMixture ruleScore, loutScore, routScore;
				GaussianMixture linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				ruleScore = grammar.getBinaryRuleScore(idParent, idlChild, idrChild);
				
				loutScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				loutScore = loutScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
				
				routScore = ruleScore.mulForInsideOutside(poutScore, GrammarRule.Unit.P, true);
				routScore = routScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, false);
				lchild.setOutsideScore(loutScore);
				rchild.setOutsideScore(routScore);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);
			}
			
			for (Tree<State> child : children) {
				outsideScoreWithTree(child);
			}
		}
	}
	
	
	protected void evalRuleCount(Tree<State> tree, Chart chart, short isample) {
		List<State> sentence = tree.getYield();
		int x0, x1, y0, y1, c0, c1, c2, nword = sentence.size();
		GaussianMixture outScore, cinScore, linScore, rinScore;
		
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getUnaryRuleMap();
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				x0 = left;
				y1 = left + ilayer;
				c2 = Chart.idx(left, nword - ilayer);
				
				// unary grammar rules
				for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) rmap.getValue();
					if (chart.containsKey(rule.lhs, c2, false) && chart.containsKey(rule.rhs, c2, true)) {
						cinScore = chart.getInsideScore(rule.rhs, c2);
						outScore = chart.getOutsideScore(rule.lhs, c2);
						
						String key = rule.lhs == 0 ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
						byte ruleType = rule.lhs == 0 ? GrammarRule.RHSPACE : GrammarRule.GENERAL;
						Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
						scores.put(GrammarRule.Unit.P, outScore);
						scores.put(key, cinScore);
						grammar.addCount(rule.lhs, rule.rhs, ruleType, scores, isample, false);
					}
				}
				
				// lexicons
				if (x0 == y1) {
					evalUnaryRuleCount(chart, c2, nword, isample, sentence);
					continue; // not necessary
				}
				
				for (int split = left; split < left + ilayer; split++) {
					y0 = split;
					x1 = split + 1;
					c0 = Chart.idx(x0, nword - (y0 - x0));
					c1 = Chart.idx(x1, nword - (y1 - x1));
					// binary grammar rules
					for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
						BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
						if (chart.containsKey(rule.lhs, c2, false) && 
							chart.containsKey(rule.lchild, c0, true) && 
							chart.containsKey(rule.rchild, c1, true)) {
							outScore = chart.getOutsideScore(rule.lhs, c2);
							linScore = chart.getInsideScore(rule.lchild, c0);
							rinScore = chart.getInsideScore(rule.rchild, c1);
							
							Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
							scores.put(GrammarRule.Unit.P, outScore);
							scores.put(GrammarRule.Unit.LC, linScore);
							scores.put(GrammarRule.Unit.RC, rinScore);
							grammar.addCount(rule.lhs, rule.lchild, rule.rchild, scores, isample, false);
						}
					}
				}
			}
		}
	}
	
	
	/**
	 * Eval the rule counts with the parse tree known.
	 * 
	 * @param tree      the parse tree
	 * @param treeScore the score of the tree
	 */
	protected void evalRuleCountWithTree(Tree<State> tree, short isample) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			evalRuleCountWithTree(child, isample);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		GaussianMixture outScore = parent.getOutsideScore();
		Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
		scores.put(GrammarRule.Unit.P, outScore);
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture cinScore = parent.getInsideScore();
			scores.put(GrammarRule.Unit.C, cinScore);
			lexicon.addCount(idParent, (short) word.wordIdx, GrammarRule.LHSPACE, scores, isample, true);
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				GaussianMixture cinScore = child.getInsideScore();
				// root, if (idParent == 0) is true
				String key = idParent == 0 ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
				byte type = idParent == 0 ? GrammarRule.RHSPACE : GrammarRule.GENERAL;
				scores.put(key, cinScore);
				grammar.addCount(idParent, idChild, type, scores, isample, true);
				break;
			}
			case 2: {
				GaussianMixture linScore, rinScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				linScore = lchild.getInsideScore();
				rinScore = rchild.getInsideScore();
				scores.put(GrammarRule.Unit.LC, linScore);
				scores.put(GrammarRule.Unit.RC, rinScore);
				grammar.addCount(idParent, idlChild, idrChild, scores, isample, true);
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	/**
	 * Compute the inside score, the recursive version.
	 * 
	 * @param chart    which stores the parse info
	 * @param sentence the sentence
	 * @param begin    left bound of the spanning range
	 * @param end      right bound of the spanning range
	 */
	protected void insideScore(Chart chart, List<State> sentence, int begin, int end) {
		int nword = sentence.size();
		
		if (begin == end) {
			int index = sentence.get(begin).wordIdx;
			int iCell = Chart.idx(begin, nword);
			Queue<Short> tagQueue = new LinkedList<Short>(); 
			List<GrammarRule> rules = lexicon.getRulesWithWord(index);
			for (GrammarRule rule : rules) {
				tagQueue.offer(rule.lhs);
				chart.addInsideScore(rule.lhs, iCell, rule.getWeight().copy(true), (short) 0);
			}
			// unary grammar rules
			insideScoreForUnaryRule(chart, tagQueue, iCell, chainurule);
		}
		
		int c2 = Chart.idx(begin, nword- (end - begin));
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBinaryRuleMap();
		
		for (int split = begin; split < end; split++) {
			
			int x0 = begin, y0 = split;
			int x1 = split + 1, y1 = end;
			int c0 = Chart.idx(x0, nword - (y0 - x0));
			int c1 = Chart.idx(x1, nword - (y1 - x1));
			
			if (!chart.getStatus(c0, true)) {
				insideScore(chart, sentence, begin, split);
			}
			
			if (!chart.getStatus(c1, true)) {
				insideScore(chart, sentence, split + 1, end);
			}
			
			GaussianMixture ruleScore, linScore, rinScore, pinScore;
			for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
				BinaryGrammarRule rule = (BinaryGrammarRule) rmap.getValue();
				if (chart.containsKey(rule.lchild, c0, true) && chart.containsKey(rule.rchild, c1, true)) {
					ruleScore = rule.getWeight();
					linScore = chart.getInsideScore(rule.lchild, c0);
					rinScore = chart.getInsideScore(rule.rchild, c1);
					pinScore = ruleScore.mulForInsideOutside(linScore, GrammarRule.Unit.LC, true);
					pinScore = pinScore.mulForInsideOutside(rinScore, GrammarRule.Unit.RC, false);
					chart.addInsideScore(rule.lhs, c2, pinScore, (short) 0);
				}
			}
		}
		chart.setStatus(c2, true, true);
	}
	
	
	/**
	 * Compute the outside score, the recursive version.
	 * 
	 * @param chart    which stores the parse info
	 * @param sentence the sentence
	 * @param begin    left bound of the spanning range
	 * @param end      right bound of the spanning range
	 */
	protected void outsideScore(Chart chart, List<State> sentence, int begin, int end) {
		
	}
	
	
	private void evalUnaryRuleCount(Chart chart, int idx, int nword, short isample, List<State> sentence) {  		
		for (int i = 0; i < nword; i++) {
			int wordIdx = sentence.get(i).wordIdx;
			int iCell = Chart.idx(i, nword);
			List<GrammarRule> rules = lexicon.getRulesWithWord(wordIdx);
			// pre-terminals
			for (GrammarRule rule : rules) {
				GaussianMixture cinScore = chart.getInsideScore(rule.lhs, iCell);
				GaussianMixture outScore = chart.getOutsideScore(rule.lhs, iCell);
				Map<String, GaussianMixture> scores = new HashMap<String, GaussianMixture>();
				scores.put(String.valueOf(isample), null);
				scores.put(GrammarRule.Unit.P, outScore);
				scores.put(GrammarRule.Unit.C, cinScore);
				lexicon.addCount(rule.lhs, (short) wordIdx, GrammarRule.LHSPACE, scores, isample, false); 
			}
		}
	}
	
	
	private void outsideScoreForUnaryRule(Chart chart, Queue<Short> tagQueue, int idx, ChainUrule identifier) {
		if (tagQueue != null) {
			switch (identifier) {
			case ALL_POSSIBLE_PATH: {
				outsideScoreForUnaryRulePP(chart, tagQueue, idx);
				break;
			}
			case PRE_COMPUTE_CHAIN: {
				outsideScoreForUnaryRuleCC(chart, tagQueue, idx);
				break;
			}
			case NOT_PRE_ADD_INTER: {
				outsideScoreForUnaryRuleAI(chart, tagQueue, idx);
				break;
			}
			case NOT_PRE_NOT_INTER: {
				outsideScoreForUnaryRuleNI(chart, tagQueue, idx);
				break;
			}
			case DEFAULT: {
				outsideScoreForUnaryRuleDefault(chart, tagQueue, idx);
				break;
			}
			default:
				logger.error("Invalid unary-rule-processing-method. ");
			}
		}
	}
	
	
	private void insideScoreForUnaryRule(Chart chart, Queue<Short> tagQueue, int idx, ChainUrule identifier) {
		if (tagQueue != null) {
			switch (identifier) {
			case ALL_POSSIBLE_PATH: {
				insideScoreForUnaryRulePP(chart, tagQueue, idx);
				break;
			}
			case PRE_COMPUTE_CHAIN: {
				insideScoreForUnaryRuleCC(chart, tagQueue, idx);
				break;
			}
			case NOT_PRE_ADD_INTER: {
				insideScoreForUnaryRuleAI(chart, tagQueue, idx);
				break;
			}
			case NOT_PRE_NOT_INTER: {
				insideScoreForUnaryRuleNI(chart, tagQueue, idx);
				break;
			}
			case DEFAULT: {
				insideScoreForUnaryRuleDefault(chart, tagQueue, idx);
				break;
			}
			default:
				logger.error("Invalid unary-rule-processing-method. ");
			}
		}
	}
	
	private void outsideScoreForUnaryRuleDefault(Chart chart, Queue<Short> tagQueue, int idx) {
		
	}
	
	private void insideScoreForUnaryRuleDefault(Chart chart, Queue<Short> tagQueue, int idx) {
	}
	
	private void outsideScoreForUnaryRuleNI(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		List<GrammarRule> rules;
		String rmKey = GrammarRule.Unit.P;
		GaussianMixture poutScore, coutScore, ruleScore;
		HashMap<Short, GaussianMixture> oset = new HashMap<Short, GaussianMixture>();
		while (!tagQueue.isEmpty()) {
			oset.clear();
			// chain unary rule of length 1
			idTag = tagQueue.poll();
			rules = grammar.getUnaryRuleWithP(idTag);
			poutScore = chart.getOutsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				ruleScore = rule.getWeight();
				// ROOT->X is valid if and only if idx == 0, which can be guaranteed naturally since rules X->ROOT do not exist
				coutScore = ruleScore.mulForInsideOutside(poutScore, rmKey, true);
				oset.put(rule.rhs, coutScore);
			}
			// chain unary rule of length 2
			for (Map.Entry<Short, GaussianMixture> one : oset.entrySet()) {
				rules = grammar.getUnaryRuleWithP(one.getKey());
				poutScore = one.getValue();
				iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					ruleScore = rule.getWeight();
					coutScore = ruleScore.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore(rule.rhs, idx, coutScore);
				}	
			}
		}
	}
	
	private void insideScoreForUnaryRuleNI(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		String rmKey;
		List<GrammarRule> rules;
		GaussianMixture pinScore, cinScore, ruleScore;
		HashMap<Short, GaussianMixture> oset = new HashMap<Short, GaussianMixture>();
		while (!tagQueue.isEmpty()) {
			oset.clear();
			// chain unary rule of length 1
			idTag = tagQueue.poll();
			rules = grammar.getUnaryRuleWithC(idTag);
			cinScore = chart.getInsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				ruleScore = rule.getWeight();
				// Root->X is valid if idx == 0, we may skip such kind of rules otherwise
				if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; }
				// double check, in case the rule.lhs is the root node
				rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
				pinScore = ruleScore.mulForInsideOutside(cinScore, rmKey, true);
				oset.put(rule.lhs, pinScore);
			}
			// chain unary rule of length 2
			for (Map.Entry<Short, GaussianMixture> one : oset.entrySet()) {
				rules = grammar.getUnaryRuleWithC(one.getKey());
				cinScore = one.getValue();
				iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					ruleScore = rule.getWeight();
					if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; }
					rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
					pinScore = ruleScore.mulForInsideOutside(cinScore, rmKey, true);
					chart.addInsideScore(rule.lhs, idx, pinScore);
				}
			}
		}
	}
	
	private void outsideScoreForUnaryRuleAI(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		List<GrammarRule> rules;
		String rmKey = GrammarRule.Unit.P;
		Set<Short> set = new HashSet<Short>();
		GaussianMixture poutScore, coutScore, ruleScore;
		while (!tagQueue.isEmpty()) {
			set.clear();
			// chain unary rule of length 1
			idTag = tagQueue.poll();
			rules = grammar.getUnaryRuleWithP(idTag);
			poutScore = chart.getOutsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				ruleScore = rule.getWeight();
				// ROOT->X is valid if and only if idx == 0, which can be guaranteed naturally since rules X->ROOT do not exist
				set.add(rule.rhs);
				coutScore = ruleScore.mulForInsideOutside(poutScore, rmKey, true);
				chart.addOutsideScore(rule.rhs, idx, coutScore);
			}
			// chain unary rule of length 2
			for (Short id : set) {
				rules = grammar.getUnaryRuleWithP(id);
				poutScore = chart.getOutsideScore(id, idx);
				iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					ruleScore = rule.getWeight();
					coutScore = ruleScore.mulForInsideOutside(poutScore, rmKey, true);
					chart.addOutsideScore(rule.rhs, idx, coutScore);
				}	
			}
		}
	}
	
	private void insideScoreForUnaryRuleAI(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		String rmKey;
		List<GrammarRule> rules;
		Set<Short> set = new HashSet<Short>();
		GaussianMixture pinScore, cinScore, ruleScore;
		while (!tagQueue.isEmpty()) {
			set.clear();
			// chain unary rule of length 1
			idTag = tagQueue.poll();
			rules = grammar.getUnaryRuleWithC(idTag);
			cinScore = chart.getInsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				ruleScore = rule.getWeight();
				// Root->X is valid if idx == 0, we may skip such kind of rules otherwise
				if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; }
				set.add(rule.lhs);
				// double check, in case the rule.lhs is the root node
				rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
				pinScore = ruleScore.mulForInsideOutside(cinScore, rmKey, true);
				chart.addInsideScore(rule.lhs, idx, pinScore);
			}
			// chain unary rule of length 2
			for (Short id : set) {	
				rules = grammar.getUnaryRuleWithC(id);
				cinScore = chart.getInsideScore(id, idx);
				iterator = rules.iterator();
				while (iterator.hasNext()) {
					UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
					ruleScore = rule.getWeight();
					if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; }
					rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
					pinScore = ruleScore.mulForInsideOutside(cinScore, rmKey, true);
					chart.addInsideScore(rule.lhs, idx, pinScore);
				}
			}
		}
	}
	
	private void outsideScoreForUnaryRuleCC(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		List<GrammarRule> rules;
		String rmKey = GrammarRule.Unit.P;
		GaussianMixture poutScore, coutScore, ruleScore;
		while (!tagQueue.isEmpty()) {
			idTag = tagQueue.poll();
			rules = grammar.getChainSumUnaryRulesWithP(idTag);
			poutScore = chart.getOutsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				ruleScore = rule.getWeight();
				// ROOT->X is valid if and only if idx == 0
				coutScore = ruleScore.mulForInsideOutside(poutScore, rmKey, true);
				chart.addOutsideScore(rule.rhs, idx, coutScore);
			}
		}
	}
	
	private void insideScoreForUnaryRuleCC(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		String rmKey;
		List<GrammarRule> rules;
		GaussianMixture pinScore, cinScore, ruleScore;
		while (!tagQueue.isEmpty()) {
			idTag = tagQueue.poll();
			rules = grammar.getChainSumUnaryRulesWithC(idTag);
			cinScore = chart.getInsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				ruleScore = rule.getWeight();
				// Root->X is valid if idx == 0, otherwise we may skip such kind of rules
				if (idx != 0 && rule.type == GrammarRule.RHSPACE) { continue; }
				// double check, in case the rule.lhs is the root node
				rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
				pinScore = ruleScore.mulForInsideOutside(cinScore, rmKey, true);
				chart.addInsideScore(rule.lhs, idx, pinScore);
			}
		}
	}
	
	/**
	 * Compute the outside score for nonterminals of unary rules.
	 * 
	 * @param chart    which records the inside and outside scores 
	 * @param tagQueue which stores the right hand sides of candidate unary rules 
	 * @param idx      the real index of the cell
	 */
	private void outsideScoreForUnaryRulePP(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		List<GrammarRule> rules;
		String rmKey = GrammarRule.Unit.P;
		GaussianMixture poutScore, coutScore, ruleScore;
		while (!tagQueue.isEmpty()) {
			idTag = tagQueue.poll();
			rules = grammar.getUnaryRuleWithP(idTag);
			poutScore = chart.getOutsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				tagQueue.offer(rule.rhs);
				ruleScore = rule.getWeight();
				coutScore = ruleScore.mulForInsideOutside(poutScore, rmKey, true);
				chart.addOutsideScore(rule.rhs, idx, coutScore);
			}
		}
	}
	
	/**
	 * Compute the inside score for nonterminals of unary rules.
	 * 
	 * @param chart    which records the inside and outside scores 
	 * @param tagQueue which stores the right hand sides of candidate unary rules 
	 * @param idx      the real index of the cell
	 */
	private void insideScoreForUnaryRulePP(Chart chart, Queue<Short> tagQueue, int idx) {
		short idTag;
		String rmKey;
		List<GrammarRule> rules;
		GaussianMixture pinScore, cinScore, ruleScore;
		// consider rules A->B->w and B->C->w. So p(w | B) = p(B->w) + p(B->C) * p(C->w).
		// TODO Could it be a dead loop? BUG.
		// DONE Grammar rules are checked before the training. See MethodUtil.checkUnaryRuleCircle().
		while (!tagQueue.isEmpty()) {
			idTag = tagQueue.poll();
			rules = grammar.getUnaryRuleWithC(idTag);
			cinScore = chart.getInsideScore(idTag, idx);
			Iterator<GrammarRule> iterator = rules.iterator();
			while (iterator.hasNext()) {
				UnaryGrammarRule rule = (UnaryGrammarRule) iterator.next();
				tagQueue.offer(rule.lhs);
				ruleScore = rule.getWeight();
				// in case when the rule.lhs is the root node
				rmKey = rule.type == GrammarRule.RHSPACE ? GrammarRule.Unit.C : GrammarRule.Unit.UC;
				pinScore = ruleScore.mulForInsideOutside(cinScore, rmKey, true);
				chart.addInsideScore(rule.lhs, idx, pinScore);
			}
		}
	}
	
	
	/**
	 * Set the outside score of the root node to 1.
	 * 
	 * @param tree the parse tree
	 */
	protected void setRootOutsideScore(Tree<State> tree) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		tree.getLabel().setOutsideScore(gm);
	}
	
	
	/**
	 * @param chart 
	 */
	protected void setRootOutsideScore(Chart chart) {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
		gm.marginalizeToOne();
		chart.addOutsideScore((short) 0, Chart.idx(0, 1), gm);
	}
}
