package edu.shanghaitech.ai.nlp.util;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

import javax.imageio.ImageIO;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.ui.TreeJPanel;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.lveg.impl.LVeGParser;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.impl.Valuator;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Cell;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.Gradient.Grads;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * Useful methods for debugging or ...
 * 
 * @author Yanpeng Zhao
 *
 */
public class FunUtil extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -9216654024276471124L;
	public final static double LOG_ZERO = -1.0e10;
	public final static double LOG_TINY = -0.5e10;
	public final static double EXP_ZERO = -Math.log(-LOG_ZERO);
	public final static NumberFormat formatter = new DecimalFormat("0.###E0");
	
	
	private static Random random = new Random(LVeGTrainer.randomseed);
	private static LVeGGrammar grammar;
	private static LVeGLexicon lexicon;
	
	
	public static void gradcheck(LVeGGrammar grammar, LVeGLexicon lexicon, LVeGParser<?, ?> lvegParser, 
			Valuator<?, ?> valuator, Tree<State> tree, double maxsample) {
		double delta = 1e-5;
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getURuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			gradcheck(grammar, lexicon, entry, lvegParser, valuator, tree, delta, maxsample);
		}
		uRuleMap = lexicon.getURuleMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			gradcheck(grammar, lexicon, entry, lvegParser, valuator, tree, delta, maxsample);
		}
	}
	
	
	public static void gradcheck(LVeGGrammar grammar, LVeGLexicon lexicon, Map.Entry<GrammarRule, GrammarRule> entry, 
			LVeGParser<?, ?> lvegParser, Valuator<?, ?> valuator, Tree<State> tree, double delta, double maxsample) {
		GaussianMixture gm = entry.getValue().getWeight();
		double src = gm.getWeight(0);
		
		double ltInit = lvegParser.doInsideOutsideWithTree(tree);
		double lsInit = lvegParser.doInsideOutside(tree);
		double llInit = ltInit - lsInit;
		
		// w.r.t. mixing weight
		gm.setWeight(0, gm.getWeight(0) + delta);
		
		
//		double llBefore = valuator.probability(tree);
		double ltBefore = lvegParser.doInsideOutsideWithTree(tree);
		double lsBefore = lvegParser.doInsideOutside(tree);
		double llBefore = ltBefore - lsBefore;
		
		
		double t1 = gm.getWeight(0);
		
		gm.setWeight(0, gm.getWeight(0) - 2 * delta);
		
		
//		double llAfter = valuator.probability(tree);
		double ltAfter = lvegParser.doInsideOutsideWithTree(tree);
		double lsAfter = lvegParser.doInsideOutside(tree);
		double llAfter = ltAfter - lsAfter;
		/*
		logger.trace(
				"\nltI: " + ltInit + "\tlsI: " + lsInit + "\tllI: " + llInit + "\n" +
				"ltB: " + ltBefore + "\tlsB: " + lsBefore + "\tllB: " + llBefore + "\n" +
				"ltA: " + ltAfter + "\tlsA: " + lsAfter + "\tllA: " + llAfter);
		*/
		double t2 = gm.getWeight(0);
		
		// restore
		gm.setWeight(0, gm.getWeight(0) + delta);	
		double des = gm.getWeight(0);
		double numericalGrad = -(llBefore - llAfter) / ((t1 - t2));
		
		logger.trace("\n-----\nRule: " + entry.getKey() + "\nGrad Weight: " + 
				numericalGrad + "=(" + llBefore + " - " + llAfter + ")/(" + (t1 - t2) + ")\n" + 
				"B : " + src + "\tA : " + des + "\t(B - A)  =" + (des - src) + "\n" +
				"t1: " + t1 + "\tt2: " + t2 + "\t(t1 - t2)=" + (t1 - t2) + "\n-----\n");
		
		
		gradcheckmu(entry, lvegParser, valuator, tree, delta);
		gradcheckvar(entry, lvegParser, valuator, tree, delta);
		
		
		Object gradients = null;
		if (entry.getKey().type != GrammarRule.LHSPACE) {
			gradients = grammar.getOptimizer().debug(entry.getKey(), false);
		} else {
			gradients = lexicon.getOptimizer().debug(entry.getKey(), false);
		}
		// divide it by # of samplings
		StringBuffer sb = new StringBuffer();
		if (gradients != null) {
			Grads grads = (Grads) gradients;
			sb.append("\n---\nWgrads: ");
			List<Double> wgrads = new ArrayList<Double>(grads.wgrads.size());
			for (Double dw : grads.wgrads) {
				wgrads.add(dw / maxsample);
			}
			List<Map<String, List<Double>>> ggrads = new ArrayList<Map<String, List<Double>>>(grads.ggrads.size());
			for (Map<String, List<Double>> comp : grads.ggrads) {
				Map<String, List<Double>> gauss = new HashMap<String, List<Double>>(comp.size(), 1);
				for (Map.Entry<String, List<Double>> gaussian : comp.entrySet()) {
					List<Double> params = new ArrayList<Double>(gaussian.getValue().size());
					for (Double dg : gaussian.getValue()) {
						params.add(dg / maxsample);
					}
					gauss.put(gaussian.getKey(), params);
				}
				ggrads.add(gauss);
			}
			logger.trace("\n---\nWgrads: " + wgrads + "\nGgrads: " + ggrads + "\n---\n");
		}
	}
	
	
	public static void gradcheckmu(Map.Entry<GrammarRule, GrammarRule> entry, 
			LVeGParser<?, ?> lvegParser, Valuator<?, ?> valuator, Tree<State> tree, double delta) {
		GaussianMixture gm = entry.getValue().getWeight();
		
		double ltInit = lvegParser.doInsideOutsideWithTree(tree);
		double lsInit = lvegParser.doInsideOutside(tree);
		double llInit = ltInit - lsInit;
		
		// w.r.t. mixing weight
		GaussianDistribution gd = gm.getComponent((short) 0).squeeze(null);
		List<Double> mus = gd.getMus();
		
		double src = mus.get(0);
		mus.set(0, mus.get(0) + delta);
		
		
//		double llBefore = valuator.probability(tree);
		double ltBefore = lvegParser.doInsideOutsideWithTree(tree);
		double lsBefore = lvegParser.doInsideOutside(tree);
		double llBefore = ltBefore - lsBefore;
		
		
		double t1 = mus.get(0);
		
		mus.set(0, mus.get(0) - 2 * delta);
		
		
//		double llAfter = valuator.probability(tree);
		double ltAfter = lvegParser.doInsideOutsideWithTree(tree);
		double lsAfter = lvegParser.doInsideOutside(tree);
		double llAfter = ltAfter - lsAfter;
		/*
		logger.trace(
				"\nltI: " + ltInit + "\tlsI: " + lsInit + "\tllI: " + llInit + "\n" +
				"ltB: " + ltBefore + "\tlsB: " + lsBefore + "\tllB: " + llBefore + "\n" +
				"ltA: " + ltAfter + "\tlsA: " + lsAfter + "\tllA: " + llAfter);
		*/
		double t2 = mus.get(0);
		
		// restore
		mus.set(0, mus.get(0) + delta);	
		double des = mus.get(0);
		double numericalGrad = -(llBefore - llAfter) / ((t1 - t2));
		
		logger.trace("\n-----\nRule: " + entry.getKey() + "\nGrad MU    : " + 
				numericalGrad + "=(" + llBefore + " - " + llAfter + ")/(" + (t1 - t2) + ")\n" + 
				"B : " + src + "\tA : " + des + "\t(B - A)  =" + (des - src) + "\n" +
				"t1: " + t1 + "\tt2: " + t2 + "\t(t1 - t2)=" + (t1 - t2) + "\n-----\n");
	}
	
	
	public static void gradcheckvar(Map.Entry<GrammarRule, GrammarRule> entry, 
			LVeGParser<?, ?> lvegParser, Valuator<?, ?> valuator, Tree<State> tree, double delta) {
		GaussianMixture gm = entry.getValue().getWeight();
		
		double ltInit = lvegParser.doInsideOutsideWithTree(tree);
		double lsInit = lvegParser.doInsideOutside(tree);
		double llInit = ltInit - lsInit;
		
		// w.r.t. mixing weight
		GaussianDistribution gd = gm.getComponent((short) 0).squeeze(null);
		List<Double> vars = gd.getVars();
		
		double src = vars.get(0);
		vars.set(0, vars.get(0) + delta);
		
		
//		double llBefore = valuator.probability(tree);
		double ltBefore = lvegParser.doInsideOutsideWithTree(tree);
		double lsBefore = lvegParser.doInsideOutside(tree);
		double llBefore = ltBefore - lsBefore;
		
		
		double t1 = vars.get(0);
		
		vars.set(0, vars.get(0) - 2 * delta);
		
		
//		double llAfter = valuator.probability(tree);
		double ltAfter = lvegParser.doInsideOutsideWithTree(tree);
		double lsAfter = lvegParser.doInsideOutside(tree);
		double llAfter = ltAfter - lsAfter;
		/*
		logger.trace(
				"\nltI: " + ltInit + "\tlsI: " + lsInit + "\tllI: " + llInit + "\n" +
				"ltB: " + ltBefore + "\tlsB: " + lsBefore + "\tllB: " + llBefore + "\n" +
				"ltA: " + ltAfter + "\tlsA: " + lsAfter + "\tllA: " + llAfter);
		*/
		double t2 = vars.get(0);
		
		// restore
		vars.set(0, vars.get(0) + delta);	
		double des = vars.get(0);
		double numericalGrad = -(llBefore - llAfter) / ((t1 - t2) /*2 * delta*/);
		
		logger.trace("\n-----\nRule: " + entry.getKey() + "\nGrad VAR   : " + 
				numericalGrad + "=(" + llBefore + " - " + llAfter + ")/(" + (t1 - t2) + ")\n" + 
				"B : " + src + "\tA : " + des + "\t(B - A)  =" + (des - src) + "\n" +
				"t1: " + t1 + "\tt2: " + t2 + "\t(t1 - t2)=" + (t1 - t2) + "\n-----\n");
	}
	
	
	public static void debugShuffle(StateTreeList trainTrees) {
		int cnt = 0;
		for (Tree<State> tree : trainTrees) {
			logger.trace(tree.getYield() + "\n\n");
			if (++cnt > 4) {
				break;
			}
		}
		trainTrees.shuffle(new Random(0));
		cnt = 0;
		for (Tree<State> tree : trainTrees) {
			logger.trace(tree.getYield() + "\n\n");
			if (++cnt > 4) {
				break;
			}
		}
	}
	
	
	public static void debugChainRule(LVeGGrammar agrammar) {
		grammar = agrammar;
		int count = 0, ncol = 1;
		StringBuffer sb = new StringBuffer();
		sb.append("\n---Chain Unary Rules---\n");
		for (int i = 0; i < grammar.ntag; i++ ) {
			System.out.println("Tag: " + i);
			List<GrammarRule> rules = grammar.getChainSumUnaryRulesWithP(i);
			for (GrammarRule rule : rules) {
				sb.append(rule + "\t" + rule.getWeight().ncomponent());
				if (++count % ncol == 0) {
					sb.append("\n");
				}
			}
		}
		logger.trace(sb.toString() + "\n");
	}
	
	
	public static void debugCount(LVeGGrammar agrammar, LVeGLexicon alexicon, Tree<State> tree, Chart chart) {
		grammar = agrammar;
		lexicon = alexicon;

		int niter = 20, iiter = 0;
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getURuleMap();
		Map<GrammarRule, GrammarRule> bRuleMap = grammar.getBRuleMap();
		// unary grammar rules
		LVeGTrainer.logger.trace("\n---Unary Grammar Rules---\n\n");
		for (Map.Entry<GrammarRule, GrammarRule> rmap : uRuleMap.entrySet()) {
			GrammarRule rule = rmap.getValue();
			Map<Short, List<Map<String, GaussianMixture>>> count = grammar.getCount(rule, false);
			LVeGTrainer.logger.trace(rule + "\tcount=" + count + "\n");
			if (++iiter >= niter) { break; }
		}
		
		iiter = 0;
		// binary grammar rules
		LVeGTrainer.logger.trace("\n---Binary Grammar Rules---\n\n");
		for (Map.Entry<GrammarRule, GrammarRule> rmap : bRuleMap.entrySet()) {
			GrammarRule rule = rmap.getValue();
			Map<Short, List<Map<String, GaussianMixture>>> count = grammar.getCount(rule, false);
			LVeGTrainer.logger.trace(rule + "\tcount=" + count + "\n");
			if (++iiter > niter) { break; }
		}
		
		iiter = 0;
		// unary rules in lexicon
		Set<GrammarRule> ruleSet = lexicon.getRuleSet();
		LVeGTrainer.logger.trace("\n---Unary Grammar Rules in Lexicon---\n");
		for (GrammarRule rule : ruleSet) {
			Map<Short, List<Map<String, GaussianMixture>>> count = lexicon.getCount(rule, false);
			LVeGTrainer.logger.trace(rule + "\tcount=" + count + "\n");
			if (++iiter >= niter) { break; }
		}
		
	}
	
	
	public static void debugRuleWeightInTheTree(LVeGGrammar agrammar, LVeGLexicon alexicon, Tree<State> tree) {
		grammar = agrammar;
		lexicon = alexicon;
		logger.trace("\n---Rules's Weights Used in The Tree---\n\n");
		checkRuleWeightInTheTree(tree);
	}
	
	
	public static void checkRuleWeightInTheTree(Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			checkRuleWeightInTheTree(child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			GaussianMixture cinScore = lexicon.score(word, idParent);
			logger.trace("Word\trule: [" + idParent + ", " + word.wordIdx + "/" + word.getName() + "] " + cinScore + "\n"); // DEBUG
			parent.setInsideScore(cinScore.copy(true));
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				GaussianMixture ruleScore;
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				byte type = idParent == 0 ? GrammarRule.RHSPACE : GrammarRule.LRURULE;
				ruleScore = grammar.getURuleWeight(idParent, idChild, type, false);
				logger.trace("Unary\trule: [" + idParent + ", " + idChild + "] " + ruleScore + "\n"); // DEBUG
				break;
			}
			case 2: {
				GaussianMixture ruleScore;
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();

				ruleScore = grammar.getBRuleWeight(idParent, idlChild, idrChild, false);
				logger.trace("Binary\trule: [" + idParent + ", " + idlChild + ", " + idrChild + "] " + ruleScore + "\n"); // DEBUG
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
	}
	
	
	public static void debugCount(LVeGGrammar agrammar, LVeGLexicon alexicon, Tree<State> tree) {
		grammar = agrammar;
		lexicon = alexicon;
		logger.trace("\n---Rule Counts in The Tree---\n\n");
		checkCount(agrammar, alexicon, tree);
	}
	
	
	public static void checkCount(LVeGGrammar grammar, LVeGLexicon lexicon, Tree<State> tree) {
		if (tree.isLeaf()) { return; }
		
		List<Tree<State>> children = tree.getChildren();
		for (Tree<State> child : children) {
			checkCount(grammar, lexicon, child);
		}
		
		State parent = tree.getLabel();
		short idParent = parent.getId();
		
		if (tree.isPreTerminal()) {
			State word = children.get(0).getLabel();
			Map<Short, List<Map<String, GaussianMixture>>> count = lexicon.getCount(idParent, (short) word.wordIdx, true, GrammarRule.LHSPACE);
			logger.trace("Word\trule: [" + idParent + ", " + word.wordIdx + "/" + word.getName() + "] count=" + count + "\n"); // DEBUG
		} else {
			switch (children.size()) {
			case 0:
				// in case there are some errors in the parse tree.
				break;
			case 1: {
				State child = children.get(0).getLabel();
				short idChild = child.getId();
				
				// root, if (idParent == 0) is true
				byte type = idParent == 0 ? GrammarRule.RHSPACE : GrammarRule.LRURULE;
				
				Map<Short, List<Map<String, GaussianMixture>>> count = grammar.getCount(idParent, idChild, true, type);
				logger.trace("Unary\trule: [" + idParent + ", " + idChild + "] count=" + count + "\n"); // DEBUG
				break;
			}
			case 2: {
				State lchild = children.get(0).getLabel();
				State rchild = children.get(1).getLabel();
				short idlChild = lchild.getId();
				short idrChild = rchild.getId();
				
				Map<Short, List<Map<String, GaussianMixture>>> count = grammar.getCount(idParent, idlChild, idrChild, true);
				logger.trace("Binary\trule: [" + idParent + ", " + idlChild + ", " + idrChild + "] count=" + count + "\n"); // DEBUG
				break;
			}
			default:
				System.err.println("Malformed tree: more than two children. Exitting...");
				System.exit(0);	
			}
		}
		
	}
	
	
	public static void debugChart(List<Cell> chart, short nfirst, int nword, Numberer numberer) {
		int size = nword * (nword + 1) / 2;
		if (chart != null) {
			for (int i = 0; i < size; i++) {
				logger.debug(i + "\t" + chart.get(i).toString(true, nfirst, true, numberer) + "\n\n");
			}
		}
	}
	
	
	public static String debugTree(Tree<State> tree, boolean simple, short nfirst, Numberer numberer, boolean onlyname) {
		StringBuilder sb = new StringBuilder();
		if (!onlyname) {
			toString(tree, simple, nfirst, sb, numberer);
		} else {
			toString(tree, sb, numberer);
		}
		return sb.toString();
	}	
	
	
	private static void toString(Tree<State> tree, StringBuilder sb, Numberer numberer) {
		if (!tree.isLeaf()) {
			sb.append('(');
		}
		State state = tree.getLabel();
		if (state != null) {
			String name = state.getName();
			name = name != null ? name : (String) numberer.object(state.getId());
			sb.append(name);
		}
		if (!tree.isLeaf()) {
			for (Tree<State> child : tree.getChildren()) {
				sb.append(' ');
				toString(child, sb, numberer);
			}
			sb.append(')');
		}
	}
	
	
	private static void toString(Tree<State> tree, boolean simple, short nfirst, StringBuilder sb, Numberer numberer) {
		if (tree.isLeaf()) { sb.append("[" + tree.getLabel().wordIdx + ", " + tree.getLabel().toString(numberer) + "]"); return; }
		sb.append('(');
		
		State state = tree.getLabel();
		if (state != null) {
			sb.append(state.toString(numberer));
			if (state.getInsideScore() != null) { 
				sb.append(" iscore=" + state.getInsideScore().toString(simple, nfirst)); 
			} else {
				sb.append(" iscore=null");
			}
			if (state.getOutsideScore() != null) {
				sb.append(" oscore=" + state.getOutsideScore().toString(simple, nfirst));
			} else {
				sb.append(" oscore=null");
			}
		}
		for (Tree<State> child : tree.getChildren()) {
			sb.append(' ');
			toString(child, simple, nfirst, sb, numberer);
		}
		sb.append(')');
	}
	
	
	/**
	 * @param stateTreeList a set of parse trees
	 * @param maxLength     find the unary chain rule with the specific length
	 * @param treeFileName  file name that is used to save the tree to the image
	 */
	public static void lenUnaryRuleChain(StateTreeList stateTreeList, short maxLength, String treeFileName, Numberer numberer) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (lenUnaryRuleChain(tree, (short) 0, maxLength)) {
				if (count == 0) {
					System.out.println("The tree contains the unary rule chain of length >= " + maxLength + ":");
					System.out.println(tree);
					try {
						saveTree2image(tree, treeFileName + maxLength + "_" + count, null, numberer);
						System.out.println("The tree has been saved to " + treeFileName + maxLength);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				count++;
			}
		}
		System.out.println("# of trees containing the unary rule chain of length >= " + maxLength + ": " + count);
	}
	
	
	/**
	 * @param tree      the parse tree
	 * @param length    stack variable
	 * @param maxLength find the unary chain rule with the specific length
	 * @return
	 */
	public static boolean lenUnaryRuleChain(Tree<State> tree, short length, short maxLength) {
		if (length >= maxLength) { return true; }
		if (tree.isPreTerminal()) { return false; }
		
		List<Tree<State>> children = tree.getChildren();
		short idParent = tree.getLabel().getId();
		switch (children.size()) {
		case 0:
			break;
		case 1: {
			if (idParent != 0) {
				length += 1;
			}
			break;
		}
		case 2: {
			length = 0;
			break;
		}
		default:
			System.err.println("Malformed tree: more than two children. Exiting...");
			System.exit(0);
		}
		for (Tree<State> child : children) {
			if (lenUnaryRuleChain(child, length, maxLength)) {
				return true;
			}
		}
		return false;
	}
	
	
	/**
	 * Check if the unary grammar rules could make a circle.
	 * 
	 * @param grammar the grammar
	 */
	public static boolean checkUnaryRuleCircle(LVeGGrammar agrammar, LVeGLexicon alexicon, boolean startWithC) {
		grammar = agrammar;
		lexicon = alexicon;
		boolean found = false;
		for (int i = 0; i < grammar.ntag; i++) {
			int repeated = 0;
			Set<Integer> visited = new LinkedHashSet<Integer>();
			// System.out.println("Tag " + i + "\t");
			if ((repeated = checkUnaryRuleCircle(i, visited, startWithC)) > 0) {
				LVeGTrainer.logger.error("Repeated item: " + repeated + "\tin the path that begins with " + i + " was found: " + visited + "\n");
				found = true;
			}
		}
		// System.out.println("No circles were found.");
		return found;
	}
	
	
	public static int checkUnaryRuleCircle(int index, Set<Integer> visited, boolean startWithC) {	
		if (visited.contains(index)) { 
			// System.out.println("Repeated item: " + index);
			return index; 
		} else {
			visited.add(index);
		}
		List<GrammarRule> rules;
		if (startWithC) {
			rules = grammar.getURuleWithC(index);
		} else {
			rules = grammar.getURuleWithP(index);
		}
		
		for (GrammarRule rule : rules) {
			UnaryGrammarRule r = (UnaryGrammarRule) rule;
			int id = startWithC ? r.lhs : r.rhs, repeat;
			if ((repeat = checkUnaryRuleCircle(id, visited, startWithC)) > 0) {
				return repeat;
			}
		}
		if (rules.isEmpty()) { /*System.out.println("Path: " + visited);*/ }
		visited.remove(index);
		return -1;
	}
	
	
	public static void isParentEqualToChild(StateTreeList stateTreeList) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (isParentEqualToChild(tree)) {
				System.out.println("The parent and children are the same in the tree " + count + ":");
				System.out.println(tree);
				return;
			}
			count++;
		}
		System.out.println("Oops!");
	}
	
	
	public static void isChildrenSizeZero(StateTreeList stateTreeList) {
		int count = 0;
		for (Tree<State> tree : stateTreeList) {
			if (isChildrenSizeZero(tree)) {
				System.out.println("Pre-terminal node has no children in the tree: " + count + ":");
				System.out.println(tree);
				return;
			}
			count++;
		}
		System.out.println("Oops!");
	}
	
	
	public static boolean containsRule(Tree<State> tree, short idp, short idc) {
		if (tree.isLeaf()) { return false; }
		
		List<Tree<State>> children = tree.getChildren();
		short idParent = tree.getLabel().getId();
		switch (children.size()) {
		case 0:
			break;
		case 1: {
			short idChild = children.get(0).getLabel().getId();
			if (idParent == idp && idChild == idc) {
				return true;
			}
			break;
		}
		case 2:
			break;
		default:
			System.err.println("Malformed tree: more than two children. Exiting...");
			System.exit(0);
		}
		for (Tree<State> child : children) {
			if (containsRule(child, idp, idc)) {
				return true;
			}
		}
		return false;
	}
	
	
	public static boolean isChildrenSizeZero(Tree<State> tree) {
		if (tree.isLeaf()) { return false; }
		
		List<Tree<State>> children = tree.getChildren();
		
		switch (children.size()) {
		case 0:
			if (tree.isPreTerminal()) {
				System.out.println("Pre-terminal: " + tree.getLabel());
			}
			System.out.println("Non-terminal: " + tree.getLabel());
			return true;
		case 1:
			break;
		case 2:
			break;
		default:
			System.err.println("Malformed tree: more than two children. Exiting...");
			System.exit(0);
		}
		
		for (Tree<State> child : children) {
			if (isChildrenSizeZero(child)) {
				return true;
			}
		}
		return false;
	}
	
	
	/**
	 * Check if the parent node and the child node of the given rule are the same.
	 * 
	 * @param tree the parse tree
	 * @return
	 */
	public static boolean isParentEqualToChild(Tree<State> tree) {
		if (tree.isLeaf() || tree.isPreTerminal()) { return false; }
		
		short idParent = tree.getLabel().getId();
		List<Tree<State>> children = tree.getChildren();
		
		switch (children.size()) {
		case 0:
			
			break;
		case 1:
			short idChild = children.get(0).getLabel().getId();
			if (idParent == idChild) {
				System.out.println("Unary rule: parent and child have the same id.");
				System.out.println(tree);
				return true;
			}
			break;
		case 2:
			short idLeftChild = children.get(0).getLabel().getId();
			short idRightChild = children.get(1).getLabel().getId();
			if (idParent == idLeftChild || idParent == idRightChild) {
				System.out.println("Binary rule: parent and children have the same id");
				System.out.println(tree);
				return true;
			}
			break;
		default:
			System.err.println("Malformed tree: more than two children. Exiting...");
			System.exit(0);
		}
		
		for (Tree<State> child : children) {
			if (isParentEqualToChild(child)) {
				return true;
			}
		}
		
		return false;
	}
	
	
	/**
	 * @param dir the dir that needs to be created
	 * @return true if created successfully
	 */
	public static boolean mkdir(String dir) {
		File file = new File(dir);
		if (!file.exists()) {
			if (file.mkdir() || file.mkdirs()) {
				return true;
			}
		}
		return false;
	}
	
	
	/**
	 * Return log(a + b) given log(a) and log(b).
	 * 
	 * @param x in logarithm
	 * @param y in logarithm
	 * @return
	 */
	public static double logAdd(double x, double y) {
		double tmp, diff;
		if (x < y) {
			tmp = x;
			x = y;
			y = tmp;
		}
		diff = y - x; // <= 0
		if (diff < EXP_ZERO) { 
			// if y is far smaller than x
			return x < LOG_TINY ? LOG_ZERO : x;
		} else {
			return x + Math.log(1.0 + Math.exp(diff));
		}
	}
	
	
	/**
	 * Match a number with optional '-' and decimal.
	 * 
	 * @param str the string
	 * @return
	 */
	public static boolean isNumeric(String str){
		return str.matches("[-+]?\\d*\\.?\\d+");  
	}
	
	
	/**
	 * @param stateTree  the state parse tree
	 * @param filename   image name
	 * @param stringTree the string parse tree
	 * @throws Exception oops
	 */
	public static void saveTree2image(Tree<State> stateTree, String filename, Tree<String> stringTree, Numberer numberer) throws Exception {
		TreeJPanel tjp = new TreeJPanel();
		if (stringTree == null) {
			stringTree = StateTreeList.stateTreeToStringTree(stateTree, numberer);
			logger.trace("\nSTRING PARSE TREE: " + stringTree + "\n");
		}
		
		tjp.setTree(stringTree);
		BufferedImage bi = new BufferedImage(tjp.width(), tjp.height(), BufferedImage.TYPE_INT_ARGB);
		
		Graphics2D g2 = bi.createGraphics();
		g2.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f));
		Rectangle2D.Double rect = new Rectangle2D.Double(0, 0, tjp.width(), tjp.height());
		g2.fill(rect);
		g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		tjp.paintComponent(g2);
		g2.dispose();
		
		ImageIO.write(bi, "png", new File(filename + ".png"));
	}
	
	
	/**
	 * @param list     item container
	 * @param type     Doubel.class or Integer.class
	 * @param length   number of items in the list
	 * @param maxint   maximum for integer, and 1 for double
	 * @param nonzero  zero inclusive (false) or exclusive (true)
	 * @param negative allow the negative (true) or not allow (false)
	 */
	public static <T> void randomInitList(Random rnd, List<T> list, Class<T> type, int length, 
			int maxint, double ratio, boolean nonzero, boolean negative) {
		Double obj = new Double(0);
		for (int i = 0; i < length; i++) {
			double tmp = rnd.nextDouble() * maxint;
			if (nonzero) { while (tmp == 0.0) { tmp = rnd.nextDouble() * maxint; } }
			if (negative && tmp < ratio) { tmp = 0 - tmp; }
			list.add(type.isInstance(obj) ? type.cast(tmp) : type.cast((int) tmp));
		}
	}
	
	
	/**
	 * @param array  item container
	 * @param type   Double.class or Integer.class
	 * @param maxint maximum for integer, and 1 for double
	 * 
	 */
	public static <T> void randomInitArray(T[] array, Class<T> type, int maxint) {
		Double obj = new Double(0);
		for (int i = 0; i < array.length; i++) {
			double tmp = random.nextDouble() * maxint;
			array[i] = type.isInstance(obj) ? type.cast(tmp) : type.cast((int) tmp);
		}
		
	}
	
	
	public static void randomInitArrayInt(int[] array, int maxint) {
		for (int i = 0; i < array.length; i++) {
			array[i] = (int) (random.nextDouble() * maxint);
		}
	}
	
	
	public static void randomInitArrayDouble(double[] array) {
		for (int i = 0; i < array.length; i++) {
			array[i] = random.nextDouble();
		}
	}
	
	
	/**
	 * @param list        a list of doubles
	 * @param precision   double precision
	 * @param nfirst      print first # of items
	 * @param exponential whether the list should be read in the exponential form or not
	 * @return
	 */
	public static List<String> double2str(List<Double> list, int precision, int nfirst, boolean exponential, boolean scientific) {
		List<String> strs = new ArrayList<String>();
		String format = "%." + precision + "f", str;
		if (nfirst < 0 || nfirst > list.size()) { nfirst = list.size(); }
		for (int i = 0; i < nfirst; i++) {
			double value = exponential ? Math.exp(list.get(i)) : list.get(i);
			str = scientific ? formatter.format(value) : String.format(format, value);
			strs.add(str);
		}
		/*
		for (Double d : list) {
			strs.add(String.format(format, d));
		}
		*/
		return strs;
	}
	
	
	/**
	 * @param list        a list of doubles
	 * @param exponential whether the list should be read in the exponential form or not
	 * @return
	 */
	public static double sum(List<Double> list, boolean exponential) {
		double sum = 0.0;
		for (Double d : list) {
			sum += exponential ? Math.exp(d) : d;
		}
		return sum;
	}
	
	
	public static void printArrayInt(int[] array) {
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		System.out.println(string);
	}
	
	
	public static void printArrayDouble(double[] array) {
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		System.out.println(string);
	}
	
	
	public static <T> void printArray(T[] array) {
		if (isEmpty(array)) { return; }
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		System.out.println(string);
	}
	
	
	public static <T> void printList(List<T> list) {
		if (isEmpty(list)) { return; }
		String string = "[";
		for (int i = 0; i < list.size() - 1; i++) {
			string += list.get(i) + ", ";
		}
		string += list.get(list.size() - 1) + "]";
		System.out.println(string);
	}
	
	
	public static <T> boolean isEmpty(T[] array) {
		if (array == null || array.length == 0) {
			System.err.println("[null or empty]");
			return true;
		}
		return false;
	}
	
	
	public static <T> boolean isEmpty(List<T> list) {
		if (list == null || list.isEmpty()) {
			System.err.println("[null or empty]");
			return true;
		}
		return false;
	}
	
	public static void testLocally() {
		String[] args = null;
		String fparams = "param.in";
		try {
			args = LearnerConfig.readFile(fparams, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		Options opts = (Options) optionParser.parse(args, true);
		// configurations
		LearnerConfig.initialize(opts, true); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = LearnerConfig.loadData(wrapper, opts);
		treeSummary(trees);
	}
	
	
	public static void treeSummary(Map<String, StateTreeList> trees) {
		StateTreeList trainTrees = trees.get(LearnerConfig.ID_TRAIN);
		StateTreeList testTrees = trees.get(LearnerConfig.ID_TEST);
		StateTreeList devTrees = trees.get(LearnerConfig.ID_DEV);
		
		logger.trace("\n---training sentence length summary---\n");
		lengthSummary(trainTrees);
		logger.trace("\n---  test sentence length summary  ---\n");
		lengthSummary(testTrees);
		logger.trace("\n---   dev sentence length summary   ---\n");
		lengthSummary(devTrees);
	}
	
	public static void lengthSummary(StateTreeList trees) {
		Map<Integer, Integer> summary = new HashMap<Integer, Integer>();
		for (Tree<State> tree : trees) {
			int len = tree.getTerminalYield().size();
			if (summary.containsKey(len)) {
				summary.put(len, summary.get(len) + 1);
			} else {
				summary.put(len, 1);
			}
		}
		logger.trace(summary + "\n");
		logger.trace(summary.keySet() + "\n");
		logger.trace(summary.values() + "\n");
		
		int nbin = 150;
		Map<Integer, Integer> lens = new HashMap<Integer, Integer>();
		for (Map.Entry<Integer, Integer> entry : summary.entrySet()) {
			for (int i = 0; i < nbin ; i += 10) {
				int len = entry.getKey();
				if (len < i) {
					if (lens.containsKey(i)) {
						lens.put(i, lens.get(i) + entry.getValue());
					} else {
						lens.put(i, entry.getValue());
					}
				}
			}
		}
		
		KeyComparator bykey = new KeyComparator(lens);
		TreeMap<Integer, Integer> sorted = new TreeMap<Integer, Integer>(bykey);
		sorted.putAll(lens);
		logger.trace(sorted.keySet() + "\n");
		logger.trace(sorted.values() + "\n");
	}
	
	
	static class KeyComparator implements Comparator<Integer> {
	    Map<Integer, Integer> map;
	    public KeyComparator(Map<Integer, Integer> map) {
	        this.map = map;
	    }
		@Override
		public int compare(Integer o1, Integer o2) {
			return o1 - o2;
		}
	}
	
	
	Comparator<Map.Entry<Integer, Integer>> keycomparator = new Comparator<Map.Entry<Integer, Integer>>() {
		@Override
		public int compare(Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) {
			return o1.getKey() - o2.getKey();
		}
	};

}
