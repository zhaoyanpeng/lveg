package edu.shanghaitech.ai.nlp.util;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.lveg.impl.LVeGParser;
import edu.shanghaitech.ai.nlp.lveg.impl.Valuator;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.optimization.Gradient.Grads;
import edu.shanghaitech.ai.nlp.syntax.State;

public class GradientChecker extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1234982154206455805L;


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
		if (entry.getKey().type != RuleType.LHSPACE) {
			gradients = grammar.getOptimizer().debug(entry.getKey(), false);
		} else {
			gradients = lexicon.getOptimizer().debug(entry.getKey(), false);
		}
		// divide it by # of samplings
		StringBuffer sb = new StringBuffer();
		if (gradients != null) {
			Grads grads = (Grads) gradients;
			sb.append("\n---\nWgrads: ");
			List<Double> wgrads = new ArrayList<>(grads.wgrads.size());
			for (Double dw : grads.wgrads) {
				wgrads.add(dw / maxsample);
			}
			List<EnumMap<RuleUnit, List<Double>>> ggrads = new ArrayList<>(grads.ggrads.size());
			for (Map<RuleUnit, List<Double>> comp : grads.ggrads) {
				EnumMap<RuleUnit, List<Double>> gauss = new EnumMap<>(RuleUnit.class);
				for (Entry<RuleUnit, List<Double>> gaussian : comp.entrySet()) {
					List<Double> params = new ArrayList<>(gaussian.getValue().size());
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
}
