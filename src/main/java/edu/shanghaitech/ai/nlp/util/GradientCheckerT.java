package edu.shanghaitech.ai.nlp.util;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.SimpleComponent;
import edu.shanghaitech.ai.nlp.lvet.impl.LVeTTagger;
import edu.shanghaitech.ai.nlp.lvet.impl.TagTPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TagWPair;
import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.impl.Valuator;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.optimization.Gradient.Grads;

public class GradientCheckerT extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -982978438137423922L;
	
	public static void gradcheck(TagTPair grammar, TagWPair lexicon, LVeTTagger<?, ?> lvetTagger, 
			Valuator<?, ?> valuator, List<TaggedWord> tree, double maxsample) {
		double delta = 1e-8;
		Map<GrammarRule, GrammarRule> uRuleMap = grammar.getEdgeMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			gradcheck(grammar, lexicon, entry, lvetTagger, valuator, tree, delta, maxsample, false);
		}
		logger.trace("\n\n--------- lexicon ---------\n\n");
		uRuleMap = lexicon.getEdgeMap();
		for (Map.Entry<GrammarRule, GrammarRule> entry : uRuleMap.entrySet()) {
			gradcheck(grammar, lexicon, entry, lvetTagger, valuator, tree, delta, maxsample, true);
		}
	}
	
	
	public static void gradcheck(TagTPair grammar, TagWPair lexicon, Map.Entry<GrammarRule, GrammarRule> entry, 
			LVeTTagger<?, ?> lvetTagger, Valuator<?, ?> valuator, List<TaggedWord> tree, 
			double delta, double maxsample, boolean islexicon) {
		GaussianMixture gm = entry.getValue().getWeight();
		double src = gm.getWeight(0);
		
		double ltInit = lvetTagger.doForwardBackwardWithTags(tree);
		double lsInit = lvetTagger.doForwardBackward(tree, -1);
		double llInit = ltInit - lsInit;
		
		// w.r.t. mixing weight
		gm.setWeight(0, gm.getWeight(0) + delta);
		
		
//		double llBefore = valuator.probability(tree);
		double ltBefore = lvetTagger.doForwardBackwardWithTags(tree);
		double lsBefore = lvetTagger.doForwardBackward(tree, -1);
		double llBefore = ltBefore - lsBefore;
		
		
		double t1 = gm.getWeight(0);
		
		gm.setWeight(0, gm.getWeight(0) - 2 * delta);
		
		
//		double llAfter = valuator.probability(tree);
		double ltAfter = lvetTagger.doForwardBackwardWithTags(tree);
		double lsAfter = lvetTagger.doForwardBackward(tree, -1);
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
		
		
		gradcheckmu(entry, lvetTagger, valuator, tree, delta);
		gradcheckvar(entry, lvetTagger, valuator, tree, delta);
		
		
		Object gradients = null;
		if (!islexicon) {
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
			
			EnumMap<RuleUnit, List<List<Double>>> ggrads = new EnumMap<>(RuleUnit.class);
			for (Entry<RuleUnit, List<List<Double>>> unit : grads.ggrads.entrySet()) {
				List<List<Double>> ugrads = new ArrayList<>();
				for (List<Double> cgrads : unit.getValue()) {
					List<Double> copy = new ArrayList<>(cgrads.size());
					copy.addAll(cgrads);
					ugrads.add(copy);
				}
				ggrads.put(unit.getKey(), ugrads);
			}
			
			logger.trace("\n---\nWgrads: " + wgrads + "\nGgrads: " + ggrads + "\n---\n");
		}
	}
	
	
	public static void gradcheckmu(Map.Entry<GrammarRule, GrammarRule> entry, 
			LVeTTagger<?, ?> lvetTagger, Valuator<?, ?> valuator, List<TaggedWord> tree, double delta) {
		GaussianMixture gm = entry.getValue().getWeight();
		
		double ltInit = lvetTagger.doForwardBackwardWithTags(tree);
		double lsInit = lvetTagger.doForwardBackward(tree, -1);
		double llInit = ltInit - lsInit;
		
		// w.r.t. mixing weight
		SimpleComponent comp = new SimpleComponent();
		gm.component(0, comp);
		
		GaussianDistribution gd = null;
		for (Entry<RuleUnit, GaussianDistribution> unit : comp.gausses.entrySet()) {
			gd = unit.getValue();
			break;
		}
		List<Double> mus = gd.getMus();
		
		double src = mus.get(0);
		mus.set(0, mus.get(0) + delta);
		
		
//		double llBefore = valuator.probability(tree);
		double ltBefore = lvetTagger.doForwardBackwardWithTags(tree);
		double lsBefore = lvetTagger.doForwardBackward(tree, -1);
		double llBefore = ltBefore - lsBefore;
		
		
		double t1 = mus.get(0);
		
		mus.set(0, mus.get(0) - 2 * delta);
		
		
//		double llAfter = valuator.probability(tree);
		double ltAfter = lvetTagger.doForwardBackwardWithTags(tree);
		double lsAfter = lvetTagger.doForwardBackward(tree, -1);
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
			LVeTTagger<?, ?> lvetTagger, Valuator<?, ?> valuator, List<TaggedWord> tree, double delta) {
		GaussianMixture gm = entry.getValue().getWeight();
		
		double ltInit = lvetTagger.doForwardBackwardWithTags(tree);
		double lsInit = lvetTagger.doForwardBackward(tree, -1);
		double llInit = ltInit - lsInit;
		
		// w.r.t. mixing weight
		SimpleComponent comp = new SimpleComponent();
		gm.component(0, comp);
		
		GaussianDistribution gd = null;
		for (Entry<RuleUnit, GaussianDistribution> unit : comp.gausses.entrySet()) {
			gd = unit.getValue();
			break;
		}
		List<Double> vars = gd.getVars();
		
		double src = vars.get(0);
		vars.set(0, vars.get(0) + delta);
		
		
//		double llBefore = valuator.probability(tree);
		double ltBefore = lvetTagger.doForwardBackwardWithTags(tree);
		double lsBefore = lvetTagger.doForwardBackward(tree, -1);
		double llBefore = ltBefore - lsBefore;
		
		
		double t1 = vars.get(0);
		
		vars.set(0, vars.get(0) - 2 * delta);
		
		
//		double llAfter = valuator.probability(tree);
		double ltAfter = lvetTagger.doForwardBackwardWithTags(tree);
		double lsAfter = lvetTagger.doForwardBackward(tree, -1);
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
