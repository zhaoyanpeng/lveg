package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.SimpleComponent;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.SimpleView;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class DiagonalGaussianMixtureTest extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4920152282468627444L;
	static short ncomp = 1, ndim = 1;
	static {
		Random rnd = new Random(0);
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		String logfile = "log/mog_test_t";
		logger = logUtil.getBothLogger(logfile);
	}
	
	
	@Test
	public void testIntegral() {
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur04 = new UnaryGrammarRule((short) 0, (short) 4, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur42 = new UnaryGrammarRule((short) 4, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		GrammarRule ur22 = new UnaryGrammarRule((short) 2, (short) 2, RuleType.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur04);
		printRule(ur12);
		printRule(ur32);
		printRule(ur42);
		printRule(ur20);
		printRule(ur21);
		printRule(ur22);
		
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.weight.copy(true);
		GaussianMixture cin12 = ur12.weight.mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		GaussianMixture cin32 = ur32.weight.mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		GaussianMixture cin42 = ur42.weight.mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		logger.trace("cin12    : " + cin12.spviews() + "\n");
		
		GaussianMixture cin01 = ur01.weight.mulAndMarginalize(cin12, null, RuleUnit.C, true);
		GaussianMixture cin03 = ur01.weight.mulAndMarginalize(cin32, null, RuleUnit.C, true);
		GaussianMixture cin04 = ur01.weight.mulAndMarginalize(cin42, null, RuleUnit.C, true);
		logger.trace("cin01    : " + cin01.spviews() + "\n");
		logger.trace("cin03    : " + cin03.spviews() + "\n");
		logger.trace("cin04    : " + cin04.spviews() + "\n");
		double scoreT = cin01.eval(null, true);
		cin01.add(cin03, false);
		cin01.add(cin04, false);
		logger.trace("cin01    : " + cin01.spviews() + "\n");
		double scoreS = cin01.eval(null, true);
		logger.trace("scoreT: " + scoreT + "\nscoreS: " + scoreS + "\n");
		
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    : " + outor + "\n");
		
		GaussianMixture out01 = ur01.weight.mulAndMarginalize(outor, null, RuleUnit.P, true);
		GaussianMixture out03 = ur03.weight.mulAndMarginalize(outor, null, RuleUnit.P, true);
		GaussianMixture out04 = ur04.weight.mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("out01    : " + out01 + "\n");
		
		GaussianMixture out12 = ur12.weight.mulAndMarginalize(out01, null, RuleUnit.P, true);
		GaussianMixture out32 = ur32.weight.mulAndMarginalize(out03, null, RuleUnit.P, true);
		GaussianMixture out42 = ur42.weight.mulAndMarginalize(out04, null, RuleUnit.P, true);
		logger.trace("out12    : " + out12.spviews() + "\n");
		logger.trace("out32    : " + out32.spviews() + "\n");
		logger.trace("out42    : " + out42.spviews() + "\n");
		
		
//		GaussianMixture outsidet = null;
//		GaussianMixture cinsidet = null;
//		GaussianMixture outsides = out04;
//		GaussianMixture cinsides = cin20;
//		evalGradients42(ur42, scoreT, scoreS, outsidet, cinsidet, outsides, cinsides);
		
		GaussianMixture outsidet = outor;
		GaussianMixture cinsidet = cin12;
		GaussianMixture outsides = outor;
		GaussianMixture cinsides = cin12;
		evalGradients01(ur01, scoreT, scoreS, outsidet, cinsidet, outsides, cinsides);
		
		
	}
	
	
	protected void evalGradients01(GrammarRule rule, double scoreT, double scoreS, 
			GaussianMixture outsidet, GaussianMixture cinsidet, GaussianMixture outsides, GaussianMixture cinsides) {
		GaussianMixture ruleW = rule.weight;
		double valt = Double.NEGATIVE_INFINITY;
		double vals = Double.NEGATIVE_INFINITY;
		double tmp;
		
		SimpleComponent comp = new SimpleComponent();
		ruleW.component(0, comp);
		GaussianDistribution ugd = comp.gausses.get(RuleUnit.C);
		
		if (outsidet != null && cinsidet != null) {
			
			double partu = Double.NEGATIVE_INFINITY;
			for (SimpleView view : cinsidet.spviews()) {
				tmp = view.weight;
				if (view != null) {
					tmp +=  Math.log(marginalize(ugd, view.gaussian));
				}
				partu = FunUtil.logAdd(partu, tmp);
			}
			
			valt = partu + comp.weight;
		}
		
		if (outsides != null && cinsides != null) {
			
			double partu = Double.NEGATIVE_INFINITY;
			for (SimpleView view : cinsides.spviews()) {
				tmp = view.weight;
				if (view != null) {
					tmp += Math.log(marginalize(ugd, view.gaussian));
				}
				partu = FunUtil.logAdd(partu, tmp);
				logger.trace("s u---" + partu + "\n");
			}
			
			vals = partu + comp.weight;
		}
		
		logger.trace(vals + "\t" + valt + "\n");
		double dMixingW = Math.exp(vals - scoreS) - Math.exp(valt - scoreT);
		logger.trace("d-mixing weight: " + dMixingW + "\n");
		
	}
	
	
	protected void evalGradients42(GrammarRule rule, double scoreT, double scoreS, 
			GaussianMixture outsidet, GaussianMixture cinsidet, GaussianMixture outsides, GaussianMixture cinsides) {
		GaussianMixture ruleW = rule.weight;
		double valt = Double.NEGATIVE_INFINITY;
		double vals = Double.NEGATIVE_INFINITY;
		double tmp;
		
		SimpleComponent comp = new SimpleComponent();
		ruleW.component(0, comp);
		GaussianDistribution pgd = comp.gausses.get(RuleUnit.P);
		GaussianDistribution ugd = comp.gausses.get(RuleUnit.UC);
		
		logger.trace("\n--- ---\n");
		
		if (outsidet != null && cinsidet != null) {
			double partp = Double.NEGATIVE_INFINITY;
			for (SimpleView view : outsidet.spviews()) {
				tmp = view.weight;
				if (view != null) {
					tmp +=  Math.log(marginalize(pgd, view.gaussian));
				}
				partp = FunUtil.logAdd(partp, tmp);
			}
			
			double partu = Double.NEGATIVE_INFINITY;
			for (SimpleView view : cinsidet.spviews()) {
				tmp = view.weight;
				if (view != null) {
					tmp +=  Math.log(marginalize(ugd, view.gaussian));
				}
				partu = FunUtil.logAdd(partu, tmp);
			}
			
			valt = partp + partu + comp.weight;
		}
		
		
		if (outsides != null && cinsides != null) {
			double partp = Double.NEGATIVE_INFINITY;
			for (SimpleView view : outsides.spviews()) {
				tmp = view.weight;
				if (view != null) {
					tmp += Math.log(marginalize(pgd, view.gaussian));
				}
				partp = FunUtil.logAdd(partp, tmp);
				logger.trace("s p---" + partp + "\n");
			}
			
			double partu = Double.NEGATIVE_INFINITY;
			for (SimpleView view : cinsides.spviews()) {
				tmp = view.weight;
				if (view != null) {
					tmp += Math.log(marginalize(ugd, view.gaussian));
				}
				partu = FunUtil.logAdd(partu, tmp);
				logger.trace("s u---" + partp + "\n");
			}
			
			vals = partp + partu + comp.weight;
		}
		
		logger.trace(vals + "\t" + valt + "\n");
		double dMixingW = Math.exp(vals - scoreS) - Math.exp(valt - scoreT);
		logger.trace("d-mixing weight: " + dMixingW + "\n");
	}
	
	
	public double marginalize(GaussianDistribution gd0, GaussianDistribution gd1) {
		int dim = gd0.getDim();
		double vdims = 1.0, epsilon = /*1e-8*/0;
		List<Double> vars0 = gd0.getVars();
		List<Double> vars1 = gd1.getVars();
		List<Double> mus0 = gd0.getMus();
		List<Double> mus1 = gd1.getMus();
		for (int i = 0; i < dim; i++) {
			vdims *= integral(mus0.get(i), mus1.get(i), vars0.get(i), vars1.get(i), epsilon);
		}
		vdims *= Math.pow(2 * Math.PI, -dim / 2.0);
		return vdims;
	}
	
	
	public double marginalizex(GaussianDistribution gd0, GaussianDistribution gd1, int idim) {
		int dim = gd0.getDim();
		double vdims = 1.0, epsilon = 0;
		List<Double> vars0 = gd0.getVars();
		List<Double> vars1 = gd1.getVars();
		List<Double> mus0 = gd0.getMus();
		List<Double> mus1 = gd1.getMus();
		for (int i = 0; i < dim; i++) {
			if (i != idim) {
				vdims *= integral(mus0.get(i), mus1.get(i), vars0.get(i), vars1.get(i), epsilon);
			} else {
				vdims *= integralx(mus0.get(i), mus1.get(i), vars0.get(i), vars1.get(i), epsilon);
			}
		}
		vdims *= Math.pow(2 * Math.PI, -dim / 2.0);
		return vdims;
	}
	
	
	public double marginalizexx(GaussianDistribution gd0, GaussianDistribution gd1, int idim) {
		int dim = gd0.getDim();
		double vdims = 1.0, epsilon = 0;
		List<Double> vars0 = gd0.getVars();
		List<Double> vars1 = gd1.getVars();
		List<Double> mus0 = gd0.getMus();
		List<Double> mus1 = gd1.getMus();
		for (int i = 0; i < dim; i++) {
			if (i != idim) {
				vdims *= integral(mus0.get(i), mus1.get(i), vars0.get(i), vars1.get(i), epsilon);
			} else {
				vdims *= integralxx(mus0.get(i), mus1.get(i), vars0.get(i), vars1.get(i), epsilon);
			}
		}
		vdims *= Math.pow(2 * Math.PI, -dim / 2.0);
		return vdims;
	}
	
	
	public double integral(double mu0, double mu1, double var0, double var1, double epsilon) {
		double mtmp = -Math.pow(mu0 - mu1, 2);
		double vtmp = Math.exp(var0 * 2) + Math.exp(var1 * 2) + epsilon;
		double vval = Math.pow(vtmp, -0.5) * Math.exp(mtmp / (2 * vtmp));
		return vval;
	}
	
	
	public double integralx(double mu0, double mu1, double var0, double var1, double epsilon) {
		double mtmp = -Math.pow(mu0 - mu1, 2);
		double vtmp = Math.exp(var0 * 2) + Math.exp(var1 * 2) + epsilon;
		double vval =  Math.pow(vtmp, -0.5) * Math.exp(mtmp / (2 * vtmp));
		double factor = (mu0 * Math.exp(var1 * 2) + mu1 * Math.exp(var0 * 2)) / vtmp;
		vval = factor * vval;
		return vval;
	}
	
	
	public double integralxx(double mu0, double mu1, double var0, double var1, double epsilon) {
		double mtmp = -Math.pow(mu0 - mu1, 2);
		double vtmp = Math.exp(var0 * 2) + Math.exp(var1 * 2) + epsilon;
		double vval =  Math.pow(vtmp, -0.5) * Math.exp(mtmp / (2 * vtmp));
		double factor0 = (mu0 * Math.exp(var1 * 2) + mu1 * Math.exp(var0 * 2)) / vtmp;
		factor0 = Math.pow(factor0, 2);
		double factor1 = Math.exp(var0 * 2) * Math.exp(var1 * 2) / vtmp;
		vval = (factor0 + factor1) * vval;
		return vval;
	}
	
	
	public void printRule(GrammarRule rule) {
		logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.weight + "\n----------\n");
	}
	
	
//	@Test
	public void testDiagonalGaussianMixture() {
		GaussianMixture gm = new DiagonalGaussianMixture((short) 1);
//		gm.setBinding(RuleType.LHSPACE);
		System.out.println(gm);
		gm.marginalizeToOne();
		System.out.println(gm);
		System.out.println(gm.spviews());
	}
}
