package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class DiagonalGaussianDistributionTest extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3752230234407177299L;


	@Test
	public void testDiagonalGaussianDistribution() {

		String logfile = "log/mog_test_t";
		logger = logUtil.getBothLogger(logfile);
		testIntegrationComp1Dim1Ana();
	}
	
	
	public void testIntegrationComp2Dim2() {
		Random rnd = new Random(0);
		short ncomp = 2, ndim = 2;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -1e-3);
		GaussianMixture cin12 = ur12.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, RuleUnit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
//		ur01.getWeight().setWeight(0, 1e-3);
		
		GaussianMixture cin01 = ur01.getWeight().mulAndMarginalize(cin12, null, RuleUnit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + FunUtil.logAdd(cin01.getWeight(0), cin01.getWeight(1)) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, RuleUnit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulAndMarginalize(cin32, null, RuleUnit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + FunUtil.logAdd(cin03.getWeight(0), cin03.getWeight(1)) + "\n");
		
		logger.trace("Score    : " + FunUtil.logAdd(cin01.marginalize(true), cin03.marginalize(true)) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulAndMarginalize(outx1, null, RuleUnit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulAndMarginalize(outx3, null, RuleUnit.P, true);
		outx2.add(out32, false);
		logger.trace("outx2    :" + outx2 + "\n");
		
		int nsample = 100000;
		GaussianMixture outsides = outor;
		GaussianMixture cinsides = cin12;
		GaussianMixture outsidet = outor;
		GaussianMixture cinsidet = cin12;
		double scoret = FunUtil.logAdd(cin01.getWeight(0), cin01.getWeight(1));
		double scores = FunUtil.logAdd(cin01.marginalize(true), cin03.marginalize(true));
		
		logger.trace("\n---Counts---\n");
		logger.trace("scoret: " + scoret + "\tscores: " + scores + "\n");
		logger.trace("outsides: " + outsides + "\n");
		logger.trace("outsidet: " + outsidet + "\n");
		logger.trace("cinsides: " + cinsides + "\n");
		logger.trace("cinsidet: " + cinsidet + "\n");
		
		Random rnd1 = new Random(0);
		evalgradientsComp2Dim2(nsample, ur01, outsides, cinsides, outsidet, cinsidet, scoret, scores, rnd1);
	}
	
	
	public void evalgradientsComp2Dim2(int nsample, GrammarRule rule, GaussianMixture outsides, GaussianMixture cinsides, 
			GaussianMixture outsidet, GaussianMixture cinsidet, double scoret, double scores, Random rnd) {
		double sum1 = 0.0, sum2 = 0.0;
		logger.trace("\nscoret: " + scoret + "\tscores: " + scores + "\n\n");
		for (int i = 0; i < nsample; i++) {
			
			double comp1dim1 = /*0.07791503650933558*/rnd.nextGaussian();
			double comp1dim2 = rnd.nextGaussian();
			double comp2dim1 = rnd.nextGaussian();
			double comp2dim2 = rnd.nextGaussian();
			
			double c1d1v = normal(comp1dim1);
			double c1d2v = normal(comp1dim2);
			double c2d1v = normal(comp2dim1);
			double c2d2v = normal(comp2dim2);
			
			double c1in = Math.exp(cinsides.getWeight(0)) * c1d1v * c1d2v * 2;
			double c2in = Math.exp(cinsides.getWeight(1)) * c2d1v * c2d2v * 2;
			
			double dc1w = Math.exp(0) * c1d1v * c1d2v;
			double dc2w = Math.exp(0) * c2d1v * c2d2v;
			
			double dc1r = Math.exp(Math.log(c1in) - scores) - Math.exp(Math.log(c1in) - scoret);
			double dc2r = Math.exp(Math.log(c2in) - scores) - Math.exp(Math.log(c2in) - scoret);
			
			sum1 += dc1w * dc1r;
			sum2 += dc2w * dc2r;
			
			/*
//			double snorm = snorms[i];
			double snorm1 = rnd.nextGaussian(); // sample from N(0, 1)
			double vnorm1 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm1 * snorm1 / 2.0);
//			logger.trace("snorm1: " + snorm1 + "\n");
//			logger.trace("vnorm1: " + vnorm1 + "\t" + Math.log(vnorm1) + "\n");
			
			double snorm2 = rnd.nextGaussian(); // sample from N(0, 1)
			double vnorm2 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm2 * snorm2 / 2.0);
//			logger.trace("snorm2: " + snorm2 + "\n");
//			logger.trace("vnorm2: " + vnorm2 + "\t" + Math.log(vnorm2) + "\n");
			
			double dW = Math.exp(0) * vnorm1 * vnorm2;
//			logger.trace("dW   : " + dW + "\t" + Math.log(dW) + "\n");
			
			double in = 1 * Math.exp(cinsides.getWeight(0)) * vnorm1 * vnorm2;
//			logger.trace("in   : " + in + "\tw: " + Math.exp(cinsides.getWeight(0)) + "\n");
			
			double dR = Math.exp(Math.log(in) - scores) - Math.exp(Math.log(in) - scoret);
//			logger.trace("dR   : " + dR + "\n");
//			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			*/
		}
		sum1 /= nsample;
		sum2 /= nsample;
		logger.trace("Evaluated Grad: [" + sum1 + ", " + sum2 + "]\n");
	}
	
	
	public void testIntegrationComp2Dim1() {
		Random rnd = new Random(0);
		short ncomp = 2, ndim = 1;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -1e-3);
		GaussianMixture cin12 = ur12.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, RuleUnit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
//		ur01.getWeight().setWeight(0, 1e-3);
		
		GaussianMixture cin01 = ur01.getWeight().mulAndMarginalize(cin12, null, RuleUnit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + FunUtil.logAdd(cin01.getWeight(0), cin01.getWeight(1)) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, RuleUnit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulAndMarginalize(cin32, null, RuleUnit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + FunUtil.logAdd(cin03.getWeight(0), cin03.getWeight(1)) + "\n");
		
		logger.trace("Score    : " + FunUtil.logAdd(cin01.marginalize(true), cin03.marginalize(true)) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulAndMarginalize(outx1, null, RuleUnit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulAndMarginalize(outx3, null, RuleUnit.P, true);
		outx2.add(out32, false);
		logger.trace("outx2    :" + outx2 + "\n");
		
		int nsample = 100000;
		GaussianMixture outsides = outor;
		GaussianMixture cinsides = cin12;
		GaussianMixture outsidet = outor;
		GaussianMixture cinsidet = cin12;
		double scoret = FunUtil.logAdd(cin01.getWeight(0), cin01.getWeight(1));
		double scores = FunUtil.logAdd(scoret, scoret);
		
		logger.trace("\n---Counts---\n");
		logger.trace("scoret: " + scoret + "\tscores: " + scores + "\n");
		logger.trace("outsides: " + outsides + "\n");
		logger.trace("outsidet: " + outsidet + "\n");
		logger.trace("cinsides: " + cinsides + "\n");
		logger.trace("cinsidet: " + cinsidet + "\n");
		
		Random rnd1 = new Random(0);
		evalgradientsComp2Dim1(nsample, ur01, outsides, cinsides, outsidet, cinsidet, scoret, scores, rnd1);
	}
	
	
	public void evalgradientsComp2Dim1(int nsample, GrammarRule rule, GaussianMixture outsides, GaussianMixture cinsides, 
			GaussianMixture outsidet, GaussianMixture cinsidet, double scoret, double scores, Random rnd) {
		double sum1 = 0.0, sum2 = 0.0;
		logger.trace("\nscoret: " + scoret + "\tscores: " + scores + "\n\n");
		for (int i = 0; i < nsample; i++) {
			
			double comp1dim1 = /*0.07791503650933558*/rnd.nextGaussian();
			double comp2dim1 = rnd.nextGaussian();
			
			double c1d1v = normal(comp1dim1);
			double c2d1v = normal(comp2dim1);
			
			double c1in = Math.exp(cinsides.getWeight(0)) * c1d1v * 2;
			double c2in = Math.exp(cinsides.getWeight(1)) * c2d1v * 2;
			
			double dc1w = Math.exp(0) * c1d1v;
			double dc2w = Math.exp(0) * c2d1v;
			
			double dc1r = Math.exp(Math.log(c1in) - scores) - Math.exp(Math.log(c1in) - scoret);
			double dc2r = Math.exp(Math.log(c2in) - scores) - Math.exp(Math.log(c2in) - scoret);
			
			sum1 += dc1w * dc1r;
			sum2 += dc2w * dc2r;
			
			/*
//			double snorm = snorms[i];
			double snorm1 = rnd.nextGaussian(); // sample from N(0, 1)
			double vnorm1 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm1 * snorm1 / 2.0);
//			logger.trace("snorm1: " + snorm1 + "\n");
//			logger.trace("vnorm1: " + vnorm1 + "\t" + Math.log(vnorm1) + "\n");
			
			double snorm2 = rnd.nextGaussian(); // sample from N(0, 1)
			double vnorm2 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm2 * snorm2 / 2.0);
//			logger.trace("snorm2: " + snorm2 + "\n");
//			logger.trace("vnorm2: " + vnorm2 + "\t" + Math.log(vnorm2) + "\n");
			
			double dW = Math.exp(0) * vnorm1 * vnorm2;
//			logger.trace("dW   : " + dW + "\t" + Math.log(dW) + "\n");
			
			double in = 1 * Math.exp(cinsides.getWeight(0)) * vnorm1 * vnorm2;
//			logger.trace("in   : " + in + "\tw: " + Math.exp(cinsides.getWeight(0)) + "\n");
			
			double dR = Math.exp(Math.log(in) - scores) - Math.exp(Math.log(in) - scoret);
//			logger.trace("dR   : " + dR + "\n");
//			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			*/
		}
		sum1 /= nsample;
		sum2 /= nsample;
		logger.trace("Evaluated Grad: [" + sum1 + ", " + sum2 + "\n");
	}
	
	
	public void testIntegrationComp1Dim2() {
		Random rnd = new Random(0);
		short ncomp = 1, ndim = 2;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -1e-3);
		GaussianMixture cin12 = ur12.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, RuleUnit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
		ur01.getWeight().setWeight(0, 1e-3);
		
		GaussianMixture cin01 = ur01.getWeight().mulAndMarginalize(cin12, null, RuleUnit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + cin01.getWeight(0) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, RuleUnit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulAndMarginalize(cin32, null, RuleUnit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + cin03.getWeight(0) + "\n");
		
		logger.trace("Score    : " + FunUtil.logAdd(cin01.getWeight(0), cin03.getWeight(0)) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulAndMarginalize(outx1, null, RuleUnit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulAndMarginalize(outx3, null, RuleUnit.P, true);
		outx2.add(out32, false);
		logger.trace("outx2    :" + outx2 + "\n");
		
		int nsample = 100000;
		GaussianMixture outsides = outor;
		GaussianMixture cinsides = cin12;
		GaussianMixture outsidet = outor;
		GaussianMixture cinsidet = cin12;
		double scoret = cin01.getWeight(0);
		double scores = FunUtil.logAdd(cin01.getWeight(0), cin03.getWeight(0));
		
		logger.trace("\n---Counts---\n");
		logger.trace("scoret: " + scoret + "\tscores: " + scores + "\n");
		logger.trace("outsides: " + outsides + "\n");
		logger.trace("outsidet: " + outsidet + "\n");
		logger.trace("cinsides: " + cinsides + "\n");
		logger.trace("cinsidet: " + cinsidet + "\n");
		
		Random rnd1 = new Random(0);
//		evalgradientsComp1Dim2(nsample, ur01, outsides, cinsides, outsidet, cinsidet, scoret, scores, rnd1);
	}
	
	
	public void evalgradientsComp1Dim2(int nsample, GrammarRule rule, GaussianMixture outsides, GaussianMixture cinsides, 
			GaussianMixture outsidet, GaussianMixture cinsidet, double scoret, double scores, Random rnd) {
		double sum = 0.0;
		logger.trace("\nscoret: " + scoret + "\tscores: " + scores + "\n\n");
		for (int i = 0; i < nsample; i++) {
			
//			double snorm = snorms[i];
			double pnorm1 = rnd.nextGaussian(); // sample from N(0, 1)
			double pvnorm1 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-pnorm1 * pnorm1 / 2.0);
//			logger.trace("snorm1: " + snorm1 + "\n");
//			logger.trace("vnorm1: " + vnorm1 + "\t" + Math.log(vnorm1) + "\n");
			
			double pnorm2 = rnd.nextGaussian(); // sample from N(0, 1)
			double pvnorm2 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-pnorm2 * pnorm2 / 2.0);
//			logger.trace("snorm2: " + snorm2 + "\n");
//			logger.trace("vnorm2: " + vnorm2 + "\t" + Math.log(vnorm2) + "\n");
			
			
			double cnorm1 = rnd.nextGaussian(); // sample from N(0, 1)
			double cvnorm1 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-cnorm1 * cnorm1 / 2.0);
//			logger.trace("snorm1: " + snorm1 + "\n");
//			logger.trace("vnorm1: " + vnorm1 + "\t" + Math.log(vnorm1) + "\n");
			
			double cnorm2 = rnd.nextGaussian(); // sample from N(0, 1)
			double cvnorm2 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-cnorm2 * cnorm2 / 2.0);
//			logger.trace("snorm2: " + snorm2 + "\n");
//			logger.trace("vnorm2: " + vnorm2 + "\t" + Math.log(vnorm2) + "\n");
			
			double pvalue = pvnorm1 * pvnorm2, cvalue = cvnorm1 * cvnorm2;
			
			double dW = Math.exp(0) * pvalue * cvalue;
//			logger.trace("dW   : " + dW + "\t" + Math.log(dW) + "\n");
			
			double in = pvalue * cvalue;
//			logger.trace("in   : " + in + "\tw: " + Math.exp(cinsides.getWeight(0)) + "\n");
			
			double dR = Math.exp(Math.log(in) - scores) - Math.exp(Math.log(in) - scoret);
//			logger.trace("dR   : " + dR + "\n");
//			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			
			
			/*
//			double snorm = snorms[i];
			double snorm1 = rnd.nextGaussian(); // sample from N(0, 1)
			double vnorm1 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm1 * snorm1 / 2.0);
//			logger.trace("snorm1: " + snorm1 + "\n");
//			logger.trace("vnorm1: " + vnorm1 + "\t" + Math.log(vnorm1) + "\n");
			
			double snorm2 = rnd.nextGaussian(); // sample from N(0, 1)
			double vnorm2 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm2 * snorm2 / 2.0);
//			logger.trace("snorm2: " + snorm2 + "\n");
//			logger.trace("vnorm2: " + vnorm2 + "\t" + Math.log(vnorm2) + "\n");
			
			double dW = Math.exp(0) * vnorm1 * vnorm2;
//			logger.trace("dW   : " + dW + "\t" + Math.log(dW) + "\n");
			
			double in = 1 * Math.exp(cinsides.getWeight(0)) * vnorm1 * vnorm2;
//			logger.trace("in   : " + in + "\tw: " + Math.exp(cinsides.getWeight(0)) + "\n");
			
			double dR = Math.exp(Math.log(in) - scores) - Math.exp(Math.log(in) - scoret);
//			logger.trace("dR   : " + dR + "\n");
//			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			*/
		}
		sum /= nsample;
		logger.trace("Evaluated Grad: " + sum + "\n");
	}
	
	
	
	public void testIntegrationComp1Dim1() {
		Random rnd = new Random(0);
		short ncomp = 1, ndim = 1;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -0.1);
		GaussianMixture cin12 = ur12.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, RuleUnit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
		GaussianMixture cin01 = ur01.getWeight().mulAndMarginalize(cin12, null, RuleUnit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + cin01.getWeight(0) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, RuleUnit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulAndMarginalize(cin32, null, RuleUnit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + cin03.getWeight(0) + "\n");
		
		double scoret = cin01.getWeight(0);;
		double scores = FunUtil.logAdd(cin01.getWeight(0), cin03.getWeight(0));
		logger.trace("Score    : " + scores + "\n");
		logger.trace("Loglh    : " + (scoret - scores) + "\t" + Math.exp(scoret - scores) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulAndMarginalize(outx1, null, RuleUnit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulAndMarginalize(outx3, null, RuleUnit.P, true);
		outx2.add(out32, false);
		logger.trace("outx2    :" + outx2 + "\n");
		
		int nsample = 300;
		GaussianMixture outsides = outor;
		GaussianMixture cinsides = cin12;
		GaussianMixture outsidet = outor;
		GaussianMixture cinsidet = cin12;
		
		logger.trace("\n---Counts---\n");
		logger.trace("scoret: " + scoret + "\tscores: " + scores + "\n");
		logger.trace("outsides: " + outsides + "\n");
		logger.trace("outsidet: " + outsidet + "\n");
		logger.trace("cinsides: " + cinsides + "\n");
		logger.trace("cinsidet: " + cinsidet + "\n");
		
		Random rnd1 = new Random(0);
		evalgradientsComp1Dim1(nsample, ur01, outsides, cinsides, outsidet, cinsidet, scoret, scores, rnd1);
	}
	
	public void evalgradientsComp1Dim1(int nsample, GrammarRule rule, GaussianMixture outsides, GaussianMixture cinsides, 
			GaussianMixture outsidet, GaussianMixture cinsidet, double scoret, double scores, Random rnd) {
		double sum = 0.0;
		logger.trace("\nscoret: " + scoret + "\tscores: " + scores + "\n\n");
		double[] snorms = /*{-0.12512048254027627, 1.1644316196215585, 9.505685819044628E-4};*/
		{0.2548576117030025, 0.24659131695413303, -0.07718962231888228};
		
		double[] pnorms = {-0.48705662466425903, 1.581991482792654, -1.0894906314581259};
		double[] cnorms = {-0.016890356724075888, 0.008265704520321576, -0.8504780918057038};
		
		double std = 1e-20;
		for (int i = 0; i < nsample; i++) {
			/*
//			double pp = pnorms[i];
//			double cp = cnorms[i];
			
			double pp = rnd.nextGaussian();
			double cp = rnd.nextGaussian();
			
			double pvnorm = normal(pp);
			double cvnorm = normal(cp);
			
			logger.trace("psnorm: " + pp + "\tcsnorm: " + cp + "\n");
			logger.trace("pvnorm: " + pvnorm + "\tcvnorm: " + cvnorm + "\n");
			
			double dW = Math.exp(0) * pvnorm * cvnorm;
			logger.trace("dW   : " + dW + "\t" + Math.log(dW) + "\n");
			
			double in = 1 * pvnorm * cvnorm;
			logger.trace("in   : " + in + "\tw: " + Math.exp(cinsides.getWeight(0)) + "\n");
			
			double dR = Math.exp(Math.log(in) - scores) - Math.exp(Math.log(in) - scoret);
			logger.trace("dR   : " + dR + "\n");
			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			*/
			
			
//			double snorm = snorms[i];
			double snorm = rnd.nextGaussian(); // sample from N(0, 1)
//			snorm = (snorm - 0) / std;
			double vnorm = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm * snorm / 2.0) / std;
//			logger.trace("snorm: " + snorm + "\n");
//			logger.trace("vnorm: " + vnorm + "\t" + Math.log(vnorm) + "\n");
			
			double dW = Math.exp(0) * vnorm;
//			logger.trace("dW   : " + dW + "\t" + Math.log(dW) + "\n");
			
			double in = Math.exp(cinsides.getWeight(0)) * vnorm;
//			logger.trace("in   : " + in + "\tw: " + Math.exp(cinsides.getWeight(0)) + "\n");
			
			double dR = Math.exp(Math.log(in) - scores) - Math.exp(Math.log(in) - scoret);
//			logger.trace("dR   : " + dR + "\n");
//			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			
		}
		sum /= nsample;
		logger.trace("Evaluated Grad: " + sum + "\n");
	}
	
	public void testIntegrationComp1Dim1V() {
		Random rnd = new Random(0);
		short ncomp = 1, ndim = 1;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		
		Component comp01 = ur01.getWeight().getComponent((short) 0);
		Component comp03 = ur03.getWeight().getComponent((short) 0);
		Component comp12 = ur12.getWeight().getComponent((short) 0);
		Component comp32 = ur32.getWeight().getComponent((short) 0);
		Component comp20 = ur20.getWeight().getComponent((short) 0);
		Component comp21 = ur21.getWeight().getComponent((short) 0);
		
		GaussianDistribution gd01 = comp01.squeeze(RuleUnit.C);
		GaussianDistribution gd03 = comp03.squeeze(RuleUnit.C);
		GaussianDistribution gd12p = comp12.squeeze(RuleUnit.P);
		GaussianDistribution gd12c = comp12.squeeze(RuleUnit.UC);
		GaussianDistribution gd32p = comp32.squeeze(RuleUnit.P);
		GaussianDistribution gd32c = comp32.squeeze(RuleUnit.UC);
		GaussianDistribution gd20 = comp20.squeeze(RuleUnit.P);
		GaussianDistribution gd21 = comp21.squeeze(RuleUnit.P);
		
		double mua = 1.0, mub = -1.0;
		gd01.getMus().set(0, mua);
		gd03.getMus().set(0, mub);
		gd12p.getMus().set(0, mua);
		gd12c.getMus().set(0, mua);
		gd32p.getMus().set(0, mub);
		gd32c.getMus().set(0, mub);
		gd20.getMus().set(0, mua);
		gd21.getMus().set(0, mub);
		
		double std = 3;
		double vara = Math.log(std), varb = Math.log(std);
		gd01.getVars().set(0, vara);
		gd03.getVars().set(0, varb);
		gd12p.getVars().set(0, vara);
		gd12c.getVars().set(0, vara);
		gd32p.getVars().set(0, varb);
		gd32c.getVars().set(0, varb);
		gd20.getVars().set(0, vara);
		gd21.getVars().set(0, varb);
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -1e-5);
		GaussianMixture cin12 = ur12.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\t" + cin12.getWeight(0) + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, RuleUnit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\t" + cin12copy.getWeight(0) + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		logger.trace("cin32    : " + cin32 + "\t" + cin32.getWeight(0) + "\n");
		GaussianMixture cin32copy = marginalize(ur32.getWeight(), cin20, RuleUnit.UC, true);
		logger.trace("cin32copy: " + cin32copy + "\t" + cin32copy.getWeight(0) + "\n");
		
//		ur01.getWeight().setWeight(0, -1e-5);
		
		GaussianMixture cin01 = ur01.getWeight().mulAndMarginalize(cin12, null, RuleUnit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + cin01.getWeight(0) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, RuleUnit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulAndMarginalize(cin32, null, RuleUnit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + cin03.getWeight(0) + "\n");
		
		double scoret = cin01.getWeight(0);
		double scores = FunUtil.logAdd(cin01.getWeight(0), cin03.getWeight(0));
		logger.trace("Score    : " + scores + "\n");
		logger.trace("Loglh    : " + (scoret - scores) + "\t" + Math.exp(scoret - scores) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulAndMarginalize(outx1, null, RuleUnit.P, true);
		GaussianMixture outsidet = outx2.copy(true);
		GaussianMixture out32 = ur32.getWeight().mulAndMarginalize(outx3, null, RuleUnit.P, true);
		outx2.add(out32, false);
		logger.trace("outx2    :" + outx2 + "\n");
		
		int nsample = 300000;
		GaussianMixture outsides = outx2;
		GaussianMixture cinsides = cin20;
		GaussianMixture cinsidet = cin20;
		
		logger.trace("\n---Counts---\n");
		logger.trace("scoret: " + scoret + "\tscores: " + scores + "\n");
		logger.trace("outsides: " + outsides + "\n");
		logger.trace("outsidet: " + outsidet + "\n");
		logger.trace("cinsides: " + cinsides + "\n");
		logger.trace("cinsidet: " + cinsidet + "\n");
		
		Random rnd1 = new Random(0);
//		evalgradientsComp1Dim1V(nsample, ur01, outsides, cinsides, outsidet, cinsidet, scoret, scores, rnd1);
		evalgradientsComp1Dim1V(nsample, ur20, outx2, cinsides, outsidet, cinsidet, scoret, scores, rnd1, std);
	}
	
	public void evalgradientsComp1Dim1V(int nsample, GrammarRule rule, GaussianMixture outsides, GaussianMixture cinsides, 
			GaussianMixture outsidet, GaussianMixture cinsidet, double scoret, double scores, Random rnd, double std) {
		double sum = 0.0;
		logger.trace("\nscoret: " + scoret + "\tscores: " + scores + "\n\n");
		for (int i = 0; i < nsample; i++) {
			
			double snorm = rnd.nextGaussian(); // sample from N(0, 1)
			double real0 = snorm * std + 1; // current distribution N(1, 1)
			double real1 = (real0 + 1) / std; // transformed to N(-1, 1)
			double vnorm0 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm * snorm / 2.0) / std;
			double vnorm1 = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-real1 * real1 / 2.0) / std;
			
			double dW = Math.exp(0) * vnorm0;
			
			double cntt = Math.exp(outsidet.getWeight(0)) * vnorm0;
			double cnts = Math.exp(outsides.getWeight(0)) * vnorm1 + cntt;
			
			double dR = Math.exp(Math.log(cnts) - scores) - Math.exp(Math.log(cntt) - scoret);
//			logger.trace("dR   : " + dR + "\n");
//			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			
		}
		sum /= nsample;
		logger.trace("Evaluated Grad: " + sum + "\n");
	}
	
	public void testIntegrationComp1Dim1Ana() {
		Random rnd = new Random(0);
		short ncomp = 1, ndim = 1;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, RuleType.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, RuleType.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, RuleType.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, RuleType.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, RuleType.LHSPACE, true);	
		
		Component comp01 = ur01.getWeight().getComponent((short) 0);
		Component comp03 = ur03.getWeight().getComponent((short) 0);
		Component comp12 = ur12.getWeight().getComponent((short) 0);
		Component comp32 = ur32.getWeight().getComponent((short) 0);
		Component comp20 = ur20.getWeight().getComponent((short) 0);
		Component comp21 = ur21.getWeight().getComponent((short) 0);
		
		comp01.setWeight(0.5);
		comp03.setWeight(-0.5);
		comp12.setWeight(0.5);
		comp32.setWeight(-0.5);
		comp20.setWeight(0.0);
		comp21.setWeight(0.0);
		
		GaussianDistribution gd01 = comp01.squeeze(RuleUnit.C);
		GaussianDistribution gd03 = comp03.squeeze(RuleUnit.C);
		GaussianDistribution gd12p = comp12.squeeze(RuleUnit.P);
		GaussianDistribution gd12c = comp12.squeeze(RuleUnit.UC);
		GaussianDistribution gd32p = comp32.squeeze(RuleUnit.P);
		GaussianDistribution gd32c = comp32.squeeze(RuleUnit.UC);
		GaussianDistribution gd20 = comp20.squeeze(RuleUnit.P);
		GaussianDistribution gd21 = comp21.squeeze(RuleUnit.P);
		
		double mua = 0.0, mub = 0.0;
		gd01.getMus().set(0, mua);
		gd03.getMus().set(0, mub);
		gd12p.getMus().set(0, mua);
		gd12c.getMus().set(0, mua);
		gd32p.getMus().set(0, mub);
		gd32c.getMus().set(0, mub);
		gd20.getMus().set(0, mua);
		gd21.getMus().set(0, mub);
		
		double std = 3;
		double vara = Math.log(std), varb = Math.log(std);
		gd01.getVars().set(0, -0.25);
		gd03.getVars().set(0, 0.25);
		gd12p.getVars().set(0, -0.25);
		gd12c.getVars().set(0, -0.25);
		gd32p.getVars().set(0, 0.25);
		gd32c.getVars().set(0, 0.25);
		gd20.getVars().set(0, 0.0);
		gd21.getVars().set(0, 0.0);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -0.1);
		GaussianMixture cin12 = ur12.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, RuleUnit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulAndMarginalize(cin20, null, RuleUnit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
		GaussianMixture cin01 = ur01.getWeight().mulAndMarginalize(cin12, null, RuleUnit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + cin01.getWeight(0) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, RuleUnit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulAndMarginalize(cin32, null, RuleUnit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + cin03.getWeight(0) + "\n");
		
		double scoret = cin03.getWeight(0);;
		double scores = FunUtil.logAdd(cin01.getWeight(0), cin03.getWeight(0));
		logger.trace("Score    : " + scores + "\n");
		logger.trace("Loglh    : " + (scoret - scores) + "\t" + Math.exp(scoret - scores) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulAndMarginalize(outor, null, RuleUnit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulAndMarginalize(outx1, null, RuleUnit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulAndMarginalize(outx3, null, RuleUnit.P, true);
		outx2.add(out32, false);
		logger.trace("outx2    :" + outx2 + "\n");
		
		int nsample = 300;
		GaussianMixture outsides = outor;
		GaussianMixture cinsides = cin32;
		GaussianMixture outsidet = outor;
		GaussianMixture cinsidet = cin32;
		
		logger.trace("\n---Counts---\n");
		logger.trace("scoret: " + scoret + "\tscores: " + scores + "\n");
		logger.trace("outsides: " + outsides + "\n");
		logger.trace("outsidet: " + outsidet + "\n");
		logger.trace("cinsides: " + cinsides + "\n");
		logger.trace("cinsidet: " + cinsidet + "\n");
		
		Random rnd1 = new Random(0);
		evalgradientsComp1Dim1Ana(nsample, ur03, outsides, cinsides, outsidet, cinsidet, scoret, scores, rnd1);
	}
	
	public void evalgradientsComp1Dim1Ana(int nsample, GrammarRule rule, GaussianMixture outsides, GaussianMixture cinsides, 
			GaussianMixture outsidet, GaussianMixture cinsidet, double scoret, double scores, Random rnd) {
		double sum = 0.0;
		logger.trace("\nscoret: " + scoret + "\tscores: " + scores + "\n");
		logger.trace("rule: " + rule.getWeight() + "\n\n");
		
		Component icomp = cinsides.getComponent((short) 0);
		GaussianDistribution iscore = icomp.squeeze(RuleUnit.P);
		Component rcomp = rule.getWeight().getComponent((short) 0);
		GaussianDistribution rscore = rcomp.squeeze(RuleUnit.C);
		
		logger.trace(icomp + "\n" + rcomp + "\n");
		
		double factor = Math.exp(icomp.getWeight() + rcomp.getWeight());
		double nn = marginalize(iscore, rscore); 
		logger.trace("cnts: " + nn + "\n");
		nn *= factor;
		double xnn = marginalizex(iscore, rscore, 0);
		logger.trace("xnn : " + xnn + "\n");
		xnn *= factor;
		double xxnn = marginalizexx(iscore, rscore, 0);
		logger.trace("xxnn: " + xxnn + "\n");
		xxnn *= factor;
		
		
		double wgrad = nn / Math.exp(scores) - nn / Math.exp(scoret);
		
		double mu0 = 0, mu1 = 0;
		double var0 = Math.exp(0.25 * 2), var1 = Math.exp(0.25 * 2);
		double vtmp = (var0 * var1) / (var0 + var1);
		xxnn = nn * vtmp / var0 - nn;
		
		double vgrad = xxnn / Math.exp(scores) - xxnn / Math.exp(scoret);
		logger.trace(xxnn + "\t" + vgrad + "\n");
		logger.trace(nn + "\t" + factor + "\n");
		logger.trace("Evaluated Grad: " + wgrad + "\n");
	}
	
	public void derivative(double order0, double order1, double order2) {
		
		
		return;
	}
	
	public double normal(double x) {
		return Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-x * x / 2.0);
	}
	
	public GaussianMixture marginalize(GaussianMixture gm0, GaussianMixture gm1, RuleUnit key, boolean deep) {
		GaussianMixture amixture = gm0.copy(deep);
		// calculating inside score can always remove some portions, but calculating outside score
		// can not, because the rule ROOT->N has the dummy outside score for ROOT (one component but
		// without gaussians) and the rule weight does not contain "P" portion. Here is hardcoding
		if (gm1.components().size() == 1 && gm1.size(0) == 0) {
			return amixture;
		}
		// the following is the general case
		for (Component comp : amixture.components()) {
			double sum = 0.0;
			GaussianDistribution gd = comp.squeeze(key);
			for (Component comp1 : gm1.components()) {
				GaussianDistribution gd1 = comp1.squeeze(null);
				double vcomp = Math.exp(comp1.getWeight()) * marginalize(gd, gd1);
				sum += vcomp;
			}
			// CHECK Math.log(Math.exp(a) * b)
			double weight = Math.exp(comp.getWeight()) * sum;
			comp.setWeight(Math.log(weight));
			comp.getMultivnd().remove(key);
		}
		return amixture;
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
		logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.getWeight() + "\n----------\n");
	}

}
