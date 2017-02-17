package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
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
		testIntegrationComp2Dim2();
	}
	
	
	public void testIntegrationComp2Dim2() {
		Random rnd = new Random(0);
		short ncomp = 2, ndim = 2;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, GrammarRule.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, GrammarRule.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, GrammarRule.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, GrammarRule.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -1e-3);
		GaussianMixture cin12 = ur12.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, GrammarRule.Unit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
//		ur01.getWeight().setWeight(0, 1e-3);
		
		GaussianMixture cin01 = ur01.getWeight().mulForInsideOutside(cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + FunUtil.logAdd(cin01.getWeight(0), cin01.getWeight(1)) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulForInsideOutside(cin32, GrammarRule.Unit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + FunUtil.logAdd(cin03.getWeight(0), cin03.getWeight(1)) + "\n");
		
		logger.trace("Score    : " + FunUtil.logAdd(cin01.marginalize(true), cin03.marginalize(true)) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulForInsideOutside(outx1, GrammarRule.Unit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulForInsideOutside(outx3, GrammarRule.Unit.P, true);
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
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, GrammarRule.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, GrammarRule.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, GrammarRule.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, GrammarRule.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -1e-3);
		GaussianMixture cin12 = ur12.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, GrammarRule.Unit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
//		ur01.getWeight().setWeight(0, 1e-3);
		
		GaussianMixture cin01 = ur01.getWeight().mulForInsideOutside(cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + FunUtil.logAdd(cin01.getWeight(0), cin01.getWeight(1)) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulForInsideOutside(cin32, GrammarRule.Unit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + FunUtil.logAdd(cin03.getWeight(0), cin03.getWeight(1)) + "\n");
		
		logger.trace("Score    : " + FunUtil.logAdd(cin01.marginalize(true), cin03.marginalize(true)) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulForInsideOutside(outx1, GrammarRule.Unit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulForInsideOutside(outx3, GrammarRule.Unit.P, true);
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
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, GrammarRule.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, GrammarRule.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, GrammarRule.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, GrammarRule.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -1e-3);
		GaussianMixture cin12 = ur12.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, GrammarRule.Unit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
		ur01.getWeight().setWeight(0, 1e-3);
		
		GaussianMixture cin01 = ur01.getWeight().mulForInsideOutside(cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + cin01.getWeight(0) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulForInsideOutside(cin32, GrammarRule.Unit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + cin03.getWeight(0) + "\n");
		
		logger.trace("Score    : " + FunUtil.logAdd(cin01.getWeight(0), cin03.getWeight(0)) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulForInsideOutside(outx1, GrammarRule.Unit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulForInsideOutside(outx3, GrammarRule.Unit.P, true);
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
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		GrammarRule ur01 = new UnaryGrammarRule((short) 0, (short) 1, GrammarRule.RHSPACE, true);	
		GrammarRule ur03 = new UnaryGrammarRule((short) 0, (short) 3, GrammarRule.RHSPACE, true);	
		GrammarRule ur12 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur32 = new UnaryGrammarRule((short) 3, (short) 2, GrammarRule.LRURULE, true);	
		GrammarRule ur20 = new UnaryGrammarRule((short) 2, (short) 0, GrammarRule.LHSPACE, true);	
		GrammarRule ur21 = new UnaryGrammarRule((short) 2, (short) 1, GrammarRule.LHSPACE, true);	
		
		printRule(ur01);
		printRule(ur03);
		printRule(ur12);
		printRule(ur32);
		printRule(ur20);
		printRule(ur21);
		
		logger.trace("\n---Inside Score---\n");
		GaussianMixture cin20 = ur20.getWeight().copy(true);
//		cin20.setWeight(0, -0.1);
		GaussianMixture cin12 = ur12.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true); // in logarithm
		logger.trace("cin12    : " + cin12 + "\n");
		GaussianMixture cin12copy = marginalize(ur12.getWeight(), cin20, GrammarRule.Unit.UC, true);    // in the normal way
		logger.trace("cin12copy: " + cin12copy + "\n");
		
		GaussianMixture cin32 = ur32.getWeight().mulForInsideOutside(cin20, GrammarRule.Unit.UC, true);
		logger.trace("cin32    : " + cin32 + "\n");
		
		GaussianMixture cin01 = ur01.getWeight().mulForInsideOutside(cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01    : " + cin01 + "\t" + cin01.getWeight(0) + "\n");
		GaussianMixture cin01copy = marginalize(ur01.getWeight(), cin12, GrammarRule.Unit.C, true);
		logger.trace("cin01copy: " + cin01copy + "\n");
		GaussianMixture cin03 = ur03.getWeight().mulForInsideOutside(cin32, GrammarRule.Unit.C, true);
		logger.trace("cin03    : " + cin03 + "\t" + cin01.getWeight(0) + "\n");
		
		logger.trace("Score    : " + FunUtil.logAdd(cin01.getWeight(0), cin03.getWeight(0)) + "\n");
		
		logger.trace("\n---Outside Score---\n");
		GaussianMixture outor = new DiagonalGaussianMixture((short) 1);
		outor.marginalizeToOne();
		logger.trace("outor    :" + outor + "\n");
		
		GaussianMixture outx1 = ur01.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx1    :" + outx1 + "\n");
		GaussianMixture outx3 = ur03.getWeight().mulForInsideOutside(outor, GrammarRule.Unit.P, true);
		logger.trace("outx3    :" + outx3 + "\n");
		GaussianMixture outx2 = ur12.getWeight().mulForInsideOutside(outx1, GrammarRule.Unit.P, true);
		GaussianMixture out32 = ur32.getWeight().mulForInsideOutside(outx3, GrammarRule.Unit.P, true);
		outx2.add(out32, false);
		logger.trace("outx2    :" + outx2 + "\n");
		
		int nsample = 300;
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
		
		for (int i = 0; i < nsample; i++) {
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
			
			
			/*
//			double snorm = snorms[i];
			double snorm = rnd.nextGaussian(); // sample from N(0, 1)
			double vnorm = Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-snorm * snorm / 2.0);
			logger.trace("snorm: " + snorm + "\n");
			logger.trace("vnorm: " + vnorm + "\t" + Math.log(vnorm) + "\n");
			
			double dW = Math.exp(0) * vnorm;
			logger.trace("dW   : " + dW + "\t" + Math.log(dW) + "\n");
			
			double in = Math.exp(cinsides.getWeight(0)) * vnorm;
			logger.trace("in   : " + in + "\tw: " + Math.exp(cinsides.getWeight(0)) + "\n");
			
			double dR = Math.exp(Math.log(in) - scores) - Math.exp(Math.log(0) - scoret);
			logger.trace("dR   : " + dR + "\n");
			logger.trace("dWdR : " + dW * dR + "\n\n");
			sum += dW * dR;
			*/
		}
		sum /= nsample;
		logger.trace("Evaluated Grad: " + sum + "\n");
	}
	
	public double normal(double x) {
		return Math.pow(2 * Math.PI * 1, -0.5) * Math.exp(-x * x / 2.0);
	}
	
	public GaussianMixture marginalize(GaussianMixture gm0, GaussianMixture gm1, String key, boolean deep) {
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
			double mtmp = -Math.pow(mus0.get(i) - mus1.get(i), 2);
			double vtmp = 2 * (Math.exp(vars0.get(i) * 2) + Math.exp(vars1.get(i) * 2)) + epsilon;
			vdims *= Math.pow(vtmp * Math.PI, -0.5) * Math.exp(mtmp / vtmp);
		}
		return vdims;
	}
	
	
	public void printRule(GrammarRule rule) {
		logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.getWeight() + "\n----------\n");
	}

}
