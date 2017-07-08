package edu.shanghaitech.ai.nlp.lveg.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;

public class GaussianMixtureTest {
	
	protected static GaussianMixture gm0;
	protected static GaussianMixture gm1;
	protected static GaussianMixture gm2;
	protected static GaussianMixture gm3;
	
	static short ncomp = 2, ndim = 2;
	
	
	@Test 
	public void testmulAndMarginalize() {
		GaussianMixture cgm0 = gm0.copy(true);
		GaussianMixture cgm2 = gm2.copy(true);
		
		System.out.println("\n---multiplication and marginalization---\n");
		GaussianMixture gm4 = cgm0.mulAndMarginalize(cgm2, null, RuleUnit.UC, true);
		System.out.println("InScore uc--" + gm4);
		System.out.println("InScore view--" + gm4.spviews);
		
		System.out.println("\n---deep copy---\n");
		System.out.println(cgm0);
		System.out.println(gm4);
		
		System.out.println("\n---shallow copy---\n");
		cgm0 = gm0.copy(true);
		System.out.println(cgm0);
		GaussianMixture gm5 = cgm0.mulAndMarginalize(cgm2, null, RuleUnit.UC, false);
		System.out.println(gm5);
		System.out.println(cgm0);
		System.out.println(cgm0.spviews);
		
		GaussianMixture cgm3 = gm3.copy(true);
		System.out.println("---cgm3---" + cgm3);
		System.out.println("---cgm3 view---" + cgm3.spviews);
		GaussianMixture gm6 = cgm3.mulAndMarginalize(cgm2, null, RuleUnit.P, true);
		
		System.out.println("\n---binary rule weight---\n");
		System.out.println("---gm6---" + gm6);
		System.out.println("---gm6 view---" + gm6.spviews);
		GaussianMixture gm7 = gm6.mulAndMarginalize(cgm2, null, RuleUnit.LC, false);
		System.out.println("---gm7---" + gm7);
		System.out.println("---gm7 view---" + gm7.spviews);
		System.out.println("---gm6---" + gm6);
		System.out.println("---gm6 view---" + gm6.spviews);
		GaussianMixture gm8 = gm6.mulAndMarginalize(cgm2, null, RuleUnit.RC, false);
		System.out.println("---gm8---" + gm8);
		System.out.println("---gm8 view---" + gm8.spviews);
		System.out.println("---gm6---" + gm6);
		System.out.println("---gm6 view---" + gm6.spviews);
	}
	
	
	static {
		int ncomponent;
		Random rnd = new Random(0);
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		gm0 = new DiagonalGaussianMixture(ncomp);
		gm1 = new DiagonalGaussianMixture(ncomp);
		gm2 = new DiagonalGaussianMixture(ncomp);
		gm3 = new DiagonalGaussianMixture(ncomp);
		
		
		List<GaussianDistribution> list0 = new ArrayList<>(5);
		List<GaussianDistribution> list1 = new ArrayList<>(5);
		for (int i = 0; i < ncomp; i++) {
			list0.add(new DiagonalGaussianDistribution(ndim));
			list1.add(new DiagonalGaussianDistribution(ndim));
		}
		gm0.add(RuleUnit.P, list0);
		gm0.add(RuleUnit.UC, list1);
		ncomponent = ncomp * ncomp;
		gm0.initMixingW(ncomponent);
		gm0.setBinding(RuleType.LRURULE);
		
		System.out.println("gm0---" + gm0);
		
		
		list0 = new ArrayList<>(5);
		list1 = new ArrayList<>(5);
		for (int i = 0; i < ncomp; i++) {
			list0.add(new DiagonalGaussianDistribution(ndim));
			list1.add(new DiagonalGaussianDistribution(ndim));
		}
		gm1.add(RuleUnit.P, list0);
		gm1.add(RuleUnit.UC, list1);
		ncomponent = ncomp * ncomp;
		gm1.initMixingW(ncomponent);
		gm1.setBinding(RuleType.LRURULE);
		gm1.buildSimpleView();
		
		System.out.println("gm1---" + gm1);
		
		
		List<GaussianDistribution> list = new ArrayList<>(5);
		for (int i = 0; i < ncomp; i++) {
			list.add(new DiagonalGaussianDistribution(ndim));
		}
		gm2.add(RuleUnit.P, list);
		ncomponent = ncomp;
		gm2.initMixingW(ncomponent);
		gm2.setBinding(RuleType.LHSPACE);
		gm2.buildSimpleView();
		
		System.out.println("gm2---" + gm2);
		System.out.println("gm2 view---" + gm2.spviews);
		
		
		list = new ArrayList<>(5);
		list0 = new ArrayList<>(5);
		list1 = new ArrayList<>(5);
		for (int i = 0; i < ncomp; i++) {
			list.add(new DiagonalGaussianDistribution(ndim));
			list0.add(new DiagonalGaussianDistribution(ndim));
			list1.add(new DiagonalGaussianDistribution(ndim));
		}
		gm3.add(RuleUnit.P, list);
		gm3.add(RuleUnit.LC, list0);
		gm3.add(RuleUnit.RC, list1);
		ncomponent = ncomp * ncomp * ncomp;
		gm3.initMixingW(ncomponent);
		gm3.setBinding(RuleType.LRBRULE);
		
		System.out.println("gm3---" + gm3);
	}
	
}
