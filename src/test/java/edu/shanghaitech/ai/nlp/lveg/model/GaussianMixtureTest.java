package edu.shanghaitech.ai.nlp.lveg.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
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
/*	
	@Test
	public void testGaussianMul() {
		Random random = new Random(0);
		short ncomponent = 1, dim = 1;
		GaussianMixture.config((short) -30, 1e-6, 4, ncomponent, 0.8, 2, -0.2, false, random, null);
		GaussianDistribution.config(1, 5, dim, 0.5, 0.8, random, null);
		
		GaussianMixture s_a = new DiagonalGaussianMixture(ncomponent);
		GaussianMixture a_b = new DiagonalGaussianMixture(ncomponent);
		GaussianMixture a_d = new DiagonalGaussianMixture(ncomponent);
		GaussianMixture b_e = new DiagonalGaussianMixture(ncomponent);
		GaussianMixture a_w = new DiagonalGaussianMixture(ncomponent);
		GaussianMixture b_w = new DiagonalGaussianMixture(ncomponent);
		GaussianMixture d_w = new DiagonalGaussianMixture(ncomponent);
		for (int i = 0; i < ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(dim));
			list1.add(new DiagonalGaussianDistribution(dim));
			map.put(RuleUnit.C, list1);
			s_a.add(i, map);
		}
		for (int i = 0; i < ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(dim));
			list1.add(new DiagonalGaussianDistribution(dim));
			map.put(RuleUnit.P, list0);
			map.put(RuleUnit.UC, list0);
			a_b.add(i, map);
		}
		
		for (int i = 0; i < ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(dim));
			list1.add(new DiagonalGaussianDistribution(dim));
			map.put(RuleUnit.P, list0);
			b_e.add(i, map);
		}
		
		for (int i = 0; i < ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(dim));
			list1.add(new DiagonalGaussianDistribution(dim));
			map.put(RuleUnit.P, list0);
			a_w.add(i, map);
		}
		for (int i = 0; i < ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(dim));
			list1.add(new DiagonalGaussianDistribution(dim));
			map.put(RuleUnit.P, list0);
			b_w.add(i, map);
		}
		
		System.out.println("\n\n");
		System.out.println("s_a---" + s_a);
		System.out.println("a_b---" + a_b);
		System.out.println("b_e---" + b_e);
		System.out.println("a_w---" + a_w);
		System.out.println("b_w---" + b_w);
		System.out.println();
		
		GaussianMixture i_a = s_a.copy(true);
		GaussianMixture i_a_w = a_w.mul(i_a, null, RuleUnit.P);
		GaussianMixture i_b = a_b.mulAndMarginalize(i_a_w, null, RuleUnit.P, true);
		GaussianMixture i_d = i_b.copy(true);
		GaussianMixture i_b_w = b_w.mul(i_b, null, RuleUnit.P);
		GaussianMixture i_d_w = i_b_w.copy(true);
		
		System.out.println("i_a---" + i_a);
		System.out.println("i_a_w-" + i_a_w);
		System.out.println("i_b---" + i_b);
		System.out.println("i_b_w-" + i_b_w);
		System.out.println("i_d---" + i_d);
		System.out.println("i_d_w-" + i_d_w);
		
		GaussianMixture i_b_e = b_e.mulAndMarginalize(i_b_w, null, RuleUnit.P, true);
		double scoreT = i_b_e.evalInsideOutside(null, false);
		
		GaussianMixture d_e = b_e.copy(true);
		GaussianMixture i_d_e = d_e.mulAndMarginalize(i_d_w, null, RuleUnit.P, true); 
		GaussianMixture i_e_all = i_b_e.copy(true);
		i_e_all.add(i_d_e, false);
		double scoreS = i_e_all.evalInsideOutside(null, false);
		
		System.out.println("i_b_e---" + i_b_e);
		System.out.println("i_d_e---" + i_d_e);
		System.out.print("\n");
		
		System.out.println("i_e---" + i_b_e);
		System.out.println("i_e_all---" + i_e_all);
		
		System.out.println("score of T: " + scoreT + "\t" + Math.log(scoreT));
		System.out.println("score of S: " + scoreS + "\t" + Math.log(scoreS));
		
		List<List<Double>> cache = new ArrayList<>(5);
		for (int i = 0; i < 4; i++) {
			cache.add(new ArrayList<>(5));
		}
		Component comp_w = i_d_w.getComponent((short) 0);
		GaussianDistribution gd_w = comp_w.squeeze(null);
		
		Component comp_e = d_e.getComponent((short) 0);
		GaussianDistribution gd_e = comp_e.squeeze(null);
		
		System.out.println("gd_w " + gd_w);
		System.out.println("gd_e " + gd_e);
		
		cache.get(3).add(comp_w.weight);
		double val = gd_e.integral(gd_w, cache);
		System.out.println(cache);
		System.out.println(val);
		
		double nn = cache.get(0).get(0);
		double nnx = cache.get(1).get(0);
		double nnxx = cache.get(2).get(0);
		double mxw = cache.get(3).get(0);
		
		GaussianMixture gnn = d_e.mul(i_d_w, null, RuleUnit.P);
		System.out.println("gnn " + gnn);
		
		
		double v0 = 0.5, v1 = 1;
		double vr0 = v0 * v0, vr1 = v1 * v1;
		double a = (vr0 + vr1) / (2 * vr0 * vr1);
		double b = 0;
		double a_nn = Math.exp(i_d_e.eval(null, true));
		double a_nnx = b / (2 * a) * a_nn;
		double a_nnxx = (b * b + 2 * a) / (4 * a * a) * a_nn;
		
		System.out.println("a_nn: " + a_nn + "\ta_nnx: " + a_nnx + "\ta_nnxx: " + a_nnxx);
		System.out.println(" _nn: " + nn + "\t _nnx: " + nnx + "\t _nnxx: " + nnxx);
		double mm = val - Math.log(2 * Math.PI) * (-dim / 2.0);
		double pi = 1 / Math.sqrt(2 * Math.PI) * Math.exp(mxw);
		System.out.println(mm + "\t" + pi + "\t" + (1 / Math.sqrt(2 * Math.PI)));
		double b_nn = Math.exp(nn) * pi;
		double b_nnx = nnx * pi;
		double b_nnxx = Math.exp(nnxx) * pi;
		System.out.println("b_nn: " + b_nn + "\tb_nnx: " + b_nnx + "\tb_nnxx: " + b_nnxx);
		
		double dw = b_nn / scoreS;
		double du = 0;
		double dv = (b_nnxx - b_nn) / scoreS;
		System.out.println(dw + "\t" + dv);
	}
*/	

	
	
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
