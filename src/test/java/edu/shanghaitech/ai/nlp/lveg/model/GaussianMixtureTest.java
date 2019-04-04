package edu.shanghaitech.ai.nlp.lveg.model;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.util.FunUtil;

public class GaussianMixtureTest {
	
	protected static GaussianMixture gm0;
	protected static GaussianMixture gm1;
	protected static GaussianMixture gm2;
	
	static {
		Random rnd = new Random(0);
		short ncomp = 2, ndim = 2;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		gm0 = new DiagonalGaussianMixture(LVeGTrainer.ncomponent);
		gm1 = new DiagonalGaussianMixture(LVeGTrainer.ncomponent);
		gm2 = new DiagonalGaussianMixture(LVeGTrainer.ncomponent);
		
		for (int i = 0; i < LVeGTrainer.ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			list1.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			map.put(RuleUnit.P, list0);
			map.put(RuleUnit.UC, list1);
			gm0.add(i, map);
		}
		System.out.println("gm0---" + gm0);
		
		for (int i = 0; i < LVeGTrainer.ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			list1.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			map.put(RuleUnit.P, list0);
			map.put(RuleUnit.UC, list1);
			gm1.add(i, map);
		}
		System.out.println("gm1---" + gm1);
		
		for (int i = 0; i < LVeGTrainer.ncomponent; i++) {
			Set<GaussianDistribution> list = new HashSet<GaussianDistribution>();
			list.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			gm2.add(i, RuleUnit.UC, list);
		}
		System.out.println("gm2---" + gm2);
	}
	
	
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
	
	
	//@Test
	public void testGaussianMixture() {
		String key = "hello", value = "world";
		Map<String, String> map = new HashMap<String, String>();
		map.put(key, value);
		System.out.println(map);
		key = "hi";
		System.out.println(map);
		testParams(map, key, value);
		System.out.println(map);
		key = "hallo";
		System.out.println(map);
	}
	
	
	public void testParams(Map<String, String> map, String key, String value) {
		map.put(key, value);
	}
	
	//@Test
	public void testHashSet() {
		
		System.out.println("\n---hash set equality test---\n");
		
		Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
		Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();
		short n = 3;
		GaussianDistribution gd0 = new DiagonalGaussianDistribution(LVeGTrainer.dim);
		GaussianDistribution gd1 = new DiagonalGaussianDistribution(LVeGTrainer.dim);
		GaussianDistribution gd2 = gd0.copy();
		
		set0.add(gd0);
		set0.add(gd1);
		
		set1.add(gd1.copy());
		set1.add(gd0);
		/*
		assertFalse(set0 == set1);
		assertTrue(set0.equals(set1));
		assertTrue(set1.contains(gd0));
		assertTrue(set1.contains(gd1));
		assertTrue(set0.containsAll(set1));
		assertTrue(set1.containsAll(set0));
		
		
		System.out.println("set0---" + set0);
		System.out.println("set1---" + set1);
		
		System.out.println("set0: " + set0.hashCode());
		for (GaussianDistribution gd : set0) {
			System.out.println(gd.hashCode() + "\t" + set1.contains(gd));
		}
		
		System.out.println("set1: " + set1.hashCode());
		for (GaussianDistribution gd : set1) {
			System.out.println(gd.hashCode() + "\t" + set0.contains(gd));
		}
		*/
		
		System.out.println("----------------------------");
		System.out.println("gd0: " + gd0.hashCode() + "\ngd1: " + gd1.hashCode() + "\ngd2: " + gd2.hashCode());
		System.out.println("----------------------------");
		
		gd0.setDim(n);
		System.out.println("set0---" + set0);
		System.out.println("set1---" + set1);
		
		for (GaussianDistribution gdd0 : set0) {
			for (GaussianDistribution gdd1 : set1) {
				System.out.println("--\n0->" + gdd0 + "\n1->" + gdd1);
				System.out.println("hash: " + (gdd0.hashCode() == gdd1.hashCode()) + "\tequals: " + gdd0.equals(gdd1));
				System.out.println("gdd1 in set0: " + set0.contains(gdd1));
			}
			System.out.println("gdd0 in set1: " + set1.contains(gdd0));
		}
		
		System.out.println("\n---the same value with different hash code---\ngd0 in set0: " + set0.contains(gd0) + "\tgd0 in set1: " + set1.contains(gd0));
		System.out.println("\n---the same hash code with different value---\ngd2 in set0: " + set0.contains(gd2) + "\tgd2 in set1: " + set1.contains(gd2));
		
		set0.add(gd2);
		set1.add(gd2);

		System.out.println("set0---" + set0);
		System.out.println("set1---" + set1);
		
		/*
		assertTrue(set0.containsAll(set1));
		assertTrue(set1.containsAll(set0));
		assertTrue(set0.equals(set1));
		*/
	}
	
	
	//@Test
	public void testMapEqual() {
		// TODO not done, discarded
		String s1 = new String("123");
		String s2 = "123";
		assertTrue(s1.equals(s2));
		assertTrue(s1.hashCode() == s2.hashCode());
	}
	
	
//	@Test
	public void testPriorityQueue() {
		GaussianMixture gmc = gm2.copy(true);
		System.out.println("gmc---" + gmc);
		
		System.out.println("addmap-gmc---" + gmc);
		EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
		Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
		list0.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
		map.put(RuleUnit.P, list0);
		gmc.add(0.0, map);
		
		List<Component> components = gmc.components();
		System.out.println("after adding-" + components.size());
		for (Component com : components) {
			System.out.println(com);
		}
//		while(!components.isEmpty()) {
//			System.out.println(components.poll());
//		}

		
		Component comp = gmc.getComponent((short) 1);
		System.out.println("get the 1st component---" + comp);
		
		comp.setWeight(-100);
		System.out.println("after setting the 1st mixing weight---" + comp);
		for (Component com : components) {
			System.out.println(com);
		}
//		while(!components.isEmpty()) {
//			System.out.println(components.poll());
//		}
		
//		components = gmc.sort();
//		while(!components.isEmpty()) {
//			System.out.println(components.poll());
//		}		
		
		
		
		
		
		
		
/*		
		int cnt = 0;
		for (Component com : components) {
			if (com.getWeight() > 0) {
				components.remove(com);
			}
			System.out.println("# " + ++cnt);
			// ConcurrentModificationException
		}
*/		
	}
	
	
	
//	@Test 
	public void testMultiplyForInsideScore() {
		GaussianMixture cgm0 = gm0.copy(true);
		GaussianMixture cgm2 = gm2.copy(true);
		
		System.out.println("---multiplication and marginalization---");
		GaussianMixture gm3 = cgm0.mulAndMarginalize(cgm2, null, RuleUnit.UC, true);
		System.out.println("InScore uc--" + gm3);
		
		System.out.println("---deep copy---");
//		cgm0.getMixture().remove(0);
		System.out.println(gm3);
		
		System.out.println("---shallow copy---");
		cgm0 = gm0.copy(true);
		System.out.println(cgm0);
		GaussianMixture gm5 = cgm0.mulAndMarginalize(cgm2, null, RuleUnit.UC, false);
		System.out.println(gm5);
		System.out.println(cgm0);
	}
	
	
//	@Test
	public void testMultiply() {
		
		GaussianMixture gm3 = gm0.multiply(gm1);
		System.out.println("gm0 X gm1---" + gm3);
		
		EnumSet<RuleUnit> keys = EnumSet.noneOf(RuleUnit.class);
		keys.add(RuleUnit.UC);
		
		gm3.marginalize(keys);
		System.out.println("Remove uc---" + gm3);
		
		GaussianMixture gm5 = GaussianMixture.merge(gm3);
		System.out.println("Merge  p ---" + gm5);
		
//		assertTrue(gm3.equals(gm4));
		
		GaussianMixture gm4 = gm0.multiply(gm2);
		System.out.println("gm0 X gm2---" + gm4);
		
		GaussianMixture gm8 = gm4.replaceAllKeys(RuleUnit.UC);
		System.out.println("Repwith uc--" + gm8);

		gm4.marginalize(keys);
		System.out.println("Remove uc---" + gm4);
		
		GaussianMixture gm6 = GaussianMixture.merge(gm4);
		System.out.println("Merge  p ---" + gm6);
		

		EnumMap<RuleUnit, RuleUnit> keys0 = new EnumMap<>(RuleUnit.class);
		EnumMap<RuleUnit, RuleUnit> keys1 = new EnumMap<>(RuleUnit.class);
		keys0.put(RuleUnit.UC, RuleUnit.RM);
		keys1.put(RuleUnit.P, RuleUnit.RM);
		GaussianMixture gm7 = GaussianMixture.mulAndMarginalize(gm0, gm1, keys0, keys1);
		System.out.println("MulAndMarginalize gm0 X gm1---" + gm7);
		
	}
	
	
//	@Test
	public void testCast() {
		System.out.println("---Efficiency Comparison---");
		
		int n = 1500;
		List<Double> weights = new ArrayList<Double>();
		FunUtil.randomInitList(LVeGTrainer.random, weights, Double.class, n, 10, 0.5, false, true);
//		gm0.setWeights(weights);
		
		System.out.println("n = " + n);
		long start = System.currentTimeMillis();
		double margin = gm0.marginalize(false);
		double matlog = Math.log(margin);
		long time = System.currentTimeMillis() - start;
		System.out.println("Math.exp: " + time);
		
		start = System.currentTimeMillis();
		double logadd = gm0.marginalize(true);
		time = System.currentTimeMillis() - start;
		System.out.println("MyLogadd: " + time);
		
		double diff = logadd - matlog;
		System.out.println("margin = " + margin + ", log margin = " + matlog + ", log add = " + logadd + ", diff = " + diff);
		
		/**
		 * n = 15000000
		 * Math.exp: 732
		 * MyLogadd: 1207
		 * margin = 3.3021925464538197E10, log margin = 24.22043758580791, log add = 24.22040516466863, diff = -3.2421139280813804E-5
		 * 
		 * n = 1500000
		 * Math.exp: 76
		 * MyLogadd: 131
		 * margin = 3.308101687207093E9, log margin = 21.91964035341989, log add = 21.919640353396428, diff = -2.3462121134798508E-11
		 * 
		 * n = 15000
		 * Math.exp: 12
		 * MyLogadd: 17
		 * margin = 3.3214069278252375E8, log margin = 19.621069210555493, log add = 19.621069210554907, diff = -5.861977570020827E-13
		 * 
		 * n = 15000
		 * Math.exp: 2
		 * MyLogadd: 3
		 * margin = 3.3221488701829065E7, log margin = 17.318707474565834, log add = 17.31870747456588, diff = 4.618527782440651E-14
		 * 
		 * n = 1500
		 * Math.exp: 0
		 * MyLogadd: 1
		 * margin = 3228167.3812916544, log margin = 14.987425159968073, log add = 14.987425159968073, diff = 0.0
		 */
	}
	
}
