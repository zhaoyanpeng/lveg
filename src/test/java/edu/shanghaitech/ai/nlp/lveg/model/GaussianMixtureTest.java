package edu.shanghaitech.ai.nlp.lveg.model;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.Unit;
import edu.shanghaitech.ai.nlp.util.FunUtil;

public class GaussianMixtureTest {
	
	protected static GaussianMixture gm0;
	protected static GaussianMixture gm1;
	protected static GaussianMixture gm2;
	
	static {
		Random rnd = new Random(0);
		short ncomp = 2, ndim = 2;
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
		
		gm0 = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		gm1 = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		gm2 = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			list1.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, list0);
			map.put(Unit.UC, list1);
			gm0.add(i, map);
		}
		System.out.println("gm0---" + gm0);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			list1.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, list0);
			map.put(Unit.UC, list1);
			gm1.add(i, map);
		}
		System.out.println("gm1---" + gm1);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Set<GaussianDistribution> list = new HashSet<GaussianDistribution>();
			list.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			gm2.add(i, Unit.UC, list);
		}
		System.out.println("gm2---" + gm2);
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
		GaussianDistribution gd0 = new DiagonalGaussianDistribution(LVeGLearner.dim);
		GaussianDistribution gd1 = new DiagonalGaussianDistribution(LVeGLearner.dim);
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
		Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
		Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
		list0.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
		map.put(Unit.P, list0);
		gmc.add(0.0, map);
		
		PriorityQueue<Component> components = gmc.components();
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
		GaussianMixture gm3 = cgm0.mulForInsideOutside(cgm2, GrammarRule.Unit.UC, true);
		System.out.println("InScore uc--" + gm3);
		
		System.out.println("---deep copy---");
//		cgm0.getMixture().remove(0);
		System.out.println(gm3);
		
		System.out.println("---shallow copy---");
		cgm0 = gm0.copy(true);
		System.out.println(cgm0);
		GaussianMixture gm5 = cgm0.mulForInsideOutside(cgm2, GrammarRule.Unit.UC, false);
		System.out.println(gm5);
		System.out.println(cgm0);
	}
	
	
	@Test
	public void testMultiply() {
		
		GaussianMixture gm3 = gm0.multiply(gm1);
		System.out.println("gm0 X gm1---" + gm3);
		
		Set<String> keys = new HashSet<String>();
		keys.add("uc");
		
		gm3.marginalize(keys);
		System.out.println("Remove uc---" + gm3);
		
		GaussianMixture gm5 = GaussianMixture.merge(gm3);
		System.out.println("Merge  p ---" + gm5);
		
//		assertTrue(gm3.equals(gm4));
		
		GaussianMixture gm4 = gm0.multiply(gm2);
		System.out.println("gm0 X gm2---" + gm4);
		
		GaussianMixture gm8 = gm4.replaceAllKeys("uc");
		System.out.println("Repwith uc--" + gm8);

		gm4.marginalize(keys);
		System.out.println("Remove uc---" + gm4);
		
		GaussianMixture gm6 = GaussianMixture.merge(gm4);
		System.out.println("Merge  p ---" + gm6);
		

		Map<String, String> keys0 = new HashMap<String, String>();
		Map<String, String> keys1 = new HashMap<String, String>();
		keys0.put("uc", "rm");
		keys1.put("p", "rm");
		GaussianMixture gm7 = GaussianMixture.mulAndMarginalize(gm0, gm1, keys0, keys1);
		System.out.println("MulAndMarginalize gm0 X gm1---" + gm7);
		
	}
	
	
//	@Test
	public void testCast() {
		System.out.println("---Efficiency Comparison---");
		
		int n = 1500;
		List<Double> weights = new ArrayList<Double>();
		FunUtil.randomInitList(LVeGLearner.random, weights, Double.class, n, 10, 0.5, false, true);
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
