package edu.shanghaitech.ai.nlp.lveg;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.GrammarRule.Unit;

public class GaussianMixtureTest {
	
	protected static GaussianMixture gm0 = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
	protected static GaussianMixture gm1 = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
	protected static GaussianMixture gm2 = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
	
	static {
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
		
		gd0.dim = n;
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
	
	
	//@Test 
	public void testMultiplyForInsideScore() {
		GaussianMixture cgm0 = gm0.copy(true);
		GaussianMixture cgm2 = gm2.copy(true);
		
		System.out.println("---multiplication and marginalization---");
		GaussianMixture gm3 = cgm0.mulForInsideOutside(cgm2, GrammarRule.Unit.UC, true);
		System.out.println("InScore uc--" + gm3);
		
		System.out.println("---deep copy---");
		cgm0.mixture.remove(0);
		System.out.println(gm3);
		
		System.out.println("---shallow copy---");
		cgm0 = gm0.copy(true);
		System.out.println(cgm0);
		GaussianMixture gm5 = cgm0.mulForInsideOutside(cgm2, GrammarRule.Unit.UC, false);
		System.out.println(gm5);
		System.out.println(cgm0);
	}
	
	
	//@Test
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
	
	
	@Test
	public void testCast() {
		Short x = 3;
		Object xo = (Object) x;
		System.out.println(xo);
	}
	
}
