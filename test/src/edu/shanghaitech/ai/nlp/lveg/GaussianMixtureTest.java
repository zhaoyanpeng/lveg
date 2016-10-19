package edu.shanghaitech.ai.nlp.lveg;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.GrammarRule.Unit;

public class GaussianMixtureTest {
	
	protected static GaussianMixture gm0 = new GaussianMixture(LVeGLearner.ncomponent);
	protected static GaussianMixture gm1 = new GaussianMixture(LVeGLearner.ncomponent);
	protected static GaussianMixture gm2 = new GaussianMixture(LVeGLearner.ncomponent);
	
	static {
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new GaussianDistribution(LVeGLearner.dim));
			list1.add(new GaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, list0);
			map.put(Unit.UC, list1);
			gm0.add(i, map);
		}
		System.out.println("gm0---" + gm0);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			list0.add(new GaussianDistribution(LVeGLearner.dim));
			list1.add(new GaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, list0);
			map.put(Unit.UC, list1);
			gm1.add(i, map);
		}
		System.out.println("gm1---" + gm1);
		
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Set<GaussianDistribution> list = new HashSet<GaussianDistribution>();
			list.add(new GaussianDistribution(LVeGLearner.dim));
			gm2.add(i, Unit.UC, list);
		}
		System.out.println("gm2---" + gm2);
	}
	
	
	// @Test
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
	
	
	@Test
	public void testHashSet() {
		Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
		Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();

		GaussianDistribution gd0 = new GaussianDistribution(LVeGLearner.dim);
		GaussianDistribution gd1 = new GaussianDistribution(LVeGLearner.dim);
		set0.add(gd0);
		set0.add(gd1);

		set1.add(gd1.copy());
		set1.add(gd0.copy());
		
		System.out.println("set0---" + set0);
		System.out.println("set1---" + set1);
		
		assertFalse(set0 == set1);
		assertTrue(set0.equals(set1));
	}
	
	
	@Test
	public void testMapEqual() {
		// TODO not done, discarded
		Map<String, Set<GaussianDistribution>> map0 = new HashMap<String, Set<GaussianDistribution>>();
		Map<String, Set<GaussianDistribution>> map1 = new HashMap<String, Set<GaussianDistribution>>();
		Map<String, Set<GaussianDistribution>> map2 = new HashMap<String, Set<GaussianDistribution>>();
	}
	
	
	@Test 
	public void testMultiplyForInsideScore() {
		GaussianMixture cgm0 = gm0.copy(true);
		GaussianMixture cgm2 = gm2.copy(true);
		
		System.out.println("---multiplication and marginalization---");
		GaussianMixture gm3 = cgm0.multiplyForInsideScore(cgm2, GrammarRule.Unit.UC, true);
		System.out.println("InScore uc--" + gm3);
		
		System.out.println("---deep copy---");
		cgm0.mixture.remove(0);
		System.out.println(gm3);
		
		System.out.println("---shallow copy---");
		cgm0 = gm0.copy(true);
		System.out.println(cgm0);
		GaussianMixture gm5 = cgm0.multiplyForInsideScore(cgm2, GrammarRule.Unit.UC, false);
		System.out.println(gm5);
		System.out.println(cgm0);
	}
	
	
	@Test
	public void testMultiply() {
		
		GaussianMixture gm3 = GaussianMixture.multiply(gm0, gm1);
		System.out.println("gm0 X gm1---" + gm3);
		
		List<String> keys = new ArrayList<String>();
		keys.add("uc");
		
		GaussianMixture.marginalize(gm3, keys);
		System.out.println("Remove uc---" + gm3);
		
		GaussianMixture gm5 = GaussianMixture.merge(gm3);
		System.out.println("Merge  p ---" + gm5);
		
//		assertTrue(gm3.equals(gm4));
		
		GaussianMixture gm4 = GaussianMixture.multiply(gm0, gm2);
		System.out.println("gm0 X gm2---" + gm4);
		
		GaussianMixture gm8 = GaussianMixture.replaceKeys(gm4, "uc");
		System.out.println("Repwith uc--" + gm8);

		GaussianMixture.marginalize(gm4, keys);
		System.out.println("Remove uc---" + gm4);
		
		GaussianMixture gm6 = GaussianMixture.merge(gm4);
		System.out.println("Merge  p ---" + gm6);
	}
	
}
