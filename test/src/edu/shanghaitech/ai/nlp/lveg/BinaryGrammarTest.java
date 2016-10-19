package edu.shanghaitech.ai.nlp.lveg;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.GrammarRule.Unit;

public class BinaryGrammarTest {
	@Test
	public void testBinaryGrammarTest() {
		GaussianMixture gm = new GaussianMixture(LVeGLearner.ncomponent);
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list2 = new HashSet<GaussianDistribution>();
			list0.add(new GaussianDistribution(LVeGLearner.dim));
			list1.add(new GaussianDistribution(LVeGLearner.dim));
			list2.add(new GaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, list0);
			map.put(Unit.LC, list1);
			map.put(Unit.RC, list2);
			gm.add(i, map);
		}
		System.out.println(gm);
	}
}
