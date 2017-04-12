package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.Unit;

public class BinaryGrammarTest {
	@Test
	public void testBinaryGrammarTest() {
		GaussianMixture gm = new DiagonalGaussianMixture(LVeGTrainer.ncomponent);
		for (int i = 0; i < LVeGTrainer.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> list0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list1 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> list2 = new HashSet<GaussianDistribution>();
			list0.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			list1.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			list2.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			map.put(Unit.P, list0);
			map.put(Unit.LC, list1);
			map.put(Unit.RC, list2);
			gm.add(i, map);
		}
		System.out.println(gm);
	}
}
