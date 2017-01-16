package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.Unit;

public class UnaryGrammarRuleTest {
	
	@Test
	public void testGMadd0() {
		GaussianMixture gm = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Set<GaussianDistribution> set = new HashSet<GaussianDistribution>();
			set.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			gm.add(i, Unit.C, set);
		}
		System.out.println(gm);
	}
	
	
	@Test
	public void testGMadd1() {
		GaussianMixture gm = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Set<GaussianDistribution> set = new HashSet<GaussianDistribution>();
			set.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			gm.add(i, Unit.P, set);
		}
		System.out.println(gm);
	}
	
	@Test
	public void testGMadd2() {
		GaussianMixture gm = new DiagonalGaussianMixture(LVeGLearner.ncomponent);
		for (int i = 0; i < LVeGLearner.ncomponent; i++) {
			Map<String, Set<GaussianDistribution>> map = new HashMap<String, Set<GaussianDistribution>>();
			Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();
			set0.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			set1.add(new DiagonalGaussianDistribution(LVeGLearner.dim));
			map.put(Unit.P, set0);
			map.put(Unit.UC, set1);
			gm.add(i, map);
		}
		System.out.println(gm);
	}

}
