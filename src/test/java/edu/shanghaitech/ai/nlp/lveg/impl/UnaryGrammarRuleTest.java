package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.EnumMap;
import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;

public class UnaryGrammarRuleTest {
	
	@Test
	public void testGMadd0() {
		GaussianMixture gm = new DiagonalGaussianMixture(LVeGTrainer.ncomponent);
		for (int i = 0; i < LVeGTrainer.ncomponent; i++) {
			Set<GaussianDistribution> set = new HashSet<GaussianDistribution>();
			set.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			gm.add(i, RuleUnit.C, set);
		}
		System.out.println(gm);
	}
	
	
	@Test
	public void testGMadd1() {
		GaussianMixture gm = new DiagonalGaussianMixture(LVeGTrainer.ncomponent);
		for (int i = 0; i < LVeGTrainer.ncomponent; i++) {
			Set<GaussianDistribution> set = new HashSet<GaussianDistribution>();
			set.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			gm.add(i, RuleUnit.P, set);
		}
		System.out.println(gm);
	}
	
	@Test
	public void testGMadd2() {
		GaussianMixture gm = new DiagonalGaussianMixture(LVeGTrainer.ncomponent);
		for (int i = 0; i < LVeGTrainer.ncomponent; i++) {
			EnumMap<RuleUnit, Set<GaussianDistribution>> map = new EnumMap<>(RuleUnit.class);
			Set<GaussianDistribution> set0 = new HashSet<GaussianDistribution>();
			Set<GaussianDistribution> set1 = new HashSet<GaussianDistribution>();
			set0.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			set1.add(new DiagonalGaussianDistribution(LVeGTrainer.dim));
			map.put(RuleUnit.P, set0);
			map.put(RuleUnit.UC, set1);
			gm.add(i, map);
		}
		System.out.println(gm);
	}

}
