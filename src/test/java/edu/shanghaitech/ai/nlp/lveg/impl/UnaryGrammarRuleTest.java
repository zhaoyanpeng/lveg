package edu.shanghaitech.ai.nlp.lveg.impl;

import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;

public class UnaryGrammarRuleTest {
	static short ncomp = 2, ndim = 2;
	static {
		Random rnd = new Random(0);
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
	}
	
	@Test
	public void testGMadd0() {
		System.out.println("\n---LHSPACE---\n");
		
		GrammarRule rule0 = new UnaryGrammarRule((short) 0, 1, RuleType.LHSPACE);
		rule0.initializeWeight(RuleType.LHSPACE, ncomp, ndim);
		System.out.println(rule0);
		System.out.println(rule0.weight);
		System.out.println(rule0.weight.spviews());
		
		assertTrue(rule0.weight.getBinding() == RuleType.LHSPACE);
		
		System.out.println("\n---RHSPACE---\n");
		
		GrammarRule rule1 = new UnaryGrammarRule((short) 0, 1, RuleType.RHSPACE);
		rule1.initializeWeight(RuleType.RHSPACE, ncomp, ndim);
		System.out.println(rule1);
		System.out.println(rule1.weight);
		System.out.println(rule1.weight.spviews());
		
		assertTrue(rule1.weight.getBinding() == RuleType.RHSPACE);
		
		System.out.println("\n---LRURULE---\n");
		
		GrammarRule rule2 = new UnaryGrammarRule((short) 0, 1, RuleType.LRURULE);
		rule2.initializeWeight(RuleType.LRURULE, ncomp, ndim);
		System.out.println(rule2);
		System.out.println(rule2.weight);
		System.out.println(rule2.weight.spviews());
		
		assertTrue(rule2.weight.getBinding() == RuleType.LRURULE);
	}
}
