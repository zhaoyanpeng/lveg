package edu.shanghaitech.ai.nlp.lveg.impl;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;

public class BinaryGrammarTest {
	static short ncomp = 2, ndim = 2;
	static {
		Random rnd = new Random(0);
		GaussianMixture.config((short) -1, 1e-6, 4, ncomp, 0.5, -1.0, -1.0, true, rnd, null);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
	}
	
	@Test
	public void testBinaryGrammarTest() {
		System.out.println("\n---LRBRULE---\n");
		
		GrammarRule rule0 = new BinaryGrammarRule((short) 0, (short) 1, (short) 2);
		rule0.initializeWeight(RuleType.LRBRULE, ncomp, ndim);
		System.out.println(rule0);
		System.out.println(rule0.weight);
		System.out.println(rule0.weight.spviews());
		
		assertTrue(rule0.weight.getBinding() == RuleType.LRBRULE);
	}
}
