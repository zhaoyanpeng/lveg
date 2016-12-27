package edu.shanghaitech.ai.nlp.syntax;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;

public class StateTest {
	
	@Test
	public void testState() {
		State state = new State("hello", (short) 0, (short) 1, (short) 2);
		GaussianMixture iscore = new DiagonalGaussianMixture((short) 2);
		GaussianMixture oscore = new DiagonalGaussianMixture((short) 2);
		state.setInsideScore(iscore);
		state.setOutsideScore(oscore);
		
//		System.out.println(state.toString(false, (short) 2));
		assertTrue(state.getInsideScore() != null);
		state.clear(false);
//		System.out.println(state.toString(false, (short) 2));
		assertTrue(state.getInsideScore() == null);
	}
}
