package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class LVeGLearnerTest {

	@Test
	public void testLVeGLearner() {
		String[] args = {"param.in"};
		try {
			LVeGTrainer.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
