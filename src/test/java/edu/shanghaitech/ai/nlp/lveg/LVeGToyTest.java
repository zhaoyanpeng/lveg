package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class LVeGToyTest {
	
	@Test
	public void testLVeGLearner() {
		String[] args = {"config/paramtoy.in"};
		try {
			LVeGToy.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
