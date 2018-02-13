package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class LVeGTesterTest {

	@Test
	public void testLVeGLearner() {
		String[] args = {"param.f1"};
		try {
//			LVeGTester.main(args);
//			LVeGTesterImp.main(args);
			LVeGTesterSim.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
