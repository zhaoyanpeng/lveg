package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class LVeGPCFGTest {

	@Test
	public void testLVeGPCFG() {
		String[] args = {"param.in"};
		try {
			LVeGPCFG.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
