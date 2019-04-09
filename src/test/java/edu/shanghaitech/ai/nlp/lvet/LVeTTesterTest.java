package edu.shanghaitech.ai.nlp.lvet;

import org.junit.Test;

public class LVeTTesterTest {
	@Test
	public void testLVeTTrainer() {
		String[] args = {"tagging.f1"};
		try {
			LVeTTester.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
