package edu.shanghaitech.ai.nlp.lvet;

import org.junit.Test;

public class LVeTTrainerTester {
	@Test
	public void testLVeTTrainer() {
		String[] args = {"param.tg"};
		try {
			LVeTTrainer.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
