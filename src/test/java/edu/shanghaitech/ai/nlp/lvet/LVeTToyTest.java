package edu.shanghaitech.ai.nlp.lvet;

import org.junit.Test;

public class LVeTToyTest {
	@Test
	public void testLVeTToy() {
		String[] args = {"config/paramtag.in"};
		try {
			LVeTToy.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
