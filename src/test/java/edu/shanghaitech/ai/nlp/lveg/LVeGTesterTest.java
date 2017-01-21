package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.util.Recorder;

public class LVeGTesterTest extends Recorder {

	private static final long serialVersionUID = 952298389200776594L;

	@Test
	public void testLVeGLearner() {
		String[] args = {"param.in"};
		try {
			LVeGTester.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
