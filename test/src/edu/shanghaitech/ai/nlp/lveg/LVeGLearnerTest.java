package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class LVeGLearnerTest {
	
	private final static String ROOT = "E:/SourceCode/ParsersData/berkeley/";
	
	@Test
	public void testLVeGLearner() {
		String corpusPath = ROOT + "treebank/combined/";
		String outputFile = ROOT + "treebank/grammar.gr";
		
		String[] args = {"-pathToCorpus", corpusPath, "-out", outputFile};
		
		LVeGLearner.main(args);
	}

}
