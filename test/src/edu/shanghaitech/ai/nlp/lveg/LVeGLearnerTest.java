package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class LVeGLearnerTest {
	
//	private final static String ROOT = "E:/SourceCode/ParsersData/berkeley/";
	private final static String ROOT = "E:/SourceCode/ParsersData/wsj/";
	
	@Test
	public void testLVeGLearner() {
		String corpusPath = ROOT + "wsj_s2-21_tree";
//		String corpusPath = ROOT + "treebank/combined/";
		String outputFile = ROOT + "treebank/grammar.gr";
		String logFile = "log/io_with_s_chain_not_precompute_not_addinter";
		
		String[] args = {"-pathToCorpus", corpusPath, "-out", outputFile, "-logType", "-logFile", logFile};
		
		LVeGLearner.main(args);
	}
}
