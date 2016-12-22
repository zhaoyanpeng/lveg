package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class LVeGLearnerTest {
	
//	private final static String ROOT = "E:/SourceCode/ParsersData/berkeley/";
	private final static String ROOT = "E:/SourceCode/ParsersData/wsj/";
	
	@Test
	public void testLVeGLearner() {
		String corpusPath = ROOT + "wsj_s2-21_tree";
//		String corpusPath = ROOT + "treebank/combined/";
//		String outputFile = ROOT + "treebank/grammar.gr";
		String logFile = "log/ll_parallel";
//		String inputFile = ROOT + "lvegrammar.gr";
		String inputFile = ROOT + "lvegrammar_1_0.gr";
		String outputFile = ROOT + "lvegrammar";
		
		
		String[] args = {"-pathToCorpus", corpusPath, "-out", outputFile, /*"-logType",*/ "-logFile", logFile,
				"-in", inputFile};
		
		try {
			LVeGLearner.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
