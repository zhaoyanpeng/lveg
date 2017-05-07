package edu.shanghaitech.ai.nlp.lvet;

import java.util.List;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.io.CoNLLFileReader;

public class CoNLLFileReaderTest {
	
	private final static String ROOT = "E:/SourceCode/ParsersData/pos/";
	
	@Test
	public void testCoNLLFileReader() throws Exception {
		String dev = ROOT + "wsj_s2-21.prd.3.pa.gs.tab";
		
		List<List<TaggedWord>> data = CoNLLFileReader.read(dev);
		for (int i = 0; i < 4; i++) {
			System.out.println(data.get(i));
		}
	}
}
