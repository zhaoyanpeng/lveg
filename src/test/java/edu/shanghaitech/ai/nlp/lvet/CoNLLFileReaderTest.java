package edu.shanghaitech.ai.nlp.lvet;

import java.util.List;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.lvet.io.CoNLLFileReader;

public class CoNLLFileReaderTest {
	
	private final static String ROOT = "/home/oops/Data/berkeley/data/Data.Prd/tag/";
	
	@Test
	public void testCoNLLFileReader() throws Exception {
		String dev = ROOT + "wsj.21.mrg.dep";
		
		CoNLLFileReader.config(3, 1);
		List<List<TaggedWord>> data = CoNLLFileReader.read(dev);
		for (int i = 0; i < 4; i++) {
			System.out.println(data.get(i));
		}
	}
}
