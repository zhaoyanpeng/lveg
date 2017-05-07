package edu.shanghaitech.ai.nlp.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import org.junit.Test;

public class PTBConverter {
	
	private final static String ROOT = "E:/SourceCode/ParsersData/wsj/";
	
	@Test
	public void testPTBConverter() throws Exception {
		String train = ROOT + "wsj_s2-21_tree";
		String test = ROOT + "wsj_s23_tree";
		String dev = ROOT + "wsj_s22_tree";
		
		
		FileWriter w1 = new FileWriter(ROOT + "wsj_s2-21.prd");
		FileWriter w2 = new FileWriter(ROOT + "wsj_s23.prd");
		FileWriter w3 = new FileWriter(ROOT + "wsj_s22.prd");
		
		String line = null;
		BufferedReader reader = new BufferedReader(new FileReader(new File(dev)));
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			if (line.length() > 0) {
				line = line.substring(6, line.length() - 1);
				w1.write(line);
				w1.write("\n");
			}
		}
		w1.close();
		
	}

}
