package edu.shanghaitech.ai.nlp.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import org.junit.Test;

public class PTBConverter {
	
	private final static String ROOT = "/home/oops/Data/berkeley/data/Data.Prd/tag/";
	
	@Test
	public void testPTBConverter() throws Exception {
		String train = ROOT + "wsj.21";
		String test = ROOT + "wsj.23";
		String dev = ROOT + "wsj.22";
		
		
		FileWriter w1 = new FileWriter(ROOT + "wsj.21.mrg");
		FileWriter w2 = new FileWriter(ROOT + "wsj.23.mrg");
		FileWriter w3 = new FileWriter(ROOT + "wsj.22.mrg");
		
		String[] filenames = {train, test, dev};
		FileWriter[] writers = {w1, w2, w3};
		
		for (int i = 0; i < 3; i++) {
			FileWriter w = writers[i];
			String filename = filenames[i];
					
			String line = null;
			BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				if (line.length() > 0) {
					line = line.substring(6, line.length() - 1);
					w.write(line);
					w.write("\n");
				}
			}
			w.close();
			reader.close();
		}
		
	}

}
