package edu.shanghaitech.ai.nlp.lvet.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class CoNLLFileReader implements TaggedFileReader {
	
	private static int tagColIdx = 1;
	private static int wordColIdx = 0;
	private static BufferedReader reader;
	private static final String SEPARATOR = "\t";
	
	public static List<List<TaggedWord>> read(String filename) throws Exception {		
		reader = new BufferedReader(new FileReader(new File(filename)));
		List<List<TaggedWord>> data = new ArrayList<List<TaggedWord>>();
		int nline = 0;
		while (true) {
			String line = "";
			while (line != null && "".equals(line.trim())) {
				line = reader.readLine();
				nline++;
			}
			if (line == null) { break; } 
			
			List<TaggedWord> sample = new ArrayList<TaggedWord>();
			while (line != null && !"".equals(line.trim())) {
				String[] tokens = line.split(SEPARATOR);
				if (tokens.length <= tagColIdx || tokens.length <= wordColIdx) {
					throw new IllegalArgumentException("file: " + filename + 
							", tag column: " + tagColIdx + ", word column: " + wordColIdx);
				}
				TaggedWord word = new TaggedWord(tokens[tagColIdx], tokens[wordColIdx]);
				sample.add(word);
				line = reader.readLine();
				nline++;
			}
			if (sample.size() > 0) { 
				data.add(sample); 
			}
		}
		reader.close();
		if (Recorder.logger == null) {
			Recorder.logger = Recorder.logUtil.getConsoleLogger();
		}
		Recorder.logger.trace("# of lines is " + nline + ", # of samples is " + data.size() + "\n");
		return data;
	}
	
}
