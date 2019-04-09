package edu.shanghaitech.ai.nlp.lvet.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.lvet.impl.TaggedWord;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class CoNLLFileReader implements TaggedFileReader {
	
	private static int defTagColIdx = 3;
	private static int defWordColIdx = 1;
	private static BufferedReader reader;
	private static final String SEPARATOR = "\t";
	
	public static void config(int tagColIdx, int wordColIdx) {
		defTagColIdx = tagColIdx;
		defWordColIdx = wordColIdx;
	}
	
	public static List<List<TaggedWord>> read(String filename) throws Exception {		
		reader = new BufferedReader(new FileReader(new File(filename)));
		List<List<TaggedWord>> data = new ArrayList<List<TaggedWord>>();
		readFromBuffer(data);
		return data;
	}
	
	public static List<List<TaggedWord>> read(BufferedReader breader) throws Exception {		
		reader = breader;
		List<List<TaggedWord>> data = new ArrayList<List<TaggedWord>>();
		readFromBuffer(data);
		return data;
	}
	
	public static void readFromBuffer(List<List<TaggedWord>> data) throws Exception {
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
				if (tokens.length <= defTagColIdx || tokens.length <= defWordColIdx) {
					throw new IllegalArgumentException("file: " + 
							", tag column: " + defTagColIdx + ", word column: " + defWordColIdx);
				}
				TaggedWord word = new TaggedWord(tokens[defTagColIdx], tokens[defWordColIdx]);
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
	}
	
}
