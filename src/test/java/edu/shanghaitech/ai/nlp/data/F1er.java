package edu.shanghaitech.ai.nlp.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.Test;

public class F1er {

	private final static String PARSED = "^(\\d+).*?(parsed).*?:(.*)";
	private final static String GOLDEN = "^(\\d+).*?(gold).*?:(.*)";
	
	
	@Test
	public void testF1er() throws Exception {
		String root = "E:/SourceCode/ParsersData/";
		String infile = root + "/gr_0508_1658_40_2_3_180_nb10_p5_f1_ep4_40_mt6.log";
		String outfile = root + "/bit";
		splitData(infile, outfile);
	}
	
	private void splitData(String infile, String outfile) throws Exception {
		String line = null;
		FileWriter writer = null;
		FileWriter[] writers = new FileWriter[4];
		writers[0] = new FileWriter(outfile + ".tst.parsed");
		writers[1] = new FileWriter(outfile + ".tst.golded");
		writers[2] = new FileWriter(outfile + ".dev.parsed");
		writers[3] = new FileWriter(outfile + ".dev.golded");
		
		Matcher matr = null;
		Pattern pat1 = Pattern.compile(PARSED);
		Pattern pat2 = Pattern.compile(GOLDEN);
		
		BufferedReader reader = new BufferedReader(new FileReader(new File(infile)));
		
		int a0 = 0, a1 = 0, cnt0 = -1, cnt1 = -1;
		while ((line = reader.readLine()) != null) {
			matr = pat1.matcher(line);
			if (matr.find()) {
				if (Integer.valueOf(matr.group(1)) == 0) {
					cnt0++;
					System.out.println("---P->cnt0: " + cnt0 + "\ta0: " + a0);
					a0 = 0;
				}
				writer = writers[cnt0 * 2];
				writer.write(matr.group(3).trim());
				writer.write("\n");
				a0++;
			} else {
				matr = pat2.matcher(line);
				if (matr.find()) {
					if (Integer.valueOf(matr.group(1)) == 0) {
						cnt1++;
						System.out.println("---G->cnt1: " + cnt1 + "\ta1: " + a1);
						a1 = 0;
					}
					writer = writers[cnt1 * 2 + 1];
					writer.write(matr.group(3).trim());
					writer.write("\n");
					a1++;
				}
			}
		}
		
		reader.close();
		for (FileWriter w : writers) {
			if (w != null) { w.close(); }
		}
		
		System.out.println("---G->cnt1: " + cnt1 + "\ta1: " + a1);
		System.out.println("---P->cnt0: " + cnt0 + "\ta0: " + a0);
	}
}
