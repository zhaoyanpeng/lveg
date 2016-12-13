package edu.shanghaitech.ai.nlp.lveg;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.syntax.State;

public class InferencerTest {
	
	private static LVeGInferencer inferencer = new LVeGInferencer(null, null);
	
//	@Test
	public void testInferencer() {
//		inferencer.insideScore(null, false);
//		inferencer.outsideScore(null);
		
		Inferencer.Chart chart = new Inferencer.Chart(5, false);
//		insideScore(chart, 5, 0, 4);
		for (int i = 0; i < 5; i++) {
			outsideScore(chart, 5, i, i);
			System.out.println();
		}
	}
	
	
	@Test
	public void testInsideLoop() {
		// inside score
		int nword = 5;
		for (int ilayer = 1; ilayer < nword; ilayer++) {
			for (int left = 0; left < nword - ilayer; left++) {
				for (int right = left; right < left + ilayer; right++) {				
					System.out.println(left + "," + (left + ilayer) + "(" + left + "," + right + "), (" + 
							(right + 1) + "," + (left + ilayer) + ")");
				}
			}
			System.out.println();
		}
	}
	
	
//	@Test
	public void insideScore(Chart chart, int nword, int begin, int end) {
		
		if (begin == end) {
			System.out.println("(" + begin + ", " + end + ")");
		}
		
		for (int split = begin; split < end; split++) {
			
			int x0 = begin, y0 = split;
			int x1 = split + 1, y1 = end;
			int l0 = nword - (y0 - x0), l1 = nword - (y1 - x1);
			
			if (!chart.getStatus(Chart.idx(x0,  l0), true)) {
				insideScore(chart, nword, begin, split);
			}
			if (!chart.getStatus(Chart.idx(x1, l1), true)) {
				insideScore(chart, nword, split + 1, end);
			}
			
			System.out.println("-(" + begin + ", " + end + ")--" + 
					"-(" + x0 + ", " + y0 + ")" + "-(" + x1 + ", " + y1 + ")");
		}
		
		chart.setStatus(Chart.idx(begin, nword - (end - begin)), true, true);
		System.out.println();
	}
	
	
//	@Test
	public void outsideScore(Chart chart, int nword, int begin, int end) {
		
		if (begin == 0 && end == nword - 1) {
			if (!chart.getStatus(Chart.idx(begin, nword - (end - begin)), true)) {
				System.out.println("(" + begin + ", " + end + ")");
				chart.setStatus(Chart.idx(begin, nword - (end - begin)), true, true);
			}
		}
		
		int x0, y0, x1, y1, l0, l1;
		
		for (int right = end + 1; right < nword; right++) {
			x0 = begin;
			y0 = right;
			x1 = end + 1;
			y1 = right;
			l0 = y0 - x0;
			l1 = y1 - x1;
			if (!chart.getStatus(Chart.idx(x0, l0), true)) {
				outsideScore(chart, nword, x0, y0);
			}
			
			System.out.println("-(" + begin + ", " + end + ")--" + 
					"-(" + x0 + ", " + y0 + ")" + "-(" + x1 + ", " + y1 + ")");
		}

		for (int left = 0; left < begin; left++) {
			x0 = left;
			y0 = end;
			x1 = left;
			y1 = begin - 1;
			l0 = y0 - x0;
			l1 = y1 - x1;
			if (!chart.getStatus(Chart.idx(x0, l0), true)) {
				outsideScore(chart, nword, x0, y0);
			}
			
			System.out.println("=(" + begin + ", " + end + ")--" + 
					"-(" + x0 + ", " + y0 + ")" + "-(" + x1 + ", " + y1 + ")");
		}
		
		chart.setStatus(Chart.idx(begin, nword - (end - begin)), true, true);
		System.out.println();
	}

	
//	@Test
	public void testOutsideLoop() {
		int nword = 5;
		for (int ilayer = nword - 1; ilayer >= 0; ilayer--) {
			for (int left = 0; left < nword - ilayer; left++) {
				
				for (int right = left + ilayer + 1; right < nword; right++) {
					System.out.println(left + "," + (left + ilayer) + "(" + left + "," + right + "), (" + 
							(left + ilayer + 1) + "," + (right) + ")");
				}
				
				
				for (int right = 0; right < left; right++) {
					System.out.println(left + "," + (left + ilayer) + "(" + right + "," + (left + ilayer) + "), (" + 
							(right) + "," + (left - 1) + ")---");
				}
			}
			System.out.println();
		}
	}
}
