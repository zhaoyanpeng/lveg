package edu.shanghaitech.ai.nlp.optimization;

import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.GrammarRule;

public class SGDForMoGTest {
	
	@Test
	public void testSGDForMoG() {
//		String key0 = GrammarRule.Unit.P;
//		String key1 = "2";
//		
//		int x = Integer.valueOf(key0);
		int a = 24;
		int b = 24;
		assert(a == b);
		
		char x = (char) -1;
		x = 1;
		System.out.println((short) x);
		System.out.println(x == 1);
		
		Byte f = -1;
		System.out.println(f);
		
		int xx = 98;
		int mm = 100;
		System.out.println(Integer.toBinaryString(xx));
		xx =  (1 << 31) + (xx << 16);
		System.out.println(Integer.toBinaryString(xx));
		xx += mm;
		System.out.println(Integer.toBinaryString(mm));
		
		System.out.println(Integer.toBinaryString(xx) + "\txx < 0: " + (xx < 0));
		
		xx = ((xx << 1) >>> 1);
		System.out.println(Integer.toBinaryString(xx));
		int xxx =  (xx >>> 16);
		System.out.println(Integer.toBinaryString(xxx));
		int mmm =  ((xx << 16) >>> 16);
		System.out.println(Integer.toBinaryString(mmm));
		
		System.out.println(Math.exp(Math.log(0.0) - 3));
		
		Set<Short> tmp = new HashSet<Short>();
		tmp.add((short) 0);
		System.out.println(tmp.contains((short) 0));
		
	}

}
