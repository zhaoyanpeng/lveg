package edu.shanghaitech.ai.nlp.util;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class MethodUtilTest {
	
	@Test
	public void testMethodUtil() {
		int maxint = 10;
		Integer[] arrayInt = new Integer[5];
		MethodUtil.randomInitArray(arrayInt, Integer.class, maxint);
		MethodUtil.printArray(arrayInt);
		
		Double[] arrayDouble = new Double[5];
		MethodUtil.randomInitArray(arrayDouble, Double.class, 1);
		MethodUtil.printArray(arrayDouble);
		
		List<Integer> listInt = new ArrayList<Integer>();
		MethodUtil.randomInitList(listInt, Integer.class, 5, maxint, false);
		MethodUtil.printList(listInt);
		
		List<Double> listDouble = new ArrayList<Double>();
		MethodUtil.randomInitList(listDouble, Double.class, 5, 1, false);
		MethodUtil.printList(listDouble);
		
		int[] arrayint = new int[5];
		MethodUtil.randomInitArrayInt(arrayint, maxint);
		MethodUtil.printArrayInt(arrayint);
		
		double[] arraydouble = new double[5];
		MethodUtil.randomInitArrayDouble(arraydouble);
		MethodUtil.printArrayDouble(arraydouble);
		
		Double[] xarray = new Double[0];
		MethodUtil.printArray(xarray);
	}
}
