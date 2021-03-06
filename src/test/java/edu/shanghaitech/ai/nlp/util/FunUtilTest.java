package edu.shanghaitech.ai.nlp.util;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig;

public class FunUtilTest {
	
	@Test
	public void testFunUtil() {
		int maxint = 10;
		Integer[] arrayInt = new Integer[5];
		FunUtil.randomInitArray(arrayInt, Integer.class, maxint);
		FunUtil.printArray(arrayInt);
		
		Double[] arrayDouble = new Double[5];
		FunUtil.randomInitArray(arrayDouble, Double.class, 1);
		FunUtil.printArray(arrayDouble);
		
		List<Integer> listInt = new ArrayList<Integer>();
		FunUtil.randomInitList(LVeGTrainer.random, listInt, Integer.class, 5, maxint, 0.5, false, true);
		FunUtil.printList(listInt);
		
		List<Double> listDouble = new ArrayList<Double>();
		FunUtil.randomInitList(LVeGTrainer.random, listDouble, Double.class, 5, 1, 0.5, false, true);
		FunUtil.printList(listDouble);
		
		int[] arrayint = new int[5];
		FunUtil.randomInitArrayInt(arrayint, maxint);
		FunUtil.printArrayInt(arrayint);
		
		double[] arraydouble = new double[5];
		FunUtil.randomInitArrayDouble(arraydouble);
		FunUtil.printArrayDouble(arraydouble);
		
		Double[] xarray = new Double[0];
		FunUtil.printArray(xarray);
		
		double x = -30;
		double y = -6;
		double z = FunUtil.logAdd(x, y);
		double m = Math.log((Math.exp(x) + Math.exp(y)));
		System.out.println("Precision of the logAdd method is: " + (m - z) + ", [m = " + m + ", z =" + z + "]");
	}
	
//	@Test
	public void testShuffle() {
		List<Integer> listInt = new ArrayList<Integer>();
		FunUtil.randomInitList(LVeGTrainer.random, listInt, Integer.class, 5, 2, 0.5, false, true);
		System.out.println("---shuffle test---");
		System.out.println(listInt);
		Collections.shuffle(listInt, new Random(0));
		System.out.println(listInt);
	}
	
	@Test
	public void stringFormat() {
		double x = Double.parseDouble("1e-3");
		System.out.println(x);
		
		
		try {
			String str = LearnerConfig.readFile("param.ini", StandardCharsets.UTF_8);
			System.out.println(str);
			String[] arr = str.split(",");
			for (int i = 0; i < arr.length; i++) {
				System.out.println(i + " | " + arr[i].trim() + ")");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
