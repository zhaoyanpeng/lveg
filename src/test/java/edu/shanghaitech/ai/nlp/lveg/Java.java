package edu.shanghaitech.ai.nlp.lveg;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;

public class Java {
	
//	@Test
	public void testJava() {
		Ghost ghost0 = new Ghost((short) 0);
		System.out.println(ghost0);
		Ghost ghost1 = new Ghost((short) 1);
		System.out.println(ghost1);
		Ghost ghost2 = new Ghost((short) 2);
		System.out.println(ghost2);
		
		System.out.println(ghost2.soul == ghost1.soul);
		ghost2.increaseSoul();
		System.out.println(ghost0);
		System.out.println(ghost1.soul == ghost0.soul);
	}
	
	
	protected static class Ghost {
		static short id;
		static Soul soul = new Soul();
		
		public Ghost(short id) {
			this.id = id;
		}
		
		public void increaseSoul() {
			soul.increase();
		}
		
		public String toString() {
			return "-" + id + soul;
		}
	}
	
	protected static class Soul {
		static short id = 0;
		String soul = "-xiasini-";
		
		public Soul() {
			id++;
			System.out.println("Do you have the soul?");
		}
		
		public void increase() {
			id++;
		}
		
		public String toString() {
			return soul + "-" + id;
		}
	}
	
	
//	@Test
	public void testQueue() {
		PriorityQueue<Integer> sorted = new PriorityQueue<Integer>(5);
		for (int i = 0; i < 5; i++) {
			sorted.add(i);
		}
		System.out.println(sorted);
		for (Integer it : sorted) {
			if (it < 3) {
				continue;
			}
			sorted.remove(it); // this is a bad coding example
		}
		System.out.println(sorted);
	}
	
	
	@Test
	public void testTest() {
		int a = 3, b = 3;
		assert(a != b);
	}
	
//	@Test
	public void testDate() {
		String timeStamp = new SimpleDateFormat(".yyyyMMddHHmmss").format(new Date());
		System.out.println(timeStamp);
		
		String xx = " ";
		System.out.println(xx.trim().equals(""));
		
//		double l1 = (-5.0620584989385815 + 4.368911318378636);
//		double l2 = (-5.0620384989385805 + 4.368891318378635);
		
//		double l3 = (-5.062048498938581 + 4.368901318378636);
//		double l1 = (-5.061048498938581 + 4.367901318378635);
//		double l2 = (-5.063048498938581 + 4.369901318378636);
		
		double l3 = ((-5.063048498938581 + 4.369401193378641) - (-5.061048498938581 + 4.368401193378641)) / (0.002);
//		System.out.println(String.format("%.10f", l1));
//		System.out.println(String.format("%.10f", l2));
		System.out.println(String.format("%.10f", l3));
		
		System.out.println(0.0 / Double.NEGATIVE_INFINITY);

	}
	
	
//	@Test
	public void testMain() {
		class LL {
			int x = 0;
			public LL(int x) {
				this.x = x;
			}
			public String toString() {
				return String.valueOf(x);
			}
		}
		Set<LL> container = new HashSet<LL>();
		LL l0 = new LL(0);
		LL l1 = new LL(1);
		container.add(l0);
		container.add(l1);
		
		LL[] array = container.toArray(new LL[0]);
		for (int i = 0; i < array.length; i++) {
			System.out.println(array[i]);
		}
		array[1].x = 50;
		System.out.println(container);
		
	}
}
