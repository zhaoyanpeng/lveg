package edu.shanghaitech.ai.nlp.lveg;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

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
	
	
	@Test
	public void testDate() {
		String timeStamp = new SimpleDateFormat(".yyyyMMddHHmmss").format(new Date());
		System.out.println(timeStamp);
		
		String xx = " ";
		System.out.println(xx.trim().equals(""));
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
