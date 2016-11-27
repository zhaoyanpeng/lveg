package edu.shanghaitech.ai.nlp.lveg;

import org.junit.Test;

public class Java {
	
	@Test
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
//		static Soul soul = new Soul();
		static Soul soul;
		
		public Ghost(short id) {
			soul = new Soul();
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
}
