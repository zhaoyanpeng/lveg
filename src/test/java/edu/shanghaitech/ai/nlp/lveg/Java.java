package edu.shanghaitech.ai.nlp.lveg;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.impl.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.util.Debugger;
import edu.shanghaitech.ai.nlp.util.FunUtil;

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
	public void c() {
//		Debugger.debugTreebank();
		
		String format = "%.3f";
		NumberFormat formatter = new DecimalFormat("0.###E0");
		double xx = Double.NEGATIVE_INFINITY;
		System.out.println(String.format(format, xx));
		System.out.println(formatter.format(xx));
		
		double yy = Double.NaN;
		System.out.println(String.format(format, yy));
		System.out.println(formatter.format(yy));
		
		GrammarRule rule0 = new BinaryGrammarRule((short) 6, (short) 8, (short)8);
		GrammarRule rule1 = new BinaryGrammarRule((short) 6, (short) 8, (short)8);

		System.out.println(rule0 = rule1);
		System.out.println(rule0.equals(rule1));
	}
	
//	@Test
	public void a() {
		int x = 1;
		b(x++);
		System.out.println("in a: " + x);
	}
	
	public void b(int a) {
		System.out.println("in b: " + a);
	}
	
//	@Test
	public void testConcurrentModifier() {
		List<String> mylist = new ArrayList<String>();
		mylist.add("1");
		mylist.add("2");
		mylist.add("3");
		mylist.add("4");
		mylist.add("5");
		Iterator<String> it = mylist.iterator();
		while (it.hasNext()) {
			String val = it.next();
			System.out.println("list val: " + val);
			if ("3".equals(val)) {
//				mylist.remove(val);
				it.remove();
			}
		}
		for (String val : mylist) {
			System.out.println(val);
		}
	}
	
//	@Test
	public void testRegx() {
		String regx = ".*?(\\d+)";
		String line = "lveg_4.gr";
		
		Pattern pat = Pattern.compile(regx);
		Matcher matcher;
		matcher = pat.matcher(line);
		if (matcher.find()) {
			System.out.println(matcher.group(1) + "\t" + matcher.groupCount());
		}
		
		
		Set<Integer> set0 = new HashSet<Integer>();
		Set<Integer> set1 = new HashSet<Integer>();
		
		set0.add(1);
		set0.add(2);
		set0.add(3);
		
		set1.add(2);
		set1.add(3);
		set1.add(4);
		
		set0.retainAll(set1);
		System.out.println(set0);
		
		double m = 1.0 - Double.NEGATIVE_INFINITY;
		System.out.println(m);
		List<Double> x = new ArrayList<Double>();
		x.add(m);
		System.out.println("scores: " + FunUtil.double2str(x, 3, -1, false, true));
	}
	
//	@Test
	public void testQueue() {
		List<Integer> list = new ArrayList<Integer>(5);
		PriorityQueue<Integer> sorted = new PriorityQueue<Integer>(5);
		for (int i = 0; i < 5; i++) {
			sorted.add(i);
			list.add(i);
		}
		System.out.println(sorted);
		for (Integer it : sorted) {
			if (it < 3) {
				continue;
			}
			sorted.remove(it); // this is a bad coding example
		}
		System.out.println(sorted);
		
//		for (int i = 1; i < 5; i++) {
//			list.remove(i);
//		}
		list.subList(0, list.size()).clear();
		list.add(4);
		list.subList(1, list.size()).clear();
		System.out.println(list);
	}
	
	
//	@Test
	public void testTest() {
		int a = 3, b = 3;
		assert(a == b);
		
//		double d0 = -15.332610488842343;
//		double d1 = -8.00196935333965;
//		double d2 = -14.834674151593733;
//		
//		double x = FunUtil.logAdd(d0, d1);
//		x = FunUtil.logAdd(x, d2);
//		System.out.println(x);
//		
//		double y = Math.log(Math.exp(d0) + Math.exp(d1) + Math.exp(d2));
//		System.out.println(y);
		
		Map<Integer, Double> map = new HashMap<Integer, Double>();
		map.put(0, 3.0);
		map.put(4, 1.0);
		map.put(2, 2.0);
		map.put(3, 9.0);
		map.put(7, 5.0);
		map.put(8, 6.0);
		int k = 4;
		Collection<Double> values = map.values();
		PriorityQueue<Double> queue = new PriorityQueue<Double>();
		for (Double d : values) {
			queue.offer(d);
			if (queue.size() > k) { queue.poll(); }
		}
		double val = queue.peek();
		System.out.println(val);
		while (!queue.isEmpty()) {
			System.out.println(queue.poll());
		}
		System.out.println();
		queue.clear();
		queue.addAll(values);
		while (!queue.isEmpty()) {
			System.out.println(queue.poll());
		}
//		Collections.sort(values);
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
