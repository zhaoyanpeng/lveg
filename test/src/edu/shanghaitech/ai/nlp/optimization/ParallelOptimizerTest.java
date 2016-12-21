package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.Test;

public class ParallelOptimizerTest {
	
	protected static int THREADS_NUM = 2;
	
	protected class Puppet {
		
		public void print(int i) {
			for (; i < 5; i++) {
//			for (int i = 0; i < 5; i++) {
				System.out.println(Thread.currentThread().getId() + ": " + i);
			}
		}
	}
	
	@Test
	public void testParallelOptimizer() {
		ExecutorService pool = Executors.newFixedThreadPool(THREADS_NUM);
		List<Callable<Boolean>> tasks = new ArrayList<Callable<Boolean>>(2);
		
		Puppet p0 = new Puppet();
		Puppet p1 = p0;
		
		tasks.add(new Callable<Boolean>() {

			@Override
			public Boolean call() throws Exception {
				p0.print(1);
				return true;
			}
			
		});
		
		tasks.add(new Callable<Boolean>() {

			@Override
			public Boolean call() throws Exception {
				p1.print(2);
				return true;
			}
			
		});
		
		try {
			pool.invokeAll(tasks);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
