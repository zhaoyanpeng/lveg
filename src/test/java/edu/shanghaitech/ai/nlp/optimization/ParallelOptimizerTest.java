package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.impl.Valuator;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer.Chart;
import edu.shanghaitech.ai.nlp.util.MethodUtil;
import edu.shanghaitech.ai.nlp.util.ThreadPool;

public class ParallelOptimizerTest {
	
	protected static int THREADS_NUM = 2;
	
	
	protected static class Muppet {
		protected static int maxiter = 5;
		
		public static void staticPrint(int i, int isample) {
			for (; i < maxiter; i++) {
//				for (int i = 0; i < 5; i++) {
					System.out.println("Muppet.static:\tisample_" + isample + "\t" + Thread.currentThread().getId() + ": " + i);
				}
		}
		
		public void nonStaticPrint(int i, int isample) {
			for (; i < maxiter; i++) {
//			for (int i = 0; i < 5; i++) {
				System.out.println("Muppet.non-static:\tisample_" + isample + "\t" + Thread.currentThread().getId() + ": " + i);
			}
		}
	}
	
	protected static class Puppet<I, O> extends Parser<I, O> {
		/**
		 * 
		 */
		private static final long serialVersionUID = 3714346641593051149L;
		protected static int maxiter = 3;
		protected Muppet muppet;
		protected String name;
		
		private Puppet(Puppet<?, ?> puppet) {
			super((short) 0, false, false);
			this.muppet = puppet.muppet;
			this.reuse = puppet.reuse;
			this.name = puppet.name;
		}
		
		public Puppet(String name) {
			super((short) 0, false, false);
			this.name = name;
			this.muppet = new Muppet();
		}
		
		public  void staticPrint(int i, int isample) {
			Muppet.staticPrint(i, isample);
		}
		
		public void nonStaticPrint(int i, int isample) {
			muppet.nonStaticPrint(i, isample);
		}
		
		protected void setContext(boolean test) {}

		@Override
		public synchronized Object call() throws Exception {
			String ll = getName();
			
//			staticPrint(idx, isample); // 0: uncomment to test static method accessing
			
			Meta<O> cache = new Meta(itask, ll);
			
			synchronized (muppet) {
				
//				staticPrint(idx, isample); // 1: uncomment to test static method accessing
				
				muppet.nonStaticPrint(idx, itask);
				muppet.notifyAll();
			}
			
			synchronized (caches) {
				
//				nonStaticPrint(idx, isample);
				
				caches.add(cache);
				caches.notifyAll();
			}
			task = null;
			return null;
		}
		
		public String getName() {
			return name + "_" + idx;
		}

		@Override
		public Parser<?, ?> newInstance() {
			return new Puppet<I, O>(this);
		}
	}
	
	
//	@Test
	public void testMultiThreadedPool() {
		// static or non-static methods accessing test
		String ll = null;
		int nthread = 2, nfailed = 0;
		Puppet<?, ?> puppet = new Puppet<Object, Double>("puppet");
		ThreadPool mpuppet = new ThreadPool(puppet, nthread);
		for (int i = 0; i < 4; i++) {
			mpuppet.execute(null);
			while (mpuppet.hasNext()) {
				ll = (String) mpuppet.getNext();
				if (ll == null) {
					nfailed++;
				} else {
					System.out.println("~~~>name: " + ll);
				}
			}
		}
		while (!mpuppet.isDone()) {
			while (mpuppet.hasNext()) {
				ll = (String) mpuppet.getNext();
				if (ll == null) {
					nfailed++;
				} else {
					System.out.println("~~~>name: " + ll);
				}
			}
		}
		System.out.println("---summary: nfailed=" + nfailed);
		mpuppet.shutdown();
	}
	
	
	@Test
	public void testParallelOptimizer() {
		ExecutorService pool = Executors.newFixedThreadPool(THREADS_NUM);
		List<Callable<Boolean>> tasks = new ArrayList<Callable<Boolean>>(2);
		
		Puppet p0 = new Puppet("0");
		Puppet p1 = p0;
		
		tasks.add(new Callable<Boolean>() {

			@Override
			public Boolean call() throws Exception {
				p0.nonStaticPrint(1, 0);
				return true;
			}
			
		});
		
		tasks.add(new Callable<Boolean>() {

			@Override
			public Boolean call() throws Exception {
				p1.nonStaticPrint(2, 1);
				return true;
			}
			
		});
		
		// see the comments in ParallelOptimizer.useCustomizedBlock();
		for (Callable<Boolean> task : tasks) {
			pool.submit(task);
		}
		boolean exit = true;
		try {
			pool.shutdown();
			exit = pool.awaitTermination(0, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		System.out.println("exit: " + exit + "... " + pool.isTerminated());
		
		/*
		try {
			pool.invokeAll(tasks);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		*/
	}

}
