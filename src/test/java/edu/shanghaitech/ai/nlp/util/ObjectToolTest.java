package edu.shanghaitech.ai.nlp.util;

import java.util.Random;

import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;
import org.junit.Before;
import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.impl.GaussFactory;
import edu.shanghaitech.ai.nlp.lveg.impl.MoGFactory;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;

public class ObjectToolTest {
	
	private ObjectPool<Short, GaussianMixture> mogPool;
	private ObjectPool<Short, GaussianDistribution> gaussPool;
	
	@Before
	public void setup() throws Exception {
		GenericKeyedObjectPoolConfig config = new GenericKeyedObjectPoolConfig();
//		config.setMaxTotalPerKey(Integer.MAX_VALUE);
//		config.setMaxTotal(Integer.MAX_VALUE);
		
		config.setMaxTotalPerKey(8);
		config.setMaxTotal(2000);
		
		
//		config.setBlockWhenExhausted(true);
//		config.setMaxWaitMillis(1500);
//		config.setTestOnBorrow(true);
//		config.setTestOnCreate(false);
//		config.setTestOnReturn(false);
		
		StringBuffer sb = new StringBuffer();
		sb.append("---max idle per key: " + config.getMaxIdlePerKey() + "\n"
				+ "---max        total: " + config.getMaxTotal() + "\n"
				+ "---max total per key: " + config.getMaxTotalPerKey() + "\n"
				+ "---min idle per key: " + config.getMinIdlePerKey() + "\n");
		
		short defaultval = 2;
		Random rnd = new Random(0);
		MoGFactory mfactory = new MoGFactory(defaultval, defaultval, 0.5, rnd);
		GaussFactory gfactory = new GaussFactory(defaultval, defaultval, 0.5, rnd);
		
		mogPool = new ObjectPool<Short, GaussianMixture>(mfactory, config);
		gaussPool = new ObjectPool<Short, GaussianDistribution>(gfactory, config);
		
		sb.append("---max active: " + mogPool.getNumActive() + "\n"
				+ "---block: " + mogPool.getBlockWhenExhausted() + "\n"
				+ "---wait: " + mogPool.getMaxWaitMillis() + "\n" 
				+ "---waitb: " + mogPool.getMaxBorrowWaitTimeMillis() + "\n");
		System.out.println(sb.toString());
	}
	
	
	public void testStress() {
//		int total = 
	}
	

//	@Test
	public void testObjectTool() {
		
		int total = 20, cnt = 0, mid = 8;
		for (int i = 0; cnt < total; i++, cnt++) {
			if (cnt < mid) {
				i = 0; 
			} else {
				i = cnt - mid + 1;
			}
			System.out.println("-----------------" + cnt + "\t" + i);
			try {
//				mogPool.borrowObject((short) i);
				gaussPool.borrowObject((short) i);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
	}

}
