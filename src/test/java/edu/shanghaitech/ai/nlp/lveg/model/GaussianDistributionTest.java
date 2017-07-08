package edu.shanghaitech.ai.nlp.lveg.model;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.impl.DiagonalGaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;

public class GaussianDistributionTest {
	static short ndim = 2;
	static {
		Random rnd = new Random(0);
		GaussianDistribution.config(1, 5, ndim, 0.5, 0.8, rnd, null);
	}
	
	@Test
	public void testGaussianDistribution() {
		GaussianDistribution gd0 = new DiagonalGaussianDistribution((short) 2);
		GaussianDistribution gd1 = new DiagonalGaussianDistribution((short) 2);
		
		System.out.println(gd0);
		System.out.println(gd1);
		
		double inte = gd0.mulAndMarginalize(gd1);
		double factor = inte + Math.log(2) + Math.log(2);
		
		System.out.println(factor);
	}
	
	
	@Test
	public void testInstanceEqual() {
		GaussianDistribution gd0 = new DiagonalGaussianDistribution();
		GaussianDistribution gd1 = new DiagonalGaussianDistribution();
		
		assertFalse(gd0 == gd1);
		assertTrue(gd0.equals(gd1));
		
		gd0.getMus().add(2.0);
		gd0.getMus().add(3.0);
		gd0.getVars().add(4.0);
		
		gd1.getMus().add(2.0);
		gd1.getMus().add(3.0);
		gd1.getVars().add(4.0);
		
		assertTrue(gd0.equals(gd1));
		
		gd1.getMus().add(2.0);
		assertFalse(gd0.equals(gd1));
	}

}
