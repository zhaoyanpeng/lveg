package edu.shanghaitech.ai.nlp.optimization;

import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;

public class SGDMinimizer {
	
	private static short nsample = 3;
	private static boolean cumulative = true;
	
	public static void applyGradientDescent(GaussianMixture gm, Random random, double cnt0, double cnt1, double learningRate) {
		for (short i = 0; i < nsample; i++) {
			double factor = 0.0;
			while (factor == 0.0) {
				gm.sample(random);
				factor = gm.eval();
			}
			factor = (cnt1 - cnt0) / factor;
			gm.derivative(factor, cumulative);
		}
		gm.update(learningRate);
	}
}
