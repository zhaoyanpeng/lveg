package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.pool2.KeyedPooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;

public class GaussFactory implements KeyedPooledObjectFactory<Short, GaussianDistribution> {
	
	protected short ndimension;
	protected short maximum;
	protected double nratio;
	protected Random rnd;
	
	public GaussFactory(short ndimension, short maximum, double nratio, Random rnd) {
		this.ndimension = ndimension;
		this.maximum = maximum;
		this.nratio = nratio;
		this.rnd = rnd;
//		ThreadLocalRandom.current().nextGaussian();
	}

	@Override
	public void activateObject(Short ndim, PooledObject<GaussianDistribution> po) throws Exception {
		ndim = ndim < 0 ? ndimension : ndim;
		List<Double> mus = po.getObject().getMus();
		for (int i = 0; i < ndim; i++) {
			double rndn = (LVeGLearner.random.nextDouble() - LVeGLearner.nratio) * maximum;
			// rndn = 0.5;
			mus.add(rndn);
		} // better initialize mu and var in the different loops
		List<Double> vars = po.getObject().getVars();
		for (int i = 0; i < ndim; i++) {
			double rndn = (LVeGLearner.random.nextDouble() - LVeGLearner.nratio) * maximum;
			// rndn = 0.5;
			vars.add(rndn);
		}	
	}

	@Override
	public void destroyObject(Short ndim, PooledObject<GaussianDistribution> po) throws Exception {
		return;
	}

	@Override
	public PooledObject<GaussianDistribution> makeObject(Short ndim) throws Exception {
		ndim = ndim < 0 ? ndimension : ndim;
		GaussianDistribution gauss = new DiagonalGaussianDistribution(ndim);
		return new DefaultPooledObject<GaussianDistribution>(gauss);
	}

	@Override
	public void passivateObject(Short ndim, PooledObject<GaussianDistribution> po) throws Exception {
		po.getObject().clear();
	}

	@Override
	public boolean validateObject(Short ndim, PooledObject<GaussianDistribution> po) {
		return po.getObject().isValid();
	}

}
