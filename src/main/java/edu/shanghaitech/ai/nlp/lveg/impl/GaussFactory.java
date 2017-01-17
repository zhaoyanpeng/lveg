package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;
import java.util.Random;

import org.apache.commons.pool2.KeyedPooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

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
	}
	
	@Override
	public void activateObject(Short key, PooledObject<GaussianDistribution> po) throws Exception {
		short ndim = key == -1 ? ndimension : key;
		List<Double> mus = po.getObject().getMus();
		for (int i = 0; i < ndim; i++) {
			double rndn = (rnd.nextDouble() - nratio) * maximum;
			// rndn = 0.5;
			mus.add(rndn);
		}
		List<Double> vars = po.getObject().getVars();
		for (int i = 0; i < ndim; i++) {
			double rndn = (rnd.nextDouble() - nratio) * maximum;
			// rndn = 0.5;
			vars.add(rndn);
		}	
	}

	@Override
	public void destroyObject(Short key, PooledObject<GaussianDistribution> po) throws Exception {
		po.getObject().destroy(key);
	}

	@Override
	public PooledObject<GaussianDistribution> makeObject(Short key) throws Exception {
		short ndim = key == -1 ? ndimension : key;
		GaussianDistribution gauss = new DiagonalGaussianDistribution(ndim, false);
		gauss.setKey(key);
		return new DefaultPooledObject<GaussianDistribution>(gauss);
	}

	@Override
	public void passivateObject(Short key, PooledObject<GaussianDistribution> po) throws Exception {
		po.getObject().clear(key);
	}

	@Override
	public boolean validateObject(Short key, PooledObject<GaussianDistribution> po) {
		GaussianDistribution obj = po.getObject();
		return obj != null && obj.isValid(key);
	}

}
