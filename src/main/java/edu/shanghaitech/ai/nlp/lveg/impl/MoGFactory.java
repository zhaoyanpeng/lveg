package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.pool2.KeyedPooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

import edu.shanghaitech.ai.nlp.lveg.LVeGLearner;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;

public class MoGFactory implements KeyedPooledObjectFactory<Short, GaussianMixture> {
	
	protected short ncomponent;
	protected short maximum;
	protected double nratio;
	protected Random rnd;
//	ThreadLocalRandom.current().nextGaussian();
	
	public MoGFactory(short ncomponent, short maximum, double nratio, Random rnd) {
		this.ncomponent = ncomponent;
		this.maximum = maximum;
		this.nratio = nratio;
		this.rnd = rnd;
	}
	
//	public 

	@Override
	public void activateObject(Short ncomp, PooledObject<GaussianMixture> po) throws Exception {
		ncomp = ncomp < 0 ? ncomponent : ncomp;
		PriorityQueue<Component> components = po.getObject().components();
		for (int i = 0; i < ncomp; i++) {
			double weight = (LVeGLearner.random.nextDouble() - LVeGLearner.nratio) * maximum;
			Map<String, Set<GaussianDistribution>> multivnd = new HashMap<String, Set<GaussianDistribution>>();
			// weight = /*-0.69314718056*/ 0; // mixing weight 0.5, 1, 2
			components.add(new Component((short) i, weight, multivnd));
		}
	}

	@Override
	public void destroyObject(Short ncomp, PooledObject<GaussianMixture> po) throws Exception {
		return;
	}

	@Override
	public PooledObject<GaussianMixture> makeObject(Short ncomp) throws Exception {
		ncomp = ncomp < 0 ? ncomponent : ncomp;
		GaussianMixture mog = new DiagonalGaussianMixture(ncomp);
		return new DefaultPooledObject<GaussianMixture>(mog);
	}

	@Override
	public void passivateObject(Short ncomp, PooledObject<GaussianMixture> po) throws Exception {
		po.getObject().clear();
	}

	@Override
	public boolean validateObject(Short ncomp, PooledObject<GaussianMixture> po) {
		return po.getObject().isValid();
	}
}
