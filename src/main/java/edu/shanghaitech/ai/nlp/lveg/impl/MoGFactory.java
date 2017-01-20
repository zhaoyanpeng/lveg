package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import org.apache.commons.pool2.KeyedPooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture.Component;

public class MoGFactory implements KeyedPooledObjectFactory<Short, GaussianMixture> {
	
	protected short ncomponent;
	protected double nwratio;
	protected double maxmw;
	protected Random rnd;
	
	public MoGFactory(short ncomponent, double maxmw, double nwratio, Random rnd) {
		this.ncomponent = ncomponent;
		this.nwratio = nwratio;
		this.maxmw = maxmw;
		this.rnd = rnd;
	}
	
	/**
	 * https://commons.apache.org/proper/commons-pool/apidocs/index.html?org/apache/commons/pool2/class-use/PooledObject.html
	 * The above link tells us that activateObject(K, org.apache.commons.pool2.PooledObject<V>) is invoked on every instance 
	 * that has been passivated before it is borrowed from the pool, which is DEFINITELY NOT consist with the codes. In line
	 * 395 of <code>GenericKeyedObjectPool</code>, every non-null empty object, including the newly-created one, will be passed
	 * into {@code activateObject(K, T)}. TODO modify and rebuild our own pool2 library.
	 */
	@Override
	public void activateObject(Short key, PooledObject<GaussianMixture> po) throws Exception {
		short ncomp = key == -1 ? ncomponent : key;
		PriorityQueue<Component> components = po.getObject().components();
		for (int i = 0; i < ncomp; i++) {
			double weight = (rnd.nextDouble() - nwratio) * maxmw;
			Map<String, Set<GaussianDistribution>> multivnd = new HashMap<String, Set<GaussianDistribution>>();
			// weight = /*-0.69314718056*/ 0; // mixing weight 0.5, 1, 2
			components.add(new Component((short) i, weight, multivnd));
		}
	}

	@Override
	public void destroyObject(Short key, PooledObject<GaussianMixture> po) throws Exception {
		po.getObject().destroy(key);
	}

	@Override
	public PooledObject<GaussianMixture> makeObject(Short key) throws Exception {
		short ncomp = key == -1 ? ncomponent : key;
		GaussianMixture mog = new DiagonalGaussianMixture(ncomp, false);
		mog.setKey(key);
		return new DefaultPooledObject<GaussianMixture>(mog);
	}

	@Override
	public void passivateObject(Short key, PooledObject<GaussianMixture> po) throws Exception {
		po.getObject().clear(key);
	}

	@Override
	public boolean validateObject(Short key, PooledObject<GaussianMixture> po) {
		GaussianMixture obj = po.getObject();
		return obj != null && obj.isValid(key);
	}
}
