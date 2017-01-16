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
	protected short maximum;
	protected double nratio;
	protected Random rnd;
	
	public MoGFactory(short ncomponent, short maximum, double nratio, Random rnd) {
		this.ncomponent = ncomponent;
		this.maximum = maximum;
		this.nratio = nratio;
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
	public void activateObject(Short ncomp, PooledObject<GaussianMixture> po) throws Exception {
		ncomp = ncomp < 0 ? ncomponent : ncomp;
		PriorityQueue<Component> components = po.getObject().components();
//		System.out.println("----------Alag B " + po.getObject());
		for (int i = 0; i < ncomp; i++) {
			double weight = (rnd.nextDouble() - nratio) * maximum;
			Map<String, Set<GaussianDistribution>> multivnd = new HashMap<String, Set<GaussianDistribution>>();
			// weight = /*-0.69314718056*/ 0; // mixing weight 0.5, 1, 2
			components.add(new Component((short) i, weight, multivnd));
		}
//		System.out.println("----------Alag A " + po.getObject());
	}

	@Override
	public void destroyObject(Short ncomp, PooledObject<GaussianMixture> po) throws Exception {
		ncomp = ncomp < 0 ? ncomponent : ncomp;
		po.getObject().destroy(ncomp);
	}

	@Override
	public PooledObject<GaussianMixture> makeObject(Short ncomp) throws Exception {
		ncomp = ncomp < 0 ? ncomponent : ncomp;
		GaussianMixture mog = new DiagonalGaussianMixture(ncomp, false);
//		System.out.println("_____________MoG: " + mog);
		return new DefaultPooledObject<GaussianMixture>(mog);
	}

	@Override
	public void passivateObject(Short ncomp, PooledObject<GaussianMixture> po) throws Exception {
		po.getObject().clear();
	}

	@Override
	public boolean validateObject(Short ncomp, PooledObject<GaussianMixture> po) {
		ncomp = ncomp < 0 ? ncomponent : ncomp;
		GaussianMixture obj = po.getObject();
		return obj != null && obj.isValid(ncomp);
//		boolean flag = obj != null && obj.isValid(ncomp);
//		System.out.println("----------Blag " + obj);
//		System.out.println("----------Flag " + flag + " = " + (obj != null) + " & " + obj.isValid(ncomp) + " nc = " + ncomp);
//		return flag;
	}
}
