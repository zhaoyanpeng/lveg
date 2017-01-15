package edu.shanghaitech.ai.nlp.util;

import java.io.Serializable;

import org.apache.commons.pool2.KeyedPooledObjectFactory;
import org.apache.commons.pool2.impl.GenericKeyedObjectPool;
import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;

public class ObjectPool<K, T> extends GenericKeyedObjectPool<K, T> implements Serializable {
	public ObjectPool(KeyedPooledObjectFactory<K, T> factory) {
		super(factory);
	}

	public ObjectPool(KeyedPooledObjectFactory<K, T> factory, GenericKeyedObjectPoolConfig config) {
		super(factory, config);
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1542223532794120768L;

}
