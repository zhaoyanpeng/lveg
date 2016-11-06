package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.nlp.util.MethodUtil;

/**
 * @author Yanpeng Zhao
 *
 */
public class GaussianDistribution implements Comparable<Object> {
	
	protected short dim;
	
	protected List<Double> mus;
	protected List<Double> sigmas;
	
	
	public GaussianDistribution() {
		this.dim = 0;
		this.mus = new ArrayList<Double>();
		this.sigmas = new ArrayList<Double>();
	}
	
	
	public GaussianDistribution(short dim) {
		this.dim = dim;
		this.mus = new ArrayList<Double>();
		this.sigmas = new ArrayList<Double>();
		initialize();
	}
	
	
	/**
	 * Memory allocation and initialization.
	 */
	private void initialize() {
		MethodUtil.randomInitList(mus, Double.class, dim, LVeGLearner.maxrandom);
		MethodUtil.randomInitList(sigmas, Double.class, dim, LVeGLearner.maxrandom);
	}
	
	
	/**
	 * Make a copy of the instance.
	 * 
	 * @return
	 */
	public GaussianDistribution copy() {
		GaussianDistribution gd = new GaussianDistribution();
		gd.dim = dim;
		for (int i = 0; i < dim; i++) {
			gd.mus.add(mus.get(i));
			gd.sigmas.add(sigmas.get(i));
		}
		return gd;
	}
	
	
	/**
	 * Memory clean.
	 */
	public void clear() {
		this.dim = 0;
		this.mus.clear();
		this.sigmas.clear();
	}
	
	
	@Override
	public int hashCode() {
		return dim ^ mus.hashCode() ^ sigmas.hashCode();
	}
	
	
	@Override
	public boolean equals(Object o) {
		if (this == o) { return true; }
		
		if (o instanceof GaussianDistribution) {
			GaussianDistribution gd = (GaussianDistribution) o;
			if (dim == gd.dim && mus.equals(gd.mus) && sigmas.equals(gd.sigmas)) {
				return true;
			}
		}
		return false;
	}
	
	
	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		GaussianDistribution gd = (GaussianDistribution) o;
		if (dim < gd.dim) { return -1; }
		if (dim > gd.dim) { return 1;  }
		/*
		if (dim > 0 && mus.get(0) < gd.mus.get(0)) { return -1; }
		if (dim > 0 && mus.get(0) > gd.mus.get(0)) { return  1; }
		if (dim > 0 && sigmas.get(0) < gd.sigmas.get(0)) { return -1; }
		if (dim > 0 && sigmas.get(0) > gd.sigmas.get(0)) { return  1; }
		*/
		if (mus.equals(gd.mus) && sigmas.equals(gd.sigmas)) { return 0; }
		return -1;
	}


	@Override
	public String toString() {
		return "GD [dim=" + dim + ", mus=" + MethodUtil.double2str(mus, LVeGLearner.precision) + 
				", sigmas=" + MethodUtil.double2str(sigmas, LVeGLearner.precision) + "]";
	}

}
