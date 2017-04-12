package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianDistribution;
import edu.shanghaitech.ai.nlp.util.FunUtil;

/**
 * Diagonal Gaussian distribution.
 * 
 * @author Yanpeng Zhao
 *
 */
public class DiagonalGaussianDistribution extends GaussianDistribution {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4484028334969182856L;

	public DiagonalGaussianDistribution() {
		super((short) 0);
	}
	
	public DiagonalGaussianDistribution(short ndimension) {
		super(ndimension);
		initialize();
	}
	
	public DiagonalGaussianDistribution(short ndimension, boolean init) {
		super(ndimension);
		if (init) { initialize(); }
	}
	
	@Override
	public GaussianDistribution instance(short ndimension, boolean init) {
		return new DiagonalGaussianDistribution(ndimension, init);
	}
	
	public static DiagonalGaussianDistribution borrowObject(short ndimension) {
		GaussianDistribution obj = null;
		try {
			obj = defObjectPool.borrowObject(ndimension);
		} catch (Exception e) {
			logger.error("---------Borrow GD " + e + "\n");
			try {
				LVeGTrainer.gaussPool.invalidateObject(ndimension, obj);
			} catch (Exception e1) {
				logger.error("---------Borrow GD(invalidate) " + e + "\n");
			}
			ndimension = ndimension == -1 ? defNdimension : ndimension;
			obj = new DiagonalGaussianDistribution(ndimension);
		}
		return (DiagonalGaussianDistribution) obj;
	}
	
	@Override
	protected void initialize() {
		for (int i = 0; i < dim; i++) {
			double rndn = (defRnd.nextDouble() - defNegMRatio) * defMaxMu;
//			 rndn = /*0.5*/ 0;
			mus.add(rndn);
		} // better initialize mu and var in the different loops
		for (int i = 0; i < dim; i++) {
			double rndn = (defRnd.nextDouble() - defNegVRatio) * defMaxVar;
			 rndn = /*0.5*/ 0 /*Math.log(1e-12)*/;
			vars.add(rndn);
		}	
	}
	
	@Override
	public DiagonalGaussianDistribution copy() {
		DiagonalGaussianDistribution gd = new DiagonalGaussianDistribution();
//		DiagonalGaussianDistribution gd = borrowObject((short) 0); // POOL
		copy(gd);
		return gd;
	}
	
	
	@Override
	protected double eval(List<Double> sample, boolean normal) { 
		if (sample != null && sample.size() == dim) {
			// sample = normalize(sample);
			double astd, norm, exps = 0.0, sinv = 0.0;
			for (int i = 0; i < dim; i++) {
				astd = Math.exp(vars.get(i));
				norm = normal ? sample.get(i) : (sample.get(i) - mus.get(i)) / astd;
				exps -= Math.pow(norm, 2) / 2;
				sinv -= vars.get(i);
			}
			double value = -(dim / 2.0) * Math.log(2 * Math.PI) + sinv + exps; // (dim / 2) - (dim / 2.0)
			return value;
			/*
			double astd, norm, exps = 0.0, sinv = 1.0;
			for (int i = 0; i < dim; i++) {
				astd = Math.exp(vars.get(i));
				norm = normal ? sample.get(i) : (sample.get(i) - mus.get(i)) / astd;
				exps -= Math.pow(norm, 2) / 2;
				sinv /= astd;
			}
			double value = Math.pow(2 * Math.PI, -dim / 2.0) * sinv * Math.exp(exps); // (dim / 2) - (dim / 2.0)
			return value;
			*/
		}
		logger.error("Invalid input sample for evaling the gaussian. sample: " + sample + ", this: " + this + "\n");
		return Double.NEGATIVE_INFINITY;
	}
	
	@Override
	public void derivative(boolean cumulative, List<Double> grads, List<Double> gradst, List<Double> gradss, double scoreT, double scoreS) {
		if (!cumulative) {
			grads.clear();
			for (int i = 0; i < dim * 2; i++) {
				grads.add(0.0);
			}
		}
		double tmps, tmpt, grad;
		for (int i = 0; i < dim * 2; i++) {
			tmpt = gradst == null ? 0 : gradst.get(i);
			tmps = gradss == null ? 0 : gradss.get(i);
			grad = tmps / scoreS - tmpt / scoreT;
			grads.set(i, grads.get(i) + grad);
		}
	}
	
	
	@Override
	public void derivative(boolean cumulative, double factor, List<Double> grads, List<List<Double>> caches) {
		if (!cumulative) {
			grads.clear();
			for (int i = 0; i < dim * 2; i++) {
				grads.add(0.0);
			}
		}
		int ncomp = caches.get(0).size() - 1;
		double mgrad, vgrad, vtmp0, vtmp1, mu, var;
		double mall = 0, vall = 0, munit = 0, vunit = 0;
		double aconst = Math.log(2 * Math.PI) * (-dim / 2.0);
		List<Double> weights = caches.get(caches.size() - 1);
		for (int i = 0; i < dim; i++) {
			mall = 0;
			vall = 0;
			mu = mus.get(i);
			var = Math.exp(vars.get(i) * 2);
			for (int icomp = 0; icomp < ncomp; icomp++) {
				vtmp0 = weights.get(icomp);
				vtmp1 = weights.get(icomp);
				munit = caches.get(dim + i).get(icomp); 
				vunit = caches.get(dim * 2 + i).get(icomp); 
				for (int j = 0; j < dim; j++) {
					if (j != i) { // in logarithmic form
						vtmp0 += caches.get(j).get(icomp);
					}
					vtmp1 += caches.get(j).get(icomp);
				}
				vtmp0 += aconst; // integrals from NN but x in the current dimension
				vtmp1 += aconst; // integrals from NN (explicit normalizer in normal distribution)
				munit = munit * Math.exp(vtmp0); // integrals from xNN, non-logarithmic form
				vunit = Math.exp(vunit + vtmp0); // integrals from xxNN, in logarithmic form
				
				vtmp1 = Math.exp(vtmp1);
				munit = munit / var; // xNN  / var 
				mgrad = munit - vtmp1 * mu / var;
				vunit = vunit / var; // xxNN / var 
				vgrad = vunit - 2 * mu * munit + (mu * mu / var - 1) * vtmp1;
				
				mall += mgrad;
				vall += vgrad;
			}
			mgrad = factor * mall;
			vgrad = factor * vall;
			grads.set(i * 2, grads.get(i * 2) + mgrad);
			grads.set(i * 2 + 1, grads.get(i * 2 + 1) + vgrad);
		}
	}
	
	
	@Override
	public double integral(GaussianDistribution gd, List<List<Double>> cache) {
		if (gd != null && gd.getDim() == dim) {
			double value = 0, vtmp = 0, epsilon = 0;
			List<Double> order0, order1, order2;
			List<Double> vars1 = gd.getVars();
			List<Double> mus1 = gd.getMus();
			for (int i = 0; i < dim; i++) {
				double mu0 = mus.get(i), mu1 = mus1.get(i);
				double vr0 = Math.exp(vars.get(i) * 2), vr1 = Math.exp(vars1.get(i) * 2);
				vtmp = vr0 + vr1 + epsilon;
				// int NN dx
				order0 = cache.get(i);
				double shared0 = -0.5 * Math.log(vtmp) - Math.pow(mu0 - mu1, 2) / (2 * vtmp);
				order0.add(shared0); // in logarithmic form
				// int xNN dx
				order1 = cache.get(dim + i);
				double shared1 = (mu0 * vr1 + mu1 * vr0) / vtmp;
				double realval = shared1 * Math.exp(shared0);
				order1.add(realval); // not in logarithmic form since shared1 may be negative
				// int xxNN dx
				order2 = cache.get(dim * 2 + i);
				double part20 = 2 * Math.log(Math.abs(shared1));
				double part21 = 2 * (vars.get(i) + vars1.get(i)) - Math.log(vtmp);
				realval = FunUtil.logAdd(part20, part21) + shared0;
				order2.add(realval); // in logarithmic form
				// complete integral
				value += shared0;
			}
			value += Math.log(2 * Math.PI) * (-dim / 2.0); // normalizer in Gaussian is implicitly cached
			return value;
		}
		logger.error("Invalid multipliers. input: " + gd + ", this: " + this + "\n");
		return Double.NEGATIVE_INFINITY; 
	}
	
	
	@Override
	public double mulAndMarginalize(GaussianDistribution gd) { 
		if (gd != null && gd.getDim() == dim) {
			double value = 0, vtmp = 0, epsilon = /*1e-8*/0;
			List<Double> vars1 = gd.getVars();
			List<Double> mus1 = gd.getMus();
			for (int i = 0; i < dim; i++) {
				vtmp = 2 * (Math.exp(vars.get(i) * 2) + Math.exp(vars1.get(i) * 2)) + epsilon;
				value += -0.5 * Math.log(vtmp * Math.PI) - Math.pow(mus.get(i) - mus1.get(i), 2) / vtmp;
			}
			return value;
		}
		logger.error("Invalid multipliers. input: " + gd + ", this: " + this + "\n");
		return Double.NEGATIVE_INFINITY; 
	};
	
	
	@Override
	protected void derivative(double factor, List<Double> sample, List<Double> grads, boolean cumulative, boolean normal) {
		if (sample != null && sample.size() == dim) {
			if (!cumulative) {
				grads.clear();
				for (int i = 0; i < dim * 2; i++) {
					grads.add(0.0);
				}
			}
			double sigma, point, mgrad, vgrad;
			for (int i = 0; i < dim; i++) {
				sigma = Math.exp(vars.get(i));
				point = normal ? sample.get(i) : (sample.get(i) - mus.get(i)) / sigma;
				mgrad = factor * point / sigma;
				// CHECK dw / dx = (dw / ds) * (ds / dx) = (factor * (point^2 - 1) / s) * (s), 
				// where s = sigma = std = exp(x)
				vgrad = factor * (Math.pow(point, 2) - 1);
				grads.set(i * 2, grads.get(i * 2) + mgrad);
				grads.set(i * 2 + 1, grads.get(i * 2 + 1) + vgrad);
			}
			return;
		}
		logger.error("Invalid input sample for taking derivative of the gaussian w.r.t mu & sigma.\n");
	}
	
}
