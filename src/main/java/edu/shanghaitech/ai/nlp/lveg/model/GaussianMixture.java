package edu.shanghaitech.ai.nlp.lveg.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.util.FunUtil;
import edu.shanghaitech.ai.nlp.util.ObjectPool;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public abstract class GaussianMixture extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -822680841484765529L;
	private static final double LOG_ZERO = -1.0e10;
	private static double EXP_ZERO = /*-Math.log(-LOG_ZERO)*/Math.log(1e-6);
	
	protected static short defMaxNbig;
	protected static short defNcomponent;
	protected static double defSexp;
	protected static double defMaxmw;
	protected static double defRetainRatio;
	protected static double defNegWRatio;
	protected static double defRiseRate;
	protected static boolean defHardCut;
	protected static ObjectPool<Short, GaussianMixture> defObjectPool;
	protected static Random defRnd;
	
	protected double bias;
	protected double prob;
	protected int ncomponent;
	
	protected GaussianMixture(short ncomponent) {
		this.bias = 0;
		this.prob = 0;
		this.ncomponent = ncomponent;
		
		// new model 
		this.binding = null;
		this.weights = new ArrayList<>();
		this.gausses = new EnumMap<>(RuleUnit.class);
		this.spviews = new ArrayList<>();
	}
	
	
	protected abstract void initialize();
	public abstract GaussianMixture instance(short ncomponent, boolean init);
	public abstract double mulAndMarginalize(EnumMap<RuleUnit, GaussianMixture> counts);
	public abstract GaussianMixture mulAndMarginalize(GaussianMixture gm, GaussianMixture des, RuleUnit key, boolean deep);
	public abstract GaussianMixture mul(GaussianMixture gm, GaussianMixture des, RuleUnit key);
	
	/**
	 * To facilitate the parameter tuning.
	 * 
	 */
	public static void config(short maxnbig, double expzero, double maxmw, short ncomponent, double negwratio, 
			double riserate, double retainratio, boolean hardcut, Random rnd, ObjectPool<Short, GaussianMixture> pool) {
		EXP_ZERO = Math.log(expzero);
		defMaxNbig = maxnbig;
		defRnd = rnd;
		defMaxmw = maxmw;
		defHardCut = hardcut;
		defRiseRate = riserate;
		defNegWRatio = negwratio;
		defRetainRatio = retainratio;
		defNcomponent = ncomponent;
		defObjectPool = pool;
	}
	
	
	/**
	 * Remove the trivial components.
	 */
	public void delTrivia() {
		if (ncomponent <= 1) { return; }
		PriorityQueue<SimpleView> sorted = sort();
		if (defMaxNbig > 0 && (defHardCut || defRetainRatio > 0 || defRiseRate > 0)) {
			int base = 0;
			if (defHardCut) {
				base = defMaxNbig;
			} else if (defRetainRatio > 0) {
				base = sorted.size();
				base = base > defMaxNbig ? (defMaxNbig + (int) (defRetainRatio * base)) : defMaxNbig;
			} else {
				base = sorted.size();
				if (base < defMaxNbig + 1) {
					base = 0;
				} else {
					base = (int) Math.floor(Math.pow(base, LVeGTrainer.squeezeexp));
				}
				base = base + defMaxNbig;
				base = base > 50 ? 50 : base; // hard coding for debugging
			}
			spviews.clear();
			if (sorted.size() > base) {
				while (!sorted.isEmpty()) {
					spviews.add(sorted.poll());
					if (spviews.size() == base) { break; }
				}
			} else {
				spviews.addAll(sorted);
			}
			ncomponent = spviews.size();
		} else {
			double maxw = sorted.peek().weight;
			for (SimpleView view : spviews) {
				if (view.weight > LOG_ZERO && (view.weight - maxw) > EXP_ZERO) { 
					continue; 
				}
				sorted.remove(view);
			}
			spviews.clear();
			spviews.addAll(sorted);
			ncomponent = sorted.size();
		}
	}
	
	/**
	 * @return the sorted components by the mixing weight in case when you have modified the mixing weights of some components.
	 */
	public PriorityQueue<SimpleView> sort() {
		PriorityQueue<SimpleView> sorter = new PriorityQueue<>(ncomponent + 1, vcomparator);
		sorter.addAll(spviews);
		return sorter;
	}
	

	/**
	 * Add the bias.
	 * 
	 * @param bias constant bias
	 */
	public void add(double bias) {
		this.bias += bias;
	}
	
	
	/**
	 * Make a copy of this MoG. This will create a new instance of MoG.
	 * 
	 * @param deep boolean value, indicating deep (true) or shallow (false) copy
	 * @return
	 */
	public GaussianMixture copy(boolean deep) { return null; }
	
	
	/**
	 * Set the outside score of the root node to 1.
	 */
	public void marginalizeToOne() {
		// sanity check
		ncomponent = 1;
		gausses.clear();
		weights.clear();
		weights.add(0.0);
		buildSimpleView();
	}
	
	
	/**
	 * Eval the MoG using the given sample. Pay attention to the second parameter.
	 * 
	 * @param sample the sample, should be null
	 * @param normal whether the sample is from N(0, 1) (true) or not (false); 
	 * 		   when the sample is null, normal is used to indicate the return value in logarithm (true) or decimal (false)
	 * @return
	 */
	public double eval(EnumMap<RuleUnit, List<Double>> sample, boolean normal) {
		if (sample == null) { return eval(normal); }
		
		double ret = 0.0;
		return ret;
	}
	
	
	/**
	 * This method assumes the MoG is equivalent to a constant, e.g., no gaussians contained in any component.
	 * 
	 * @param normal whether the sample is from N(0, 1) (true) or not (false)
	 * @return
	 */
	private double eval(boolean logarithm) {
		if (logarithm) {
			double logval = Double.NEGATIVE_INFINITY;
			for (SimpleView view : spviews) {
				if (view.gaussian != null) {
					throw new RuntimeException("You are not supposed to call this method if the MoG contains any Gaussian.\n");
				}
				logval = FunUtil.logAdd(logval, view.weight);
			}
			return logval;
		} else {
			double value = 0;
			for (SimpleView view : spviews) {
				if (view.gaussian != null) {
					throw new RuntimeException("You are not supposed to call this method if the MoG contains any Gaussian.\n");
				}
				value += Math.exp(view.weight);
			}
			return value;
		}
	}
	
	
	/**
	 * Derivative w.r.t. mixing weight & mu & sigma.
	 */
	public void derivative(boolean cumulative, double scoreT, double scoreS, EnumMap<RuleUnit, List<List<Double>>> gradst, 
			EnumMap<RuleUnit, List<List<Double>>> gradss, EnumMap<RuleUnit, List<List<Double>>> grads, List<Double> wgrads, 
			List<EnumMap<RuleUnit, GaussianMixture>> cntsWithT, List<EnumMap<RuleUnit, GaussianMixture>> cntsWithS, 
			List<EnumMap<RuleUnit, List<List<List<Double>>>>> cachesWithT, List<EnumMap<RuleUnit, List<List<List<Double>>>>> cachesWithS) {
		if (!cumulative) { // CHECK stupid if...else..., wgrads is used by all components.
			wgrads.clear();
			for (int i = 0; i < ncomponent; i++) {
				wgrads.add(0.0);
			}
		}
		// memo
		allocateMemory(cntsWithT, cntsWithS, cachesWithT, cachesWithS);
		computeCaches(cntsWithT, cachesWithT);
		computeCaches(cntsWithS, cachesWithS);
		
		// w.r.t. mixing weight
		derivative(wgrads, cntsWithT, cntsWithS, cachesWithT, cachesWithS, scoreT, scoreS);
		
		// w.r.t. mu & sigma
		boolean zeroflagt = derivative(gradst, cntsWithT, cachesWithT);
		boolean zeroflags = derivative(gradss, cntsWithS, cachesWithS);
		// note that either Math.exp(scoreT) or Math.exp(scoreS) could be 0. Here we have to pass them in decimal format
		// because it does not guarantee that gradst and gradss must be positive, which indicates that we have to store
		// both of them in decimal format. See details in DiagonalGaussianDistribution.derivative(...).
		derivative(cumulative, zeroflagt, zeroflags, scoreT, scoreS, gradst, gradss, grads);
	}
	
	
	/**
	 * Update cumulative gradients w.r.t mixing weights.
	 * 
	 * @param iComponent index of the component
	 * @param wgrads     which stores the cumulative gradients w.r.t mixing weights
	 * @param logw       mixing weight in logarithmic form
	 * @param valt       integrals given parse tree
	 * @param vals       integrals given sentence
	 * @param scoreT     score of the parse tree
	 * @param scoreS     score of the sentence
	 */
	private void updateWgrads(int iComponent, List<Double> wgrads, 
			double logw, double valt, double vals, double scoreT, double scoreS) {
		double weight = Math.exp(logw);
		double dMixingW = Math.exp(vals - scoreS) - Math.exp(valt - scoreT);
		double dPenalty = Params.reg ? (Params.l1 ? Params.wdecay * weight : Params.wdecay * Math.pow(weight, 2)) : 0.0;
		dMixingW += dPenalty;
		dMixingW = Math.abs(dMixingW) > Params.absmax ? Params.absmax * Math.signum(dMixingW) : dMixingW;
		wgrads.set(iComponent, wgrads.get(iComponent) + dMixingW);
	}
	
	
	/**
	 * Derivatives w.r.t the mixing weights.
	 * 
	 * @param wgrads      cumulative gradients of mixing weights
	 * @param cachesWithT see {@link #computeCaches(List, List)}
	 * @param cachesWithS see {@link #computeCaches(List, List)}
	 */
	public void derivative(List<Double> wgrads, List<EnumMap<RuleUnit, GaussianMixture>> countsWithT, 
			List<EnumMap<RuleUnit, GaussianMixture>> countsWithS, List<EnumMap<RuleUnit, List<List<List<Double>>>>> cachesWithT, 
			List<EnumMap<RuleUnit, List<List<List<Double>>>>> cachesWithS, double scoreT, double scoreS) {
		double val, vals, valt, logw;
		switch (binding) {
		case RHSPACE: {
			for (int i = 0; i < ncomponent; i++) {
				vals = Double.NEGATIVE_INFINITY;
				valt = Double.NEGATIVE_INFINITY;
				logw = weights.get(i);
				
				if (countsWithT != null) {
					for (int j = 0; j < countsWithT.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesT = cachesWithT.get(j);
						List<List<List<Double>>> caches = cachesT.get(RuleUnit.C);
						List<Double> cache = caches.get(i).get(0);
						val = cache.get(cache.size() - 1) + logw;
						valt = FunUtil.logAdd(valt, val);
					}
				}
				
				if (countsWithS != null) {
					for (int j = 0; j < countsWithS.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesS = cachesWithS.get(j);
						List<List<List<Double>>> caches = cachesS.get(RuleUnit.C);
						List<Double> cache = caches.get(i).get(0);
						val = cache.get(cache.size() - 1) + logw;
						vals = FunUtil.logAdd(vals, val);
					}
				}
				updateWgrads(i, wgrads, logw, valt, vals, scoreT, scoreS);
			}
			break;
		}
		case LHSPACE: {
			for (int i = 0; i < ncomponent; i++) {
				vals = Double.NEGATIVE_INFINITY;
				valt = Double.NEGATIVE_INFINITY;
				logw = weights.get(i);
				
				if (countsWithT != null) {
					for (int j = 0; j < countsWithT.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesT = cachesWithT.get(j);
						List<List<List<Double>>> caches = cachesT.get(RuleUnit.P);
						List<Double> cache = caches.get(i).get(0);
						val = cache.get(cache.size() - 1) + logw;
						valt = FunUtil.logAdd(valt, val);
					}
				}
				
				if (countsWithS != null) {
					for (int j = 0; j < countsWithS.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesS = cachesWithS.get(j);
						List<List<List<Double>>> caches = cachesS.get(RuleUnit.P);
						List<Double> cache = caches.get(i).get(0);
						val = cache.get(cache.size() - 1) + logw;
						vals = FunUtil.logAdd(vals, val);
					}
				}
				updateWgrads(i, wgrads, logw, valt, vals, scoreT, scoreS);
			}
			break;
		}
		case LRURULE: {
			int nuc = gausses.get(RuleUnit.UC).size();
			for (int i = 0; i < ncomponent; i++) {
				vals = Double.NEGATIVE_INFINITY;
				valt = Double.NEGATIVE_INFINITY;
				logw = weights.get(i);
				
				if (countsWithT != null) {
					for (int j = 0; j < countsWithT.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesT = cachesWithT.get(j);
						List<List<List<Double>>> caches0 = cachesT.get(RuleUnit.P);
						List<List<List<Double>>> caches1 = cachesT.get(RuleUnit.UC);
						List<Double> cache0 = caches0.get(i / nuc).get(0);
						List<Double> cache1 = caches1.get(i % nuc).get(0);
						val = cache0.get(cache0.size() - 1) + cache1.get(cache1.size() - 1) + logw;
						valt = FunUtil.logAdd(valt, val);
					}
				}
				
				if (countsWithS != null) {
					for (int j = 0; j < countsWithS.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesS = cachesWithS.get(j);
						List<List<List<Double>>> caches0 = cachesS.get(RuleUnit.P);
						List<List<List<Double>>> caches1 = cachesS.get(RuleUnit.UC);
						List<Double> cache0 = caches0.get(i / nuc).get(0);
						List<Double> cache1 = caches1.get(i % nuc).get(0);
						val = cache0.get(cache0.size() - 1) + cache1.get(cache1.size() - 1) + logw;
						vals = FunUtil.logAdd(vals, val);
					}
				}
				updateWgrads(i, wgrads, logw, valt, vals, scoreT, scoreS);
			}
			break;
		}
		case LRBRULE: { // idx = np * |nlc| * |nrc| + nlc * |nrc| + nrc
			int nlc = gausses.get(RuleUnit.LC).size(), ip;
			int nrc = gausses.get(RuleUnit.RC).size(), ic;
			for (int i = 0; i < ncomponent; i++) {
				vals = Double.NEGATIVE_INFINITY;
				valt = Double.NEGATIVE_INFINITY;
				logw = weights.get(i);
				ip = i / (nlc * nrc);
				ic = i % (nlc * nrc);
				
				if (countsWithT != null) {
					for (int j = 0; j < countsWithT.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesT = cachesWithT.get(j);
						List<List<List<Double>>> caches0 = cachesT.get(RuleUnit.P);
						List<List<List<Double>>> caches1 = cachesT.get(RuleUnit.LC);
						List<List<List<Double>>> caches2 = cachesT.get(RuleUnit.RC);
						List<Double> cache0 = caches0.get(ip).get(0);
						List<Double> cache1 = caches1.get(ic / nrc).get(0);
						List<Double> cache2 = caches2.get(ic % nrc).get(0);
						val = cache0.get(cache0.size() - 1) + cache1.get(cache1.size() - 1) + cache2.get(cache2.size() - 1) + logw;
						valt = FunUtil.logAdd(valt, val);
					}
				}
				
				if (countsWithS != null) {
					for (int j = 0; j < countsWithS.size(); j++) {
						EnumMap<RuleUnit, List<List<List<Double>>>> cachesS = cachesWithS.get(j);
						List<List<List<Double>>> caches0 = cachesS.get(RuleUnit.P);
						List<List<List<Double>>> caches1 = cachesS.get(RuleUnit.LC);
						List<List<List<Double>>> caches2 = cachesS.get(RuleUnit.RC);
						List<Double> cache0 = caches0.get(ip).get(0);
						List<Double> cache1 = caches1.get(ic / nrc).get(0);
						List<Double> cache2 = caches2.get(ic % nrc).get(0);
						val = cache0.get(cache0.size() - 1) + cache1.get(cache1.size() - 1) + cache2.get(cache2.size() - 1) + logw;
						vals = FunUtil.logAdd(vals, val);
					}
				}
				updateWgrads(i, wgrads, logw, valt, vals, scoreT, scoreS);
			}
			break;
		}
		default: {
			throw new RuntimeException("Not consistent with any grammar rule type. Existing type: " + binding);
		}
		}
	}
	
	
	/**
	 * @param comp       the component of the rule weight
	 * @param cumulative accumulate gradients (true) or not (false)
	 * @param zeroflagt  if the expected counts with parse tree exist (true) or not (false)
	 * @param zeroflags  if the expected counts with sentence exist (true) or not (false)
	 * @param scoreT     score of the parse tree, should be in non-logarithmic form
	 * @param scoreS     score of the sentence, should be in non-logarithmic form
	 * @param gradst     intermediate values (with parse tree) from {@link #derivative(Map, Component, List, List)}
	 * @param gradss     intermediate values (with sentence) from {@link #derivative(Map, Component, List, List)}
	 * @param grads      which holds gradients of mu & sigma
	 */
	protected void derivative(boolean cumulative, boolean zeroflagt, boolean zeroflags, double scoreT, double scoreS,
			EnumMap<RuleUnit, List<List<Double>>> gradst, EnumMap<RuleUnit, List<List<Double>>> gradss, EnumMap<RuleUnit, List<List<Double>>> grads) {
		if (!(zeroflagt || zeroflags)) { logger.error("There must be something wrong.\n"); }
		for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
			List<List<Double>> ugrads = grads.get(unit.getKey());
			List<List<Double>> ugradst = gradst.get(unit.getKey());
			List<List<Double>> ugradss = gradss.get(unit.getKey());
			List<GaussianDistribution> gaussians = unit.getValue();
			for (int k = 0; k < gaussians.size(); k++) {
				GaussianDistribution gaussian = gaussians.get(k);
				List<Double> agrads = ugrads.get(k);
				List<Double> agradst = zeroflagt ? ugradst.get(k) : null;
				List<Double> agradss = zeroflags ? ugradss.get(k) : null;
				gaussian.derivative(cumulative, agrads, agradst, agradss, scoreT, scoreS);
			}
		}
	}
	

	
	/**
	 * Compute the intermediate values needed in gradients computation.
	 * 
	 * @param ggrads which stores intermediate values in computing gradients
	 * @param counts which decides if expected counts exist (true) or not (false)
	 * @param caches see {@link #computeCaches(List, List)}
	 * @return
	 */
	public boolean derivative(EnumMap<RuleUnit, List<List<Double>>> ggrads, 
			List<EnumMap<RuleUnit, GaussianMixture>> counts, List<EnumMap<RuleUnit, List<List<List<Double>>>>> caches) {
		if (counts == null) { return false; }
		double logw, factor;
		boolean cumulative = false;
		switch (binding) {
		case RHSPACE: {
			List<GaussianDistribution> gaussians = gausses.get(RuleUnit.C);
			for (int i = 0; i < counts.size(); i++) {
				List<List<List<Double>>> cache = caches.get(i).get(RuleUnit.C);
				List<List<Double>> grads = ggrads.get(RuleUnit.C);
				for (int k = 0; k < ncomponent; k++) {
					logw = weights.get(k);
					GaussianDistribution gaussian = gaussians.get(k);
					List<List<Double>> kcache = cache.get(k);
					List<Double> kgrad = grads.get(k);
					factor = logw;
					factor = Math.exp(factor);
					cumulative = (i != 0);
					gaussian.derivative(cumulative, factor, kgrad, kcache);
				}
			}
			break;
		}
		case LHSPACE: {
			List<GaussianDistribution> gaussians = gausses.get(RuleUnit.P);
			for (int i = 0; i < counts.size(); i++) {
				List<List<List<Double>>> cache = caches.get(i).get(RuleUnit.P);
				List<List<Double>> grads = ggrads.get(RuleUnit.P);
				for (int k = 0; k < ncomponent; k++) {
					logw = weights.get(k);
					GaussianDistribution gaussian = gaussians.get(k);
					List<List<Double>> kcache = cache.get(k);
					List<Double> kgrad = grads.get(k);
					factor = logw;
					factor = Math.exp(factor);
					cumulative = (i != 0);
					gaussian.derivative(cumulative, factor, kgrad, kcache);
				}
			}
			break;
		}
		case LRURULE: {
			List<GaussianDistribution> pGaussians = gausses.get(RuleUnit.P);
			List<GaussianDistribution> ucGaussians = gausses.get(RuleUnit.UC);
			int nuc = ucGaussians.size(), np = pGaussians.size(), idx;
			GaussianDistribution gaussian;
			for (int i = 0; i < counts.size(); i++) {
				List<List<Double>> grads = ggrads.get(RuleUnit.P);
				List<List<List<Double>>> pCache = caches.get(i).get(RuleUnit.P);
				List<List<List<Double>>> ucCache = caches.get(i).get(RuleUnit.UC);
				
				for (int ip = 0; ip < np; ip++) {
					gaussian = pGaussians.get(ip);
					List<Double> kgrad = grads.get(ip);
					List<List<Double>> kcache = pCache.get(ip);
					for (int iuc = 0; iuc < nuc; iuc++) {
						idx = ip * nuc + iuc;
						logw = weights.get(idx);
						List<Double> cache = ucCache.get(iuc).get(0);
						factor = logw + cache.get(cache.size() - 1);
						factor = Math.exp(factor);
						cumulative = ((i != 0) || (iuc != 0));
						gaussian.derivative(cumulative, factor, kgrad, kcache);
					}
				}
				
				grads = ggrads.get(RuleUnit.UC);
				for (int iuc = 0; iuc < nuc; iuc++) {
					gaussian = ucGaussians.get(iuc);
					List<Double> kgrad = grads.get(iuc);
					List<List<Double>> kcache = ucCache.get(iuc);
					for (int ip = 0; ip < np; ip++) {
						idx = ip * nuc + iuc;
						logw = weights.get(idx);
						List<Double> cache = pCache.get(ip).get(0);
						factor = logw + cache.get(cache.size() - 1);
						factor = Math.exp(factor);
						cumulative = ((i != 0) || (ip != 0));
						gaussian.derivative(cumulative, factor, kgrad, kcache);
					}
				}
			}
			break;
		}
		case LRBRULE: { // idx = np * |nlc| * |nrc| + nlc * |nrc| + nrc
			List<GaussianDistribution> pGaussians = gausses.get(RuleUnit.P);
			List<GaussianDistribution> lcGaussians = gausses.get(RuleUnit.LC);
			List<GaussianDistribution> rcGaussians = gausses.get(RuleUnit.RC);
			int nlc = lcGaussians.size(), nrc = rcGaussians.size(), np = pGaussians.size(), idx;
			
			GaussianDistribution gaussian;
			for (int i = 0; i < counts.size(); i++) {
				List<List<Double>> grads = ggrads.get(RuleUnit.P);
				List<List<List<Double>>> pCache = caches.get(i).get(RuleUnit.P);
				List<List<List<Double>>> lcCache = caches.get(i).get(RuleUnit.LC);
				List<List<List<Double>>> rcCache = caches.get(i).get(RuleUnit.RC);
				
				for (int ip = 0; ip < np; ip++) {
					gaussian = pGaussians.get(ip);
					List<Double> kgrad = grads.get(ip);
					List<List<Double>> kcache = pCache.get(ip);
					for (int ilc = 0; ilc < nlc; ilc++) {
						for (int irc = 0; irc < nrc; irc++) {
							idx = ip * nlc * nrc + ilc * nrc + irc;
							logw = weights.get(idx);
							List<Double> lcache = lcCache.get(ilc).get(0);
							List<Double> rcache = rcCache.get(irc).get(0);
							factor = logw + lcache.get(lcache.size() - 1) + rcache.get(rcache.size() - 1);
							factor = Math.exp(factor);
							cumulative = ((i != 0) || (ilc != 0) || (irc != 0));
							gaussian.derivative(cumulative, factor, kgrad, kcache);
						}
					}
				}
				
				grads = ggrads.get(RuleUnit.LC);
				for (int ilc = 0; ilc < nlc; ilc++) {
					gaussian = lcGaussians.get(ilc);
					List<Double> kgrad = grads.get(ilc);
					List<List<Double>> kcache = lcCache.get(ilc);
					for (int ip = 0; ip < np; ip++) {
						for (int irc = 0; irc < nrc; irc++) {
							idx = ip * nlc * nrc + ilc * nrc + irc;
							logw = weights.get(idx);
							List<Double> cache = pCache.get(ip).get(0);
							List<Double> rcache = rcCache.get(irc).get(0);
							factor = logw + cache.get(cache.size() - 1) + rcache.get(rcache.size() - 1);
							factor = Math.exp(factor);
							cumulative = ((i != 0) || (ip != 0) || (irc != 0));
							gaussian.derivative(cumulative, factor, kgrad, kcache);
						}
					}
				}
				
				
				grads = ggrads.get(RuleUnit.RC);
				for (int irc = 0; irc < nrc; irc++) {
					gaussian = rcGaussians.get(irc);
					List<Double> kgrad = grads.get(irc);
					List<List<Double>> kcache = rcCache.get(irc);
					for (int ip = 0; ip < np; ip++) {
						for (int ilc = 0; ilc < nlc; ilc++) {
							idx = ip * nlc * nrc + ilc * nrc + irc;
							logw = weights.get(idx);
							List<Double> cache = pCache.get(ip).get(0);
							List<Double> lcache = lcCache.get(ilc).get(0);
							factor = logw + cache.get(cache.size() - 1) + lcache.get(lcache.size() - 1);
							factor = Math.exp(factor);
							cumulative = ((i != 0) || (ip != 0) || (ilc != 0));
							gaussian.derivative(cumulative, factor, kgrad, kcache);
						}
					}
				}
			}
			break;
		}
		default: {
			throw new RuntimeException("Not consistent with any grammar rule type. Existing type: " + binding);
		}
		}
		return true;
	}
	
	
	/**
	 * Compute integrals of NN, xNN, xxNN, where x is the variable in some dimension, N is the d-dimensional Gaussian.
	 * 
	 * @param comp   current component
	 * @param counts rule counts given parse tree or sentence
	 * @param caches integrals holder 
	 * @return       in logarithmic form
	 */
	protected void computeCaches(List<EnumMap<RuleUnit, GaussianMixture>> counts, List<EnumMap<RuleUnit, List<List<List<Double>>>>> caches) {
		if (counts == null) { return; }
		for (int i = 0; i < counts.size(); i++) {
			EnumMap<RuleUnit, GaussianMixture> count = counts.get(i);
			EnumMap<RuleUnit, List<List<List<Double>>>> cache = caches.get(i);
			for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
				List<GaussianDistribution> gaussians = unit.getValue();
				List<List<List<Double>>> cunit = cache.get(unit.getKey());
				GaussianMixture ios = count.get(unit.getKey());
				for (int j = 0; j < gaussians.size(); j++) {
					List<List<Double>> gunit = cunit.get(j);
					integral(gaussians.get(j), ios, gunit);
				}
			}
		}
	}
	
	
	/**
	 * Integrals of NN, xNN, xxNN from a specific portion (outside score & head or inside score & tail variable).
	 * 
	 * @param gd    a specific portion of the grammar rule weight (head variable or tail variable)
	 * @param gm    inside/outside score, we shall restrict # of components because it is memory-consuming
	 * @param cache memory space
	 * @return      integrals in logarithmic form, I will give an example
	 */
	protected double integral(GaussianDistribution gd, GaussianMixture gm, List<List<Double>> cache) {
		double value = Double.NEGATIVE_INFINITY, vtmp;
		List<Double> weights = cache.get(cache.size() - 1);
		for (SimpleView view : gm.spviews) {
			vtmp = gd.integral(view.gaussian, cache);
			vtmp += view.weight; // integral contributed by one component
			value = FunUtil.logAdd(value, vtmp);
			weights.add(view.weight);
		}
		List<Double> sumvals = cache.get(0); // the last item in the first row
		sumvals.add(value); // sum of integrals from the current unit, in logarithmic form
		return value;
	}
	
	/**
	 * Assure the equality between the size of counts and that of the caches. 
	 * 
	 * @param cntsWithT   rule counts given parse tree
	 * @param cntsWithS   rule counts given the sentence
	 * @param cachesWithT see {@link #computeCaches(List, List)}
	 * @param cachesWithS see {@link #computeCaches(List, List)}
	 */
	protected void allocateMemory(List<EnumMap<RuleUnit, GaussianMixture>> cntsWithT, List<EnumMap<RuleUnit, GaussianMixture>> cntsWithS, 
			List<EnumMap<RuleUnit, List<List<List<Double>>>>> cachesWithT, List<EnumMap<RuleUnit, List<List<List<Double>>>>> cachesWithS) {
		int delta = -1;
		if (cntsWithT != null && (delta = cntsWithT.size() - cachesWithT.size()) > 0) {
			List<EnumMap<RuleUnit, List<List<List<Double>>>>> wantage = cachelike(delta, 50);
			cachesWithT.addAll(wantage);
		}
		delta = -1;
		if (cntsWithS != null && (delta = cntsWithS.size() - cachesWithS.size()) > 0) {
			List<EnumMap<RuleUnit, List<List<List<Double>>>>> wantage = cachelike(delta, 50);
			cachesWithS.addAll(wantage);
		}
	}
	
	
	/**
	 * Update parameters using the gradient.
	 * 
	 * @param icomponent index of the component of MoG
	 * @param ggrads     gradients of the parameters of gaussians
	 * @param wgrads     gradients of the mixing weights of MoG
	 * @param minexp     minimum exponent representing the exponential mixing weight
	 */
	public void update(EnumMap<RuleUnit, List<List<Double>>> ggrads, List<Double> wgrads, double minexp) {
		for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
			List<GaussianDistribution> gaussians = unit.getValue();
			List<List<Double>> grads = ggrads.get(unit.getKey());
			assert(gaussians.size() == grads.size());
			for (int i = 0; i < grads.size(); i++) {
				gaussians.get(i).update(grads.get(i));
			}
		}
		double weight = 0.0;
		for (int i = 0; i < weights.size(); i++) {
			weight = weights.get(i) + wgrads.get(i);
			weight = weight < minexp ? minexp : weight;
			weights.set(i, weight);
		}
		updateSimpleView();
	}
	
	
	/**
	 * Allocate memory space for gradients.
	 * 
	 * @param pad pad the allocated memory (true) or not (false)
	 * @return gradients holder
	 */
	public EnumMap<RuleUnit, List<List<Double>>> zeroslike(boolean pad) {
		EnumMap<RuleUnit, List<List<Double>>> grads = new EnumMap<>(RuleUnit.class);
		for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
			List<GaussianDistribution> gaussians = unit.getValue();
			List<List<Double>> gunit = new ArrayList<>(gaussians.size());
			for (GaussianDistribution gd : gaussians) {
				List<Double> grad = new ArrayList<>(gd.dim * 2 + 1);
				if (pad) {
					for (int i = 0; i < gd.dim * 2 + 1; i++) {
						grad.add(0.0); // preallocate memo
					}
				}
				gunit.add(grad);
			}
			grads.put(unit.getKey(), gunit);
		}
		return grads;
	}
	
	
	/**
	 * Allocate memory space for caches to be used for gradients calculation.
	 * 
	 * @param iComponent 0 by default, since all components have the same portions.
	 * @return caches holder
	 */
	public List<EnumMap<RuleUnit, List<List<List<Double>>>>> cachelike(int ncnt, int capacity) {
		int size = 0;
		List<EnumMap<RuleUnit, List<List<List<Double>>>>> caches = new ArrayList<>(ncnt);
		for (int i = 0; i < ncnt; i++) {
			EnumMap<RuleUnit, List<List<List<Double>>>> cache = new EnumMap<>(RuleUnit.class);
			for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
				List<GaussianDistribution> gaussians = unit.getValue();
				List<List<List<Double>>> cunit = new ArrayList<>();
				for (GaussianDistribution gd : gaussians) {
					size = gd.dim * 3 + 1;
					List<List<Double>> gunit = new ArrayList<>(size);
					for (int j = 0; j < size; j++) {
						gunit.add(new ArrayList<>(capacity));
					}
					cunit.add(gunit);
				}
				cache.put(unit.getKey(), cunit);
			}
			caches.add(cache);
		}
		return caches;
	}
	
	
	public int ncomponent() {
		return ncomponent;
	}
	
	
	public void setWeight(int iComponent, double weight) {
		weights.set(iComponent, weight);
	}
	
	
	public double getWeight(int iComponent) {
		return weights.get(iComponent);
	}
	
	
	public void setWeights(double weight) {
		for (int i = 0; i < weights.size(); i++) {
			weights.set(i, weight);
		}
	}
	
	
	public List<Double> getWeights() {
		return weights;
	}
	
	
	public List<?> getMixture() {
		if (binding == null || spviews.size() > 0) {
			return spviews;
		} else {
			List<SimpleComponent> mixture = new ArrayList<>(ncomponent + 1);
			for (int i = 0; i < ncomponent; i++) {
				SimpleComponent comp = new SimpleComponent();
				component(i, comp);
				mixture.add(comp);
			}
			return mixture;
		}
	}
	
	
	public void clear(boolean deep) {
		if (deep) {
			clear();
		} else {
			this.bias = 0.0;
			this.ncomponent = 0;
			gausses.clear();
			weights.clear();
			spviews.clear();
		}
	}
	
	
	private void clear() {
		this.bias = 0.0;
		this.ncomponent = 0;
		gausses.clear();
		weights.clear();
		spviews.clear();
	}
	
	
	public void clear(short ncomp) {
		clear();
	}
	
	
	public double getProb() {
		return prob;
	}
	
	
	public void setProb(double prob) {
		this.prob = prob;
	}
	
	
	public double getBias() {
		return bias;
	}
	

	public void setBias(double bias) {
		this.bias = bias;
	}
	

	public String toString(boolean simple, int nfirst) {
		if (simple) {
			StringBuffer sb = new StringBuffer();
			sb.append("GM [ncomponent=" + ncomponent + ", weights=" + 
//					FunUtil.double2str(getWeights(), 16, -1, false, false) + "<->" +
					FunUtil.double2str(getWeights(), LVeGTrainer.precision, nfirst, true, true));
			sb.append("]");
			return sb.toString();
		} else {
			return toString();
		}
	}
	
	
	@Override
	public String toString() {
		return "GM [bias=" + bias + ", ncomp=" + ncomponent + ", weights=" + 
//				FunUtil.double2str(getWeights(), 16, -1, false, false) + "<->" +
				FunUtil.double2str(getWeights(), LVeGTrainer.precision, -1, true, true) + ", mixture=" + getMixture() + "]";
	}

	
	
	protected RuleType binding; // to which kind of rule type the GM is bounded
	protected List<Double> weights;
	protected EnumMap<RuleUnit, List<GaussianDistribution>> gausses;
	
	// which is valid only for single variated MoG, we need this for the sake of 
	// representing in(out)-side scores regardless of which unit these Gaussians
	// exactly correspond to.
	protected List<SimpleView> spviews; 
	
	public RuleType getBinding() {
		return binding;
	}
	
	public void setBinding(RuleType binding) {
		this.binding = binding;
	}
	
	public List<SimpleView> spviews() {
		return spviews;
	}
	
	public void initMixingW(int ncomponent) {
		this.ncomponent = ncomponent;
		for (int i = 0; i < ncomponent; i++) {
			weights.add(0.0);
		}
	}
	
	public void add(GaussianMixture gm, boolean prune) {
		ncomponent += gm.ncomponent;
		spviews.addAll(gm.spviews);
		if (prune) { delTrivia(); }
	}
	
	/**
	 * The mixing weights should also be adjusted after calling this method.
	 * 
	 * @param key       id of the gaussian unit
	 * @param gaussians a set of gaussians corresponding to the 'key'
	 */
	protected void add(RuleUnit key, List<GaussianDistribution> gaussians) {
		List<GaussianDistribution> unit = null;
		if (gausses.containsKey(key) && (unit = gausses.get(key)) != null) {
			unit.addAll(gaussians);
		} else {
			gausses.put(key, gaussians);
		}
	}
	
	/**
	 * Make a copy of the MoG, only copy the internal data (gausses and weights).
	 * 
	 * @param des  the placeholder of MoG
	 * @param deep boolean value, indicating deep (true) or shallow (false) copy
	 */
	protected void copy(GaussianMixture des, boolean deep) {
		des.binding = binding;
		des.ncomponent = ncomponent;
		for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
			if (deep) {
				List<GaussianDistribution> gaussians = unit.getValue();
				List<GaussianDistribution> holder = new ArrayList<>(gaussians.size());
				for (GaussianDistribution gaussian : gaussians) {
					holder.add(gaussian.copy());
				}
				des.gausses.put(unit.getKey(), holder);
			} else {
				des.gausses.put(unit.getKey(), unit.getValue());
			}
		}
		des.weights.addAll(weights);
		des.buildSimpleView();
	}
	
	/**
	 * Build the view for the single variated MoG, which may facilitate the computation of in(out)-scores.
	 * 
	 * @return whether the view has been built (true) or not (false)
	 */
	protected boolean buildSimpleView() {
		spviews.clear();
		if (gausses.size() == 1) {
			for (Entry<RuleUnit, List<GaussianDistribution>> unit : gausses.entrySet()) {
				List<GaussianDistribution> gaussians = unit.getValue();
				assert(gaussians.size() == weights.size());
				for (int i = 0; i < weights.size(); i++) {
					spviews.add(new SimpleView(weights.get(i), gaussians.get(i)));
				}
				// break; // should break 
			}
			return true;
		} else if (gausses.size() == 0) {
			for (int i = 0; i < weights.size(); i++) {
				spviews.add(new SimpleView(weights.get(i), null));
			}
			return true;
		} else {
			return false;
		}
	}
	
	/**
	 * Note that views are valid only for single variated MoG, and the Gaussians are stored via 
	 * references while weights are not, thus weights of views must be updated after performing
	 * the modifications on them.
	 * 
	 * @return whether the view has been built (true) or not (false)
	 */
	protected boolean updateSimpleView() {
		int size = gausses.size();
		if (size == 0 || size == 1) {
			for (int i = 0; i < weights.size(); i++) {
				spviews.get(i).weight = weights.get(i);
			}
			return true;
		} else {
			return false;
		}
	}
	
	/**
	 * Marginalize the specific unit of the MoG.
	 * 
	 * @param key the unit to be marginalized
	 * @return 
	 */
	public boolean marginalize(RuleUnit key, List<Double> factors) {
		switch (binding) {
		case RHSPACE:
		case LHSPACE: {
			for (int i = 0; i < weights.size(); i++) {
				weights.set(i, weights.get(i) + factors.get(i));
			}
			gausses.remove(key);
			break;
		}
		case LRURULE: { // idx = np * |nuc| + nuc
			int nuc = 1, np = 1, idx;
			boolean valid = false;
			List<GaussianDistribution> gaussians = null;
			if ((gaussians = gausses.get(RuleUnit.UC)) != null && gaussians.size() > 0) {
				nuc = gaussians.size();
				valid = true;
			}
			if ((gaussians = gausses.get(RuleUnit.P)) != null && gaussians.size() > 0) {
				np = gaussians.size();
				valid = true;
			}
			if (valid) { // valid only when there is something available to be marginalized
				double logsum, tmp;
				switch (key) {
				case P: {
					for (int iuc = 0; iuc < nuc; iuc++) {
						logsum = Double.NEGATIVE_INFINITY;
						for (int ip = 0; ip < np; ip++) {
							idx = ip * nuc + iuc;
							tmp = weights.get(idx) + factors.get(ip);
							logsum = FunUtil.logAdd(logsum, tmp);
						}
						weights.add(logsum);
					}
					break;
				}
				case UC: {
					for (int ip = 0; ip < np; ip++) {
						logsum = Double.NEGATIVE_INFINITY;
						for (int iuc = 0; iuc < nuc; iuc++) {
							idx = ip * nuc + iuc;
							tmp = weights.get(idx) + factors.get(iuc);
							logsum = FunUtil.logAdd(logsum, tmp);
						}
						weights.add(logsum);
					}
					break;
				}
				default:
					break;
				}
				weights.subList(0, ncomponent).clear();
				ncomponent = weights.size();
				gausses.remove(key);
			} else {
				//
			}
			break;
		}
		case LRBRULE: { // idx = np * |nlc| * |nrc| + nlc * |nrc| + nrc
			int nlc = 1, nrc = 1, np = 1, idx;
			boolean valid = false;
			List<GaussianDistribution> gaussians = null;
			if ((gaussians = gausses.get(RuleUnit.RC)) != null && gaussians.size() > 0) {
				nrc = gaussians.size();
				valid = true;
			}
			if ((gaussians = gausses.get(RuleUnit.LC)) != null && gaussians.size() > 0) {
				nlc = gaussians.size();
				valid = true;
			}
			if ((gaussians = gausses.get(RuleUnit.P)) != null && gaussians.size() > 0) {
				np = gaussians.size();
				valid = true;
			}
			if (valid) {
				double logsum, tmp;
				switch (key) {
				case P: {
					for (int ilc = 0; ilc < nlc; ilc++) {
						for (int irc = 0; irc < nrc; irc++) {
							logsum = Double.NEGATIVE_INFINITY;
							for (int ip = 0; ip < np; ip++) {
								idx = ip * nlc * nrc + ilc * nrc + irc;
								tmp = weights.get(idx) + factors.get(ip);
								logsum = FunUtil.logAdd(logsum, tmp);
							}
							weights.add(logsum);
						}
					}
					break;
				}
				case LC: {
					for (int ip = 0; ip < np; ip++) {
						for (int irc = 0; irc < nrc; irc++) {
							logsum = Double.NEGATIVE_INFINITY;
							for (int ilc = 0; ilc < nlc; ilc++) {
								idx = ip * nlc * nrc + ilc * nrc + irc;
								tmp = weights.get(idx) + factors.get(ilc);
								logsum = FunUtil.logAdd(logsum, tmp);
							}
							weights.add(logsum);
						}
					}
					break;
				}
				case RC: {
					for (int ip = 0; ip < np; ip++) {
						for (int ilc = 0; ilc < nlc; ilc++) {
							logsum = Double.NEGATIVE_INFINITY;
							for (int irc = 0; irc < nrc; irc++) {
								idx = ip * nlc * nrc + ilc * nrc + irc;
								tmp = weights.get(idx) + factors.get(irc);
								logsum = FunUtil.logAdd(logsum, tmp);
							}
							weights.add(logsum);
						}
					}
					break;
				}
				default:
					break;
				}
				weights.subList(0, ncomponent).clear();
				ncomponent = weights.size();
				gausses.remove(key);
			} else {
				//
			}
			break;
		}
		default: {
			throw new RuntimeException("Not consistent with any grammar rule type. Existing type: " + binding);
		}
		}
		return buildSimpleView(); // build views for single variated MoG
	}
	
	/**
	 * Evaluate the value of MoG given a set of assignments.
	 * 
	 * @param factors the inputs used to instantiate MoG
	 * @return in logarithmic form
	 */
	public double eval(EnumMap<RuleUnit, List<Double>> factors) {
		double values = Double.NEGATIVE_INFINITY, value;
		switch (binding) {
		case RHSPACE: {
			List<Double> integrals = factors.get(RuleUnit.C);
			for (int i = 0; i < ncomponent; i++) {
				value = weights.get(i) + integrals.get(i);
				values = FunUtil.logAdd(values, value);
			}
			break;
		}
		case LHSPACE: {
			List<Double> integrals = factors.get(RuleUnit.P);
			for (int i = 0; i < ncomponent; i++) {
				value = weights.get(i) + integrals.get(i);
				values = FunUtil.logAdd(values, value);
			}
			break;
		}
		case LRURULE: { // idx = np * |nuc| + nuc
			int nuc = 1, np = 1, idx;
			boolean valid = false;
			List<Double> pIntegrals = null, ucIntegrals = null;
			if ((pIntegrals = factors.get(RuleUnit.P)) != null && pIntegrals.size() > 0) {
				nuc = pIntegrals.size();
				valid = true;
			}
			if ((ucIntegrals = factors.get(RuleUnit.UC)) != null && ucIntegrals.size() > 0) {
				np = ucIntegrals.size();
				valid = true;
			}
			if (valid) {
				for (int ip = 0; ip < np; ip++) {
					for (int iuc = 0; iuc < nuc; iuc++) {
						idx = ip * nuc + iuc;
						value = weights.get(idx) + pIntegrals.get(ip) + ucIntegrals.get(iuc);
						values = FunUtil.logAdd(values, value);
					}
				}
			} else {
				//
			}
			break;
		}
		case LRBRULE: { // idx = np * |nlc| * |nrc| + nlc * |nrc| + nrc
			int nlc = 1, nrc = 1, np = 1, idx;
			boolean valid = false;
			List<Double> pIntegrals = null, lcIntegrals = null, rcIntegrals = null;
			if ((rcIntegrals = factors.get(RuleUnit.RC)) != null && rcIntegrals.size() > 0) {
				nrc = rcIntegrals.size();
				valid = true;
			}
			if ((lcIntegrals = factors.get(RuleUnit.LC)) != null && lcIntegrals.size() > 0) {
				nlc = lcIntegrals.size();
				valid = true;
			}
			if ((pIntegrals = factors.get(RuleUnit.P)) != null && pIntegrals.size() > 0) {
				np = pIntegrals.size();
				valid = true;
			}
			if (valid) {
				for (int ip = 0; ip < np; ip++) {
					for (int ilc = 0; ilc < nlc; ilc++) {
						for (int irc = 0; irc < nrc; irc++) {
							idx = ip * nlc * nrc + ilc * nrc + irc;
							value = weights.get(idx) + pIntegrals.get(ip) + lcIntegrals.get(ilc) + rcIntegrals.get(irc);
							values = FunUtil.logAdd(values, value);
						}
					}
				}
			} else {
				//
			}
			break;
		}
		default: {
			throw new RuntimeException("Not consistent with any grammar rule type. Existing type: " + binding);
		}
		}
		return values;
	}
	
	/**
	 * Return a specific component of the mixture of Gaussians.
	 * 
	 * @param iComponent which specifies the id of the component to be queried
	 * @param holder     in which the returned data is stored
	 */
	public void component(int iComponent, SimpleComponent holder) {
		if (iComponent >= ncomponent) { // sanity check
			throw new RuntimeException("Invalid component index. Queried: " + iComponent + ", Maximum: " + ncomponent);
		} 
		holder.clear();
		holder.weight = weights.get(iComponent);
		switch (binding) {
		case RHSPACE: {
			List<GaussianDistribution> gaussians = null;
			if ((gaussians = gausses.get(RuleUnit.C)) != null) {
				holder.gausses.put(RuleUnit.C, gaussians.get(iComponent));
			}
			break;
		}
		case LHSPACE: {
			List<GaussianDistribution> gaussians = null;
			if ((gaussians = gausses.get(RuleUnit.P)) != null) {
				holder.gausses.put(RuleUnit.P, gaussians.get(iComponent));
			}
			break;
		}
		case LRURULE: { // idx = np * |nuc| + nuc
			int nuc = 1;
			List<GaussianDistribution> gaussians = null;
			if ((gaussians = gausses.get(RuleUnit.UC)) != null && gaussians.size() > 0) {
				nuc = gaussians.size();
				holder.gausses.put(RuleUnit.UC, gaussians.get(iComponent % nuc));
			}
			if ((gaussians = gausses.get(RuleUnit.P)) != null && gaussians.size() > 0) {
				holder.gausses.put(RuleUnit.P, gaussians.get(iComponent / nuc));
			}
			break;
		}
		case LRBRULE: { // idx = np * |nlc| * |nrc| + nlc * |nrc| + nrc
			int nlc = 1, nrc = 1, ip;
			List<GaussianDistribution> gaussians, lcGaussians, rcGaussians;
			if ((rcGaussians = gausses.get(RuleUnit.RC)) != null && rcGaussians.size() > 0) {
				nrc = rcGaussians.size();
			}
			if ((lcGaussians = gausses.get(RuleUnit.LC)) != null && lcGaussians.size() > 0) {
				nlc = rcGaussians.size();
			}
			ip = iComponent / (nlc * nrc);
			if ((gaussians = gausses.get(RuleUnit.P)) != null && gaussians.size() > 0) {
				holder.gausses.put(RuleUnit.P, gaussians.get(ip));
			}
			iComponent %= (nlc * nrc);
			if (rcGaussians != null && rcGaussians.size() > 0) {
				holder.gausses.put(RuleUnit.RC, rcGaussians.get(iComponent % nrc));
			}
			if (lcGaussians != null && lcGaussians.size() > 0) {
				holder.gausses.put(RuleUnit.LC, lcGaussians.get(iComponent / nrc));
			}
			break;
		}
		default: {
			throw new RuntimeException("Not consistent with any grammar rule type. Existing type: " + binding);
		}
		}
	}
	
	public static class SimpleView implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = -2293951438943599036L;
		public double weight;
		public GaussianDistribution gaussian;
		
		public SimpleView(double weight, GaussianDistribution gaussian) {
			this.weight = weight;
			this.gaussian = gaussian;
		}

		@Override
		public String toString() {
			return "VIEW [W=" + String.format( "%." + LVeGTrainer.precision + "f", weight) + ", G=" + gaussian + "]";
		}
		
	}
	
	public static class SimpleComponent implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 2562366397356025676L;
		public double weight;
		public EnumMap<RuleUnit, GaussianDistribution> gausses;
		
		public SimpleComponent() {
			this.gausses = new EnumMap<>(RuleUnit.class);
		}
		
		public SimpleComponent(double weight, EnumMap<RuleUnit, GaussianDistribution> gausses) {
			this.weight = weight;
			this.gausses = gausses;
		}
		
		public void clear() {
			this.weight = Double.NEGATIVE_INFINITY;
			this.gausses.clear();
		}

		@Override
		public String toString() {
			return "COMP [W=" + String.format( "%." + LVeGTrainer.precision + "f", weight) + ", G=" + gausses + "]";
		}
	}
	
	protected static Comparator<SimpleView> vcomparator = new SimpleViewComparator<>();
	protected static class SimpleViewComparator<T> implements Comparator<T>, Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = -8813462045781155697L;
		@Override
		public int compare(T o1, T o2) {
			double diff = ((SimpleView) o1).weight - ((SimpleView) o2).weight;
			return diff > 0 ? -1 : (diff < 0 ? 1 : 0);
		}
	}
}
