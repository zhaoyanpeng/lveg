package edu.shanghaitech.ai.nlp.optimization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Params;
import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.optimization.Optimizer.OptChoice;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * @author Yanpeng Zhao
 *
 */
public class Gradient extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2620919219751675203L;
	public static class Grads {
		public List<Double> wgrads;
		public List<EnumMap<RuleUnit, List<Double>>> ggrads;
		public Grads(List<Double> wgrads, List<EnumMap<RuleUnit, List<Double>>> ggrads) {
			this.wgrads = wgrads;
			this.ggrads = ggrads;
		}
	}
	/*
	 * It always holds the counts calculated from only one sample, 
	 * which is equally saying that the batch size is always 1.
	 */
	protected static final short MAX_BATCH_SIZE = 1;
	protected boolean updated;
	protected boolean cumulative;
	protected double cntUpdate;
	protected double partition;
	protected List<Double> wgrads, wgrads1, wgrads2;
	protected List<EnumMap<RuleUnit, List<Double>>> ggrads, ggrads1, ggrads2, ggradst, ggradss;
	
	protected EnumMap<RuleUnit, List<Double>> truths;
	protected EnumMap<RuleUnit, List<Double>> sample;
	
	protected List<EnumMap<RuleUnit, List<List<Double>>>> cachesWithT;
	protected List<EnumMap<RuleUnit, List<List<Double>>>> cachesWithS;
	
	public Gradient(GrammarRule rule, Random random, int msample, short bsize) {
		GaussianMixture ruleW = rule.getWeight();
		initialize(ruleW);
		this.cntUpdate = 0;
		this.partition = Optimizer.batchsize * Optimizer.maxsample;
		// TODO use lazy initialization?
		this.wgrads1 = new ArrayList<>(ruleW.ncomponent());
		this.wgrads2 = new ArrayList<>(ruleW.ncomponent());
		this.ggrads1 = ruleW.zeroslike(true);
		this.ggrads2 = ruleW.zeroslike(true);
		this.ggradst = ruleW.zeroslike(false);
		this.ggradss = ruleW.zeroslike(false);
		for (int i = 0; i < ruleW.ncomponent(); i++) {
			wgrads1.add(0.0);
			wgrads2.add(0.0);
		}
	}
	
	
	private void initialize(GaussianMixture ruleW) {
		this.ggrads = ruleW.zeroslike(false);
		this.wgrads = new ArrayList<>(ruleW.ncomponent());
		List<EnumMap<RuleUnit, List<Double>>> holder = ruleW.zeroslike(0); 
		this.sample = holder.get(0);
		this.truths = holder.get(1);
		this.cumulative = false;
		this.updated = false;
		this.cachesWithT = ruleW.cachelike(0, 2, 50);
		this.cachesWithS = ruleW.cachelike(0, 10, 50);
	}
	
	
	protected boolean apply(GrammarRule rule) {
		if (!updated) { return false; } // no need to update because no gradients could be applied
		cntUpdate++; // count of gradient update
		update(Optimizer.choice);
		GaussianMixture ruleW = rule.getWeight();
		for (int icomponent = 0; icomponent < ruleW.ncomponent(); icomponent++) {
			ruleW.update(icomponent, ggrads.get(icomponent), wgrads, Optimizer.minexp);
		}
		reset();
		return true;
	}
	
	
	protected void reset() {
		wgrads.clear();
		for (Map<RuleUnit, List<Double>> ggrad : ggrads) {
			for (Entry<RuleUnit, List<Double>> part : ggrad.entrySet()) {
				part.getValue().clear();
			}
		}
		cumulative = false;
		updated  = false;
	}
	
	
	protected void clearCaches() {
		for (Map<RuleUnit, List<List<Double>>> caches : cachesWithT) {
			for (Entry<RuleUnit, List<List<Double>>> part : caches.entrySet()) {
				for (List<Double> cache : part.getValue()) {
					cache.clear();
				}
			}
		}
		for (Map<RuleUnit, List<List<Double>>> caches : cachesWithS) {
			for (Entry<RuleUnit, List<List<Double>>> part : caches.entrySet()) {
				for (List<Double> cache : part.getValue()) {
					cache.clear();
				}
			}
		}
	}
	
	
	protected Object debug(GrammarRule rule, boolean beloved) {
		if (beloved) { // debug gradients
			logger.trace("\n----------\nRule: " + rule + "\nRule Weight: " + rule.getWeight() + 
					"\nGrad Weight: " + wgrads + "\nGrad Gauss : " + ggrads + "\n----------\n");
			return null;
		} else {
			return new Grads(wgrads, ggrads);
		}
	}
	
	
	protected boolean eval(GrammarRule rule, Batch ioScoreWithT, Batch ioScoreWithS, List<Double> scoreSandT) {
		try {
			if (Optimizer.sampling) {
				return evalsampling(rule, ioScoreWithT, ioScoreWithS, scoreSandT);
			} else {
				return evalintegral(rule, ioScoreWithT, ioScoreWithS, scoreSandT);
			}
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}
	
	
	/**
	 * Eval and accumulate the gradients.
	 * 
	 * @param rule
	 * @param ioScoreWithT
	 * @param ioScoreWithS
	 * @param scoreSandT
	 */
	protected boolean evalintegral(GrammarRule rule, Batch ioScoreWithT, Batch ioScoreWithS, List<Double> scoreSandT) {
		List<EnumMap<RuleUnit, GaussianMixture>> iosWithT, iosWithS;
		boolean removed = false, allocated, iallocated;
		EnumMap<RuleUnit, List<Double>> grads, gradst, gradss;
		double scoreT, scoreS;
		
		GaussianMixture ruleW = rule.getWeight();
		for (int icomponent = 0; icomponent < ruleW.ncomponent(); icomponent++) {
			iallocated = false; // ggrad below is assumed to have not been allocated
			grads = ggrads.get(icomponent);
			gradst = ggradst.get(icomponent);
			gradss = ggradss.get(icomponent);
			for (short i = 0; i < MAX_BATCH_SIZE; i++) {
				iosWithT = ioScoreWithT.get(i);
				iosWithS = ioScoreWithS.get(i);
				if (iosWithT == null && iosWithS == null) { continue; } // zero counts
				scoreT = scoreSandT.get(i * 2);
				scoreS = scoreSandT.get(i * 2 + 1);
				// cancel the denominator w(r) 
				if (!removed && rule.getType() == RuleType.LHSPACE) { 
					if (iosWithT != null) { 
						for (Map<RuleUnit, GaussianMixture> ios : iosWithT) { ios.remove(RuleUnit.C); } 
					}
					if (iosWithS != null) { 
						for (Map<RuleUnit, GaussianMixture> ios : iosWithS) { ios.remove(RuleUnit.C); } 
					}
				}
				// remove the meaningless outside score of ROOT
				if (!removed && rule.getType() == RuleType.RHSPACE) { 
					if (iosWithT != null) { 
						for (Map<RuleUnit, GaussianMixture> ios : iosWithT) { ios.remove(RuleUnit.P); } 
					}
					if (iosWithS != null) { 
						for (Map<RuleUnit, GaussianMixture> ios : iosWithS) { ios.remove(RuleUnit.P); } 
					}
				}
				allocated = cumulative ? true : (iallocated ? true : false);
				ruleW.derivative(allocated, icomponent, scoreT, scoreS, gradst, gradss, grads, wgrads, iosWithT, iosWithS, cachesWithT, cachesWithS);
				iallocated = true; // ggrad must have been allocated after derivative() is invoked
				updated = true; // we can apply gradient descent with the gradients
			}
			removed = true; // avoid useless checking and removing
			clearCaches(); // clear caches of this component
		}
		cumulative = true; // accumulate gradients in a batch
		return updated;
	}
	
	
	/**
	 * Eval and accumulate the gradients.
	 * 
	 * @param rule
	 * @param ioScoreWithT
	 * @param ioScoreWithS
	 * @param scoreSandT
	 */
	protected boolean evalsampling(GrammarRule rule, Batch ioScoreWithT, Batch ioScoreWithS, List<Double> scoreSandT) {
		List<EnumMap<RuleUnit, GaussianMixture>> iosWithT, iosWithS;
		GaussianMixture ruleW = rule.getWeight();
		EnumMap<RuleUnit, List<Double>> ggrad;
		boolean removed = false, allocated, iallocated;
		double scoreT, scoreS, dRuleW;
		
		for (int icomponent = 0; icomponent < ruleW.ncomponent(); icomponent++) {
			ggrad = ggrads.get(icomponent);
			iallocated = false; // assume ggrad has not been allocated
			for (int isample = 0; isample < Optimizer.maxsample; isample++) {
				ruleW.sample(icomponent, sample, truths, Optimizer.rnd);
				
//				logger.trace("\n" + rule + ", icomp" + icomponent + ", isample: " + isample + "\nsample: " + sample + "\ntruths: " + truths + "\n"); // DEBUG
				
				for (short i = 0; i < MAX_BATCH_SIZE; i++) {
					iosWithT = ioScoreWithT.get(i);
					iosWithS = ioScoreWithS.get(i);
					if (iosWithT == null && iosWithS == null) { continue; } // zero counts
					scoreT = scoreSandT.get(i * 2);
					scoreS = scoreSandT.get(i * 2 + 1);
					/* 
					 * For the rule A->w, count(A->w) = o(A->w) * i(A->w) = o(A->w) * w(A->w).
					 * For the sub-type rule r of A->w, count(a->w) = o(a->w) * w(a->w). The derivative
					 * of the objective function w.r.t w(r) is (count(r | T_S) - count(r | S)) / w(r), 
					 * which contains the term 1 / w(r), thus we could eliminate w(r) when computing it.
					 */
					if (!removed && rule.getType() == RuleType.LHSPACE) { 
						if (iosWithT != null) { 
							for (Map<RuleUnit, GaussianMixture> ios : iosWithT) { ios.remove(RuleUnit.C); } 
						}
						if (iosWithS != null) { 
							for (Map<RuleUnit, GaussianMixture> ios : iosWithS) { ios.remove(RuleUnit.C); } 
						}
					}
					/* 
					 * If i-th count is skipped, the memory space for ggrad would not be allocated.
					 * So when cumulative is false, we should use the second, correct version. And
					 * notice that wgrad should be clear only when icomponent = 1.
					 * Incorrect version: local = cumulative ? true : (isample + i) > 0;
					 */
					allocated = cumulative ? true : (isample > 0 ? true : (iallocated ? true : false));
					dRuleW = derivateRuleWeight(scoreT, scoreS, iosWithT, iosWithS);
					ruleW.derivative(allocated, icomponent, dRuleW, sample, ggrad, wgrads, true);
					iallocated = true; // ggrad must have been allocated after derivative() is invoked
					updated = true; // we can apply gradient descent with the gradients
				}
				removed = true; // avoid replicate checking and removing
				
//				logger.trace(rule + ", icomp" + icomponent + ", isample: " + isample + "\nwgrads: " + wgrads + "\nggrads: " + ggrads + "\n"); // DEBUG
			
			}
		}
		cumulative = true; // 
		return updated;
	}
	
	
	/**
	 * TODO revise the computation in logarithm.
	 * 
	 * @param scoreT in logarithmic form
	 * @param scoreS in logarithmic form
	 * @param ioScoreWithT
	 * @param ioScoreWithS
	 * @return
	 */
	private double derivateRuleWeight(
			double scoreT, 
			double scoreS, 
			List<EnumMap<RuleUnit, GaussianMixture>> ioScoreWithT,
			List<EnumMap<RuleUnit, GaussianMixture>> ioScoreWithS) {
		double countWithT = 0.0, countWithS = 0.0, cnt, part, dRuleW;
		if (ioScoreWithT != null) {
			for (Map<RuleUnit, GaussianMixture> iosWithT : ioScoreWithT) {
				cnt = 1.0;
				boolean found = false;
				for (Entry<RuleUnit, GaussianMixture> ios : iosWithT.entrySet()) {
					part = ios.getValue().evalInsideOutside(truths.get(ios.getKey()), false);
					cnt *= part;
					found = true;
				}
				if (found) { countWithT += cnt; }
			}
		}
		if (ioScoreWithS != null) {
			for (Map<RuleUnit, GaussianMixture> iosWithS : ioScoreWithS) {
				cnt = 1.0;
				boolean found = false;
				for (Entry<RuleUnit, GaussianMixture> ios : iosWithS.entrySet()) {
					part = ios.getValue().evalInsideOutside(truths.get(ios.getKey()), false);
					cnt *= part;
					found = true;
				}
				if (found) { countWithS += cnt; }
			}
		}
//		logger.trace("cnts: " + countWithS + "\tcntt: " + countWithT + "\n"); // DEBUG
		dRuleW = Math.exp(Math.log(countWithS) - scoreS) - Math.exp(Math.log(countWithT) - scoreT);
		return dRuleW;
	}

	
	private void update(OptChoice choice) {
		switch (choice) {
		case NORMALIZED: { 
			normalize();
			break;
		}
		case SGD: {
			sgd();
			break;
		}
		case MOMENTUM: {
			momentum();
			break;
		}
		case ADAGRAD: {
			adagrad();
			break;
		}
		case RMSPROP: {
			rmsprop();
			break;
		}
		case ADADELTA: {
			adadelta();
			break;
		}
		case ADAM: {
			adam();
			break;
		}
		default: {
			logger.error("unmatched optimization choice.\n");
		}
		}
	}
	
	
	private void normalize() {
		double grad;
		for (int k = 0; k < wgrads.size(); k++) { // component k
			EnumMap<RuleUnit, List<Double>> gcomp = ggrads.get(k);
			for (Entry<RuleUnit, List<Double>> grads : gcomp.entrySet()) {
				List<Double> grads0 = grads.getValue();
				for (int d = 0; d < grads0.size(); d++) { // dimension d
					grad = -Params.lr * Math.signum(grads0.get(d)) / partition;
					grad = clip(grad);
					grads0.set(d, grad);
				}
			}
			grad = -Params.lr * Math.signum(wgrads.get(k)) / partition;
			grad = clip(grad);
			wgrads.set(k, grad);
		}
	}
	
	
	private void sgd() {
		double g1st, grad;
		for (int k = 0; k < wgrads.size(); k++) { // component k
			EnumMap<RuleUnit, List<Double>> gcomp = ggrads.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp1 = ggrads1.get(k);
			for (Entry<RuleUnit, List<Double>> grads : gcomp.entrySet()) {
				List<Double> grads0 = grads.getValue();
				List<Double> grads1 = gcomp1.get(grads.getKey());
				for (int d = 0; d < grads0.size(); d++) { // dimension d
					grad = grads0.get(d) / partition;
					g1st = -Params.lr * grad;
					grads1.set(d, g1st);
					grad = g1st;
					grad = clip(grad);
					grads0.set(d, grad);
				}
			}
			grad = wgrads.get(k) / partition;
			g1st = -Params.lr * grad;
			wgrads1.set(k, g1st);
			grad = g1st;
			grad = clip(grad);
			wgrads.set(k, grad);
		}
	}
	
	
	private void momentum() {
		double g1st, grad;
		for (int k = 0; k < wgrads.size(); k++) { // component k
			EnumMap<RuleUnit, List<Double>> gcomp = ggrads.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp1 = ggrads1.get(k);
			for (Entry<RuleUnit, List<Double>> grads : gcomp.entrySet()) {
				List<Double> grads0 = grads.getValue();
				List<Double> grads1 = gcomp1.get(grads.getKey());
				for (int d = 0; d < grads0.size(); d++) { // dimension d
					grad = grads0.get(d) / partition;
					g1st = Params.lambda * grads1.get(d) - Params.lr * grad;
					grads1.set(d, g1st);
					grad = g1st;
					grad = clip(grad);
					grads0.set(d, grad);
				}
			}
			grad = wgrads.get(k) / partition;
			g1st = Params.lambda * wgrads1.get(k) - Params.lr * grad;
			wgrads1.set(k, g1st);
			grad = g1st;
			grad = clip(grad);
			wgrads.set(k, grad);
		}
	}
	
	
	private void adagrad() {
		double g2nd, grad;
		for (int k = 0; k < wgrads.size(); k++) { // component k
			EnumMap<RuleUnit, List<Double>> gcomp = ggrads.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp2 = ggrads2.get(k);
			for (Entry<RuleUnit, List<Double>> grads : gcomp.entrySet()) {
				List<Double> grads0 = grads.getValue();
				List<Double> grads2 = gcomp2.get(grads.getKey());
				for (int d = 0; d < grads0.size(); d++) { // dimension d
					grad = grads0.get(d) / partition;
					g2nd = grads2.get(d) + grad * grad;
					grads2.set(d, g2nd);
					grad = -Params.lr * grad / Math.sqrt(g2nd+ Params.epsilon);
					grad = clip(grad);
					grads0.set(d, grad);
				}
			}
			grad = wgrads.get(k) / partition;
			g2nd = wgrads2.get(k) + grad * grad;
			wgrads2.set(k, g2nd);
			grad = -Params.lr * grad / Math.sqrt(g2nd + Params.epsilon);
			grad = clip(grad);
			wgrads.set(k, grad);
		}
	}
	
	
	private void rmsprop() {
		double g2nd, grad;
		for (int k = 0; k < wgrads.size(); k++) { // component k
			EnumMap<RuleUnit, List<Double>> gcomp = ggrads.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp2 = ggrads2.get(k);
			for (Entry<RuleUnit, List<Double>> grads : gcomp.entrySet()) {
				List<Double> grads0 = grads.getValue();
				List<Double> grads2 = gcomp2.get(grads.getKey());
				for (int d = 0; d < grads0.size(); d++) { // dimension d
					grad = grads0.get(d) / partition;
					g2nd = Params.lambda * grads2.get(d) + (1 - Params.lambda) * grad * grad;
					grads2.set(d, g2nd);
					grad = -Params.lr * grad / Math.sqrt(g2nd + Params.epsilon);
					grad = clip(grad);
					grads0.set(d, grad);
				}
			}
			grad = wgrads.get(k) / partition;
			g2nd = Params.lambda * wgrads2.get(k) + (1 - Params.lambda) * grad * grad;
			wgrads2.set(k, g2nd);
			grad = -Params.lr * grad / (Math.sqrt(g2nd) + Params.epsilon);
			grad = clip(grad);
			wgrads.set(k, grad);
		}
	}
	
	
	private void adadelta() {
		double v2nd, g2nd, grad;
		for (int k = 0; k < wgrads.size(); k++) { // component k
			EnumMap<RuleUnit, List<Double>> gcomp = ggrads.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp1 = ggrads1.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp2 = ggrads2.get(k);
			for (Entry<RuleUnit, List<Double>> grads : gcomp.entrySet()) {
				List<Double> grads0 = grads.getValue();
				List<Double> grads1 = gcomp1.get(grads.getKey());
				List<Double> grads2 = gcomp2.get(grads.getKey());
				for (int d = 0; d < grads0.size(); d++) { // dimension d
					grad = grads0.get(d) / partition;
					v2nd = grads1.get(d);
					g2nd = Params.lambda * grads2.get(d) + (1 - Params.lambda) * grad * grad;
					grad = - Math.sqrt(v2nd + Params.epsilon) * grad / Math.sqrt(g2nd + Params.epsilon);
					grad = clip(grad);
					v2nd = Params.lambda * v2nd + (1 - Params.lambda) * grad * grad;
					grads1.set(d, v2nd);
					grads2.set(d, g2nd);
					grads0.set(d, grad);
				}
			}
			grad = wgrads.get(k) / partition;
			v2nd = wgrads1.get(k);
			g2nd = Params.lambda * wgrads2.get(k) + (1 - Params.lambda) * grad * grad;
			grad = - Math.sqrt(v2nd + Params.epsilon) * grad / Math.sqrt(g2nd + Params.epsilon);
			grad = clip(grad);
			v2nd = Params.lambda * v2nd + (1 - Params.lambda) * grad * grad;
			wgrads1.set(k, v2nd);
			wgrads2.set(k, g2nd);
			wgrads.set(k, grad);
		}
	}
	
	
	private void adam() {
		double g1st, g2nd, grad;
		double ldecay1 = 1 - Math.pow(Params.lambda1, cntUpdate);
		double ldecay2 = 1 - Math.pow(Params.lambda2, cntUpdate);
		for (int k = 0; k < wgrads.size(); k++) { // component k
			EnumMap<RuleUnit, List<Double>> gcomp = ggrads.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp1 = ggrads1.get(k);
			EnumMap<RuleUnit, List<Double>> gcomp2 = ggrads2.get(k);
			for (Entry<RuleUnit, List<Double>> grads : gcomp.entrySet()) {
				List<Double> grads0 = grads.getValue();
				List<Double> grads1 = gcomp1.get(grads.getKey());
				List<Double> grads2 = gcomp2.get(grads.getKey());
				for (int d = 0; d < grads0.size(); d++) { // dimension d
					grad = grads0.get(d) / partition;
					g1st = Params.lambda1 * grads1.get(d) + (1 - Params.lambda1) * grad;
					g2nd = Params.lambda2 * grads2.get(d) + (1 - Params.lambda2) * grad * grad;
					grads1.set(d, g1st);
					grads2.set(d, g2nd);
					g1st /= ldecay1;
					g2nd /= ldecay2;
					grad = -Params.lr * g1st / (Math.sqrt(g2nd) + Params.epsilon);
					grad = clip(grad);
					grads0.set(d, grad);
				}
			}
			grad = wgrads.get(k) / partition;
			g1st = Params.lambda1 * wgrads1.get(k) + (1 - Params.lambda1) * grad;
			g2nd = Params.lambda2 * wgrads2.get(k) + (1 - Params.lambda2) * grad * grad;
			wgrads1.set(k, g1st);
			wgrads2.set(k, g2nd);
			g1st /= ldecay1;
			g2nd /= ldecay2;
			grad = -Params.lr * g1st / (Math.sqrt(g2nd) + Params.epsilon);
			grad = clip(grad);
			wgrads.set(k, grad);
		}
	}
	
	
	private static double clip(double grad) {
		return Params.clip ? (Math.abs(grad) > Params.absmax ? Params.absmax * Math.signum(grad) : grad) : grad;
	}
	
}
