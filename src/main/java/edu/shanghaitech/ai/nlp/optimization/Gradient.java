package edu.shanghaitech.ai.nlp.optimization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
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
	/*
	 * It always holds the counts calculated from only one sample, 
	 * which is equally saying that the batch size is always 1.
	 */
	protected static final short MAX_BATCH_SIZE = 1;
	/*
	protected static short maxsample = Optimizer.maxsample;
	protected static short batchsize = Optimizer.batchsize;
	protected static Random rnd = Optimizer.rnd;
	*/
	protected boolean updated;
	protected boolean cumulative;
	protected List<Double> wgrads;
	protected List<Map<String, List<Double>>> ggrads;
	
	protected Map<String, List<Double>> truths;
	protected Map<String, List<Double>> sample;
	
	
	public Gradient(GrammarRule rule, Random random, short msample, short bsize) {
		/*
		rnd = random;
		batchsize = bsize;
		maxsample = msample;
		*/
		initialize(rule);
	}
	
	
	private void initialize(GrammarRule rule) {
		GaussianMixture ruleW = rule.getWeight();
		this.ggrads = ruleW.zeroslike();
		this.wgrads = new ArrayList<Double>(ruleW.ncomponent());
		List<HashMap<String, List<Double>>> holder = ruleW.zeroslike(0); 
		this.sample = holder.get(0);
		this.truths = holder.get(1);
		this.cumulative = false;
		this.updated = false;
	}
	
	
	protected boolean apply(GrammarRule rule) {
		if (!updated) { return false; } // no need to update because no gradients could be applied
		GaussianMixture ruleW = rule.getWeight();
		for (int icomponent = 0; icomponent < ruleW.ncomponent(); icomponent++) {
			ruleW.update(icomponent, ggrads.get(icomponent), wgrads, Optimizer.maxsample/*maxsample * batchsize*/);
		}
		reset();
		return true;
	}
	
	
	protected void reset() {
		wgrads.clear();
		for (Map<String, List<Double>> ggrad : ggrads) {
			for (Map.Entry<String, List<Double>> part : ggrad.entrySet()) {
				part.getValue().clear();
			}
		}
		cumulative = false;
		updated  = false;
	}
	
	
	/**
	 * Eval and accumulate the gradients.
	 * 
	 * @param rule
	 * @param ioScoreWithT
	 * @param ioScoreWithS
	 * @param scoreSandT
	 */
	protected boolean eval(GrammarRule rule, Batch ioScoreWithT, Batch ioScoreWithS, List<Double> scoreSandT) {
		List<Map<String, GaussianMixture>> iosWithT, iosWithS;
		GaussianMixture ruleW = rule.getWeight();
		Map<String, List<Double>> ggrad;
		boolean removed = false, allocated, iallocated;
		double scoreT, scoreS, dRuleW;
		for (int icomponent = 0; icomponent < ruleW.ncomponent(); icomponent++) {
			ggrad = ggrads.get(icomponent);
			iallocated = false; // assume ggrad has not been allocated
			for (short isample = 0; isample < Optimizer.maxsample; isample++) {
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
					if (!removed && rule.getType() == GrammarRule.LHSPACE) { 
						if (iosWithT != null) { 
							for (Map<String, GaussianMixture> ios : iosWithT) { ios.remove(GrammarRule.Unit.C); } 
						}
						if (iosWithS != null) { 
							for (Map<String, GaussianMixture> ios : iosWithS) { ios.remove(GrammarRule.Unit.C); } 
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
			List<Map<String, GaussianMixture>> ioScoreWithT,
			List<Map<String, GaussianMixture>> ioScoreWithS) {
		double countWithT = 0.0, countWithS = 0.0, cnt, part, dRuleW;
		if (ioScoreWithT != null) {
			for (Map<String, GaussianMixture> iosWithT : ioScoreWithT) {
				cnt = 1.0;
				boolean found = false;
				for (Map.Entry<String, GaussianMixture> ios : iosWithT.entrySet()) {
					part = ios.getValue().evalInsideOutside(truths.get(ios.getKey()), false);
					cnt *= part;
					found = true;
				}
				if (found) { countWithT += cnt; }
			}
		}
		if (ioScoreWithS != null) {
			for (Map<String, GaussianMixture> iosWithS : ioScoreWithS) {
				cnt = 1.0;
				boolean found = false;
				for (Map.Entry<String, GaussianMixture> ios : iosWithS.entrySet()) {
					part = ios.getValue().evalInsideOutside(truths.get(ios.getKey()), false);
					cnt *= part;
					found = true;
				}
				if (found) { countWithS += cnt; }
			}
		}
		dRuleW = Math.exp(Math.log(countWithS) - scoreS) - Math.exp(Math.log(countWithT) - scoreT);
		return dRuleW;
	}
	
}
