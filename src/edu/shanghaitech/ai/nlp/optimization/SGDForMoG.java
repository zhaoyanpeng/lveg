package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.optimization.Optimizer.Batch;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * Naive. The dedicated implementation is used to enable the program to run asap.
 * 
 * @author Yanpeng Zhao
 *
 */
public class SGDForMoG extends Recorder {
	
	protected Random random;
	protected short nsample;
	
	protected Map<String, List<Double>> truths;
	protected Map<String, List<Double>> sample;
	protected Map<String, List<Double>> ggrads;
	protected List<Double> wgrads;
	protected double wgrad;
	
	
	/**
	 * To avoid the excessive 'new' operations.
	 */
	public SGDForMoG() {
		this.sample = new HashMap<String, List<Double>>();
		sample.put(GrammarRule.Unit.P, new ArrayList<Double>());
		sample.put(GrammarRule.Unit.C, new ArrayList<Double>());
		sample.put(GrammarRule.Unit.UC, new ArrayList<Double>());
		sample.put(GrammarRule.Unit.LC, new ArrayList<Double>());
		sample.put(GrammarRule.Unit.RC, new ArrayList<Double>());
		this.truths = new HashMap<String, List<Double>>();
		truths.put(GrammarRule.Unit.P, new ArrayList<Double>());
		truths.put(GrammarRule.Unit.C, new ArrayList<Double>());
		truths.put(GrammarRule.Unit.UC, new ArrayList<Double>());
		truths.put(GrammarRule.Unit.LC, new ArrayList<Double>());
		truths.put(GrammarRule.Unit.RC, new ArrayList<Double>());
		this.wgrads = new ArrayList<Double>();
		this.ggrads = new HashMap<String, List<Double>>();
		ggrads.put(GrammarRule.Unit.P, new ArrayList<Double>());
		ggrads.put(GrammarRule.Unit.C, new ArrayList<Double>());
		ggrads.put(GrammarRule.Unit.UC, new ArrayList<Double>());
		ggrads.put(GrammarRule.Unit.LC, new ArrayList<Double>());
		ggrads.put(GrammarRule.Unit.RC, new ArrayList<Double>());
	}
	
	
	public SGDForMoG(Random random) {
		this();
		this.random = random;
		this.nsample = 1;
	}
	
	
	public SGDForMoG(Random random, short nsample, double lr) {
		this();
		this.random = random;
		this.nsample = nsample;
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
	
	
	/**
	 * @param rule
	 * @param ioScoreWithT
	 * @param ioScoreWithS
	 * @param scoresOfSAndT
	 */
	public void optimize(
			GrammarRule rule, 
			Batch ioScoreWithT,
			Batch ioScoreWithS,
			List<Double> scoresSandT) {
		int batchsize = scoresSandT.size() / 2;
		GaussianMixture ruleW = rule.getWeight();
		List<Map<String, GaussianMixture>> iosWithT, iosWithS;
		boolean removed = false, cumulative, updated;
		double scoreT, scoreS, dRuleW;
		byte uRuleType = -1;
		
		for (int icomponent = 0; icomponent < ruleW.getNcomponent(); icomponent++) {
			updated = false; // 
			for (short isample = 0; isample < nsample; isample++) {
				clearSample(); // to ensure the correct sample is in use
				if (rule.isUnary()) {
					UnaryGrammarRule urule = (UnaryGrammarRule) rule;
					switch (urule.getType()) {
					case GrammarRule.GENERAL: {
						sample(sample.get(GrammarRule.Unit.P), ruleW.getDim(icomponent, GrammarRule.Unit.P));
						sample(sample.get(GrammarRule.Unit.UC), ruleW.getDim(icomponent, GrammarRule.Unit.UC));
						break;
					}
					case GrammarRule.LHSPACE: {
						sample(sample.get(GrammarRule.Unit.P), ruleW.getDim(icomponent, GrammarRule.Unit.P));
						uRuleType = GrammarRule.LHSPACE;
						break;
					}
					case GrammarRule.RHSPACE: {
						sample(sample.get(GrammarRule.Unit.C), ruleW.getDim(icomponent, GrammarRule.Unit.C));
						break;
					}
					default: {
						logger.error("Not a valid unary grammar rule.\n");
					}
					}
				} else {
					sample(sample.get(GrammarRule.Unit.P), ruleW.getDim(icomponent, GrammarRule.Unit.P));
					sample(sample.get(GrammarRule.Unit.LC), ruleW.getDim(icomponent, GrammarRule.Unit.LC));
					sample(sample.get(GrammarRule.Unit.RC), ruleW.getDim(icomponent, GrammarRule.Unit.RC));
				}
				ruleW.restoreSample(icomponent, sample, truths);
				for (short i = 0; i < batchsize; i++) {
					iosWithT = ioScoreWithT.get(i);
					iosWithS = ioScoreWithS.get(i);
					if (iosWithT == null && iosWithS == null) { continue; } // zero counts
					scoreT = scoresSandT.get(i * 2);
					scoreS = scoresSandT.get(i * 2 + 1);
					/**
					 * For the rule A->w, count(A->w) = o(A->w) * i(A->w) = o(A->w) * w(A->w).
					 * For the sub-type rule r of A->w, count(a->w) = o(a->w) * w(a->w). The derivative
					 * of the objective function w.r.t w(r) is (count(r | T_S) - count(r | S)) / w(r), 
					 * which contains the term 1 / w(r), thus we could eliminate w(r) when computing it.
					 */
					if (!removed && uRuleType == GrammarRule.LHSPACE) { 
						if (iosWithT != null) { for (Map<String, GaussianMixture> ios : iosWithT) { ios.remove(GrammarRule.Unit.C); } }
						if (iosWithS != null) { for (Map<String, GaussianMixture> ios : iosWithS) { ios.remove(GrammarRule.Unit.C); } }
					}
					cumulative = (isample + i) > 0; // CHECK when to clear old gradients and accumulate new gradients
					dRuleW = derivateRuleWeight(scoreT, scoreS, iosWithT, iosWithS);
					ruleW.derivative(cumulative, icomponent, dRuleW, sample, ggrads, wgrads, true);
					updated = true; // CHECK do we need to update in the case where derivative() was not invoked.
				}
				removed = true; // CHECK avoid impossible remove
			}
			if (updated) {
				ruleW.update(icomponent, ggrads, wgrads);
			}
		}
	}
	
	
	protected void sample(List<Double> slice, int dim) {
		slice.clear();
		for (int i = 0; i < dim; i++) {
			slice.add(random.nextGaussian());
		}
	}
	
	
	protected void clearSample() {
		for (Map.Entry<String, List<Double>> slice : sample.entrySet()) {
			slice.getValue().clear();
		}
		for (Map.Entry<String, List<Double>> truth : truths.entrySet()) {
			truth.getValue().clear();
		}
	}
	
}
