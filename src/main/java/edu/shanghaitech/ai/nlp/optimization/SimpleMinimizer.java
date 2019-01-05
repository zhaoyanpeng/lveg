package edu.shanghaitech.ai.nlp.optimization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.model.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleType;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule.RuleUnit;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * Naive. The dedicated implementation is used to enable the program to run asap.
 * 
 * @author Yanpeng Zhao
 *
 */
public class SimpleMinimizer extends Recorder implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4030933989131874669L;
	protected EnumMap<RuleUnit, List<Double>> truths;
	protected EnumMap<RuleUnit, List<Double>> sample;
	protected EnumMap<RuleUnit, List<Double>> ggrads;
	protected List<Double> wgrads;
	protected double wgrad;
	
	
	/**
	 * To avoid the excessive 'new' operations.
	 */
	public SimpleMinimizer() {
		this.sample = new EnumMap<RuleUnit, List<Double>>(RuleUnit.class);
		sample.put(RuleUnit.P, new ArrayList<>());
		sample.put(RuleUnit.C, new ArrayList<>());
		sample.put(RuleUnit.UC, new ArrayList<>());
		sample.put(RuleUnit.LC, new ArrayList<>());
		sample.put(RuleUnit.RC, new ArrayList<>());
		this.truths = new EnumMap<>(RuleUnit.class);
		truths.put(RuleUnit.P, new ArrayList<>());
		truths.put(RuleUnit.C, new ArrayList<>());
		truths.put(RuleUnit.UC, new ArrayList<>());
		truths.put(RuleUnit.LC, new ArrayList<>());
		truths.put(RuleUnit.RC, new ArrayList<>());
		this.wgrads = new ArrayList<Double>();
		this.ggrads = new EnumMap<RuleUnit, List<Double>>(RuleUnit.class);
		ggrads.put(RuleUnit.P, new ArrayList<>());
		ggrads.put(RuleUnit.C, new ArrayList<>());
		ggrads.put(RuleUnit.UC, new ArrayList<>());
		ggrads.put(RuleUnit.LC, new ArrayList<>());
		ggrads.put(RuleUnit.RC, new ArrayList<>());
	}
	
	
	public SimpleMinimizer(Random random, int msample, short bsize) {
		this();
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
		List<EnumMap<RuleUnit, GaussianMixture>> iosWithT, iosWithS;
		boolean removed = false, cumulative, updated;
		double scoreT, scoreS, dRuleW;
		RuleType uRuleType = null;
		
		for (int icomponent = 0; icomponent < ruleW.ncomponent(); icomponent++) {
			updated = false; // 
			for (short isample = 0; isample < Optimizer.maxsample; isample++) {
				switch (rule.getType()) {
				case LRBRULE: {
					sample(sample.get(RuleUnit.P), ruleW.dim(icomponent, RuleUnit.P));
					sample(sample.get(RuleUnit.LC), ruleW.dim(icomponent, RuleUnit.LC));
					sample(sample.get(RuleUnit.RC), ruleW.dim(icomponent, RuleUnit.RC));
					break;
				}
				case LRURULE: {
					sample(sample.get(RuleUnit.P), ruleW.dim(icomponent, RuleUnit.P));
					sample(sample.get(RuleUnit.UC), ruleW.dim(icomponent, RuleUnit.UC));
					break;
				}
				case LHSPACE: {
					sample(sample.get(RuleUnit.P), ruleW.dim(icomponent, RuleUnit.P));
					uRuleType = RuleType.LHSPACE;
					break;
				}
				case RHSPACE: {
					sample(sample.get(RuleUnit.C), ruleW.dim(icomponent, RuleUnit.C));
					break;
				}
				default: {
					logger.error("Not a valid unary grammar rule.\n");
				}
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
					if (!removed && uRuleType == RuleType.LHSPACE) { 
						if (iosWithT != null) { 
							for (Map<RuleUnit, GaussianMixture> ios : iosWithT) { ios.remove(RuleUnit.C); } 
						}
						if (iosWithS != null) { 
							for (Map<RuleUnit, GaussianMixture> ios : iosWithS) { ios.remove(RuleUnit.C); } 
						}
					}
					// cumulative = (isample + i) > 0; // incorrect version
					// CHECK when to clear old gradients and accumulate new gradients
					cumulative = isample > 0 ? true : (updated ? true : false);
					dRuleW = derivateRuleWeight(scoreT, scoreS, iosWithT, iosWithS);
					ruleW.derivative(cumulative, icomponent, dRuleW, sample, ggrads, wgrads, true);
					updated = true; // CHECK do we need to update in the case where derivative() was not invoked.
				}
				removed = true; // CHECK avoid impossible remove
			}
			if (updated) {
				ruleW.update(icomponent, ggrads, wgrads, Optimizer.minexp);
			}
		}
	}
	
	
	protected void sample(List<Double> slice, int dim) {
		slice.clear();
		for (int i = 0; i < dim; i++) {
			slice.add(Optimizer.rnd.nextGaussian());
		}
	}
	
}
