package edu.shanghaitech.ai.nlp.optimization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.shanghaitech.ai.nlp.lveg.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.GaussianMixture;
import edu.shanghaitech.ai.nlp.lveg.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.util.Recorder;

/**
 * Naive. The dedicated implementation is used to enable the program to run asap.
 * 
 * @author Yanpeng Zhao
 *
 */
public class SGDForMoG extends Recorder {
	
	protected Random random;
	protected double lr;
	protected short nsample;
	
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
		this.wgrads = new ArrayList<Double>();
		this.ggrads = new HashMap<String, List<Double>>();
		ggrads.put(GrammarRule.Unit.P, new ArrayList<Double>());
		ggrads.put(GrammarRule.Unit.C, new ArrayList<Double>());
		ggrads.put(GrammarRule.Unit.UC, new ArrayList<Double>());
		ggrads.put(GrammarRule.Unit.LC, new ArrayList<Double>());
		sample.put(GrammarRule.Unit.RC, new ArrayList<Double>());
	}
	
	
	public SGDForMoG(Random random) {
		this();
		this.random = random;
		this.nsample = 3;
		this.lr = 0.02;
	}
	
	
	public SGDForMoG(Random random, short nsample, double lr) {
		this();
		this.random = random;
		this.nsample = nsample;
		this.lr = lr;
	}
	
	
	private void allocate(int ncomponent) {
		for (int i = 0; i < ncomponent; i++) {
			wgrads.add(0.0);
		}
	}
	
	
	private void deallocate() {
		for (int i = 0; i < wgrads.size(); i++) {
			wgrads.set(i, 0.0);
		}
	}
	
	
	private void sample(List<Double> slice, int dim) {
		slice.clear();
		for (int i = 0; i < dim; i++) {
			slice.add(random.nextGaussian());
		}
	}
	
	
	private int getIdx(Set<String> strsWithT, Set<String> strsWithS) {
		int idx = -1;
		for (String str : strsWithT) {
			if (isNumeric(str) && strsWithS.contains(str)) {
				idx = Integer.valueOf(str);
				break;
			}
		}
		if (idx < 0) { logger.error("Not found the score of the tree or the score of the sentence."); }
		return idx;
	}
	
	
	private double derivateOfRuleWeight(
			double scoreT, 
			double scoreS, 
			Map<String, GaussianMixture> iosWithT,
			Map<String, GaussianMixture> iosWithS) {
		
		
		return -0.0;
	}
	
	
	public void optimize(
			GrammarRule rule, 
			List<Map<String, GaussianMixture>> ioScoreWithT,
			List<Map<String, GaussianMixture>> ioScoreWithS,
			List<Double> scoresOfSAndT) {
		int batchsize = ioScoreWithT.size();
		if (batchsize != ioScoreWithT.size()) {
			logger.error("Rule count with the tree is not equal to that with the sentence.");
			return;
		}
		
		GaussianMixture ruleW = rule.getWeight();
		int ncomponent = ruleW.getNcomponent(), idx = -1;
		
		allocate(ncomponent); // TODO not necessary, one double is in fact enough
		
		double scoreT, scoreS;
		double dRuleW, iMixingW, dMixingW, factor;
		Map<String, GaussianMixture> iosWithT, iosWithS;
		UnaryGrammarRule urule = (UnaryGrammarRule) rule;
		
		for (int i = 0; i < batchsize; i++) {
			iosWithT = ioScoreWithT.get(i);
			iosWithS = ioScoreWithS.get(i);
			idx = getIdx(iosWithT.keySet(), iosWithS.keySet());
			scoreT = scoresOfSAndT.get(idx * 2);
			scoreS = scoresOfSAndT.get(idx * 2 + 1);
			for (int icomponent = 0; icomponent < ncomponent; icomponent++) {
				for (short isample = 0; isample < nsample; isample++) {
					if (rule instanceof UnaryGrammarRule) {
						switch (urule.getType()) {
						case GrammarRule.LHSPACE: {
							sample(sample.get(GrammarRule.Unit.P), ruleW.getDim(icomponent, GrammarRule.Unit.P));
							break;
						}
						case GrammarRule.RHSPACE: {
							sample(sample.get(GrammarRule.Unit.C), ruleW.getDim(icomponent, GrammarRule.Unit.C));
							break;
						}
						case GrammarRule.GENERAL: {
							sample(sample.get(GrammarRule.Unit.P), ruleW.getDim(icomponent, GrammarRule.Unit.P));
							sample(sample.get(GrammarRule.Unit.UC), ruleW.getDim(icomponent, GrammarRule.Unit.UC));
							break;
						}
						default: {
							logger.error("Not a valid unary grammar rule.");
						}
						}
					} else if (rule instanceof BinaryGrammarRule) {
						sample(sample.get(GrammarRule.Unit.P), ruleW.getDim(icomponent, GrammarRule.Unit.P));
						sample(sample.get(GrammarRule.Unit.LC), ruleW.getDim(icomponent, GrammarRule.Unit.LC));
						sample(sample.get(GrammarRule.Unit.RC), ruleW.getDim(icomponent, GrammarRule.Unit.UC));
					} else {
						logger.error("Not a valid grammar rule.");
					}
					dRuleW = derivateOfRuleWeight(scoreT, scoreS, iosWithT, iosWithS);
					ruleW.derivative(isample, icomponent, dRuleW, sample, ggrads, wgrads);
					wgrads.set(icomponent, wgrads.get(icomponent) + 1);
				}
				ruleW.update(icomponent, lr, ggrads, wgrads);
			}
		}
	}
	
	
	public static boolean isNumeric(String str){
	  return str.matches("[-+]?\\d*\\.?\\d+");  //match a number with optional '-' and decimal.
	}
	
}
