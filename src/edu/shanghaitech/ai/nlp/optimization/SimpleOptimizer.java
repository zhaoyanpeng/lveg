package edu.shanghaitech.ai.nlp.optimization;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.GrammarRule;

/**
 * @author Yanpeng Zhao
 *
 */
public class SimpleOptimizer extends Optimizer {
	private SimpleMinimizer minimizer;
	
	private SimpleOptimizer() {
		this.cntsWithS = new HashMap<GrammarRule, Batch>();
		this.cntsWithT = new HashMap<GrammarRule, Batch>();
		this.ruleSet = new HashSet<GrammarRule>();
	}
	
	
	public SimpleOptimizer(Random random) {
		this();
		rnd = random;
		this.minimizer = new SimpleMinimizer(random, maxsample);
	}
	
	
	public SimpleOptimizer(Random random, short nsample) {
		this();
		rnd = random;
		maxsample = nsample;
		this.minimizer = new SimpleMinimizer(random, maxsample);
	}
	
	
	@Override
	public void applyGradientDescent(List<Double> scoresST) {
		if (scoresST.size() == 0) { return; }
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			cntWithT = cntsWithT.get(rule);
			cntWithS = cntsWithS.get(rule);
			if (cntWithT.size() == 0 && cntWithS.size() == 0) { continue; }
			minimizer.optimize(rule, cntWithT, cntWithS, scoresST);
		}
		reset();
	}
	
	
	@Override
	public void evalGradients(List<Double> scoreSandT, boolean parallel) {
		return;
	}
	
	
	@Override
	public void addRule(GrammarRule rule) {
		ruleSet.add(rule);
		Batch batchWithT = new Batch(-1);
		Batch batchWithS = new Batch(-1);
		cntsWithT.put(rule, batchWithT);
		cntsWithS.put(rule, batchWithS);
	}

	
	@Override
	public void reset() { 
		Batch cntWithT, cntWithS;
		for (GrammarRule rule : ruleSet) {
			if ((cntWithT = cntsWithT.get(rule)) != null) { cntWithT.clear(); }
			if ((cntWithS = cntsWithS.get(rule)) != null) { cntWithS.clear(); }
		}
	}
	
}
