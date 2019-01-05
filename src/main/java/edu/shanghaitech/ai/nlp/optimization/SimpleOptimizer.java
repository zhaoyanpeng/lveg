package edu.shanghaitech.ai.nlp.optimization;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;

/**
 * @author Yanpeng Zhao
 *
 */
public class SimpleOptimizer extends Optimizer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1474006225613054835L;
	private SimpleMinimizer minimizer;
	
	private SimpleOptimizer() {
		this.cntsWithS = new HashMap<>();
		this.cntsWithT = new HashMap<>();
		this.ruleSet = new HashSet<>();
	}
	
	
	public SimpleOptimizer(Random random) {
		this();
		rnd = random;
		this.minimizer = new SimpleMinimizer(random, maxsample, batchsize);
	}
	
	
	public SimpleOptimizer(Random random, short msample, short bsize) {
		this();
		rnd = random;
		batchsize = bsize;
		maxsample = msample;
		this.minimizer = new SimpleMinimizer(random, maxsample, batchsize);
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
	public void evalGradients(List<Double> scoreSandT) {
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
