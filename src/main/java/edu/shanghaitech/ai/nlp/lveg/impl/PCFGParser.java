package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.Inferencer;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Executor;

public class PCFGParser<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2668797817745391783L;
	private PCFGInferencer inferencer;
	
	
	private PCFGParser(PCFGParser<?, ?> parser) {
		super(parser.maxslen, parser.nthread, parser.parallel, parser.iosprune, parser.usemask);
		this.inferencer = parser.inferencer; // shared by multiple threads
		this.chart = new Chart(parser.maxslen, false, true, false);
	}
	
	
	public PCFGParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxslen, short nthread,
			boolean parallel, boolean iosprune, boolean usemasks) {
		super(maxslen, nthread, parallel, iosprune, usemasks);
		this.inferencer = new PCFGInferencer(grammar, lexicon);
		this.chart = new Chart(maxslen, false, true, false);
	}

	
	@Override
	public Executor<?, ?> newInstance() {
		return new PCFGParser<I, O>(this);
	}

	
	@Override
	public synchronized Object call() throws Exception {
		Tree<State> sample = (Tree<State>) task;
		Tree<String> parsed = null;
		synchronized (sample) {
			parsed = parse(sample);
		}
		Meta<O> cache = new Meta(itask, parsed);
		synchronized (caches) {
			caches.add(cache);
			caches.notify();
		}
		task = null;
		return itask;
	}
	
	
	/**
	 * Dedicated to error handling.
	 * 
	 * @param tree the golden parse tree
	 * @return     parse tree given the sentence
	 */
	public Tree<String> parse(Tree<State> tree) {
		Tree<String> parsed = null;
		try { // do NOT expect it to crash
			viterbiParse(tree);
			parsed = StateTreeList.stateTreeToStringTree(tree, Inferencer.grammar.numberer);
			parsed = Inferencer.extractBestMaxRuleParse(chart, parsed.getYield());
		} catch (Exception e) {
			parsed = new Tree<String>(Inferencer.DUMMY_TAG);
			e.printStackTrace();
		}
		return parsed;
	}
	
	
	/**
	 * Compute and store a viterbi parse path.
	 * 
	 * @param tree the golden parse tree
	 */
	protected void viterbiParse(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, false, true, false);
		}
//		synchronized (inferencer) { // inferencer is read-only
			inferencer.viterbiParse(chart, sentence, nword);
//		}
	}

}
