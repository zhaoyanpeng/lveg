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
	
	
	public PCFGParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxLenParsing, short nthread,
			boolean parallel, boolean iosprune, boolean usemasks) {
		super(maxLenParsing, nthread, parallel, iosprune, usemasks);
		this.inferencer = new PCFGInferencer(grammar, lexicon);
		this.chart = new Chart(maxLenParsing, false, true, false);
	}

	@Override
	public Executor<?, ?> newInstance() {
		return new PCFGParser<I, O>(this);
	}

	@Override
	public Object call() throws Exception {
		Tree<State> sample = (Tree<State>) task;
		viterbiParsing(sample);
		Tree<String> tree = StateTreeList.stateTreeToStringTree(sample, Inferencer.grammar.numberer);
		tree = Inferencer.extractBestMaxRuleParse(chart, tree.getYield());
		Meta<O> cache = new Meta(itask, tree);
		synchronized (caches) {
			caches.add(cache);
			caches.notifyAll();
		}
		task = null;
		return null;
	}
	
	public Tree<String> parse(Tree<State> tree) {
		viterbiParsing(tree);
		Tree<String> strTree = StateTreeList.stateTreeToStringTree(tree, Inferencer.grammar.numberer);
		Tree<String> parseTree = Inferencer.extractBestMaxRuleParse(chart, strTree.getYield());
		return parseTree;
	}
	
	protected void viterbiParsing(Tree<State> tree) {
		List<State> sentence = tree.getYield();
		int nword = sentence.size();
		if (chart != null) {
			chart.clear(nword);
		} else {
			chart = new Chart(nword, false, true, false);
		}
		inferencer.viterbiParsing(chart, sentence, nword);
	}

}
