package edu.shanghaitech.ai.nlp.lveg.impl;

import edu.shanghaitech.ai.nlp.lveg.model.ChartCell.Chart;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.lveg.model.Parser;
import edu.shanghaitech.ai.nlp.util.Executor;

public class PCFGParser<I, O> extends Parser<I, O> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2668797817745391783L;
	
	
	private PCFGParser(PCFGParser<?, ?> parser) {
		super(parser.maxLenParsing, parser.nthread, parser.parallel, parser.iosprune, parser.usemasks);
		this.chart = new Chart(parser.maxLenParsing, false, false, false);
	}
	
	
	public PCFGParser(LVeGGrammar grammar, LVeGLexicon lexicon, short maxLenParsing, short nthread,
			boolean parallel, boolean iosprune, boolean usemasks) {
		super(maxLenParsing, nthread, parallel, iosprune, usemasks);
		this.chart = new Chart(maxLenParsing, false, false, false);
	}

	@Override
	public Executor<?, ?> newInstance() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object call() throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

}
