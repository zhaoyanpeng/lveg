package edu.shanghaitech.ai.nlp.lveg;

public abstract class GrammarRule {

	/**
	 * the ID of the left-hand side nonterminal
	 */
	protected int lhs;

	protected DiagonalGaussianMixture weightFunction;
}
