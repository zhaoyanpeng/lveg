package edu.shanghaitech.ai.nlp.lveg;

import java.util.ArrayList;

public class DiagonalGaussianMixture {

	protected ArrayList<Double> weights;

	/**
	 * The set of Gaussian components. Each component consists of one or more
	 * independent Gaussian distributions.
	 */
	protected ArrayList<DiagonalGaussianDistribution[]> gaussians;

	/**
	 * The terminal or nonterminal IDs represented by the independent Gaussian
	 * distributions.
	 */
	protected int[] nodeIDs;

	/**
	 * Multiply two Gaussian mixtures, whose variables (node IDs) may or may not overlap.
	 * 
	 * @param m
	 * @return the product
	 */
	public DiagonalGaussianMixture multiply(DiagonalGaussianMixture m) {
		// TODO
		return null;
	}

	/**
	 * Marginalize the variables specified by <code>ids</code>.
	 * 
	 * @param ids the IDs of the variables to be marginalized
	 */
	public void marginalize(int[] ids) {
		// TODO
	}
}
