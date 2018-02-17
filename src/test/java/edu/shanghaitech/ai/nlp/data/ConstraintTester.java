package edu.shanghaitech.ai.nlp.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.Corpus;
import edu.berkeley.nlp.PCFGLA.ParserData;
import edu.berkeley.nlp.PCFGLA.StateSetTreeList;
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType;
import edu.berkeley.nlp.syntax.StateSet;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import edu.berkeley.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.data.ObjectFileManager.Constraint;


public class ConstraintTester {
	private static final String TREE_ROOT = "F:/sourcecode/Data.Prd";
	
	
	@Test
	public void calConstraints() throws Exception {
		String root = "F:/sourcecode/Rets.Prd/gr.sophis";
//		testF1erEntry(40000, root);
//		isGoldenParseIncludedEntry(40000, root);
		
//		loadConstraints(40000, root);
		
		averageNumOfTrees(root + "/gr.200.final.best.200");
		
//		String root = "F:/sourcecode/Package/stanford-parser-use";
//		testF1erCacheEntry(40000, root);
//		isGoldenParseIncludedCacheEntry(40000, root);
	}
	
	
	public void loadConstraints(int N, String DATA_ROOT) {
		Set<String>[][][] constraints = new HashSet[N][][];
		Set<String>[][][] goldenconst = new HashSet[N][][];
		
		/*
		Set<String>[][] acon = constraints[0];
		int len = acon.length;
		for (int i = 0; i < len; i++) {
			for (int j = i; j < len; j++) {
				System.out.print(acon[i][j].size() + " ");
			}
			System.out.println();
		}
		*/
		String constr = DATA_ROOT + "/gr.200.21.cons";
		String grfile = DATA_ROOT + "/gr.gr";
		String treebk = TREE_ROOT + "/wsj/";
		
		ParserData pData = ParserData.Load(grfile);
		Numberer.setNumberers(pData.getNumbs());
		Numberer numberer = Numberer.getGlobalNumberer("tags");
		short[] numSubStatesArray = pData.getGrammar().numSubStates;
		
		Corpus corpus = new Corpus(treebk, TreeBankType.WSJ, 1.0, false);
		List<Tree<String>> train = corpus.getTrainTrees();
		
		Constraint cons = (Constraint) ObjectFileManager.ObjectFile.load(constr);
		constraints = cons.getConstraints();
		
		int itree = 0, nerr = 0;
		for (Tree<String> tree : train) {
			List<Tree<String>> curTree = new ArrayList<Tree<String>>();
			
			curTree.add(tree);
			Set<String>[][] constraint = candidate(curTree, numberer, numSubStatesArray);
			goldenconst[itree] = constraint;
			itree += 1;
		}
		System.out.println("--total " + train.size());
		
		for (int i = 0; i < train.size(); i++) {
			Set<String>[][] constraint = constraints[i];
			Set<String>[][] goldencons = goldenconst[i];
			
			System.out.print("--itest " + i + " done; ");
			if (!isIncluded(constraint, goldencons)) {
				nerr += 1;
				System.out.print("err ");
			}
			System.out.println();
		}
		System.out.println("--ntest " + itree + ", nerr " + nerr + " / " + train.size());
		
	}
	
	
	public void isGoldenParseIncludedCacheEntry(int N, String DATA_ROOT) throws Exception {
//		String DATA_ROOT = "F:/sourcecode/Data.Prd";
		
		String infile = DATA_ROOT + "/wsj.cache.40.200.23.pred";
		String grfile = DATA_ROOT + "/gr.tb.gr";
		String constr = DATA_ROOT + "/wsj.cache.40.200.23.cons";
		String treebk = TREE_ROOT + "/wsj/";
		
		String section = "dev";
		List<Tree<String>> trees = getTrees(treebk, section);
		
		isGoldenParseIncluded(N, infile, grfile, constr, trees);
	}
	
	
	public void isGoldenParseIncludedEntry(int N, String DATA_ROOT) throws Exception {
//		String DATA_ROOT = "F:/sourcecode/Data.Prd";
		
		String infile = DATA_ROOT + "/gr.200.train.best.200";
		String grfile = DATA_ROOT + "/gr.gr";
		String constr = DATA_ROOT + "/gr.200.21.cons";
		String treebk = TREE_ROOT + "/wsj/";
		
		String section = "train";
		List<Tree<String>> trees = getTrees(treebk, section);
		
		isGoldenParseIncluded(N, infile, grfile, constr, trees);
	}
	
	
	public void isGoldenParseIncluded(int N, String infile, String grfile, String constr, List<Tree<String>> trees) throws Exception {

		ParserData pData = ParserData.Load(grfile);
		
		Numberer.setNumberers(pData.getNumbs());
		Numberer numberer = Numberer.getGlobalNumberer("tags");
		short[] numSubStatesArray = pData.getGrammar().numSubStates;
		
		Set<String>[][][] constraints = new HashSet[N][][];
		Set<String>[][][] goldenconst = new HashSet[N][][];
		
		int itree = 0;
		for (Tree<String> tree : trees) {
			List<Tree<String>> curTree = new ArrayList<Tree<String>>();
			
			curTree.add(tree);
			Set<String>[][] constraint = candidate(curTree, numberer, numSubStatesArray);
			goldenconst[itree] = constraint;
			itree += 1;
		}
		System.out.println("--total " + trees.size());
		
		
		try {
			List<Tree<String>> someTrees = new ArrayList<Tree<String>>();
			BufferedReader br = new BufferedReader(new FileReader(infile));
			String line;
			int itest = 0, idx = 0, iline = 0, nerr = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals("")) {
					iline += 1;
					itest += 1;
					
					System.out.print("--itest " + itest + " done; ");
					if (idx != 200) {
						System.out.print("iline " + iline + ", idx " + idx);
					}
					
					Set<String>[][] constraint;
					if (someTrees.size() == 0) {
						Tree<String> gold = trees.get(itest - 1);
						int len = gold.getYield().size();
						constraint = getMem(len);
					} else {
						constraint = candidate(someTrees, numberer, numSubStatesArray);
					}
					constraints[itest - 1] = constraint;
					
					Set<String>[][] goldencons = goldenconst[itest - 1];
					
					if (!isIncluded(constraint, goldencons)) {
						nerr += 1;
						System.out.print("err ");
					}
					System.out.println();
					
					
					idx = 0;
					someTrees.clear();
					continue;
				}
				idx += 1;
				iline += 1;
				someTrees.add((new Trees.PennTreeReader(new StringReader(line))).next());
			}
			System.out.println("--ntest " + itest + ", nerr " + nerr + " / " + trees.size());
			br.close();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	
	public static boolean isIncluded(Set<String>[][] pred, Set<String>[][] gold) {
		int len = pred.length;
		for (int i = 0; i < len; i++) {
			for (int j = i; j < len; j++) {
				if (!pred[i][j].containsAll(gold[i][j])) {
					return false;
				}
			}
		}
		return true;
	}
	
	
	public static void testF1erCacheEntry(int N, String DATA_ROOT) throws Exception {
//		String DATA_ROOT = "F:/sourcecode/Data.Prd";
		
		String infile = DATA_ROOT + "/wsj.cache.40.200.21.pred";
		String grfile = DATA_ROOT + "/gr.tb.gr";
		String constr = DATA_ROOT + "/wsj.cache.40.200.21.cons";
		String treebk = TREE_ROOT + "/wsj/";
		
		String section = "final";
		List<Tree<String>> trees = getTrees(treebk, section);
		if ("train".equals(section)) {
			testF1erTraindata(N, infile, grfile, constr, trees);
		} else {
			testF1er(N, infile, grfile, constr, trees);
		}
	}
	
	
	
	private static List<Tree<String>> getTrees(String treebk, String section) {
		Corpus corpus = new Corpus(treebk, TreeBankType.WSJ, 1.0, false);
		List<Tree<String>> trees = null;
		if ("train".equals(section)) {
			trees = corpus.getTrainTrees();
		} else if ("final".equals(section)) {
			trees = corpus.getFinalTestingTrees();
		} else if ("dev".equals(section)) {
			trees = corpus.getDevTestingTrees();
		}
		return trees;
	}
	
	public static void testF1erEntry(int N, String DATA_ROOT) throws Exception {
		String infile = DATA_ROOT + "/gr.200.train.best.200";
		String grfile = DATA_ROOT + "/gr.gr";
		String constr = DATA_ROOT + "/gr.200.21.cons";
		String treebk = TREE_ROOT + "/wsj/";
		
		String section = "train";
		List<Tree<String>> trees = getTrees(treebk, section);
		if ("train".equals(section)) {
			testF1erTraindata(N, infile, grfile, constr, trees);
		} else {
			testF1er(N, infile, grfile, constr, trees);
		}
	}
	
	
	public static void testF1erTraindata(int N, String infile, String grfile, String constr, List<Tree<String>> trees) throws Exception {
		ParserData pData = ParserData.Load(grfile);
		Numberer.setNumberers(pData.getNumbs());
		Numberer numberer = Numberer.getGlobalNumberer("tags");
		short[] numSubStatesArray = pData.getGrammar().numSubStates;
		
		Set<String>[][][] constraints = new HashSet[N][][];
		
		try {
			List<Tree<String>> someTrees = new ArrayList<Tree<String>>();
			BufferedReader br = new BufferedReader(new FileReader(infile));
			String line;
			int itest = 0, idx = 0, iline = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals("")) {
					iline += 1;
					itest += 1;
					
					System.out.print("--itest " + itest + " done; ");
					if (idx != 200) {
						System.out.print("iline " + iline + ", idx " + idx);
					}
					System.out.println();
					
					Tree<String> gold = trees.get(itest - 1);
					someTrees.add(gold);
					
					Set<String>[][] constraint = candidate(someTrees, numberer, numSubStatesArray);
					constraints[itest - 1] = constraint;
					
					idx = 0;
					someTrees.clear();
					continue;
				}
				idx += 1;
				iline += 1;
				someTrees.add((new Trees.PennTreeReader(new StringReader(line))).next());
			}
			System.out.println("--ntest " + itest);
			br.close();
			
			Constraint cons = new Constraint(constraints);
			cons.save(constr);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	
	public static void testF1er(int N, String infile, String grfile, String constr, List<Tree<String>> trees) throws Exception {
		ParserData pData = ParserData.Load(grfile);
		Numberer.setNumberers(pData.getNumbs());
		Numberer numberer = Numberer.getGlobalNumberer("tags");
		short[] numSubStatesArray = pData.getGrammar().numSubStates;
		
		Set<String>[][][] constraints = new HashSet[N][][];
		
		try {
			List<Tree<String>> someTrees = new ArrayList<Tree<String>>();
			BufferedReader br = new BufferedReader(new FileReader(infile));
			String line;
			int itest = 0, idx = 0, iline = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals("")) {
					iline += 1;
					itest += 1;
					
					System.out.print("--itest " + itest + " done; ");
					if (idx != 200) {
						System.out.print("iline " + iline + ", idx " + idx);
					}
					System.out.println();
					
					if (someTrees.size() == 0) {
						Tree<String> gold = trees.get(itest - 1);
						int len = gold.getYield().size();
						constraints[itest - 1] = getMem(len);
					} else {
						Set<String>[][] constraint = candidate(someTrees, numberer, numSubStatesArray);
						constraints[itest - 1] = constraint;
					}
					
					idx = 0;
					someTrees.clear();
					continue;
				}
				idx += 1;
				iline += 1;
				someTrees.add((new Trees.PennTreeReader(new StringReader(line))).next());
			}
			System.out.println("--ntest " + itest);
			br.close();
			
			Constraint cons = new Constraint(constraints);
			cons.save(constr);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * @param infile
	 * @throws Exception
	 * Inf: F:/sourcecode/Rets.Prd/gr.sophis//gr.200.final.best.200
	 * Ret: iline 323691, idx 12--itest 2416 done; --ntest 2416, nerr 0 / 2416, ntree = 321476, average # of parses 133.0612582781457
	 */
	public void averageNumOfTrees(String infile) throws Exception {
		try {
			List<Tree<String>> someTrees = new ArrayList<Tree<String>>();
			BufferedReader br = new BufferedReader(new FileReader(infile));
			String line;
			int itest = 0, idx = 0, iline = 0, nerr = 0, ntree = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals("")) {
					iline += 1;
					itest += 1;
					
					System.out.print("--itest " + itest + " done; ");
					if (idx != 200) {
						System.out.print("iline " + iline + ", idx " + idx);
					}
					System.out.println();
					
					ntree += someTrees.size();
					
					idx = 0;
					someTrees.clear();
					continue;
				}
				idx += 1;
				iline += 1;
				someTrees.add((new Trees.PennTreeReader(new StringReader(line))).next());
			}
			double average = ntree / (double) itest;
			System.out.println("--ntest " + itest + ", nerr " + nerr + " / " + itest + 
					", ntree = " + ntree + ", average # of parses " + average);
			br.close();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	
	private static Set<String>[][] getMem(int nword) {
		Set<String>[][] mask = new HashSet[nword][nword];
		for (int i = 0; i < nword; i++) {
			for (int j = i; j < nword; j++) {
				mask[i][j] = new HashSet<String>();
			}
		}
		return mask;
	}
	
	
	public static Set<String>[][] candidate(List<Tree<String>> data, Numberer numberer, short[] numSubStatesArray) {
		List<Tree<String>> trees = Corpus.binarizeAndFilterTrees(
				data, 1, 0, 10000, Binarization.RIGHT, false, false);
		StateSetTreeList stateTreeList = new StateSetTreeList(
				trees, numSubStatesArray, false, numberer);
		
		int len = stateTreeList.get(0).getYield().size();
		Set<String>[][] mask = getMem(len);
		
		for (Tree<StateSet> tree : stateTreeList) {
			if (tree.getYield().size() != len) {
				System.out.println("--something must be wrong.");
				continue;
			}
			fillChat(tree, mask, numberer);
		}
		
		return mask;
	}
	
	
	public static void fillChat(Tree<StateSet> tree, Set<String>[][] mask, Numberer numberer) {
		if (tree.isLeaf()) { return; }
		List<Tree<StateSet>> children = tree.getChildren();
		for (Tree<StateSet> child : children) {
			fillChat(child, mask, numberer);
		}
		
		StateSet parent = tree.getLabel();
		short idParent = parent.getState();
		
		short x = parent.from, y = (short) (parent.to - 1);
		mask[x][y].add((String) numberer.object(idParent));
		
		if (tree.isPreTerminal()) {
			// StateSet word = children.get(0).getLabel();
		} else {
			switch (children.size()) {
			case 0:
				break;
			case 1: {
				StateSet child = children.get(0).getLabel();
				short idChild = child.getState();
				
				x = child.from;
				y = (short) (child.to - 1);
				mask[x][y].add((String) numberer.object(idChild));
				
				break;
			}
			case 2: {
				StateSet lchild = children.get(0).getLabel();
				StateSet rchild = children.get(1).getLabel();
				short idlChild = lchild.getState();
				short idrChild = rchild.getState();
				
				x = lchild.from;
				y = (short) (lchild.to - 1);
				mask[x][y].add((String) numberer.object(idlChild));
				
				x = rchild.from;
				y = (short) (rchild.to - 1);
				mask[x][y].add((String) numberer.object(idrChild));

				break;
			}
			default:
				throw new RuntimeException("Malformed tree: invalid # of children. # children: " + children.size());
			}
		}
	}
}
