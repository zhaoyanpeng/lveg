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
//		testF1er(3000);
//		loadConstraints();
//		isGoldenParseIncludedEntry(3000);
		
		String root = "F:/sourcecode/Package/stanford-parser-use";
		testF1erCacheEntry(3000, root);
//		isGoldenParseIncludedCacheEntry(3000, root);
	}
	
	
	public void loadConstraints(String DATA_ROOT) {
		String constr = DATA_ROOT + "/gr.tb.200.cons";
		Constraint cons = (Constraint) ObjectFileManager.ObjectFile.load(constr);
		Set<String>[][][] constraints = cons.getConstraints();
		
		Set<String>[][] acon = constraints[0];
		int len = acon.length;
		for (int i = 0; i < len; i++) {
			for (int j = i; j < len; j++) {
				System.out.print(acon[i][j].size() + " ");
			}
			System.out.println();
		}
	}
	
	
	public void isGoldenParseIncludedCacheEntry(int N, String DATA_ROOT) throws Exception {
//		String DATA_ROOT = "F:/sourcecode/Data.Prd";
		
		String infile = DATA_ROOT + "/wsj.cache.40.500.23.pred";
		String grfile = DATA_ROOT + "/gr.tb.gr";
		String constr = DATA_ROOT + "/wsj.cache.40.500.23.cons";
		String treebk = TREE_ROOT + "/wsj/";
		
		isGoldenParseIncluded(N, infile, grfile, constr, treebk);
	}
	
	
	public void isGoldenParseIncludedEntry(int N, String DATA_ROOT) throws Exception {
//		String DATA_ROOT = "F:/sourcecode/Data.Prd";
		
		String infile = DATA_ROOT + "/gr.tb.200.final.best.200";
		String grfile = DATA_ROOT + "/gr.tb.gr";
		String constr = DATA_ROOT + "/gr.tb.200.cons";
		String treebk = DATA_ROOT + "/wsj/";
		
		isGoldenParseIncluded(N, infile, grfile, constr, treebk);
	}
	
	
	public void isGoldenParseIncluded(int N, String infile, String grfile, String constr, String treebk) throws Exception {

		ParserData pData = ParserData.Load(grfile);
		
		Numberer.setNumberers(pData.getNumbs());
		Numberer numberer = Numberer.getGlobalNumberer("tags");
		short[] numSubStatesArray = pData.getGrammar().numSubStates;
		
		Set<String>[][][] constraints = new HashSet[N][][];
		Set<String>[][][] goldenconst = new HashSet[N][][];
		
		int itree = 0;
		Corpus corpus = new Corpus(treebk, TreeBankType.WSJ, 1.0, false);
		List<Tree<String>> train = corpus.getTrainTrees();
		List<Tree<String>> test = corpus.getFinalTestingTrees();
		List<Tree<String>> val = corpus.getValidationTrees();
		List<Tree<String>> dev = corpus.getDevTestingTrees();
		
		System.out.println("|train| = " + train.size() + ", |test| = " + test.size() +
				", |val| = " + val.size() + ", |dev| = " + dev.size());
		
		for (Tree<String> tree : test) {
			List<Tree<String>> curTree = new ArrayList<Tree<String>>();
			
			curTree.add(tree);
			Set<String>[][] constraint = candidate(curTree, numberer, numSubStatesArray);
			goldenconst[itree] = constraint;
			itree += 1;
		}
		System.out.println("--total " + test.size());
		
		
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
					
					
					Set<String>[][] constraint = candidate(someTrees, numberer, numSubStatesArray);
					Set<String>[][] goldencons = goldenconst[itest - 1];
					constraints[itest - 1] = constraint;
					
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
			System.out.println("--ntest " + itest + ", nerr " + nerr + " / " + test.size());
			br.close();
			
//			Constraint cons = new Constraint(constraints);
//			cons.save(constr);
			
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
		
		String infile = DATA_ROOT + "/wsj.cache.40.500.23.pred";
		String grfile = DATA_ROOT + "/gr.tb.gr";
		String constr = DATA_ROOT + "/wsj.cache.40.500.23.cons";
		
		testF1er(N, infile, grfile, constr);
	}
	
	
	
	public static void testF1erEntry(int N, String DATA_ROOT) throws Exception {
//		String DATA_ROOT = "F:/sourcecode/Data.Prd";
		
		String infile = DATA_ROOT + "/gr.tb.200.final.best.200";
		String grfile = DATA_ROOT + "/gr.tb.gr";
		String constr = DATA_ROOT + "/gr.tb.200.cons";
		
		testF1er(N, infile, grfile, constr);
	}
	
	
	public static void testF1er(int N, String infile, String grfile, String constr) throws Exception {
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
	
	
	public static Set<String>[][] candidate(List<Tree<String>> data, Numberer numberer, short[] numSubStatesArray) {
		List<Tree<String>> trees = Corpus.binarizeAndFilterTrees(
				data, 1, 0, 10000, Binarization.RIGHT, false, false);
		StateSetTreeList stateTreeList = new StateSetTreeList(
				trees, numSubStatesArray, false, numberer);
		
		int len = stateTreeList.get(0).getYield().size();
		Set<String>[][] mask = new HashSet[len][len];
		for (int i = 0; i < len; i++) {
			for (int j = i; j < len; j++) {
				mask[i][j] = new HashSet<String>();
			}
		}
		
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
