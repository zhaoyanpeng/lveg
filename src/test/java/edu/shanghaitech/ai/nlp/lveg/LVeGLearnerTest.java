package edu.shanghaitech.ai.nlp.lveg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.junit.Test;

import edu.berkeley.nlp.syntax.Tree;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LearnerConfig.Options;
import edu.shanghaitech.ai.nlp.syntax.State;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.util.OptionParser;
import edu.shanghaitech.ai.nlp.util.Recorder;

public class LVeGLearnerTest extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = 952298389200776594L;

	@Test
	public void testLVeGLearner() {
		String[] args = {"param.in"};
		try {
			LVeGLearner.main(args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
//	@Test
	public void testLocally() {
		String[] args = null;
		String fparams = "param.in";
		try {
			args = LearnerConfig.readFile(fparams, StandardCharsets.UTF_8).split(",");
		} catch (IOException e) {
			e.printStackTrace();
		}
		OptionParser optionParser = new OptionParser(Options.class);
		Options opts = (Options) optionParser.parse(args, true);
		// configurations
		LearnerConfig.initialize(opts); // logger can only be used after the initialization
		logger.info("Calling with " + optionParser.getParsedOptions() + "\n");
		// loading data
		Numberer wrapper = new Numberer();
		Map<String, StateTreeList> trees = LearnerConfig.loadData(wrapper, opts);
		treeSummary(trees);
	}
	
	
	public void treeSummary(Map<String, StateTreeList> trees) {
		StateTreeList trainTrees = trees.get(LearnerConfig.ID_TRAIN);
		StateTreeList testTrees = trees.get(LearnerConfig.ID_TEST);
		StateTreeList devTrees = trees.get(LearnerConfig.ID_DEV);
		
		logger.trace("\n---training sentence length summary---\n");
		lengthSummary(trainTrees);
		logger.trace("\n---  test sentence length summary  ---\n");
		lengthSummary(testTrees);
		logger.trace("\n---   dev sentence length summary   ---\n");
		lengthSummary(devTrees);
	}
	
	public void lengthSummary(StateTreeList trees) {
		Map<Integer, Integer> summary = new HashMap<Integer, Integer>();
		for (Tree<State> tree : trees) {
			int len = tree.getTerminalYield().size();
			if (summary.containsKey(len)) {
				summary.put(len, summary.get(len) + 1);
			} else {
				summary.put(len, 1);
			}
		}
		logger.trace(summary + "\n");
		logger.trace(summary.keySet() + "\n");
		logger.trace(summary.values() + "\n");
		
		int nbin = 150;
		Map<Integer, Integer> lens = new HashMap<Integer, Integer>();
		for (Map.Entry<Integer, Integer> entry : summary.entrySet()) {
			for (int i = 0; i < nbin ; i += 10) {
				int len = entry.getKey();
				if (len < i) {
					if (lens.containsKey(i)) {
						lens.put(i, lens.get(i) + entry.getValue());
					} else {
						lens.put(i, entry.getValue());
					}
				}
			}
		}
		
		KeyComparator bykey = new KeyComparator(lens);
		TreeMap<Integer, Integer> sorted = new TreeMap<Integer, Integer>(bykey);
		sorted.putAll(lens);
		logger.trace(sorted.keySet() + "\n");
		logger.trace(sorted.values() + "\n");
	}
	
	
	class KeyComparator implements Comparator<Integer> {
	    Map<Integer, Integer> map;
	    public KeyComparator(Map<Integer, Integer> map) {
	        this.map = map;
	    }
		@Override
		public int compare(Integer o1, Integer o2) {
			return o1 - o2;
		}
	}
	
	
	Comparator<Map.Entry<Integer, Integer>> keycomparator = new Comparator<Map.Entry<Integer, Integer>>() {
		@Override
		public int compare(Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) {
			return o1.getKey() - o2.getKey();
		}
	};
}
