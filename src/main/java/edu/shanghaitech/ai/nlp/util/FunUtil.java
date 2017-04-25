package edu.shanghaitech.ai.nlp.util;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import javax.imageio.ImageIO;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.ui.TreeJPanel;
import edu.shanghaitech.ai.nlp.data.StateTreeList;
import edu.shanghaitech.ai.nlp.lveg.LVeGTrainer;
import edu.shanghaitech.ai.nlp.util.Numberer;
import edu.shanghaitech.ai.nlp.syntax.State;

/**
 * Useful methods for debugging or ...
 * 
 * @author Yanpeng Zhao
 *
 */
public class FunUtil extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -9216654024276471124L;
	public final static double LOG_ZERO = -1.0e20;
	public final static double LOG_TINY = -0.5e20;
	public final static double EXP_ZERO = -Math.log(-LOG_ZERO);
	public final static NumberFormat formatter = new DecimalFormat("0.###E0");
	
	private static Random random = new Random(LVeGTrainer.randomseed);

	public static Comparator<Map.Entry<Integer, Integer>> keycomparator = new Comparator<Map.Entry<Integer, Integer>>() {
		@Override
		public int compare(Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) {
			return o1.getKey() - o2.getKey();
		}
	};
	
	public static class KeyComparator implements Comparator<Integer> {
	    Map<Integer, Integer> map;
	    public KeyComparator(Map<Integer, Integer> map) {
	        this.map = map;
	    }
		@Override
		public int compare(Integer o1, Integer o2) {
			return o1 - o2;
		}
	}
	
	
	/**
	 * @param dir the dir that needs to be created
	 * @return true if created successfully
	 */
	public static boolean mkdir(String dir) {
		File file = new File(dir);
		if (!file.exists()) {
			if (file.mkdir() || file.mkdirs()) {
				return true;
			}
		}
		return false;
	}
	
	
	/**
	 * Return log(a + b) given log(a) and log(b).
	 * 
	 * @param x in logarithm
	 * @param y in logarithm
	 * @return
	 */
	public static double logAdd(double x, double y) {
		double tmp, diff;
		if (x < y) {
			tmp = x;
			x = y;
			y = tmp;
		}
		diff = y - x; // <= 0
		if (diff < EXP_ZERO) { 
			// if y is far smaller than x
			return x < LOG_TINY ? LOG_ZERO : x;
		} else {
			return x + Math.log(1.0 + Math.exp(diff));
		}
	}
	
	
	/**
	 * Match a number with optional '-' and decimal.
	 * 
	 * @param str the string
	 * @return
	 */
	public static boolean isNumeric(String str){
		return str.matches("[-+]?\\d*\\.?\\d+");  
	}
	
	
	/**
	 * @param stateTree  the state parse tree
	 * @param filename   image name
	 * @param stringTree the string parse tree
	 * @throws Exception oops
	 */
	public static void saveTree2image(Tree<State> stateTree, String filename, Tree<String> stringTree, Numberer numberer) throws Exception {
		TreeJPanel tjp = new TreeJPanel();
		if (stringTree == null) {
			stringTree = StateTreeList.stateTreeToStringTree(stateTree, numberer);
			logger.trace("\nSTRING PARSE TREE: " + stringTree + "\n");
		}
		
		tjp.setTree(stringTree);
		BufferedImage bi = new BufferedImage(tjp.width(), tjp.height(), BufferedImage.TYPE_INT_ARGB);
		
		Graphics2D g2 = bi.createGraphics();
		g2.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f));
		Rectangle2D.Double rect = new Rectangle2D.Double(0, 0, tjp.width(), tjp.height());
		g2.fill(rect);
		g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		tjp.paintComponent(g2);
		g2.dispose();
		
		ImageIO.write(bi, "png", new File(filename + ".png"));
	}
	
	
	/**
	 * @param list     item container
	 * @param type     Doubel.class or Integer.class
	 * @param length   number of items in the list
	 * @param maxint   maximum for integer, and 1 for double
	 * @param nonzero  zero inclusive (false) or exclusive (true)
	 * @param negative allow the negative (true) or not allow (false)
	 */
	public static <T> void randomInitList(Random rnd, List<T> list, Class<T> type, int length, 
			int maxint, double ratio, boolean nonzero, boolean negative) {
		Double obj = new Double(0);
		for (int i = 0; i < length; i++) {
			double tmp = rnd.nextDouble() * maxint;
			if (nonzero) { while (tmp == 0.0) { tmp = rnd.nextDouble() * maxint; } }
			if (negative && tmp < ratio) { tmp = 0 - tmp; }
			list.add(type.isInstance(obj) ? type.cast(tmp) : type.cast((int) tmp));
		}
	}
	
	
	/**
	 * @param array  item container
	 * @param type   Double.class or Integer.class
	 * @param maxint maximum for integer, and 1 for double
	 * 
	 */
	public static <T> void randomInitArray(T[] array, Class<T> type, int maxint) {
		Double obj = new Double(0);
		for (int i = 0; i < array.length; i++) {
			double tmp = random.nextDouble() * maxint;
			array[i] = type.isInstance(obj) ? type.cast(tmp) : type.cast((int) tmp);
		}
		
	}
	
	
	public static void randomInitArrayInt(int[] array, int maxint) {
		for (int i = 0; i < array.length; i++) {
			array[i] = (int) (random.nextDouble() * maxint);
		}
	}
	
	
	public static void randomInitArrayDouble(double[] array) {
		for (int i = 0; i < array.length; i++) {
			array[i] = random.nextDouble();
		}
	}
	
	
	/**
	 * @param list        a list of doubles
	 * @param precision   double precision
	 * @param nfirst      print first # of items
	 * @param exponential whether the list should be read in the exponential form or not
	 * @return
	 */
	public static List<String> double2str(List<Double> list, int precision, int nfirst, boolean exponential, boolean scientific) {
		List<String> strs = new ArrayList<String>();
		String format = "%." + precision + "f", str;
		if (nfirst < 0 || nfirst > list.size()) { nfirst = list.size(); }
		for (int i = 0; i < nfirst; i++) {
			double value = exponential ? Math.exp(list.get(i)) : list.get(i);
			str = scientific ? formatter.format(value) : String.format(format, value);
			strs.add(str);
		}
		return strs;
	}
	
	
	/**
	 * @param list        a list of doubles
	 * @param exponential whether the list should be read in the exponential form or not
	 * @return
	 */
	public static double sum(List<Double> list, boolean exponential) {
		double sum = 0.0;
		for (Double d : list) {
			sum += exponential ? Math.exp(d) : d;
		}
		return sum;
	}
	
	
	public static void printArrayInt(int[] array) {
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		logger.trace(string + "\n");
	}
	
	
	public static void printArrayDouble(double[] array) {
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		logger.trace(string + "\n");
	}
	
	
	public static <T> void printArray(T[] array) {
		if (isEmpty(array)) { return; }
		String string = "[";
		for (int i = 0; i < array.length - 1; i++) {
			string += array[i] + ", ";
		}
		string += array[array.length - 1] + "]";
		logger.trace(string + "\n");
	}
	
	
	public static <T> void printList(List<T> list) {
		if (isEmpty(list)) { return; }
		String string = "[";
		for (int i = 0; i < list.size() - 1; i++) {
			string += list.get(i) + ", ";
		}
		string += list.get(list.size() - 1) + "]";
		logger.trace(string + "\n");
	}
	
	
	public static <T> boolean isEmpty(T[] array) {
		if (array == null || array.length == 0) {
			logger.error("[null or empty]\n");
			return true;
		}
		return false;
	}
	
	
	public static <T> boolean isEmpty(List<T> list) {
		if (list == null || list.isEmpty()) {
			logger.error("[null or empty]\n");
			return true;
		}
		return false;
	}
	
}
