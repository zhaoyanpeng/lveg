package edu.shanghaitech.ai.nlp.data;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGLexicon;
import edu.shanghaitech.ai.nlp.util.Numberer;

public class ObjectFileManager {
	
	public static class ObjectFile implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1852249590891181238L;
		
		public boolean save(String filename) {
			if (new File(filename).exists()) { 
				filename += new SimpleDateFormat(".yyyyMMddHHmmss").format(new Date());
			}
			try {
				FileOutputStream fos = new FileOutputStream(filename);
				GZIPOutputStream gos = new GZIPOutputStream(fos);
				ObjectOutputStream oos = new ObjectOutputStream(gos);
				oos.writeObject(this);
				oos.flush();
				oos.close();
				gos.close();
				fos.close();
			} catch (IOException e) {
				e.printStackTrace();
				return false;
			} 
			return true;
		}
		
		public static Object load(String filename) {
			Object o = null;
			try {
				FileInputStream fis = new FileInputStream(filename);
				GZIPInputStream gis = new GZIPInputStream(fis);
				ObjectInputStream ois = new ObjectInputStream(gis);
				o = ois.readObject();
				ois.close();
				gis.close();
				fis.close();
			} catch (IOException | ClassNotFoundException e) {
				e.printStackTrace();
				return null;
			}
			return o;
		}
	}
	
	
	public static class CorpusFile extends ObjectFile {
		/**
		 * 
		 */
		private static final long serialVersionUID = -5871763111246836457L;
		private StateTreeList train;
		private StateTreeList test;
		private StateTreeList dev;
		private Numberer numberer;
		
		public CorpusFile(StateTreeList train, StateTreeList test, StateTreeList dev, Numberer numberer) {
			this.numberer = numberer;
			this.train = train;
			this.test = test;
			this.dev = dev;
		}

		public StateTreeList getTrain() {
			return train;
		}

		public StateTreeList getTest() {
			return test;
		}

		public StateTreeList getDev() {
			return dev;
		}
		
		public Numberer getNumberer() {
			return numberer;
		}
	}
	
	
	public static class GrammarFile extends ObjectFile {
		/**
		 * 
		 */
		private static final long serialVersionUID = -8119623200836693006L;
		private LVeGGrammar grammar;
		private LVeGLexicon lexicon;
		
		public GrammarFile(LVeGGrammar grammar, LVeGLexicon lexicon) {
			this.grammar = grammar;
			this.lexicon = lexicon;
		}
		
		public LVeGGrammar getGrammar() {
			return grammar;
		}
		
		public LVeGLexicon getLexicon() {
			return lexicon;
		}
	}
}
