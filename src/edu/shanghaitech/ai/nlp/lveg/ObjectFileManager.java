package edu.shanghaitech.ai.nlp.lveg;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class ObjectFileManager {
	
	public static class ObjectFile implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1852249590891181238L;
		
		public boolean save(String filename) {
			if (new File(filename).exists()) { filename += ".new"; }
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
		
		public static GrammarFile load(String filename) {
			GrammarFile gfile = null;
			try {
				FileInputStream fis = new FileInputStream(filename);
				GZIPInputStream gis = new GZIPInputStream(fis);
				ObjectInputStream ois = new ObjectInputStream(gis);
				gfile = (GrammarFile) ois.readObject();
				ois.close();
				gis.close();
				fis.close();
			} catch (IOException | ClassNotFoundException e) {
				e.printStackTrace();
				return null;
			}
			return gfile;
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
		
		public CorpusFile(StateTreeList train, StateTreeList test, StateTreeList dev) {
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
