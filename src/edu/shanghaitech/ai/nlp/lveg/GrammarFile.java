package edu.shanghaitech.ai.nlp.lveg;

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

public class GrammarFile implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -8119623200836693006L;
	LVeGGrammar grammar;
	LVeGLexicon lexicon;
	
	
	public GrammarFile(LVeGGrammar grammar, LVeGLexicon lexicon) {
		this.grammar = grammar;
		this.lexicon = lexicon;
	}
	
	
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
	
	public LVeGGrammar getGrammar() {
		return grammar;
	}
	
	
	public LVeGLexicon getLexicon() {
		return lexicon;
	}
}
