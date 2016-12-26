package edu.shanghaitech.ai.nlp.util;

import java.io.Serializable;

import org.apache.log4j.Logger;
import org.apache.log4j.xml.DOMConfigurator;

/**
 * It contains two kinds of loggers, console logger and file logger, which 
 * are supposed not to be used at the same time. If you must use both of 
 * them together, please invoke <code>getConsoleLogger</code> before <code>
 * getFileLogger</code>. That is the best way I can think of.
 * 
 * @author Yanpeng Zhao
 *
 */
public class LogUtil implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -113853579415943859L;

	private static LogUtil instance;
	
	private static Logger logCons = null;
	private static Logger logFile = null;
	
	private final static String KEY = "log.name";
	private final static String LOGGER_XML = "config/log4j.xml";
	

	private LogUtil() {
		logFile = Logger.getLogger("FILE");
		logCons = Logger.getLogger("CONSOLE");
	}
	

	public static LogUtil getLogger() {
		if (instance == null) {
			instance = new LogUtil();
		}
		return instance;
	}
	
	
	public Logger getConsoleLogger() {
		DOMConfigurator.configure(LOGGER_XML);
		return logCons;
	}
	
	
	public Logger getFileLogger(String log) {
		System.setProperty(KEY, log);
		DOMConfigurator.configure(LOGGER_XML);
		return logFile;
	}
}
