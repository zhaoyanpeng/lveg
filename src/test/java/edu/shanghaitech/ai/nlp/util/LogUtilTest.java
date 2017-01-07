package edu.shanghaitech.ai.nlp.util;

import org.apache.log4j.Logger;
import org.junit.Test;

public class LogUtilTest extends Recorder {

	@Test
	public void testLogUtils() {
		String logFile = "log/0_logger_test"; 
		/* get console logger before the file logger */
		
//		Logger logger0 = logUtil.getConsoleLogger();
		Logger logger1 = logUtil.getFileLogger(logFile);
		Logger logger0 = logUtil.getConsoleLogger();
		
		logger1.fatal("File Fatal 1\n");
		
		logger0.trace("Trace\n");
		logger0.debug("Debug\n");
		logger0.info("Info\n");
		logger0.warn("Warn\n");
		logger0.error("Error\n");
		logger0.fatal("Fatal\n");
		
		logger1.trace("Trace\n");
		logger1.debug("Debug\n");
		logger1.info("Info\n");
		logger1.warn("Warn\n");
		logger1.error("Error\n");
		logger1.fatal("Fatal\n");
		
		logger0.fatal("File Fatal 1\n");
		
	}
	
//	@Test
	public void testBothLogger() {
		String logfile = "log/0_logger_test"; 
		Logger logger0 = logUtil.getBothLogger(logfile);
		
		logger0.trace("Trace\n");
		logger0.debug("Debug\n");
		logger0.info("Info\n");
		logger0.warn("Warn\n");
		logger0.error("Error\n");
		logger0.fatal("Fatal\n");
	}
}
