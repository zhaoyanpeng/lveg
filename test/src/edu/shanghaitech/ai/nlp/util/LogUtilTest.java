package edu.shanghaitech.ai.nlp.util;

import org.apache.log4j.Logger;
import org.junit.Test;

public class LogUtilTest extends Recorder {

	@Test
	public void testLogUtils() {
		String logFile = "log/unittest"; 
		/* get console logger before the file logger */
		Logger logger0 = logUtil.getConsoleLogger();
		Logger logger1 = logUtil.getFileLogger(logFile);
		
		logger1.fatal("File Fatal 1");
		
		logger0.trace("Trace");
		logger0.debug("Debug");
		logger0.info("Info");
		logger0.warn("Warn");
		logger0.error("Error");
		logger0.fatal("Fatal");
		
		logger1.trace("Trace");
		logger1.debug("Debug");
		logger1.info("Info");
		logger1.warn("Warn");
		logger1.error("Error");
		logger1.fatal("Fatal");
		
		logger0.fatal("File Fatal 1");
	}
}
