package edu.shanghaitech.ai.nlp.lveg;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import edu.shanghaitech.ai.nlp.util.Recorder;

public class OptionParser extends Recorder {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3167531289091868339L;
	private final Map<String, Option> options = new HashMap<String, Option>();
	private final Map<String, Field> fields = new HashMap<String, Field>();
	private final Map<Class<?>, Type> types = new HashMap<Class<?>, Type>();
	private final Set<String> required = new HashSet<String>();
	private final Class<?> clsOption;
	private enum Type {
		CHAR, BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, BOOLEAN, STRING
	}
	
	private StringBuilder parsedOpts;
	
	public OptionParser(Class<?> clsOption) {
		this.clsOption = clsOption;
		for (Field field : clsOption.getDeclaredFields()) {
			Option option = field.getAnnotation(Option.class);
			if (option == null) { continue; }
			options.put(option.name(), option);
			fields.put(option.name(), field);
			if (option.required()) {
				required.add(option.name());
			}
		}
		types.put(char.class, Type.CHAR);
		types.put(byte.class, Type.BYTE);
		types.put(short.class, Type.SHORT);
		types.put(int.class, Type.INT);
		types.put(long.class, Type.LONG);
		types.put(float.class, Type.FLOAT);
		types.put(double.class, Type.DOUBLE);
		types.put(boolean.class, Type.BOOLEAN);
		types.put(String.class, Type.STRING);
	}
	
	public String getParsedOptions() {
		return parsedOpts.toString();
	}
	
	public Object parse(String[] args) {
		return parse(args, false, false);
	}
	
	public Object parse(String[] args, boolean exitIfFailed) {
		return parse(args, exitIfFailed, false);
	}
	
	private void usage() {
		System.out.println();
		for (Option option : options.values()) {
			System.out.printf("%-30s%s", option.name(), option.usage());
			if (option.required()) { System.out.printf(" [required]"); }
			System.out.println();
		}
		System.out.printf("%-30shelp message\n", "-h");
		System.out.println();
		System.exit(2);
	}
	
	private void error(boolean exitIfFailed, String value) {
		if (exitIfFailed) {
			throw new RuntimeException("Cannot recognize option |" + value + "|");
		} else {
			logger.warn("Cannot recognize option |" + value + "|");
		}
	}
	
	public Object parse(String[] args, boolean exitIfFailed, boolean parrot) {
		if (parrot) { logger.info("Calling with " + Arrays.deepToString(args)); }
		try {
			Set<String> seenOpts = new HashSet<String>();
			parsedOpts = new StringBuilder("{");
			Object option = clsOption.newInstance();
			for (int i = 0; i < args.length; i++) {
				String key = args[i], value = args[i + 1];
				if (key != null) { key = key.trim(); }
				if (value != null) { value = value.trim(); }
				if ("-h".equals(key)) { usage(); }
				seenOpts.add(key);
				Option opt = options.get(key);
				if (opt == null) { 
					error(exitIfFailed, key); 
					continue;
				}
				
				Field field = fields.get(key);
				Class<?> ftype = field.getType();
				if (!ftype.isEnum()) {
					switch (types.get(ftype)) {
					case CHAR: {
						field.setChar(option, value.charAt(0));
						break;
					}
					case BYTE: {
						field.setByte(option, Byte.parseByte(value));
						break;
					}
					case SHORT: {
						field.setShort(option, Short.parseShort(value));
						break;
					}
					case INT: {
						field.setInt(option, Integer.parseInt(value));
						break;
					}
					case LONG: {
						field.setLong(option, Long.parseLong(value));
						break;
					}
					case FLOAT: {
						field.setFloat(option, Float.parseFloat(value));
						break;
					}
					case DOUBLE: {
						field.setDouble(option, Double.parseDouble(value));
						break;
					}
					case STRING: {
						field.set(option, value);
						break;
					}
					case BOOLEAN: {
						field.setBoolean(option, Boolean.parseBoolean(value));
						break;
					}
					default: {
						try {
							Constructor<?> cst = ftype.getConstructor(new Class<?>[] {String.class});
							field.set(option, cst.newInstance((Object[]) (new String[] { value })));
						} catch (NoSuchMethodException e) {
							logger.error("Cannot construct object of type " + ftype.getCanonicalName() + " from the string.\n");
						}
					}
					}
				} else {
					Object[] values = ftype.getEnumConstants();
					boolean found = false;
					for (Object val : values) {
						String name = ((Enum<?>) val).name();
						if (value.equals(name)) {
							field.set(option, val);
							found = true;
							break;
						}
					}
					if (!found) { error(exitIfFailed, value); }
				}
				parsedOpts.append(String.format(" %s => %s", opt.name(), value));
				i++;
			}
			parsedOpts.append(" }");
			Set<String> leftOpts = new HashSet<String>(required);
			leftOpts.removeAll(seenOpts);
			if (!leftOpts.isEmpty()) { 
				logger.error("Failed to specify: " + leftOpts + "\n"); 
				usage();
			}
			return option;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
}
