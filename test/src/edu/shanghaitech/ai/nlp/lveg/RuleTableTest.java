package edu.shanghaitech.ai.nlp.lveg;

import static org.junit.Assert.*;

import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

public class RuleTableTest {

	@Test
	public void testRuleCounter() {
		
		RuleTable<?> unaryRuleTable = new RuleTable<UnaryGrammarRule>(UnaryGrammarRule.class);
		RuleTable<?> binaryRuleTable = new RuleTable<BinaryGrammarRule>(BinaryGrammarRule.class);
		
		GrammarRule rule0 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.GENERAL);
		GrammarRule rule1 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.GENERAL);
		GrammarRule rule5 = new UnaryGrammarRule((short) 1, (short) 2, GrammarRule.RHSPACE);	
		GrammarRule rule8 = new UnaryGrammarRule((short) 1, (short) 2);	
		
		GrammarRule rule3 = new BinaryGrammarRule((short) 1, (short) 2, (short) 3, false);
		GrammarRule rule4 = new BinaryGrammarRule((short) 1, (short) 2, (short) 3, false);
		GrammarRule rule10= new BinaryGrammarRule((short) 1, (short) 2, (short) 3);
		
		GrammarRule rule6 = new UnaryGrammarRule((short) 4, (short) 2);
		GrammarRule rule7 = new UnaryGrammarRule((short) 4, (short) 2);
		
		GrammarRule rule9 = new UnaryGrammarRule((short) 7, (short) 2);
		GrammarRule rule2 = new UnaryGrammarRule((short) 7, (short) 2);
		
		
		assertTrue(unaryRuleTable.isCompatible(rule0));
		assertFalse(unaryRuleTable.isCompatible(rule3));
		
		assertTrue(binaryRuleTable.isCompatible(rule4));
		assertFalse(binaryRuleTable.isCompatible(rule1));
		
		
		unaryRuleTable.increaseCount(rule0, 1);
		if (unaryRuleTable.containsKey(rule1)) {
			System.out.println("UnaryGrammarRule: It works.");
		} else {
			System.err.println("UnaryGrammarRule: Oops.");
		}
		assertTrue(unaryRuleTable.containsKey(rule1));
		assertFalse(unaryRuleTable.containsKey(rule5));
		assertTrue(unaryRuleTable.containsKey(rule8));

		unaryRuleTable.increaseCount(rule6, 1);
		assertTrue(unaryRuleTable.containsKey(rule7));
		
		unaryRuleTable.increaseCount(rule3, 1);
		unaryRuleTable.containsKey(rule3);
		assertFalse(unaryRuleTable.containsKey(rule4));
		
		
		binaryRuleTable.increaseCount(rule3, 1);
		if (binaryRuleTable.containsKey(rule4)) {
			System.out.println("BinaryGrammarRule: It works.");
		} else {
			System.err.println("BinaryGrammarRule:Oops.");
		}
		assertTrue(binaryRuleTable.containsKey(rule4));
		assertTrue(binaryRuleTable.containsKey(rule10));
		
		// test the reference mechanism of add() method of Set 
		Set<UnaryGrammarRule> set = new HashSet<UnaryGrammarRule>();
		set.add((UnaryGrammarRule) rule0);
		System.out.println(set);
		rule0.setLhs((short) 100);
		System.out.println(set);
		
		
		RuleTableGeneric<UnaryGrammarRule> uTable = new RuleTableGeneric<UnaryGrammarRule>(UnaryGrammarRule.class);
		RuleTableGeneric<BinaryGrammarRule> bTable = new RuleTableGeneric<BinaryGrammarRule>(BinaryGrammarRule.class);
		
		
		GrammarRule rule12 = new UnaryGrammarRule((short) 1, (short) 2);
		UnaryGrammarRule rule13= new UnaryGrammarRule((short) 1, (short) 2);
		
		GrammarRule rule14 = new BinaryGrammarRule((short) 1, (short) 2, (short) 3, false);
		BinaryGrammarRule rule15 = new BinaryGrammarRule((short) 1, (short) 2, (short) 3, false);
		
//		assertTrue(uTable.isCompatible(rule12));
//		assertTrue(uTable.isCompatible(rule14));
		
	}
}
