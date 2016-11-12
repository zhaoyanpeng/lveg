package edu.shanghaitech.ai.nlp.lveg;

import java.util.List;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.util.MethodUtil;

public class LVeGGrammarTest {
	
//	static {
	// circle -3-5-7-
	GrammarRule r0 = new UnaryGrammarRule((short) 1, (short) 0, GrammarRule.GENERAL);
	GrammarRule r1 = new UnaryGrammarRule((short) 3, (short) 0, GrammarRule.GENERAL);
	GrammarRule r2 = new UnaryGrammarRule((short) 2, (short) 1, GrammarRule.GENERAL);
	GrammarRule r3 = new UnaryGrammarRule((short) 4, (short) 1, GrammarRule.GENERAL);
	GrammarRule r4 = new UnaryGrammarRule((short) 8, (short) 2, GrammarRule.GENERAL);
	UnaryGrammarRule r5 = new UnaryGrammarRule((short) 9, (short) 2, GrammarRule.GENERAL);
	UnaryGrammarRule r6 = new UnaryGrammarRule((short) 5, (short) 3, GrammarRule.GENERAL);
	UnaryGrammarRule r7 = new UnaryGrammarRule((short) 8, (short) 3, GrammarRule.GENERAL);
	UnaryGrammarRule r8 = new UnaryGrammarRule((short) 9, (short) 3, GrammarRule.GENERAL);
	UnaryGrammarRule r9 = new UnaryGrammarRule((short) 6, (short) 4, GrammarRule.GENERAL);
	UnaryGrammarRule r10 = new UnaryGrammarRule((short) 9, (short) 4, GrammarRule.GENERAL);
	UnaryGrammarRule r11 = new UnaryGrammarRule((short) 7, (short) 5, GrammarRule.GENERAL);
	UnaryGrammarRule r12 = new UnaryGrammarRule((short) 8, (short) 6, GrammarRule.GENERAL);
	UnaryGrammarRule r13 = new UnaryGrammarRule((short) 3, (short) 7, GrammarRule.GENERAL);
	UnaryGrammarRule r14 = new UnaryGrammarRule((short) 8, (short) 7, GrammarRule.GENERAL);
	UnaryGrammarRule r15 = new UnaryGrammarRule((short) 9, (short) 8, GrammarRule.GENERAL);
	UnaryGrammarRule r25 = new UnaryGrammarRule((short) 9, (short) 8, GrammarRule.GENERAL);
	
	BinaryGrammarRule rule16= new BinaryGrammarRule((short) 1, (short) 2, (short) 3);
	BinaryGrammarRule rule17= new BinaryGrammarRule((short) 2, (short) 4, (short) 6);
	BinaryGrammarRule rule18= new BinaryGrammarRule((short) 3, (short) 5, (short) 8);
	BinaryGrammarRule rule19= new BinaryGrammarRule((short) 2, (short) 8, (short) 9);
	BinaryGrammarRule rule20= new BinaryGrammarRule((short) 5, (short) 1, (short) 2);
	BinaryGrammarRule rule21= new BinaryGrammarRule((short) 8, (short) 4, (short) 7);
	BinaryGrammarRule rule23= new BinaryGrammarRule((short) 3, (short) 6, (short) 7);
	BinaryGrammarRule rule24= new BinaryGrammarRule((short) 7, (short) 8, (short) 9);

//	}
	
	int nTag = 10;
	LVeGGrammar grammar = new LVeGGrammar(null, 0.0, nTag);
	
	@Test
	public void testLVeGGrammar() {
		grammar.addUnaryRule((UnaryGrammarRule) r0);
		grammar.addUnaryRule((UnaryGrammarRule) r1);
		grammar.addUnaryRule((UnaryGrammarRule) r2);
		grammar.addUnaryRule((UnaryGrammarRule) r3);
		grammar.addUnaryRule((UnaryGrammarRule) r4);
		grammar.addUnaryRule(r5);
		grammar.addUnaryRule(r6);
		grammar.addUnaryRule(r7);
		grammar.addUnaryRule(r8);
		grammar.addUnaryRule(r9);
		grammar.addUnaryRule(r10);
		grammar.addUnaryRule(r11);
		grammar.addUnaryRule(r12);
		grammar.addUnaryRule(r13);
		grammar.addUnaryRule(r14);
		grammar.addUnaryRule(r15);
		// addUnaryRule() test
		grammar.addUnaryRule(r15);
		grammar.addUnaryRule(r25);
		
		/*
		for (int i = 0; i < nTag; i++) {
			List<GrammarRule> rules = grammar.getUnaryRuleWithC(i);
			System.out.println("Rules with child: " + i);
			System.out.println(rules);
		}
		*/
		
		MethodUtil.checkUnaryRuleCircle(grammar, null, true);
	}
}
