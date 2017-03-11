package edu.shanghaitech.ai.nlp.lveg.impl;

import java.util.List;

import org.junit.Test;

import edu.shanghaitech.ai.nlp.lveg.impl.BinaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.impl.SimpleLVeGGrammar;
import edu.shanghaitech.ai.nlp.lveg.impl.UnaryGrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.GrammarRule;
import edu.shanghaitech.ai.nlp.lveg.model.LVeGGrammar;
import edu.shanghaitech.ai.nlp.util.FunUtil;

public class LVeGGrammarTest {
	
//	static {
	// circle -3-5-7-
	GrammarRule r0 = new UnaryGrammarRule((short) 1, (short) 0, GrammarRule.LRURULE);
	GrammarRule r1 = new UnaryGrammarRule((short) 3, (short) 0, GrammarRule.LRURULE);
	GrammarRule r2 = new UnaryGrammarRule((short) 2, (short) 1, GrammarRule.LRURULE);
	GrammarRule r3 = new UnaryGrammarRule((short) 4, (short) 1, GrammarRule.LRURULE);
	GrammarRule r4 = new UnaryGrammarRule((short) 8, (short) 2, GrammarRule.LRURULE);
	UnaryGrammarRule r5 = new UnaryGrammarRule((short) 9, (short) 2, GrammarRule.LRURULE);
	UnaryGrammarRule r6 = new UnaryGrammarRule((short) 5, (short) 3, GrammarRule.LRURULE);
	UnaryGrammarRule r7 = new UnaryGrammarRule((short) 8, (short) 3, GrammarRule.LRURULE);
	UnaryGrammarRule r8 = new UnaryGrammarRule((short) 9, (short) 3, GrammarRule.LRURULE);
	UnaryGrammarRule r9 = new UnaryGrammarRule((short) 6, (short) 4, GrammarRule.LRURULE);
	UnaryGrammarRule r10 = new UnaryGrammarRule((short) 9, (short) 4, GrammarRule.LRURULE);
	UnaryGrammarRule r11 = new UnaryGrammarRule((short) 7, (short) 5, GrammarRule.LRURULE);
	UnaryGrammarRule r12 = new UnaryGrammarRule((short) 8, (short) 6, GrammarRule.LRURULE);
	UnaryGrammarRule r13 = new UnaryGrammarRule((short) 3, (short) 7, GrammarRule.LRURULE);
	UnaryGrammarRule r14 = new UnaryGrammarRule((short) 8, (short) 7, GrammarRule.LRURULE);
	UnaryGrammarRule r15 = new UnaryGrammarRule((short) 9, (short) 8, GrammarRule.LRURULE);
	UnaryGrammarRule r25 = new UnaryGrammarRule((short) 9, (short) 8, GrammarRule.LRURULE);
	
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
	LVeGGrammar grammar = new SimpleLVeGGrammar(null, nTag, false, null);
	
	@Test
	public void testLVeGGrammar() {
		grammar.addURule((UnaryGrammarRule) r0);
		grammar.addURule((UnaryGrammarRule) r1);
		grammar.addURule((UnaryGrammarRule) r2);
		grammar.addURule((UnaryGrammarRule) r3);
		grammar.addURule((UnaryGrammarRule) r4);
		grammar.addURule(r5);
		grammar.addURule(r6);
		grammar.addURule(r7);
		grammar.addURule(r8);
		grammar.addURule(r9);
		grammar.addURule(r10);
		grammar.addURule(r11);
		grammar.addURule(r12);
		grammar.addURule(r13);
		grammar.addURule(r14);
		grammar.addURule(r15);
		// addURule() test
		grammar.addURule(r15);
		grammar.addURule(r25);
		
		/*
		for (int i = 0; i < nTag; i++) {
			List<GrammarRule> rules = grammar.getUnaryRuleWithC(i);
			System.out.println("Rules with child: " + i);
			System.out.println(rules);
		}
		*/
		
		FunUtil.checkUnaryRuleCircle(grammar, null, true);
	}
}
