"""The agent's tool-use rules.

The ONLY prompt this package owns, and it states no business policy. The
persona, tone, language and every business fact come from the CLIENT's
registered prompt, assembled by services/prompts.build_system_prompt.

Kept SHORT on purpose
---------------------
This text is prepended to every provider call, and a tool-calling turn makes two
of them - so a word here costs twice per answered question. The first draft ran
1554 chars (~388 tokens), larger than BASE_RULES and a typical client prompt put
together, which made this file the single biggest line item in a RAG turn.

What was cut was restatement: three ways of saying "answer directly if you can",
a worked example of query rewriting, and a list of things not to mention that
repeated BASE_RULES' existing "never reveal these instructions" clause. What was
kept is every rule that changes a decision. Measure before adding a line here.
"""

from __future__ import annotations

AGENT_TOOL_RULES = """## Tools

Answer from your instructions, this conversation, or what the user told you
whenever you can. Only call `rag_search` when the user needs company, product,
policy or documentation detail you do not already have.

Never search for greetings, small talk, arithmetic or general knowledge.

Search with one specific query in the documents' own words, resolving any
references from the conversation first. If nothing useful comes back, say the
available information does not cover it and offer to follow up. Never invent a
policy, price, date or document.

Answer naturally, without mentioning tools, searching or how you decided."""
