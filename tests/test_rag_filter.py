"""Tests for rule-based RAG pre-filtering (app/services/rag_filter.py)."""

import unittest
from app.services.rag_filter import should_skip_rag


class TestRagFilter(unittest.TestCase):
    def test_casual_queries_skip_rag(self):
        casual_queries = [
            "hello",
            "Hi",
            "HEY!",
            "Good morning",
            "Good evening...",
            "thanks",
            "Thank you so much!",
            "thx",
            "how are you",
            "what's up?",
            "who are you",
            "ok",
            "OKAY",
            "got it",
            "sure",
            "yes",
            "no",
            "tell me a joke",
            "Tell a joke!",
        ]
        for query in casual_queries:
            with self.subTest(query=query):
                self.assertTrue(should_skip_rag(query), f"Expected '{query}' to skip RAG")

    def test_knowledge_queries_eligible_for_rag(self):
        knowledge_queries = [
            "What is RideOn Taxi?",
            "How do I request a refund?",
            "What is your pricing policy for extended licenses?",
            "Do you support custom Flutter setup?",
            "What are the system requirements for installation?",
            "Where can I find the documentation?",
        ]
        for query in knowledge_queries:
            with self.subTest(query=query):
                self.assertFalse(should_skip_rag(query), f"Expected '{query}' to be eligible for RAG")


if __name__ == "__main__":
    unittest.main()
