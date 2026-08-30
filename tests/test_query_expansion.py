import os
import unittest
from unittest.mock import patch

from langchain_agent.model import create_model
from langchain_agent.repository_knowledge import QueryExpansionError
from langchain_agent.repository_knowledge._internal.retrieval.multi_query_retriever import (
    MultiQueryRetriever,
)
from langchain_agent.repository_knowledge.query_expansion import (
    FallbackQueryExpander,
    IdentityQueryExpander,
    LLMQueryExpander,
    QueryExpansionPayload,
)


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.config = None

    def invoke(self, messages, config=None):
        self.messages = messages
        self.config = config
        return self.response


class FakeChatModel:
    def __init__(self, response):
        self.structured_model = FakeStructuredModel(response)
        self.schema = None
        self.include_raw = None

    def with_structured_output(self, schema, *, include_raw=False):
        self.schema = schema
        self.include_raw = include_raw
        return self.structured_model


class RaisingExpander:
    def __init__(self, error):
        self.error = error

    def expand(self, query, *, limit):
        raise self.error


class RecordingExpander:
    def __init__(self, rewrites):
        self.rewrites = rewrites
        self.calls = []

    def expand(self, query, *, limit):
        self.calls.append((query, limit))
        return self.rewrites


class RecordingRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=5, minimum_score=None):
        self.queries.append(query)
        return []


class QueryExpansionTests(unittest.TestCase):
    def test_llm_expander_uses_structured_output_and_sanitizes_rewrites(self):
        model = FakeChatModel(
            {
                "raw": object(),
                "parsed": QueryExpansionPayload(
                    rewrites=[
                        " original query ",
                        " permission middleware ",
                        "PERMISSION MIDDLEWARE",
                        "tool denial policy",
                        "unused third rewrite",
                    ]
                ),
                "parsing_error": None,
            }
        )
        expander = LLMQueryExpander(model=model)

        rewrites = expander.expand(" original query ", limit=2)

        self.assertEqual(
            rewrites,
            (
                "permission middleware",
                "tool denial policy",
            ),
        )
        self.assertIs(model.schema, QueryExpansionPayload)
        self.assertTrue(model.include_raw)
        self.assertEqual(
            model.structured_model.config["run_name"],
            "rag_query_expansion",
        )

    def test_structured_parsing_failure_becomes_query_expansion_error(self):
        parsing_error = ValueError("invalid payload")
        model = FakeChatModel(
            {
                "raw": object(),
                "parsed": None,
                "parsing_error": parsing_error,
            }
        )

        with self.assertRaises(QueryExpansionError) as raised:
            LLMQueryExpander(model=model).expand("query", limit=2)

        self.assertIs(raised.exception.__cause__, parsing_error)

    def test_fallback_handles_expected_expansion_failure(self):
        expander = FallbackQueryExpander(
            primary=RaisingExpander(QueryExpansionError("provider unavailable")),
            fallback=IdentityQueryExpander(),
        )

        self.assertEqual(expander.expand("query", limit=2), ())

    def test_fallback_does_not_hide_programming_errors(self):
        expander = FallbackQueryExpander(
            primary=RaisingExpander(TypeError("implementation bug")),
            fallback=IdentityQueryExpander(),
        )

        with self.assertRaises(TypeError):
            expander.expand("query", limit=2)

    def test_multi_query_passes_budget_and_enforces_it(self):
        base_retriever = RecordingRetriever()
        query_expander = RecordingExpander(
            [
                "rewrite one",
                "rewrite two",
                "rewrite three",
            ]
        )
        retriever = MultiQueryRetriever(
            base_retriever=base_retriever,
            query_expander=query_expander,
            max_query_rewrites=2,
        )

        retriever.retrieve("original", top_k=3)

        self.assertEqual(query_expander.calls, [("original", 2)])
        self.assertEqual(
            base_retriever.queries,
            [
                "original",
                "rewrite one",
                "rewrite two",
            ],
        )


class ModelFactoryTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "MODEL_NAME": "deepseek-v4-flash",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://example.invalid",
        },
    )
    @patch("langchain_agent.model.ChatDeepSeek")
    def test_query_model_disables_thinking_without_changing_model_name(
        self,
        chat_deepseek,
    ):
        create_model(thinking=False)

        options = chat_deepseek.call_args.kwargs
        self.assertEqual(options["model"], "deepseek-v4-flash")
        self.assertEqual(
            options["extra_body"],
            {
                "thinking": {
                    "type": "disabled",
                }
            },
        )

    @patch.dict(
        os.environ,
        {
            "MODEL_NAME": "deepseek-v4-flash",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://example.invalid",
        },
    )
    @patch("langchain_agent.model.ChatDeepSeek")
    def test_default_model_leaves_thinking_mode_unspecified(self, chat_deepseek):
        create_model()

        self.assertNotIn(
            "extra_body",
            chat_deepseek.call_args.kwargs,
        )


if __name__ == "__main__":
    unittest.main()
