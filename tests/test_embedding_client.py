import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import numpy as np

from langchain_agent.repository_knowledge import EmbeddingError
from langchain_agent.repository_knowledge.embedding import (
    SentenceTransformerEmbeddingClient,
)


class EmbeddingClientTests(unittest.TestCase):
    def test_model_encode_error_becomes_embedding_error(self) -> None:
        expected_error = RuntimeError("model execution failed")

        class FailingModel:
            def encode(self, texts, **kwargs):
                raise expected_error

        client = SentenceTransformerEmbeddingClient("fake-model")
        client._model = FailingModel()
        client._dimension = 2

        with self.assertRaises(EmbeddingError) as raised:
            client.embed_query("repository search")

        self.assertIs(raised.exception.__cause__, expected_error)

    def test_shared_client_serializes_model_encode_calls(self) -> None:
        first_entered = Event()
        release_first = Event()
        second_started = Event()
        second_entered = Event()

        class BlockingModel:
            def __init__(self) -> None:
                self.call_count = 0
                self.counter_lock = Lock()

            def encode(self, texts, **kwargs):
                with self.counter_lock:
                    self.call_count += 1
                    call_number = self.call_count

                if call_number == 1:
                    first_entered.set()
                    if not release_first.wait(timeout=5):
                        raise TimeoutError("test did not release first encode")
                else:
                    second_entered.set()

                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        model = BlockingModel()
        client = SentenceTransformerEmbeddingClient("fake-model")
        client._model = model
        client._dimension = 2

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.embed_query, "first query")
            self.assertTrue(first_entered.wait(timeout=5))

            def encode_second_query():
                second_started.set()
                return client.embed_query("second query")

            second = executor.submit(encode_second_query)
            self.assertTrue(second_started.wait(timeout=5))
            self.assertFalse(second_entered.wait(timeout=0.1))
            release_first.set()
            first.result(timeout=5)
            second.result(timeout=5)

        self.assertTrue(second_entered.is_set())
        self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
