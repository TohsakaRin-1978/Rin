import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.chatbot_engine import ChatbotEngine
from core.document_loader import load_txt_files
from core.text_splitter import split_text
from core.vector_store import VectorStore
from database.db import DB_PATH, get_recent_history, init_db, save_message


class FakeUploadedFile:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class CoreTests(unittest.TestCase):
    def test_load_txt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.txt").write_text("hello", encoding="utf-8")
            Path(tmp, "b.txt").write_text("world", encoding="utf-8")
            docs = load_txt_files(tmp)
            self.assertEqual([doc["filename"] for doc in docs], ["a.txt", "b.txt"])
            self.assertEqual([doc["content"] for doc in docs], ["hello", "world"])

    def test_split_text_with_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = split_text(text, chunk_size=10, overlap=2)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0], "abcdefghij")
        self.assertTrue(chunks[1].startswith("ijkl"))

    def test_database_round_trip(self):
        init_db()
        save_message("u1", "Q1", "A1")
        save_message("u1", "Q2", "A2")
        history = get_recent_history("u1", limit=2)
        self.assertEqual(history[-1], ("Q2", "A2"))
        self.assertEqual(history[0], ("Q1", "A1"))

    def test_vector_store_search_returns_relevant_text(self):
        store = VectorStore(persist_path=os.path.join(tempfile.gettempdir(), "edu_test_chroma"))
        store.add_documents(["Python is a language.", "Databases store data."])
        results = store.search("What is Python?", top_k=1)
        self.assertTrue(results)

    def test_chatbot_uses_history_and_retrieval(self):
        class StubStore:
            def search(self, query, top_k=3):
                return ["Python is a language."]

        engine = ChatbotEngine(StubStore())
        with patch.object(engine, "generator", return_value=[{"generated_text": "Python is a language."}]):
            answer, sources = engine.answer("What is Python?", [("Hi", "Hello")])
        self.assertEqual(answer, "Python is a language.")
        self.assertEqual(sources, ["Python is a language."])

    def test_chatbot_returns_no_answer_when_no_sources(self):
        class EmptyStore:
            def search(self, query, top_k=3):
                return []

        engine = ChatbotEngine(EmptyStore())
        answer, sources = engine.answer("Unknown question", [])
        self.assertIn("not available", answer)
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()
