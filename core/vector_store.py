import hashlib
from pathlib import Path


def _safe_import_chromadb():
    try:
        import chromadb

        return chromadb
    except Exception:
        return None


def _safe_import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except Exception:
        return None


class _FallbackEmbeddingModel:
    def encode(self, text):
        if isinstance(text, list):
            return [self._encode_one(t) for t in text]
        return self._encode_one(text)

    def _encode_one(self, text):
        vector = [0.0] * 16
        for index, token in enumerate((text or "").lower().split()):
            bucket = index % len(vector)
            vector[bucket] += (sum(ord(ch) for ch in token) % 997) / 997.0
        return vector


class _FallbackCollection:
    def __init__(self):
        self._items = []

    def add(self, ids, embeddings, documents, metadatas):
        existing = {item["id"] for item in self._items}
        for item_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            if item_id in existing:
                continue
            existing.add(item_id)
            self._items.append({
                "id": item_id,
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            })

    def upsert(self, ids, embeddings, documents, metadatas):
        index = {item["id"]: i for i, item in enumerate(self._items)}
        for item_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            payload = {
                "id": item_id,
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }
            if item_id in index:
                self._items[index[item_id]] = payload
            else:
                index[item_id] = len(self._items)
                self._items.append(payload)

    def query(self, query_embeddings, n_results=3):
        if not self._items:
            return {"documents": [[]]}
        query = query_embeddings[0]
        ranked = sorted(
            self._items,
            key=lambda item: _cosine_similarity(query, item["embedding"]),
            reverse=True,
        )[:n_results]
        return {"documents": [[item["document"] for item in ranked]]}


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class VectorStore:
    def __init__(self, persist_path="data/chroma_db"):
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        chromadb = _safe_import_chromadb()
        SentenceTransformer = _safe_import_sentence_transformer()
        self.embedding_model = (
            SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            if SentenceTransformer
            else _FallbackEmbeddingModel()
        )
        if chromadb:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(
                "edu_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self.client = None
            self.collection = _FallbackCollection()

    def _make_id(self, text: str, index: int):
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"chunk_{index}_{text_hash}"

    def add_documents(self, chunks):
        """Add text chunks to ChromaDB."""
        if not chunks:
            return 0
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            ids.append(self._make_id(chunk, i))
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({"source": "local_knowledge_base"})
        if hasattr(self.collection, "upsert"):
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        else:
            self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(chunks)

    def search(self, query, top_k=3):
        """Search relevant knowledge chunks by semantic similarity."""
        query_embedding = self.embedding_model.encode(query)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["documents"][0] if results.get("documents") else []
