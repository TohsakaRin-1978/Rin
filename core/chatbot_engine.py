def _safe_import_pipeline():
    try:
        from transformers import pipeline

        return pipeline
    except Exception:
        return None


def _create_t5_generator():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        def generate(prompt):
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=150)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return [{"generated_text": text}]

        return generate
    except Exception:
        return None


class _FallbackGenerator:
    def __call__(self, prompt):
        return [{"generated_text": self._generate(prompt)}]

    def _generate(self, prompt):
        if "Course knowledge:" in prompt:
            tail = prompt.split("Course knowledge:", 1)[1]
            knowledge = tail.split("Student question:", 1)[0].strip()
            if knowledge:
                return knowledge.splitlines()[0][:500]
        return "The answer is not available in the current knowledge base."


class ChatbotEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        pipeline = _safe_import_pipeline()
        t5_gen = _create_t5_generator()

        if t5_gen:
            self.generator = t5_gen
        elif pipeline:
            try:
                self.generator = pipeline(
                    "text-generation",
                    model="gpt2",
                    max_new_tokens=150,
                )
            except Exception:
                self.generator = _FallbackGenerator()
        else:
            self.generator = _FallbackGenerator()

    def build_prompt(self, question, retrieved_docs, history):
        context_text = "\n".join(retrieved_docs)
        history_text = ""
        for user_msg, bot_msg in history:
            history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        prompt = f"""You are an educational assistant. Answer the student's question based on the
course knowledge below. If the knowledge base does not contain the answer,
say that the answer is not available in the current knowledge base.

Conversation history:
{history_text}

Course knowledge:
{context_text}

Student question:
{question}

Answer:"""
        return prompt

    def answer(self, question, history):
        retrieved_docs = self.vector_store.search(question, top_k=3)
        if not retrieved_docs:
            return "The answer is not available in the current knowledge base.", []
        prompt = self.build_prompt(question, retrieved_docs, history)
        result = self.generator(prompt)
        answer_text = result[0]["generated_text"]
        return answer_text, retrieved_docs
