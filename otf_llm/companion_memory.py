# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# companion_memory.py

import os
import json
import time
import re
from typing import List, Dict, Optional
import math


class CompanionMemoryManager:
    """
    Lightweight, zero-VRAM user long-term memory store.
    Extracts, stores, and retrieves relevant user facts using TF-IDF cosine similarity.
    Designed for instant < 2ms retrieval in CPU RAM.
    """

    def __init__(self, storage_filepath: str = "companion_user_memory.json"):
        self.storage_filepath = storage_filepath
        self.memories: List[Dict[str, str]] = []
        self._load_memory()

    def _load_memory(self) -> None:
        """Loads memory facts from a local JSON storage file."""
        if os.path.exists(self.storage_filepath):
            try:
                with open(self.storage_filepath, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
                print(f"[CompanionMemory] Successfully loaded {len(self.memories)} memory entries.")
            except Exception as e:
                print(f"[CompanionMemory] Warning: Failed to load storage file: {e}")
                self.memories = []
        else:
            self.memories = []

    def _save_memory(self) -> None:
        """Persists current memory facts to the local JSON storage file."""
        try:
            with open(self.storage_filepath, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CompanionMemory] Error: Failed to save memory to disk: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Normalizes and tokenizes text for TF-IDF matching."""
        text_clean = re.sub(r"[^\w\s]", "", text.lower())
        tokens = [word for word in text_clean.split() if len(word) > 2]
        return tokens

    def auto_extract_and_store(self, user_text: str) -> bool:
        """
        Scans incoming user messages for explicit facts (e.g., 'My name is...', 'I live in...', 'My favorite language is...').
        Saves new facts to memory if triggered.
        """
        fact_triggers = [
            r"my name is (.+)",
            r"i am a (.+)",
            r"i work as (.+)",
            r"i live in (.+)",
            r"my favorite (.+) is (.+)",
            r"меня зовут (.+)",
            r"я работаю (.+)",
            r"мой любимый (.+) это (.+)",
            r"я живу в (.+)"
        ]

        text_lower = user_text.strip()
        new_fact_found = False

        for trigger in fact_triggers:
            match = re.search(trigger, text_lower, re.IGNORECASE)
            if match:
                fact_str = user_text.strip()
                # Check for duplicates
                if not any(m["fact"].lower() == fact_str.lower() for m in self.memories):
                    entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "fact": fact_str
                    }
                    self.memories.append(entry)
                    self._save_memory()
                    print(f"[CompanionMemory] 🧠 New fact stored: '{fact_str}'")
                    new_fact_found = True
                break

        return new_fact_found

    def add_explicit_fact(self, fact_text: str) -> None:
        """Manually adds a fact entry to the memory bank."""
        fact_clean = fact_text.strip()
        if fact_clean and not any(m["fact"].lower() == fact_clean.lower() for m in self.memories):
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fact": fact_clean
            }
            self.memories.append(entry)
            self._save_memory()
            print(f"[CompanionMemory] 🧠 Explicit fact added: '{fact_clean}'")

    def retrieve_relevant_memories(self, query: str, top_k: int = 3, threshold: float = 0.1) -> List[str]:
        """
        Calculates cosine TF-IDF similarity between query and stored memories on CPU.
        Returns Top-K relevant background facts in < 2ms.
        """
        if not self.memories:
            return []

        t0 = time.time()
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build vocabulary across query and stored facts
        corpus = [m["fact"] for m in self.memories]
        corpus_tokens = [self._tokenize(doc) for doc in corpus]

        # Calculate TF-IDF vectors
        doc_count = len(corpus) + 1
        idf_dict = {}

        all_words = set(query_tokens)
        for tokens in corpus_tokens:
            all_words.update(tokens)

        for word in all_words:
            matching_docs = sum(1 for tokens in corpus_tokens if word in tokens)
            if word in query_tokens:
                matching_docs += 1
            idf_dict[word] = math.log((doc_count + 1) / (matching_docs + 1)) + 1.0

        def get_vector(tokens: List[str]) -> Dict[str, float]:
            tf_dict = {}
            for t in tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1
            vec = {}
            for t, count in tf_dict.items():
                vec[t] = (count / len(tokens)) * idf_dict.get(t, 1.0)
            return vec

        query_vec = get_vector(query_tokens)

        # Calculate cosine similarity
        scores = []
        for idx, doc_tok in enumerate(corpus_tokens):
            if not doc_tok:
                continue
            doc_vec = get_vector(doc_tok)

            # Dot product
            dot_product = sum(query_vec.get(w, 0.0) * doc_vec.get(w, 0.0) for w in query_vec)

            # Norms
            norm_q = math.sqrt(sum(v ** 2 for v in query_vec.values()))
            norm_d = math.sqrt(sum(v ** 2 for v in doc_vec.values()))

            similarity = dot_product / (norm_q * norm_d) if (norm_q * norm_d) > 0 else 0.0

            if similarity >= threshold:
                scores.append((similarity, corpus[idx]))

        scores.sort(key=lambda x: x[0], reverse=True)
        relevant_facts = [fact for score, fact in scores[:top_k]]

        elapsed_ms = (time.time() - t0) * 1000
        if relevant_facts:
            print(f"[CompanionMemory] ⚡ Retrieved {len(relevant_facts)} relevant memories in {elapsed_ms:.2f} ms.")

        return relevant_facts

    def inject_memory_into_system_prompt(self, system_prompt: str, user_query: str) -> str:
        """
        Searches for relevant facts and injects them into the system prompt structure.
        """
        relevant_facts = self.retrieve_relevant_memories(user_query, top_k=3)
        if not relevant_facts:
            return system_prompt

        memory_context = "\n\n[USER LONG-TERM PERSONAL MEMORY]:\n" + "\n".join(f"- {f}" for f in relevant_facts)
        return system_prompt + memory_context


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING COMPANION LONG-TERM MEMORY SYSTEM")
    print("=" * 70)

    memory_manager = CompanionMemoryManager(storage_filepath="test_memory.json")

    # 1. Test Auto Extraction
    memory_manager.auto_extract_and_store("Hello! My name is Gleb Tikhiy and I am an AI Engineer.")
    memory_manager.auto_extract_and_store("I live in Munich and my favorite language is Python.")

    # 2. Test Manual Addition
    memory_manager.add_explicit_fact("User is working on OTF-LLM Engine project at GT Labs AI.")

    # 3. Test Relevant Retrieval
    test_query = "Can you write a Python script for my AI project?"
    retrieved = memory_manager.retrieve_relevant_memories(test_query)

    print("\n📝 Retained Memory Output for Query:")
    for fact in retrieved:
        print(f"  • {fact}")

    # 4. Test System Prompt Injection
    base_sys = "You are a helpful Senior AI Assistant."
    enhanced_sys = memory_manager.inject_memory_into_system_prompt(base_sys, test_query)

    print("\n🤖 Enhanced System Prompt:")
    print(enhanced_sys)

    # Cleanup test file
    if os.path.exists("test_memory.json"):
        os.remove("test_memory.json")