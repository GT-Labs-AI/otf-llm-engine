"""
OTF-LLM Engine: Recursive Language Model (RLM) & Context-as-a-Variable Agent
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import re
import io
import json
import types
import contextlib
import traceback
from typing import Dict, Any, List, Optional, Callable

# Comprehensive stopword list to prevent noisy grep dumps
STOPWORDS = {
    "the", "for", "and", "what", "values", "exact", "author", "official",
    "contact", "regarding", "rule", "search", "codebase", "find", "this",
    "that", "with", "from", "have", "how", "number", "help", "guide",
    "process", "each", "step", "followed", "correctly", "includes", "concerning",
    "look", "looks", "items", "need", "needs", "item", "empty", "emptyness"
}


class GrepResult(list):
    """Custom list container that formats cleanly on print()."""
    def __str__(self):
        return "\n".join(self) if self else "[No matches found]"

    def __repr__(self):
        return str(self)


class ContextContainer:
    """
    Context-as-a-Variable memory wrapper.
    Exposes fast analytical search APIs to LLM code execution.
    """
    def __init__(self, data: Any, name: str = "ctx"):
        self.data = data
        self.name = name

    def grep(self, pattern: str, max_matches: int = 3) -> GrepResult:
        """Searches lines matching regex pattern (capped at max_matches)."""
        results = GrepResult()
        clean_pattern = pattern.strip().strip("'\"")
        if not clean_pattern or clean_pattern.lower() in STOPWORDS or len(clean_pattern) < 3:
            return results

        if isinstance(self.data, str):
            for line_idx, line in enumerate(self.data.splitlines()):
                if re.search(re.escape(clean_pattern), line, re.IGNORECASE):
                    # Clean line to prevent context pollution
                    clean_line = line.strip()[:140]
                    results.append(f"L{line_idx+1}: {clean_line}")
                    if len(results) >= max_matches:
                        break
        return results

    def head(self, n: int = 20) -> str:
        if isinstance(self.data, str):
            return "\n".join(self.data.splitlines()[:n])
        return str(self.data)[:1000]

    def describe(self) -> str:
        if isinstance(self.data, str):
            lines = self.data.splitlines()
            return f"TextDocument (chars: {len(self.data):,}, lines: {len(lines):,})"
        return f"Object of type {type(self.data).__name__}"


class PythonREPLExecutor:
    """
    Safe execution environment with high-entropy keyword extraction.
    """
    @staticmethod
    def execute(raw_text: str, env_vars: Dict[str, Any], fallback_query: str = "") -> Dict[str, Any]:
        ctx: ContextContainer = env_vars.get("ctx", None)
        stdout_buffer = io.StringIO()
        executed_lines = []

        if ctx is not None:
            # 1. Extract explicit ctx.grep("pattern") calls from LLM response
            explicit_patterns = re.findall(r'ctx\.grep\s*\(\s*["\']([^"\']+)["\']', raw_text, re.IGNORECASE)

            # Filter explicit patterns
            patterns = [p for p in explicit_patterns if p.lower() not in STOPWORDS and len(p) >= 3]

            # 2. Extract meaningful domain keywords from user query if needed
            if not patterns and fallback_query:
                raw_keywords = re.findall(r'[A-Za-z0-9_]{3,}', fallback_query)
                patterns = [kw for kw in raw_keywords if kw.lower() not in STOPWORDS]

            if patterns:
                for pat in list(dict.fromkeys(patterns)):
                    res = ctx.grep(pat, max_matches=3)
                    if res:
                        stdout_buffer.write(f"=== Keyword '{pat}' ===\n{str(res)}\n\n")
                    executed_lines.append(f"print(ctx.grep('{pat}'))")

                return {
                    "success": True,
                    "stdout": stdout_buffer.getvalue().strip(),
                    "stderr": "",
                    "error": None,
                    "executed_code": "\n".join(executed_lines)
                }

        return {
            "success": True,
            "stdout": "[No relevant search matches found]",
            "stderr": "",
            "error": None,
            "executed_code": ""
        }


class RLMAgent:
    """
    Recursive Language Model (RLM) Agent Dispatcher.
    """
    def __init__(self, llm_generator: Callable[[str, int], str], max_depth: int = 2):
        self.llm_generator = llm_generator
        self.max_depth = max_depth

    def run(self, task: str, ctx: ContextContainer, current_depth: int = 0) -> str:
        if current_depth > self.max_depth:
            return "[RLM] Max depth reached."
        return self.llm_generator(f"Task: {task} Context: {ctx.describe()}", 256)