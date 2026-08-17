"""
OTF-LLM Engine: RLM & Context-as-a-Variable Verification Suite
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import time

# Add repository root to python module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from otf_llm.rlm_agent import ContextContainer, RLMAgent


def mock_llm_engine(prompt: str, max_tokens: int) -> str:
    """
    Mock LLM engine simulating intelligent RLM actions and sub-agent queries.
    """
    if "grep" not in prompt and "Observation" not in prompt:
        # First turn: search for security anomalies in logs
        return """I need to investigate the system log context `ctx` for security violations.
Let's search for authentication failures and privilege escalations.

```python
matches = ctx.grep("CRITICAL_SECURITY_ALERT")
print(f"Found {len(matches)} alerts:")
for m in matches:
    print(m)
```
"""
    elif "CRITICAL_SECURITY_ALERT" in prompt and "rlm" not in prompt:
        # Second turn: spawn an RLM sub-agent to audit the compromised user
        return """I found a critical injection attack on User ID 9481.
I will spawn a dedicated sub-agent to perform a focused security audit on this trace.

```python
audit_report = rlm("Perform deep vulnerability analysis on unauthorized SQL access", sub_data=matches[0])
print(f"Sub-agent Audit Result: {audit_report}")
final_answer(f"Security incident confirmed. Details: {audit_report}")
```
"""
    elif "Perform deep vulnerability analysis" in prompt:
        # Sub-agent execution
        return """[Sub-Agent] Identified SQL injection vulnerability in endpoint /api/v2/auth. 
Root cause: Unsanitized input parameter 'session_token'.
```python
final_answer("Vulnerability: SQL Injection in /api/v2/auth via 'session_token'. Status: High Severity.")
```
"""
    else:
        return "Task complete. final_answer('Investigation concluded.')"


def test_rlm_infinite_context_workflow():
    print("=" * 75)
    print("🚀 GT Labs AI — Testing Recursive Language Model (RLM) Architecture")
    print("=" * 75)

    # 1. Generate massive 100,000-line synthetic log data (>5,000,000 characters)
    print("\n📦 [1/3] Generating massive synthetic payload (100,000 log lines)...")
    log_lines = [f"2026-08-17 12:00:{i%60:02d} [INFO] Worker-{i%8} processed request #{i}" for i in range(100000)]
    
    # Inject security anomaly at line 42,891
    log_lines[42891] = "2026-08-17 12:34:56 [CRITICAL_SECURITY_ALERT] Unauthorized SQL injection attempt on user_id=9481 endpoint=/api/v2/auth"
    
    massive_log_data = "\n".join(log_lines)
    print(f"✅ Generated payload size: {len(massive_log_data) / (1024 ** 2):.2f} MB (Total Lines: {len(log_lines):,})")

    # 2. Encapsulate data into ContextContainer
    print("\n📦 [2/3] Initializing Context-as-a-Variable container...")
    ctx = ContextContainer(massive_log_data, name="server_logs")

    # 3. Launch RLM Agent
    print("\n⚙️ [3/3] Launching OTF-RLM Agent Workflow...")
    agent = RLMAgent(llm_generator=mock_llm_engine, max_depth=2, max_iterations=3)

    start_time = time.time()
    result = agent.run(
        task="Locate any critical security breaches in server_logs and audit the root cause.",
        ctx=ctx
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 75)
    print(f"📊 RLM EXECUTION SUMMARY:")
    print(f"   • Execution Time: {elapsed:.3f} seconds")
    print(f"   • Final Answer:   {result}")
    print("=" * 75)
    
    assert "SQL Injection" in result, "Test failed: Sub-agent analysis was not propagated!"
    print("🏆 ALL RLM TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_rlm_infinite_context_workflow()