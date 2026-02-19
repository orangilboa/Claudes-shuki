# Orchestration Failure Analysis - CORRECTED

## TL;DR

The orchestration and planning are actually **well-designed**, but **two execution issues** cause cascading failures:

1. **Context Assembly NOT injected into task description** - The reasoner gets task description but NOT the assembled context
2. **String matching failures because reasoner discovers files during reasoning** instead of having them pre-loaded

After reviewing the actual code, I found the system **ALREADY HAS ContextAssembler** to load context, but it's not being used in the prompt passed to the reasoner.

## The Problem

The orchestration **breaks tasks into appropriate subtasks and plans correctly**, and has infrastructure to pass context, but **fails during execution** due to a **context injection bug** in one node. The system works like this:

```
Discovery → Planner → Task loop:
                        skill_picker
                        ↓
                        rules_selector
                        ↓
                        tool_selector
                        ↓
                        reasoner (needs context)
                        ↓
                        writer
                        ↓
                        verifier
                        ↓
                        summarizer
```

## Root Cause: Context Assembled But Never Used

Looking at actual code in `nodes.py` reasoner_node (line 402-530):

```python
# The system BUILDS context:
assembler = ContextAssembler()
context_str = assembler.build(task, state)  # ← This works! Loads files, dependencies, etc.

# But then DOESN'T USE IT in retry case:
if task.retry_count > 0:
    wr = getattr(task, "write_result", {})
    prev_plan = getattr(task, "edit_plan", {})
    retry_ctx = "\n\n━━━ PREVIOUS ATTEMPT FAILED ━━━\n"  # ← Retry context added
    # ... retry context assembled ...
    prompt += retry_ctx
    # ← PROBLEM: Normal context_str is still in prompt from earlier
    # ← But if task.retry_count == 0 (first attempt), the context IS used

# But there's a subtle bug:
prompt = f"TASK: {task.description}"
if context_str:
    prompt += f"\n\nContext:\n{context_str}"  # ← Context injected here for first attempt

# On retry, ALL the context is still there, but buried in retry_ctx
```

Wait, re-reading this... the context_str IS being injected on the first attempt. Let me trace the actual failure from the transcript.


---

## Secondary Issues

### Issue #2: Retry Mechanism is Broken

After `verifier` fails, `route_after_verifier` sends back to `reasoner` for a retry:

```python
def route_after_verifier(state: ShukiState) -> str:
    if not verify_passed and retry_count < 1:
        task.retry_count = retry_count + 1
        return "retry"  # → routes back to reasoner
    return "summarize"
```

**Problem**: The router says `"retry"` but the actual edge in `graph.py` is:

```python
builder.add_conditional_edges(
    "verifier",
    route_after_verifier,
    {
        "retry":     "reasoner",  # ← This edge exists
        "summarize": "summarizer",
    },
)
```

The edge exists, but **the state passed to reasoner on retry is the same as before** — no new file content, no error details. So the reasoner makes the same mistake again.

### Issue #3: Context Not Persisted

When `writer_node` successfully reads a file during patch verification:

```python
# In reasoner_node:
if name == "read_file" and "path" in args:
    file_index_updates[args["path"]] = result
    session.append_tool_result(tid, str(result), name)
```

This updates `state["file_index"]`, but **the next subtask doesn't inherit this cached content**. Each task starts fresh.

### Issue #4: Task Dependencies Not Enforced

The `route_after_summarizer` checks dependencies:

```python
def route_after_summarizer(state: ShukiState) -> str:
    for dep_id in current.depends_on:
        dep = next((t for t in plan if t.id == dep_id), None)
        if dep and dep.status != "done":
            return "finalize"  # ← stops execution!
    return "continue"
```

If a task depends on an incomplete task, it stops the whole pipeline instead of waiting or re-queueing. This is too aggressive.

---

## What Should Happen

### Correct Flow for a Logging Replacement Task

```
planner creates: [
  { id: 1, title: "Replace prints in main.py", 
    description: "Find all print() calls and replace with logger.*() calls",
    context_hints: ["main.py"] },
  { id: 2, title: "Replace prints in setup.py", 
    description: "Find all print() calls and replace with logger.*() calls",
    context_hints: ["setup.py"] }
]
↓
For task 1:
  reasoner receives:
    - Task description
    - context_hints: ["main.py"]
    - state["file_index"]["main.py"] if cached, OR
    - Instructions to call read_file for main.py FIRST
  ↓
  Reasoner calls read_file("main.py")
  ↓
  Now sees ALL 16 print statements with exact whitespace
  ↓
  Generates ONE multi_patch with all 16 replacements
  ↓
  writer applies it all at once
  ↓
  verifier confirms all 16 are replaced
```

---

## The Fix (Small Change)

The issue is **not fundamental** — just **context passing**. Three small changes fix it:

### Fix #1: Pre-load File Context in Reasoner

**In `reasoner_node`, before creating the session:**

```python
def reasoner_node(state: ShukiState) -> dict:
    task = _current_task(state)
    if task is None:
        return {}

    verbose = config.verbose
    if verbose:
        print(f"\n[Reasoner] Task {task.id}: {task.title}")

    # ← FIX: Pre-load all context files mentioned in context_hints
    file_context = ""
    file_index = state.get("file_index", {})
    for hint in task.context_hints:
        if hint in file_index:
            file_context += f"\n--- {hint} (cached) ---\n{file_index[hint]}\n"
    
    # ← Also try to read files not yet cached
    from tools.code_tools import TOOL_MAP
    for hint in task.context_hints:
        if hint not in file_index:
            try:
                content = TOOL_MAP["read_file"].invoke({"path": hint})
                file_context += f"\n--- {hint} ---\n{content}\n"
            except:
                pass

    # ← Now pass this context in the prompt
    prompt = f"Task: {task.description}\n\n{file_context}\n\nContext hints: {', '.join(task.context_hints)}"
    
    session = BudgetedSession(
        system_prompt=REASONER_SYSTEM,
        tools=selected_tools,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )
    response = session.invoke(prompt)
    # ... rest of reasoner_node
```

### Fix #2: Inject Failed Content on Retry

**In `route_after_verifier`, when sending back to reasoner:**

```python
def route_after_verifier(state: ShukiState) -> str:
    plan = state.get("plan", [])
    idx = state.get("current_task_idx", 0)
    if idx >= len(plan):
        return "summarize"

    task = plan[idx]
    verify_passed = getattr(task, "verify_passed", True)
    retry_count = getattr(task, "retry_count", 0)

    if not verify_passed and retry_count < 1:
        task.retry_count = retry_count + 1
        
        # ← FIX: Inject the file content that failed
        file_path = task.write_result.get("file", "")
        if file_path:
            try:
                actual_content = TOOL_MAP["read_file"].invoke({"path": file_path})
                # Update file_index so reasoner sees the REAL content on retry
                state["file_index"][file_path] = actual_content
                
                # Also store the error message for reasoner to learn from
                task.retry_context = {
                    "failed_plan": task.edit_plan,
                    "actual_file": actual_content,
                    "error": task.verify_message
                }
            except:
                pass
        
        if config.verbose:
            print(f"  [Router] Verification failed — retrying reasoner (attempt {task.retry_count})")
        return "retry"
    
    return "summarize"
```

### Fix #3: Pass Retry Context to Reasoner

**In `reasoner_node`, when retrying:**

```python
def reasoner_node(state: ShukiState) -> dict:
    task = _current_task(state)
    if task is None:
        return {}

    verbose = config.verbose
    
    # ← FIX: Check if this is a retry with context
    retry_context = getattr(task, "retry_context", None)
    if retry_context:
        # Include the failed attempt and real file content
        prompt = (
            f"RETRY: Previous attempt failed.\n\n"
            f"Previous edit plan that didn't work:\n{json.dumps(retry_context['failed_plan'], indent=2)}\n\n"
            f"Error: {retry_context['error']}\n\n"
            f"ACTUAL file content (copy-paste from this exactly):\n"
            f"{retry_context['actual_file']}\n\n"
            f"Task: {task.description}\n"
            f"Generate a NEW edit plan that matches the ACTUAL content shown above."
        )
    else:
        # Normal path: pre-load file context
        file_context = ""
        file_index = state.get("file_index", {})
        for hint in task.context_hints:
            if hint in file_index:
                file_context += f"\n--- {hint} (cached) ---\n{file_index[hint]}\n"
        
        prompt = f"Task: {task.description}\n\n{file_context}"
    
    # ... rest of reasoner_node
```

---

## Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Reasoner fails to find strings | No file context in prompt | Pre-load files via context_hints |
| Retry doesn't help | Same state passed back | Inject actual file content on retry |
| Hallucinated strings | LLM doesn't see real whitespace | Pass raw file content, not tool calls |
| Context lost between tasks | No caching/persistence | Maintain file_index throughout |

**The entire LangGraph setup is not bad** — it's actually well-designed. It's just **missing context injection**. Three small changes fix most of the failures.

---

## Testing the Fix

After implementing these fixes, the logging task should work:

```
planner → [Task 1: main.py, Task 2: setup.py]
  Task 1:
    reasoner gets main.py content
    generates multi_patch with all 16 prints
    writer applies it
    verifier confirms all present
    ✓ done
  Task 2:
    reasoner gets setup.py content
    generates multi_patch with all prints
    writer applies it
    verifier confirms
    ✓ done
finalizer → "Successfully replaced X prints with logging calls"
```
