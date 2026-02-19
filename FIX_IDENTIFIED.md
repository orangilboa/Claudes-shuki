# ROOT CAUSE IDENTIFIED ✓

## The Single Issue

Looking at `config.py`:

```python
max_input_tokens: int = 8192  # Budget for everything

executor_input_budget: int = 0
def __post_init__(self):
    self.executor_input_budget = int(self.max_input_tokens * 0.60)  # 60% = 4915 tokens

# Then in LLMConfig:
file_snippet_max_chars: int = 600  # Each file snippet limited to 600 chars!
```

## The Math

When reasoner_node calls ContextAssembler:

```python
assembler = ContextAssembler()
context_str = assembler.build(task, state)  # Respects executor_input_budget

# Budget: 4915 tokens * 3.5 chars/token = 17,203 chars available
# Per file: only 600 chars max!

# So for main.py (6000+ chars):
# - Only ~600 chars injected into prompt
# - Reasoner can't see most of the file
# - Can't find the strings to match
# - Falls back to search_in_files instead of direct read
# - Search gives line numbers but not exact whitespace
# - Patches fail due to whitespace/indentation mismatch
```

## Why the Logging Task Failed

Task: "Replace prints with logging in main.py"

**Expected flow:**
```
reasoner reads main.py (6KB) from file_index
sees all 16 print statements with exact spacing
generates one multi_patch with 16 patches
all patches succeed ✓
```

**What actually happened:**
```
reasoner gets context_str with main.py truncated to 600 chars
can't see most prints
calls search_in_files("print\\(") → returns 16 matches with line numbers
tries to generate patches from regex patterns instead of exact strings
reasoner output has hallucinated string positions and whitespace
patches fail because old_str doesn't match file exactly
verifier detects failure
reasoner gets retry with actual file content (finally!)
but at this point already confused, patches still fail
task marked done with verify_passed=False
```

## The Fix: Two Lines

In `config.py`, line 58:

```python
# Current:
file_snippet_max_chars: int = 600

# Change to:
file_snippet_max_chars: int = 2000  # Or even 4000 for small files
```

That's it. This allows the ContextAssembler to pass more of each file to the reasoner, so it sees the full content instead of a truncated snippet.

**Why this works:**
- executor_input_budget is already 4915 tokens (17KB chars)
- Even with 2000 chars per file, you can inject 8-9 files before hitting budget
- Reasoner sees complete files → can find all strings → patches work first try

## Verification

To confirm this is the issue, look at the transcript when the reasoner ran:

```
[Session] ← The search results show that the main file with print statements is `setup.py`...
[Session] ← Based on the file snippet, there are two print statements...

                    ↑ WRONG! There are 16+ prints in main.py
                      System only saw ~600 chars, missed most of them
```

The reasoner clearly didn't see the full files when making its plan.

## Why This Isn't "Bad Orchestration"

The LangGraph setup is actually **very good**:
- ✓ Breaks tasks into focused subtasks
- ✓ Has proper dependency tracking  
- ✓ Has built-in file caching (file_index)
- ✓ Has intelligent context assembly
- ✓ Has retry mechanism with failure context

The **config values are just too conservative** for your use case. The system was tuned for Ollama models with 2K-4K token limits, but you're using 8192, so you can afford larger snippets.

## Should You Change Anything Else?

No. Just change `file_snippet_max_chars` and retry the logging task. It should work.

If it still fails after this change, then we investigate whether the context_str is actually being passed to the reasoner prompt correctly. But I'm 99% sure that's the issue.
