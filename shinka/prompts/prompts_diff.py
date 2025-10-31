DIFF_SYS_FORMAT = """
You must reply using the exact format and rules below.
Follow everything literally. Do not add extra text, explanations, Markdown code fences, or language tags.
You MUST repond using a **edit name, description**, and the exact **SEARCH/REPLACE diff format** shown below.

---

#### 🔹 REQUIRED OUTPUT STRUCTURE

<NAME>
Short name for the edited program (lowercase, no spaces, underscores allowed).
</NAME>

<DESCRIPTION>
Explain and justify the change you are proposing.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE
</DIFF>

---

#### 🔹 RULES (MANDATORY)

1. You may **only** edit code located **between** `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END`.
   Everything outside these markers is **read-only**.
2. **Never** include the marker lines (`EVOLVE-BLOCK-START` or `EVOLVE-BLOCK-END`) inside your diff.
3. In the `SEARCH` section, copy the original code **verbatim** — including indentation and spacing.
4. The `REPLACE` section must contain valid, runnable code.

   * Do **not** wrap it in triple backticks (```) or language labels (e.g., `python`).
   * Plain text only.
5. Keep the same **function or class entry point** as the original code so it can still be called correctly from its existing call site.
6. Maintain identical **inputs and outputs** — you may only change internal logic.
7. You may propose **multiple edits**; each SEARCH/REPLACE block appears one after another, with **no extra text** between them.
8. The resulting file must still **run without errors** after applying all changes.

---

#### 🔹 EXAMPLE (CORRECT FORMAT)

Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

<<<<<<< SEARCH
    total = 0
    for v in values:
        total += v
    return total
=======
    # Use NumPy's optimized sum for numerical arrays
    return np.sum(values)
>>>>>>> REPLACE

---

#### ❌ **BAD EXAMPLES (DO NOT DO THIS)**

**1. Using Markdown fences or language tags**

````
<<<<<<< SEARCH
```python
x = foo(bar)
````

=======

# Avoid repeated work by caching

_cache_key = (bar,)
if _cache_key in cache:
x = cache[_cache_key]
else:
x = foo(bar)
cache[_cache_key] = x

> > > > > > > REPLACE
````

🚫 Wrong: uses ```python``` and code fences, use spaces between each greater-than sign in.

---

**2. Breaking the entry point**

````
<<<<<<< SEARCH
def process_data(data):
    return transform(data)
======================

def new_function_name(data):  # ❌ entry point renamed
    return transform(data)
>>>>>>> REPLACE
```

🚫 Wrong: changed the callable name — breaks the call site.  

---

**3. Editing outside EVOLVE blocks or altering markers**

```
<<<<<<< SEARCH
EVOLVE-BLOCK-START
def compute_total(values):
    total = 0
    for v in values:
        total += v
    return total
EVOLVE-BLOCK-END
=======
# ❌ Wrong: EVOLVE markers must not appear inside your diff
EVOLVE-BLOCK-START
def compute_total(values):
    return sum(values)
EVOLVE-BLOCK-END
>>>>>>> REPLACE
```

🚫 Wrong: EVOLVE markets appear inside the diff.

"""


DIFF_ITER_MSG = """# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```{language}
{code_content}
```

Here are the performance metrics of the program:

{performance_metrics}{text_feedback_section}

# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""
