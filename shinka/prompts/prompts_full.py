# Multiple Full Rewrite Prompt Variants
# 1. Default
# 2. Different Algorithm
# 3. Context Motivated
# 4. Structural Redesign
# 5. Parametric Design

# FULL REWRITE RULES
FULL_SYS_OUTPUT_RULES = """
⚠️ STRICT RULES (failure to follow any of these will cause automatic rejection):
1. You MUST include the three tags exactly as shown: <NAME>, <DESCRIPTION>, and <CODE>.  
2. You MUST preserve the “EVOLVE-BLOCK-START” and “EVOLVE-BLOCK-END” comments exactly.  
3. You MUST NOT modify or remove any code outside of these markers.  
4. You MUST keep the same function/class entry point used in the original code so that the program can still be called from the same method during evaluation.  
5. You MUST maintain identical inputs and outputs — only internal implementation may change.  
6. You MUST produce valid, runnable code.

---

### OUTPUT FORMAT (mandatory):

<NAME>
A short, lowercase name summarizing the rewritten code. Use underscores if needed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the code you are proposing.
</DESCRIPTION>

<CODE>
```{language}
# The new rewritten program here, with EVOLVE-BLOCK markers preserved.
```
</CODE>

---

Failure to exactly match this structure, or changing the callable entry point, will result in your output being rejected, even if the rewritten code is otherwise correct.
"""

# Original/Default Full Rewrite
FULL_SYS_FORMAT_DEFAULT = """
Rewrite the program to improve its performance on the specified metrics.
You must produce the new full program following the exact required format below.
""" + FULL_SYS_OUTPUT_RULES

# Variant 1: Completely Different Algorithm
FULL_SYS_FORMAT_DIFFERENT = """
Design a completely different algorithm approach to solve the same problem.
Ignore the current implementation and think of alternative algorithmic strategies that could achieve better performance.
""" + FULL_SYS_OUTPUT_RULES


# Variant 2: Motivated by Context but Different
FULL_SYS_FORMAT_MOTIVATED = """
Create a novel algorithm that draws inspiration from the provided context programs but implements a fundamentally different approach.
Study the patterns and techniques from the examples, then design something new.
""" + FULL_SYS_OUTPUT_RULES


# Variant 3: Structural Modification
FULL_SYS_FORMAT_STRUCTURAL = """
Redesign the program with a different structural approach while potentially using similar core concepts.
Focus on changing the overall architecture, data flow, or program organization.
""" + FULL_SYS_OUTPUT_RULES


# Variant 4: Parameter-Based Algorithm Design
FULL_SYS_FORMAT_PARAMETRIC = """
Analyze the current program to identify its key parameters and algorithmic components, then design a new algorithm with different parameter settings and configurations.
""" + FULL_SYS_OUTPUT_RULES

# List of all variants for sampling
FULL_SYS_FORMATS = [
    FULL_SYS_FORMAT_DEFAULT,
    FULL_SYS_FORMAT_DIFFERENT,
    FULL_SYS_FORMAT_MOTIVATED,
    FULL_SYS_FORMAT_STRUCTURAL,
    FULL_SYS_FORMAT_PARAMETRIC,
]

# Variant names for debugging/logging
FULL_SYS_FORMAT_NAMES = [
    "default",
    "different_algorithm",
    "context_motivated",
    "structural_redesign",
    "parametric_design",
]

FULL_ITER_MSG = """# Current program

Here is the current program we are trying to improve (you will need to 
propose a new program with the same inputs and outputs as the original 
program, but with improved internal implementation):

```{language}
{code_content}
```

Here are the performance metrics of the program:

{performance_metrics}{text_feedback_section}

# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: You must follow the required output format exactly. Your rewritten program must keep the same inputs, outputs, and callable entry point as the original, while improving the internal implementation. Any deviation from the format or entry point will cause the answer to be rejected, even if the code is otherwise correct.
"""
