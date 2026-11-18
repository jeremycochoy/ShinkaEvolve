"""extract_between: a tag with no end tag, and fenced code with "<word>" text."""

import pytest

from shinka.llm.llm import extract_between


def test_tag_with_no_end_tag_ends_at_the_next_tag():
    # Small models and DeepSeek can leave out the end tag
    content = "<NAME>\nfast_lr\n<DESCRIPTION>\nRaise lr.\n</DESCRIPTION>"
    assert extract_between(content, "<NAME>", "</NAME>", False) == "fast_lr"
    assert extract_between(content, "<DESCRIPTION>", "</DESCRIPTION>", False) == (
        "Raise lr."
    )


def test_tag_with_its_end_tag_keeps_its_angle_bracket_text():
    content = "<DESCRIPTION>Use Vec<Order> here.</DESCRIPTION>"
    assert extract_between(content, "<DESCRIPTION>", "</DESCRIPTION>", False) == (
        "Use Vec<Order> here."
    )


@pytest.mark.parametrize(
    ("fence", "code"),
    [
        ("cpp", "#include <cstdio>\nint main() { return 0; }"),
        ("rust", "fn orders() -> Vec<Order> {\n    Vec::new()\n}"),
    ],
)
def test_fenced_code_keeps_its_angle_bracket_text(fence, code):
    # A full rewrite holds the whole file: the "<cstdio>" or "<Order>" text of
    # the code does not end the code
    content = f"Here is the program:\n```{fence}\n{code}\n```\nDone."
    assert extract_between(content, f"```{fence}", "```", False) == code
