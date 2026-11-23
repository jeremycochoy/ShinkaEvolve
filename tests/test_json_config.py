"""The JSON config of a json task: the key of the copy check, and the diffs
of the earlier configs in the proposal prompt."""

import random
from pathlib import Path

import pytest

from shinka.core.sampler import PromptSampler
from shinka.database import Program
from shinka.prompts import construct_eval_history_msg, get_cross_component
from shinka.utils.json_config import (
    CanonicalizerError,
    canonical_config,
    config_diff_lines,
    config_key,
    is_json_language,
    load_canonicalizer,
)

CONFIG = """{
    "data": {"freq": "240s"},
// EVOLVE-BLOCK-START
    "train": {"lr": 0.001, "seed": 16777217, "fees": [0.0004, 0.0002]}
// EVOLVE-BLOCK-END
}
"""
EARLIER = CONFIG.replace('"lr": 0.001', '"lr": 0.003')
DIFF_LINE = "train.lr: 0.003 -> 0.001"
FULL_CONFIG_TEXT = '"freq": "240s"'


def _program(program_id, code, language="json", correct=True):
    return Program(
        id=program_id,
        code=code,
        language=language,
        combined_score=0.4,
        public_metrics={"alpha": 0.5},
        correct=correct,
    )


def test_json_and_json5_are_json_languages():
    assert is_json_language("json")
    assert is_json_language("JSON5")
    assert not is_json_language("python")


def test_float_noise_gives_the_same_key():
    assert config_key('{"lr": 0.0010000000474974513}') == config_key('{"lr": 0.001}')


def test_key_order_white_space_and_comment_lines_give_the_same_key():
    text = '{"a": 1, "b": {"c": 0.5, "d": "x"}}'
    other = (
        '{\n\t"b": {"d": "x",   "c": 0.5},\n'
        "// EVOLVE-BLOCK-START\n"
        '  "a": 1\n'
        "  // EVOLVE-BLOCK-END\n}"
    )
    assert config_key(text) == config_key(other)


def test_an_int_and_a_float_are_different_keys():
    assert config_key('{"a": 1}') != config_key('{"a": 1.0}')


def test_two_ints_above_2_24_are_different_keys():
    assert config_key('{"seed": 16777217}') != config_key('{"seed": 16777216}')


def test_a_bool_is_not_equal_to_a_number():
    assert config_key('{"a": true}') != config_key('{"a": 1}')
    assert config_key('{"a": false}') != config_key('{"a": 0}')


@pytest.mark.parametrize(
    "text",
    [
        CONFIG.replace('{"freq": "240s"},', '{"freq": "240s"}, // 4-minute bars'),
        CONFIG.replace('"lr": 0.001', '"lr": /* was 0.003 */ 0.001'),
        CONFIG.replace("[0.0004, 0.0002]}", "[0.0004, 0.0002,]},"),
    ],
    ids=["inline // comment", "/* */ comment", "trailing comma"],
)
def test_a_json5_form_of_an_unchanged_config_gives_the_key_of_the_config(text):
    clean_key = config_key(CONFIG)
    assert clean_key is not None
    assert config_key(text) == clean_key


def test_a_double_slash_in_a_string_stays_in_the_value():
    text = '{"url": "http://example.com/a//b"} // comment'
    assert canonical_config(text) == {"url": "http://example.com/a//b"}


@pytest.mark.parametrize("text", ['{"a": 1', "not json", ""])
def test_a_text_that_does_not_parse_has_no_key(text):
    assert config_key(text) is None


SPEC = """{
    "data_configuration": {"freq": "240s"},
    "model_configuration": {"T": 32, "leverage": 1},
    "train_configuration": {"lr": 0.001}
}
"""
NO_LEVERAGE = SPEC.replace(', "leverage": 1', "")


@pytest.fixture
def spec_key(canonicalizer_dir):
    return load_canonicalizer("spec_keys:spec_key", canonicalizer_dir)


def test_a_canonicalizer_gives_one_key_to_a_key_at_its_default_and_no_key(
    spec_key,
):
    assert config_key(NO_LEVERAGE, spec_key) == config_key(SPEC, spec_key)


def test_with_a_canonicalizer_a_change_of_lr_is_a_new_key(spec_key):
    other = SPEC.replace('"lr": 0.001', '"lr": 0.003')
    assert config_key(other, spec_key) != config_key(SPEC, spec_key)


def test_with_a_canonicalizer_a_change_of_a_data_value_is_a_new_key(spec_key):
    other = SPEC.replace('"240s"', '"120s"')
    assert config_key(other, spec_key) != config_key(SPEC, spec_key)


def test_without_a_canonicalizer_a_key_at_its_default_is_a_new_key():
    assert config_key(NO_LEVERAGE) != config_key(SPEC)


def test_an_error_of_the_canonicalizer_gives_a_canonicalizer_error(
    canonicalizer_dir,
):
    strict = load_canonicalizer("spec_keys:strict_spec_key", canonicalizer_dir)
    with pytest.raises(CanonicalizerError, match="ValueError: model_configuration.T"):
        config_key(SPEC.replace('"T": 32', '"T": "32"'), strict)


def test_a_canonicalizer_value_with_no_json_text_gives_a_canonicalizer_error():
    with pytest.raises(CanonicalizerError, match="TypeError"):
        config_key(SPEC, lambda config: {"keys": {1, 2}})


def test_a_text_that_does_not_parse_has_no_key_with_a_canonicalizer():
    def no_call(config):
        raise AssertionError("the canonicalizer gets only a parsed config")

    assert config_key('{"a": 1', no_call) is None


def test_a_relative_canonicalizer_path_starts_at_the_working_directory(
    canonicalizer_dir, monkeypatch
):
    monkeypatch.chdir(Path(canonicalizer_dir).parent)
    function = load_canonicalizer("spec_keys:spec_key", "canonicalizer")
    assert config_key(NO_LEVERAGE, function) == config_key(SPEC, function)


@pytest.mark.parametrize(
    "target, error",
    [
        ("spec_keys", "write module:function"),
        ("spec_keys:no_such_function", "AttributeError"),
        ("no_such_module:spec_key", "ModuleNotFoundError"),
        ("spec_keys:DEFAULT_LEVERAGE", "not a function"),
    ],
)
def test_a_target_that_gives_no_function_stops_with_its_name(
    canonicalizer_dir, target, error
):
    with pytest.raises(ValueError) as info:
        load_canonicalizer(target, canonicalizer_dir)
    assert repr(target) in str(info.value)
    assert error in str(info.value)


def test_diff_gives_the_dotted_path_of_a_nested_leaf():
    earlier = '{"model": {"block": {"width": 64, "mode": "cat"}}}'
    current = '{"model": {"block": {"width": 128, "mode": "cat"}}}'
    assert config_diff_lines(earlier, current) == ["model.block.width: 64 -> 128"]


def test_diff_gives_absent_for_a_key_that_one_side_does_not_have():
    earlier = '{"a": 1, "old": "x"}'
    current = '{"a": 1, "new": {"b": 0.5}}'
    assert config_diff_lines(earlier, current) == [
        'new: (absent) -> {"b": 0.5}',
        'old: "x" -> (absent)',
    ]


def test_diff_gives_a_list_as_one_leaf():
    earlier = '{"view_dd": [10, 100, 500]}'
    current = '{"view_dd": [10, 100, 1000]}'
    assert config_diff_lines(earlier, current) == [
        "view_dd: [10, 100, 500] -> [10, 100, 1000]"
    ]


def test_float_noise_is_no_change_and_a_float_prints_as_short_float32_text():
    earlier = '{"lr": 0.0010000000474974513, "wd": 0.05000000074505806}'
    current = '{"lr": 0.001, "wd": 0.1, "fees": [0.00047999998787418]}'
    assert config_diff_lines(earlier, current) == [
        "wd: 0.05 -> 0.1",
        "fees: (absent) -> [0.00048]",
    ]


def test_diff_keeps_the_type_of_a_value():
    earlier = '{"a": 1, "b": true, "c": null}'
    current = '{"a": 1.0, "b": 1, "c": "null"}'
    assert config_diff_lines(earlier, current) == [
        "a: 1 -> 1.0",
        "b: true -> 1",
        'c: null -> "null"',
    ]


def test_the_same_config_gives_an_empty_diff():
    assert config_diff_lines(CONFIG, CONFIG.replace("0.001", "0.0010000000474974513")) == []


def test_the_diff_of_a_text_that_does_not_parse_is_none():
    assert config_diff_lines("{", CONFIG) is None
    assert config_diff_lines(CONFIG, "{") is None


def test_json_eval_history_gives_each_earlier_config_as_a_diff():
    parent = _program("parent", CONFIG)
    message = construct_eval_history_msg(
        [_program("earlier", EARLIER)], language="json", parent=parent
    )
    assert (
        "Changed keys (value in this program -> value in the current program):\n"
        f"{DIFF_LINE}\n\n"
        "Performance metrics:\nCombined score to maximize: 0.40\nalpha: 0.50\n\n"
    ) in message
    assert "```" not in message
    assert FULL_CONFIG_TEXT not in message


def test_an_earlier_config_equal_to_the_config_to_edit_gives_no_changed_keys():
    parent = _program("parent", CONFIG)
    message = construct_eval_history_msg(
        [_program("island-copy", CONFIG)], language="json", parent=parent
    )
    assert "-> value in the current program):\nno changed keys\n\n" in message


def test_an_earlier_config_that_does_not_parse_goes_in_full():
    parent = _program("parent", CONFIG)
    message = construct_eval_history_msg(
        [_program("broken", '{"lr": 0.003,')], language="json", parent=parent
    )
    assert '```json\n{"lr": 0.003,\n```\n\n' in message
    assert "Changed keys" not in message


@pytest.mark.parametrize("patch_type", ["diff", "full", "cross"])
def test_json_proposal_prompt_holds_diffs_and_only_the_config_to_edit_in_full(
    patch_type,
):
    sampler = PromptSampler(
        language="json", patch_types=[patch_type], patch_type_probs=[1.0]
    )
    _, user_msg, sampled_type = sampler.sample(
        parent=_program("parent", CONFIG),
        archive_inspirations=[_program("archive", EARLIER)],
        top_k_inspirations=[_program("top-k", EARLIER.replace("0.0002", "0.0003"))],
    )
    assert sampled_type == patch_type
    assert DIFF_LINE in user_msg
    assert "train.fees: [0.0004, 0.0003] -> [0.0004, 0.0002]" in user_msg
    assert user_msg.count(FULL_CONFIG_TEXT) == 1
    if patch_type == "cross":
        cross_part = user_msg.split("# Crossover Inspiration Programs")[1]
        assert DIFF_LINE in cross_part
        assert FULL_CONFIG_TEXT not in cross_part


def test_json_fix_prompt_gives_the_ancestors_as_diffs():
    sampler = PromptSampler(language="json")
    _, user_msg, patch_type = sampler.sample_fix(
        incorrect_program=_program("incorrect", CONFIG, correct=False),
        ancestor_inspirations=[_program("ancestor", EARLIER, correct=False)],
    )
    assert patch_type == "fix"
    assert (
        "Changed keys (value in this program -> value in the program to fix):\n"
        f"{DIFF_LINE}\n\n"
    ) in user_msg
    assert user_msg.count(FULL_CONFIG_TEXT) == 1


def test_python_prompts_do_not_change():
    parent = _program("parent", "x = 1\n", language="python")
    earlier = _program("earlier", "x = 2\n", language="python")
    history = construct_eval_history_msg([earlier], language="python", parent=parent)
    assert history == construct_eval_history_msg([earlier], language="python")
    assert "```python\nx = 2\n\n```\n\n" in history

    random.seed(0)
    cross = get_cross_component([earlier], [], language="python", parent=parent)
    random.seed(0)
    assert cross == get_cross_component([earlier], [], language="python")
    assert "```python\nx = 2\n\n```\n\n" in cross
