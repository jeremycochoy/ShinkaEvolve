from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.utils.languages import (
    get_code_fence_languages,
    get_evolve_comment_prefix,
    get_language_extension,
    normalize_language,
)


def test_apply_diff_patch_supports_fortran(tmp_path):
    original_content = """! EVOLVE-BLOCK-START
integer function score(x)
    integer, intent(in) :: x
    score = x + 1
end function score
! EVOLVE-BLOCK-END"""

    patch_content = (
        "! EVOLVE-BLOCK-START\n"
        "<<<<<<< SEARCH\n"
        "score = x + 1\n"
        "=======\n"
        "score = x + 2\n"
        ">>>>>>> REPLACE\n"
        "! EVOLVE-BLOCK-END"
    )

    patch_dir = tmp_path / "fortran_diff_patch"
    result = apply_diff_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="fortran",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "score = x + 2" in updated_content
    assert output_path == patch_dir / "main.f90"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "original.f90").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "score = x + 2" in patch_txt

    search_replace_txt = (patch_dir / "search_replace.txt").read_text("utf-8")
    assert "EVOLVE-BLOCK-START" not in search_replace_txt
    assert "EVOLVE-BLOCK-END" not in search_replace_txt


def test_apply_full_patch_supports_fortran_and_f90_fence(tmp_path):
    original_content = """! EVOLVE-BLOCK-START
integer function score(x)
    integer, intent(in) :: x
    score = x + 1
end function score
! EVOLVE-BLOCK-END
"""

    patch_content = """```f90
! EVOLVE-BLOCK-START
integer function score(x)
    integer, intent(in) :: x
    score = x + 10
end function score
! EVOLVE-BLOCK-END
```"""

    patch_dir = tmp_path / "fortran_full_patch"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="fortran",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "score = x + 10" in updated_content
    assert output_path == patch_dir / "main.f90"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "rewrite.txt").exists()
    assert (patch_dir / "original.f90").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "score = x + 10" in patch_txt


def test_apply_full_patch_supports_fortran_fence(tmp_path):
    original_content = """! EVOLVE-BLOCK-START
integer function score(x)
    integer, intent(in) :: x
    score = x + 1
end function score
! EVOLVE-BLOCK-END
"""

    patch_content = """```fortran
! EVOLVE-BLOCK-START
integer function score(x)
    integer, intent(in) :: x
    score = x + 20
end function score
! EVOLVE-BLOCK-END
```"""

    patch_dir = tmp_path / "fortran_named_fence"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="f90",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, _ = result

    assert error is None
    assert num_applied == 1
    assert "score = x + 20" in updated_content
    assert output_path == patch_dir / "main.f90"
    assert patch_txt is not None and "score = x + 20" in patch_txt


def test_language_helpers_support_fortran_aliases():
    for alias in ("f90", "f95", "f03", "f08"):
        assert normalize_language(alias) == "fortran"
        assert get_language_extension(alias) == "f90"
        assert get_evolve_comment_prefix(alias) == "!"

    assert get_language_extension("fortran") == "f90"
    assert get_evolve_comment_prefix("fortran") == "!"

    fences = get_code_fence_languages("f95")
    assert fences[0] == "f95"
    assert "fortran" in fences
    assert "f90" in fences
    assert "f03" in fences
    assert "f08" in fences
