"""PBS wrapper must not mask ezpz launch exit code (#246)."""

from ezpz.submit import generate_pbs_script, generate_slurm_script


def _non_comment_lines(script: str) -> list[str]:
    return [
        ln.strip()
        for ln in script.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_pbs_script_preserves_exit_code():
    script = generate_pbs_script("python3 -m ezpz.examples.test --model debug")
    assert "RC=$?" in script
    assert 'exit "$RC"' in script or "exit $RC" in script
    lines = _non_comment_lines(script)
    # strict adds set +e / set -e guard: last non-comment lines are set -e then exit
    assert lines[-3] == "RC=$?"
    assert lines[-1] in ('exit "$RC"', "exit $RC")


def test_slurm_script_preserves_exit_code():
    script = generate_slurm_script(
        "python3 -m ezpz.examples.test --model debug"
    )
    assert "RC=$?" in script
    assert 'exit "$RC"' in script or "exit $RC" in script
    lines = _non_comment_lines(script)
    assert lines[-3] == "RC=$?"
    assert lines[-1] in ('exit "$RC"', "exit $RC")


def test_pbs_strict_guards_errexit():
    script = generate_pbs_script("echo hi", strict=True)
    assert "set +e" in script
    # second set -e restores errexit after capture
    assert script.count("set -e") >= 1


def test_no_wrap_still_preserves_exit_code():
    script = generate_pbs_script("echo hi", wrap_with_launch=False)
    assert "RC=$?" in script
    assert 'exit "$RC"' in script or "exit $RC" in script
