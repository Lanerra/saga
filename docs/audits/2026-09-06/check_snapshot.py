"""Read-only drift check for this SAGA audit; never imports application code.

Exit 0 means source hashes and observed Git metadata match the audit baseline.
This is not a test run, a backup, a clean-worktree assertion, or live-service proof.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]
AUDIT_REL = str(AUDIT.relative_to(ROOT)) + "/"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).rstrip("\n")


def comparable_status(text):
    return sorted(
        line for line in text.splitlines()
        if line and not line.startswith("## ") and not line[3:].startswith(AUDIT_REL)
    )


def main():
    rows = json.loads((AUDIT / "source-inventory.json").read_text())
    baseline = json.loads((AUDIT / "inventory-summary.json").read_text())
    changed = []
    missing = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            missing.append(row["path"])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            changed.append(row["path"])
    observed_head = git("rev-parse", "HEAD")
    observed_branch = git("branch", "--show-current")
    observed_status = comparable_status(git("status", "--porcelain=v1", "--untracked-files=all"))
    baseline_status = comparable_status((AUDIT / "git-status-before.txt").read_text())
    status_added = sorted(set(observed_status) - set(baseline_status))
    status_removed = sorted(set(baseline_status) - set(observed_status))
    result = {
        "repository": str(ROOT),
        "checked_files": len(rows),
        "changed_source": changed,
        "missing_source": missing,
        "head": observed_head,
        "branch": observed_branch,
        "head_matches": observed_head == baseline["head"],
        "branch_matches": observed_branch == baseline["branch"],
        "status_added_excluding_audit": status_added,
        "status_removed_excluding_audit": status_removed,
        "scope": "First-party source/test/prompt hashes and Git metadata; ignores audit-only additions, not user-data changes.",
    }
    result["matches_audit_baseline"] = (
        not changed and not missing and not status_added and not status_removed
        and result["head_matches"] and result["branch_matches"]
    )
    print(json.dumps(result, indent=2))
    return 0 if result["matches_audit_baseline"] else 1


if __name__ == "__main__":
    sys.exit(main())
