"""
Change Analysis Stage for Decisio Pipeline.

Analyzes what actually changed in a pull request deterministically before
consulting the LLM. This provides structural context (e.g., this is a
pure dependency bump, or a test-only PR, or a config update).

By separating CHANGE classification from DECISION detection, we ensure
that simple maintenance updates do not get flagged as decisions unless
they contain explicit reasoning/discussion.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.pipeline.models import PRContext, ChangeAnalysisResult

logger = logging.getLogger(__name__)


# The ChangeAnalysisResult model is imported from models.py


# Re-use patterns from detection or define them cleanly here
DEPENDENCY_PATTERNS = [
    r"^bump\s+\S+\s+from\s+",
    r"^update\s+\S+\s+to\s+",
    r"^upgrade\s+\S+\s+from\s+",
    r"^chore\(deps\)",
    r"^chore:\s*bump",
    r"^dependabot",
    r"^renovate",
]

DOCS_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
DOCS_FOLDERS = {"docs/", ".github/"}

TEST_FILE_PATTERNS = [
    r"test_",
    r"_test",
    r"\.spec\.",
    r"\.test\.",
    r"^tests/",
    r"^test/",
]

CONFIG_FILES = {
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "requirements.txt", "pipfile", "pyproject.toml", "setup.cfg",
    "alembic.ini", "tsconfig.json", "webpack.config.js", "vite.config.ts",
    "vite.config.js", "next.config.js", "next.config.mjs", "go.sum",
    "cargo.lock", "gemfile.lock", "makefile", "build.gradle", "pom.xml"
}


class ChangeAnalyzer:
    """Classifies the type of changes in a pull request deterministically."""

    def analyze(self, ctx: PRContext) -> ChangeAnalysisResult:
        """Analyze the PRContext to determine change types."""
        title_lower = ctx.title.lower()
        body_lower = ctx.body.lower()
        files = ctx.files

        change_types: list[str] = []

        # 1. Check Dependency Bump
        is_dep = False
        if any(re.search(pat, title_lower, re.IGNORECASE) for pat in DEPENDENCY_PATTERNS):
            is_dep = True
        elif any(f.filename.lower() in CONFIG_FILES for f in files) and "bump" in title_lower:
            is_dep = True

        if is_dep:
            change_types.append("dependency_bump")

        # 2. Check Docs Only
        is_docs = False
        if files:
            is_docs = all(
                self._is_doc_file(f.filename) for f in files
            )
        if is_docs:
            change_types.append("documentation")

        # 3. Check Tests Only
        is_tests = False
        if files and not is_docs:
            is_tests = all(
                self._is_test_file(f.filename) or self._is_doc_file(f.filename) for f in files
            )
            # Ensure there is actually at least one test file
            if is_tests and any(self._is_test_file(f.filename) for f in files):
                change_types.append("tests")
            else:
                is_tests = False

        # 4. Check Config Only
        is_config = False
        if files and not is_docs and not is_tests:
            is_config = all(
                self._is_config_file(f.filename) or self._is_doc_file(f.filename) for f in files
            )
            if is_config:
                change_types.append("configuration")

        # 5. Check Bug Fix
        is_bug = False
        if title_lower.startswith(("fix", "bug", "patch", "hotfix")) or "bug" in title_lower:
            is_bug = True
            change_types.append("bug_fix")

        # 6. Check Refactor
        is_refactor = False
        if "refactor" in title_lower or "cleanup" in title_lower or "reorganize" in title_lower:
            is_refactor = True
            change_types.append("refactor")

        # 7. Check Security
        is_security = False
        if any(word in title_lower or word in body_lower for word in ["security", "vuln", "cve", "exploit", "cwe", "audit"]):
            is_security = True
            change_types.append("security")

        # 8. Check Performance
        is_perf = False
        if any(word in title_lower or word in body_lower for word in ["perf", "optimize", "speed", "latency", "benchmark"]):
            is_perf = True
            change_types.append("performance")

        # Determine primary change type
        if is_dep:
            primary = "dependency_bump"
        elif is_docs:
            primary = "documentation"
        elif is_tests:
            primary = "tests"
        elif is_config:
            primary = "configuration"
        elif is_security:
            primary = "security"
        elif is_perf:
            primary = "performance"
        elif is_refactor:
            primary = "refactor"
        elif is_bug:
            primary = "bug_fix"
        else:
            primary = "feature"

        # Generate summary
        file_summary = f"{len(files)} files changed (+{ctx.total_additions}/-{ctx.total_deletions})"
        summary = f"Primary change classified as {primary.upper()} ({file_summary})."

        return ChangeAnalysisResult(
            change_types=change_types,
            is_dependency_bump=is_dep,
            is_docs_only=is_docs,
            is_tests_only=is_tests,
            is_config_only=is_config,
            is_bug_fix=is_bug,
            is_refactor=is_refactor,
            is_security_related=is_security,
            is_performance_related=is_perf,
            primary_change_type=primary,
            summary=summary,
        )

    def _is_doc_file(self, filename: str) -> bool:
        """Check if file is documentation."""
        lower = filename.lower()
        if any(lower.endswith(ext) for ext in DOCS_EXTENSIONS):
            return True
        if any(lower.startswith(folder) for folder in DOCS_FOLDERS):
            return True
        if lower.split("/")[-1] in {"license", "changelog", "readme", "gitignore"}:
            return True
        return False

    def _is_test_file(self, filename: str) -> bool:
        """Check if file is a test file."""
        lower = filename.lower()
        return any(re.search(pat, lower) for pat in TEST_FILE_PATTERNS)

    def _is_config_file(self, filename: str) -> bool:
        """Check if file is a config or build file."""
        basename = filename.lower().split("/")[-1]
        if basename in CONFIG_FILES:
            return True
        if filename.startswith((".github/workflows/", ".circleci/", ".vscode/")):
            return True
        return False
