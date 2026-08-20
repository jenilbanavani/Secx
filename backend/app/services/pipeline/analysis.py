"""
Impact Analysis for Decisio Pipeline.

Determines what files and technologies are affected by a decision
using DETERMINISTIC logic — not LLM guesses.

Affected files come directly from the GitHub API data.
Technologies are detected via file extensions, imports, and config file patterns.

The LLM is not involved in this stage at all.
"""

from __future__ import annotations

import logging
import re

from app.services.pipeline.models import (
    AffectedFile,
    DetectedTechnology,
    ExtractedDecision,
    PRContext,
)

logger = logging.getLogger(__name__)


# ── Technology detection patterns ────────────────────────────

# Map file extensions to technologies
EXTENSION_TECH_MAP: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React",
    ".tsx": "React",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".swift": "Swift",
    ".sql": "SQL",
    ".graphql": "GraphQL",
    ".proto": "gRPC/Protobuf",
    ".yaml": "YAML Config",
    ".yml": "YAML Config",
    ".toml": "TOML Config",
    ".dockerfile": "Docker",
}

# Map config filenames to technologies
CONFIG_TECH_MAP: dict[str, str] = {
    "dockerfile": "Docker",
    "docker-compose": "Docker Compose",
    "package.json": "Node.js",
    "tsconfig.json": "TypeScript",
    "webpack.config": "Webpack",
    "vite.config": "Vite",
    "next.config": "Next.js",
    "pyproject.toml": "Python",
    "requirements.txt": "Python/pip",
    "cargo.toml": "Rust/Cargo",
    "go.mod": "Go Modules",
    "gemfile": "Ruby/Bundler",
    "pom.xml": "Maven/Java",
    "build.gradle": "Gradle/Java",
    ".eslintrc": "ESLint",
    ".prettierrc": "Prettier",
    "alembic.ini": "Alembic/SQLAlchemy",
    "jest.config": "Jest",
    "pytest.ini": "pytest",
    "nginx.conf": "Nginx",
    "redis.conf": "Redis",
    "kafka": "Kafka",
}

# Keywords in text that map to specific technologies
TEXT_TECH_PATTERNS: dict[str, list[str]] = {
    "PostgreSQL": ["postgresql", "postgres", "psql", "pg_", "asyncpg"],
    "MySQL": ["mysql", "mariadb"],
    "Redis": ["redis", "redis cluster", "redis sentinel"],
    "MongoDB": ["mongodb", "mongoose", "pymongo"],
    "Elasticsearch": ["elasticsearch", "elastic search", "opensearch"],
    "Kafka": ["kafka", "confluent", "ksqldb"],
    "RabbitMQ": ["rabbitmq", "amqp", "pika"],
    "GraphQL": ["graphql", "apollo", "gql"],
    "gRPC": ["grpc", "protobuf", "proto3"],
    "Docker": ["docker", "container", "dockerfile"],
    "Kubernetes": ["kubernetes", "k8s", "helm", "kubectl"],
    "AWS": ["aws", "s3", "ec2", "lambda", "dynamodb", "sqs", "sns"],
    "SQLAlchemy": ["sqlalchemy", "alembic"],
    "FastAPI": ["fastapi"],
    "Django": ["django"],
    "Flask": ["flask"],
    "React": ["react", "jsx", "tsx", "next.js", "nextjs"],
    "Vue": ["vue", "vuex", "nuxt"],
    "Celery": ["celery"],
    "SQLite": ["sqlite", "aiosqlite"],
}


class ImpactAnalyzer:
    """Analyzes the impact of a decision using deterministic methods."""

    def analyze_affected_files(self, ctx: PRContext, relevant_files: list[str] | None = None) -> list[AffectedFile]:
        """Determine affected files from actual GitHub data, filtered by relevance.

        Maps from PRContext.files, preserving the real status, additions, and deletions.
        If relevant_files are provided, only includes files that match.
        Otherwise, falls back to top 5 files with the most lines changed.
        """
        target_files = ctx.files
        
        if relevant_files:
            filtered_files = []
            normalized_rel = [r.lower().strip().lstrip("/") for r in relevant_files]
            
            for f in ctx.files:
                f_lower = f.filename.lower()
                # Check for exact match or suffix match (e.g. "store.py" matches "session/store.py")
                if any(f_lower == r or f_lower.endswith("/" + r) for r in normalized_rel):
                    filtered_files.append(f)
            
            # If we found matches, use them
            if filtered_files:
                target_files = filtered_files
            else:
                # Fallback: if none matched, select top 5 modified files
                target_files = sorted(ctx.files, key=lambda f: f.additions + f.deletions, reverse=True)[:5]
        else:
            # No relevant files specified, take top 5 modified files to avoid overwhelming output
            target_files = sorted(ctx.files, key=lambda f: f.additions + f.deletions, reverse=True)[:5]

        affected = []
        for f in target_files:
            relationship = "changed"
            if f.status == "added":
                relationship = "created"
            elif f.status == "removed":
                relationship = "deleted"
            elif f.status == "renamed":
                relationship = "renamed"

            affected.append(
                AffectedFile(
                    file_path=f.filename,
                    status=f.status,
                    additions=f.additions,
                    deletions=f.deletions,
                    relationship=relationship,
                )
            )
        return affected

    def detect_technologies(
        self,
        ctx: PRContext,
        decisions: list[ExtractedDecision] | None = None,
    ) -> list[DetectedTechnology]:
        """Detect technologies involved in the PR.

        Uses three deterministic methods:
        1. File extensions → technology mapping
        2. Config file names → technology mapping
        3. Text keyword matching in title, body, and comments

        Deduplicates and returns unique technologies.
        """
        seen: set[str] = set()
        technologies: list[DetectedTechnology] = []

        # Method 1: File extensions
        for f in ctx.files:
            ext = self._get_extension(f.filename)
            if ext in EXTENSION_TECH_MAP:
                tech = EXTENSION_TECH_MAP[ext]
                if tech not in seen:
                    seen.add(tech)
                    technologies.append(
                        DetectedTechnology(
                            name=tech,
                            detection_source="file_extension",
                            relationship="used",
                        )
                    )

        # Method 2: Config file names
        for f in ctx.files:
            basename = f.filename.lower().split("/")[-1]
            for config_pattern, tech in CONFIG_TECH_MAP.items():
                if config_pattern in basename and tech not in seen:
                    seen.add(tech)
                    technologies.append(
                        DetectedTechnology(
                            name=tech,
                            detection_source="config_file",
                            relationship="configured",
                        )
                    )

        # Method 3: Text keyword matching
        all_text = " ".join([
            ctx.title.lower(),
            ctx.body.lower(),
            *[c.body.lower() for c in ctx.all_comments],
            *[commit.message.lower() for commit in ctx.commits],
        ])

        for tech, keywords in TEXT_TECH_PATTERNS.items():
            if tech not in seen:
                for kw in keywords:
                    if kw in all_text:
                        # Try to infer the relationship from context
                        relationship = self._infer_tech_relationship(kw, all_text)
                        seen.add(tech)
                        technologies.append(
                            DetectedTechnology(
                                name=tech,
                                detection_source="text_keyword",
                                relationship=relationship,
                            )
                        )
                        break

        return technologies

    def _get_extension(self, filename: str) -> str:
        """Extract file extension, handling special cases like Dockerfile."""
        lower = filename.lower()
        if "dockerfile" in lower:
            return ".dockerfile"
        parts = lower.rsplit(".", 1)
        if len(parts) == 2:
            return f".{parts[1]}"
        return ""

    def _infer_tech_relationship(self, keyword: str, text: str) -> str:
        """Infer the relationship between a technology and the PR.

        Looks for context words near the keyword to determine if the technology
        is being adopted, replaced, deprecated, or just used.
        """
        # Find the keyword position and examine surrounding context
        idx = text.find(keyword)
        if idx == -1:
            return "used"

        # Get a window of text around the keyword
        start = max(0, idx - 100)
        end = min(len(text), idx + len(keyword) + 100)
        window = text[start:end]

        adopt_words = ["adopt", "introduce", "add", "switch to", "move to", "migrate to", "use"]
        replace_words = ["replace", "switch from", "move from", "instead of", "migrate from"]
        deprecate_words = ["deprecat", "remov", "drop", "sunset", "phase out"]

        for word in deprecate_words:
            if word in window:
                return "deprecated"
        for word in replace_words:
            if word in window:
                return "replaced"
        for word in adopt_words:
            if word in window:
                return "adopted"

        return "used"
