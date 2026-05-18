#!/usr/bin/env python3
"""Summarize LLVM documentation format usage and edit history.

This script is intentionally lightweight: it only depends on Python and git.
It prints Markdown tables suitable for pasting into Discourse or an RFC.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
import subprocess
import sys


MARKUP_EXTS = (".rst", ".md")

# These directories heavily skew raw .rst counts. AMDGPU contains many small
# target reference fragments, and clang-tidy check pages follow a generated-ish
# one-page-per-check structure. Keep them in the raw totals, but make it easy
# to see the hand-authored narrative trend without them.
STRUCTURED_DOC_PREFIXES = (
    "llvm/docs/AMDGPU/",
    "clang-tools-extra/docs/clang-tidy/checks/",
)


@dataclass
class Commit:
    hash: str
    date: str
    author: str
    email: str
    subject: str
    files: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class NumstatCommit:
    hash: str
    date: str
    author: str
    email: str
    subject: str
    stats: list[tuple[int, int, str]] = field(default_factory=list)


def run_git(args: list[str], repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, errors="replace"
        )
    except subprocess.CalledProcessError as err:
        print(f"error: git {' '.join(args)} failed", file=sys.stderr)
        raise SystemExit(err.returncode)


def git_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], Path.cwd()).strip())


def head_date(repo: Path) -> date:
    value = run_git(["log", "-1", "--format=%cs"], repo).strip()
    return datetime.strptime(value, "%Y-%m-%d").date()


def markdown_escape(value: object) -> str:
    text = str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(markdown_escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * part / total:.1f}%"


def ext(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return suffix if suffix else "[none]"


def is_docs_path(path: str) -> bool:
    return "docs" in Path(path).parts


def docs_root(path: str) -> str:
    parts = Path(path).parts
    try:
        index = parts.index("docs")
    except ValueError:
        return ""
    return "/".join(parts[: index + 1])


def line_count(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"\n") + (0 if not data or data.endswith(b"\n") else 1)


def is_release_note(path: str) -> bool:
    return (
        "/ReleaseNotes/" in path
        or path.endswith("/ReleaseNotes.rst")
        or path.endswith("/ReleaseNotes.md")
        or path.endswith("ReleaseNotes.rst")
        or path.endswith("ReleaseNotes.md")
    )


def is_flang_meeting_note(path: str) -> bool:
    return path.startswith("flang/docs/MeetingNotes/")


def passes_filter(path: str, level: int) -> bool:
    if level >= 1 and path.startswith(STRUCTURED_DOC_PREFIXES):
        return False
    if level >= 2 and is_release_note(path):
        return False
    if level >= 3 and is_flang_meeting_note(path):
        return False
    return True


def filter_label(level: int) -> str:
    labels = {
        0: "All tracked docs markup",
        1: "Excluding AMDGPU reference and clang-tidy check pages",
        2: "Also excluding release notes",
        3: "Also excluding Flang meeting notes",
    }
    return labels[level]


def current_files(repo: Path, include_third_party: bool) -> list[str]:
    # Use the working tree for the current footprint so the script remains useful
    # while a docs-format migration is in progress. `git ls-files` includes
    # deleted tracked files until the deletion is staged, and omits untracked
    # replacement files.
    files = sorted(
        set(run_git(["ls-files"], repo).splitlines())
        | set(run_git(["ls-files", "--others", "--exclude-standard"], repo).splitlines())
    )
    result = []
    for path in files:
        if not include_third_party and path.startswith("third-party/"):
            continue
        if not (repo / path).is_file():
            continue
        if is_docs_path(path):
            result.append(path)
    return result


def parse_name_status_log(repo: Path, since: str, until: str | None) -> list[Commit]:
    args = [
        "log",
        f"--since={since}",
        "--date=short",
        "--format=@@@%H%x1f%cd%x1f%an%x1f%ae%x1f%s",
        "--name-status",
        "--no-renames",
    ]
    if until:
        args.insert(2, f"--until={until}")
    text = run_git(args, repo)
    commits: list[Commit] = []
    current: Commit | None = None
    for line in text.splitlines():
        if line.startswith("@@@"):
            if current:
                commits.append(current)
            commit_hash, commit_date, author, email, subject = line[3:].split(
                "\x1f", 4
            )
            current = Commit(commit_hash, commit_date, author, email, subject)
            continue
        if not current or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            current.files.append((parts[0], parts[-1]))
    if current:
        commits.append(current)
    return commits


def parse_numstat_log(repo: Path, since: str, until: str | None) -> list[NumstatCommit]:
    args = [
        "log",
        f"--since={since}",
        "--date=short",
        "--format=@@@%H%x1f%cd%x1f%an%x1f%ae%x1f%s",
        "--numstat",
        "--no-renames",
    ]
    if until:
        args.insert(2, f"--until={until}")
    text = run_git(args, repo)
    commits: list[NumstatCommit] = []
    current: NumstatCommit | None = None
    for line in text.splitlines():
        if line.startswith("@@@"):
            if current:
                commits.append(current)
            commit_hash, commit_date, author, email, subject = line[3:].split(
                "\x1f", 4
            )
            current = NumstatCommit(commit_hash, commit_date, author, email, subject)
            continue
        if not current or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        current.stats.append((int(parts[0]), int(parts[1]), parts[2]))
    if current:
        commits.append(current)
    return commits


def print_section(title: str) -> None:
    print(f"\n## {title}\n")


def emit_current_footprint(repo: Path, files: list[str], max_roots: int) -> None:
    print_section("Current Documentation Footprint")
    rows = []
    for level in range(4):
        counts = Counter()
        lines = Counter()
        for path in files:
            if ext(path) not in MARKUP_EXTS or not passes_filter(path, level):
                continue
            counts[ext(path)] += 1
            lines[ext(path)] += line_count(repo / path)
        total_files = counts[".rst"] + counts[".md"]
        total_lines = lines[".rst"] + lines[".md"]
        for extension in MARKUP_EXTS:
            rows.append(
                [
                    filter_label(level),
                    extension,
                    fmt_int(counts[extension]),
                    fmt_pct(counts[extension], total_files),
                    fmt_int(lines[extension]),
                    fmt_pct(lines[extension], total_lines),
                ]
            )
    print(markdown_table(["Scope", "Format", "Files", "File share", "Lines", "Line share"], rows))

    root_counts: dict[str, Counter] = defaultdict(Counter)
    root_lines: dict[str, Counter] = defaultdict(Counter)
    for path in files:
        if ext(path) not in MARKUP_EXTS:
            continue
        root = docs_root(path)
        root_counts[root][ext(path)] += 1
        root_lines[root][ext(path)] += line_count(repo / path)

    root_rows = []
    for root, counts in sorted(
        root_counts.items(), key=lambda item: item[1][".md"] + item[1][".rst"], reverse=True
    )[:max_roots]:
        root_rows.append(
            [
                root,
                fmt_int(counts[".md"]),
                fmt_int(root_lines[root][".md"]),
                fmt_int(counts[".rst"]),
                fmt_int(root_lines[root][".rst"]),
            ]
        )
    print("Top docs roots by tracked Markdown/RST files:\n")
    print(markdown_table(["Docs root", ".md files", ".md lines", ".rst files", ".rst lines"], root_rows))


def emit_sphinx_support(repo: Path) -> None:
    print_section("Sphinx Markdown Support")
    confs = [
        path
        for path in run_git(["ls-files"], repo).splitlines()
        if path.endswith("conf.py") and is_docs_path(path)
    ]
    rows = []
    for path in confs:
        text = (repo / path).read_text(errors="ignore")
        rows.append(
            [
                str(Path(path).parent),
                "yes" if "myst_parser" in text else "no",
                "yes" if "source_suffix" in text and ".md" in text else "no",
                "yes"
                if 'source_suffix = ".rst"' in text or "source_suffix = '.rst'" in text
                else "no",
            ]
        )
    print(markdown_table(["Docs root", "Imports MyST", "Explicit .md source suffix", "Literal RST-only suffix"], rows))


def commit_docs_files(
    commit: Commit, include_third_party: bool, level: int | None = None
) -> list[tuple[str, str]]:
    docs_files = []
    for status, path in commit.files:
        if not include_third_party and path.startswith("third-party/"):
            continue
        if not is_docs_path(path):
            continue
        if level is not None and not passes_filter(path, level):
            continue
        docs_files.append((status, path))
    return docs_files


def emit_recent_activity(
    commits: list[Commit],
    numstat_commits: list[NumstatCommit],
    include_third_party: bool,
) -> None:
    print_section("Recent Documentation Activity")
    doc_commits = [c for c in commits if commit_docs_files(c, include_third_party)]
    authors = {c.email.lower() for c in doc_commits}
    md_only = rst_only = both = neither = 0
    commit_by_ext = Counter()
    authors_by_ext: dict[str, set[str]] = defaultdict(set)
    file_touches = Counter()
    status_by_ext: dict[str, Counter] = defaultdict(Counter)

    for commit in doc_commits:
        docs_files = commit_docs_files(commit, include_third_party)
        exts = {ext(path) for _, path in docs_files}
        if ".md" in exts and ".rst" in exts:
            both += 1
        elif ".md" in exts:
            md_only += 1
        elif ".rst" in exts:
            rst_only += 1
        else:
            neither += 1
        for extension in exts:
            commit_by_ext[extension] += 1
            authors_by_ext[extension].add(commit.email.lower())
        for status, path in docs_files:
            extension = ext(path)
            file_touches[extension] += 1
            status_by_ext[extension][status[:1]] += 1

    summary_rows = [
        ["All docs commits", fmt_int(len(doc_commits))],
        ["Unique docs authors", fmt_int(len(authors))],
        ["Markdown-only docs commits", fmt_int(md_only)],
        ["RST-only docs commits", fmt_int(rst_only)],
        ["Commits touching both Markdown and RST docs", fmt_int(both)],
        ["Docs commits touching neither Markdown nor RST", fmt_int(neither)],
    ]
    print(markdown_table(["Metric", "Value"], summary_rows))

    ext_rows = []
    for extension in MARKUP_EXTS:
        ext_rows.append(
            [
                extension,
                fmt_int(commit_by_ext[extension]),
                fmt_int(len(authors_by_ext[extension])),
                fmt_int(file_touches[extension]),
                fmt_int(status_by_ext[extension]["A"]),
                fmt_int(status_by_ext[extension]["M"]),
                fmt_int(status_by_ext[extension]["D"]),
            ]
        )
    print("Markdown/RST activity by extension:\n")
    print(markdown_table(["Format", "Commits", "Authors", "File touches", "Adds", "Mods", "Deletes"], ext_rows))

    line_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    line_authors: dict[str, set[str]] = defaultdict(set)
    for commit in numstat_commits:
        for adds, deletes, path in commit.stats:
            if not include_third_party and path.startswith("third-party/"):
                continue
            if not is_docs_path(path):
                continue
            extension = ext(path)
            if extension not in MARKUP_EXTS:
                continue
            line_stats[extension][0] += adds
            line_stats[extension][1] += deletes
            line_stats[extension][2] += 1
            line_authors[extension].add(commit.email.lower())
    line_rows = []
    for extension in MARKUP_EXTS:
        adds, deletes, entries = line_stats[extension]
        line_rows.append(
            [
                extension,
                fmt_int(adds),
                fmt_int(deletes),
                fmt_int(entries),
                fmt_int(len(line_authors[extension])),
            ]
        )
    print("Line churn by extension:\n")
    print(markdown_table(["Format", "Added lines", "Deleted lines", "File entries", "Authors"], line_rows))


def emit_filtered_additions(
    commits: list[Commit], include_third_party: bool, max_examples: int
) -> None:
    print_section("New Markdown/RST Files")
    for level in range(4):
        counts = Counter()
        authors: dict[str, set[str]] = defaultdict(set)
        roots: dict[str, Counter] = defaultdict(Counter)
        examples: dict[str, list[list[str]]] = defaultdict(list)
        for commit in commits:
            for status, path in commit_docs_files(commit, include_third_party, level):
                extension = ext(path)
                if not status.startswith("A") or extension not in MARKUP_EXTS:
                    continue
                counts[extension] += 1
                authors[extension].add(commit.email.lower())
                roots[extension][docs_root(path)] += 1
                if len(examples[extension]) < max_examples:
                    examples[extension].append(
                        [
                            commit.date,
                            path,
                            commit.author,
                            commit.subject,
                            commit.hash[:12],
                        ]
                    )
        rows = []
        for extension in MARKUP_EXTS:
            top_roots = ", ".join(
                f"{root}: {count}" for root, count in roots[extension].most_common(5)
            )
            rows.append(
                [
                    extension,
                    fmt_int(counts[extension]),
                    fmt_int(len(authors[extension])),
                    top_roots,
                ]
            )
        print(f"{filter_label(level)}:\n")
        print(markdown_table(["Format", "Added files", "Authors", "Top roots"], rows))

    print(f"Recent Markdown additions, limited to {max_examples} rows:\n")
    print(
        markdown_table(
            ["Date", "Path", "Author", "Subject", "Commit"],
            examples_for_additions(commits, include_third_party, ".md", max_examples),
        )
    )
    print(f"Recent RST additions, limited to {max_examples} rows:\n")
    print(
        markdown_table(
            ["Date", "Path", "Author", "Subject", "Commit"],
            examples_for_additions(commits, include_third_party, ".rst", max_examples),
        )
    )


def examples_for_additions(
    commits: list[Commit], include_third_party: bool, extension: str, max_examples: int
) -> list[list[str]]:
    rows = []
    for commit in commits:
        for status, path in commit_docs_files(commit, include_third_party):
            if status.startswith("A") and ext(path) == extension:
                rows.append([commit.date, path, commit.author, commit.subject, commit.hash[:12]])
                if len(rows) >= max_examples:
                    return rows
    return rows


def emit_author_signals(commits: list[Commit], include_third_party: bool) -> None:
    print_section("Author Format Signals")
    author_counts: dict[str, Counter] = defaultdict(Counter)
    author_names: dict[str, str] = {}
    root_author_ext: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for commit in commits:
        docs_files = commit_docs_files(commit, include_third_party)
        if not docs_files:
            continue
        email = commit.email.lower()
        author_names[email] = f"{commit.author} <{commit.email}>"
        exts = {ext(path) for _, path in docs_files}
        author_counts[email]["docs"] += 1
        for extension in MARKUP_EXTS:
            if extension in exts:
                author_counts[email][extension] += 1
        for _, path in docs_files:
            extension = ext(path)
            if extension in MARKUP_EXTS:
                root_author_ext[docs_root(path)][extension].add(email)

    only_md = sum(1 for counts in author_counts.values() if counts[".md"] and not counts[".rst"])
    only_rst = sum(1 for counts in author_counts.values() if counts[".rst"] and not counts[".md"])
    both = sum(1 for counts in author_counts.values() if counts[".md"] and counts[".rst"])
    neither = sum(
        1 for counts in author_counts.values() if not counts[".md"] and not counts[".rst"]
    )
    print(
        markdown_table(
            ["Author bucket", "Authors"],
            [
                ["Touched Markdown docs only", fmt_int(only_md)],
                ["Touched RST docs only", fmt_int(only_rst)],
                ["Touched both Markdown and RST docs", fmt_int(both)],
                ["Touched docs, but neither Markdown nor RST", fmt_int(neither)],
            ],
        )
    )

    root_rows = []
    for root in sorted(root_author_ext):
        root_rows.append(
            [
                root,
                fmt_int(len(root_author_ext[root][".md"])),
                fmt_int(len(root_author_ext[root][".rst"])),
            ]
        )
    print("Unique authors by docs root and format:\n")
    print(markdown_table(["Docs root", "Markdown authors", "RST authors"], root_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown tables for LLVM docs format migration analysis."
    )
    parser.add_argument(
        "--since",
        help="Start date for git history, default is one year before HEAD committer date.",
    )
    parser.add_argument(
        "--until",
        help="End date for git history, default is HEAD committer date.",
    )
    parser.add_argument(
        "--include-third-party",
        action="store_true",
        help="Include third-party docs in the analysis.",
    )
    parser.add_argument(
        "--max-roots",
        type=int,
        default=30,
        help="Maximum docs roots to show in the current-footprint root table.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum recent added-file examples per format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = git_root()
    default_until = head_date(repo)
    since = args.since or (default_until - timedelta(days=365)).isoformat()
    until = args.until or default_until.isoformat()

    print("# LLVM Documentation Format Analysis")
    print()
    print(f"- Repository: `{repo}`")
    print(f"- HEAD: `{run_git(['rev-parse', '--short', 'HEAD'], repo).strip()}`")
    print(f"- History window: `{since}` through `{until}` using commit dates")
    print(f"- Third-party docs: {'included' if args.include_third_party else 'excluded'}")
    print("- Current footprint: existing tracked and untracked working-tree files")

    files = current_files(repo, args.include_third_party)
    emit_current_footprint(repo, files, args.max_roots)
    emit_sphinx_support(repo)

    commits = parse_name_status_log(repo, since, until)
    numstat_commits = parse_numstat_log(repo, since, until)
    emit_recent_activity(commits, numstat_commits, args.include_third_party)
    emit_filtered_additions(commits, args.include_third_party, args.max_examples)
    emit_author_signals(commits, args.include_third_party)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
