#!/usr/bin/env python3
"""Reflow converted MyST Markdown to resemble the original reST wrapping.

This is a diff-reduction helper for documentation format migrations. It reads a
converted Markdown file, finds the corresponding original reStructuredText file
from git history, matches prose blocks by normalized rendered-ish text, and
rewraps matched Markdown blocks using the original block's line-length profile.

The script is intentionally conservative. It skips code fences, literal-looking
lines, tables, directives, labels, headings, and raw HTML. By default it only
prints a report; pass --in-place to write the reflowed Markdown.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from html import unescape
from pathlib import Path
import re
import subprocess
import sys
from difflib import SequenceMatcher


DEFAULT_WIDTH = 79

MARKDOWN_SKIP_PREFIXES = (
    "#",
    "```",
    "````",
    "~~~",
    "|",
    "+",
    "<",
    "</",
    "<!",
    ":local:",
    ":depth:",
    ":name:",
    ":class:",
    ":align:",
    ":widths:",
    ":header-rows:",
    ":caption:",
    ":glob:",
    ":maxdepth:",
    ":hidden:",
    ":orphan:",
    ":start-after:",
    ":end-before:",
    "---",
    "===",
    "***",
    ".. ",
    ":::",
)

CODEISH_PREFIXES = ("@", "%", "$", "!", ";", "[", "]", "{", "}", ",", ".", "/", "\\", "<", "=")
CODEISH_WORDS = (
    "declare ",
    "define ",
    "ret ",
    "br ",
    "switch ",
    "invoke ",
    "call ",
    "store ",
    "load ",
    "fence ",
    "cmpxchg ",
    "atomicrmw ",
    "landingpad ",
    "resume ",
    "catchpad ",
    "cleanuppad ",
    "catchret ",
    "cleanupret ",
    "catchswitch ",
    "getelementptr ",
    "extractvalue ",
    "insertvalue ",
    "phi ",
    "select ",
)

LABEL_RE = re.compile(r"^\([A-Za-z0-9_. -]+\)=$")
LINK_DEF_RE = re.compile(r"^\[[^\]]+\]:")
MD_BULLET_RE = re.compile(r"^(\s*[-*+]\s{2,})(\S.*)$")
MD_NUMBER_RE = re.compile(r"^(\s*\d+\.\s{2,})(\S.*)$")
MD_DEFINITION_RE = re.compile(r"^(\s*:   )(\S.*)$")
MD_BLOCKQUOTE_BULLET_RE = re.compile(r"^(>\s*[-*+]\s{2,})(\S.*)$")
MD_BLOCKQUOTE_NUMBER_RE = re.compile(r"^(>\s*\d+\.\s{2,})(\S.*)$")
MD_BLOCKQUOTE_RE = re.compile(r"^(>\s?)(\S.*)$")
MD_INDENT_RE = re.compile(r"^(\s{4,})(\S.*)$")

RST_BULLET_RE = re.compile(r"^(\s*[-*+]\s+)(\S.*)$")
RST_NUMBER_RE = re.compile(r"^(\s*(?:\d+\.|#\.)\s+)(\S.*)$")
RST_FIELD_RE = re.compile(r"^(\s*:[^:]+:\s*)(\S.*)$")
RST_INDENT_RE = re.compile(r"^(\s{3,})(\S.*)$")


@dataclass
class LineItem:
    content: str
    initial_prefix: str
    subsequent_prefix: str
    starts_block: bool
    source_width: int


@dataclass
class Block:
    start: int
    end: int
    text: str
    key: str
    widths: list[int]
    initial_prefix: str = ""
    subsequent_prefix: str = ""


@dataclass
class FileStats:
    source_blocks: int = 0
    markdown_blocks: int = 0
    matched: int = 0
    fuzzy_matched: int = 0
    unmatched: int = 0
    changed: int = 0
    unchanged: int = 0


def run_git(args: list[str], repo: Path, *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def git_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], Path.cwd()).strip())


def git_path_exists(repo: Path, ref: str, path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def git_show(repo: Path, ref: str, path: str) -> str:
    return run_git(["show", f"{ref}:{path}"], repo)


def repo_relative(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def inferred_rst_path(md_path: str) -> str:
    path = Path(md_path)
    if path.suffix != ".md":
        raise ValueError(f"expected a .md file, got {md_path}")
    return path.with_suffix(".rst").as_posix()


def find_rst_source(
    repo: Path,
    md_path: str,
    rst_ref: str | None,
    rst_file: Path | None,
    scan_ancestors: int,
) -> tuple[str, str]:
    if rst_file is not None:
        return f"file:{rst_file}", rst_file.read_text(errors="replace")

    rst_path = inferred_rst_path(md_path)
    if rst_ref:
        return f"{rst_ref}:{rst_path}", git_show(repo, rst_ref, rst_path)

    local = repo / rst_path
    if local.is_file():
        return f"working-tree:{rst_path}", local.read_text(errors="replace")

    for depth in range(scan_ancestors + 1):
        ref = "HEAD" if depth == 0 else f"HEAD~{depth}"
        if git_path_exists(repo, ref, rst_path):
            return f"{ref}:{rst_path}", git_show(repo, ref, rst_path)

    raise SystemExit(
        f"error: could not find {rst_path} in the working tree or HEAD~0.."
        f"HEAD~{scan_ancestors}; pass --rst-ref or --rst-file"
    )


def is_heading_adornment(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 3 and len(set(stripped)) == 1 and stripped[0] in '=-~^"#+*'


def is_fence_start(line: str) -> tuple[str, int] | None:
    match = re.match(r"^\s*(`{3,}|~{3,})", line)
    if not match:
        return None
    marker = match.group(1)
    return marker[0], len(marker)


def is_fence_end(line: str, fence: tuple[str, int]) -> bool:
    ch, count = fence
    return bool(re.match(r"^\s*" + re.escape(ch) + "{" + str(count) + r",}\s*$", line))


def role_text(value: str) -> str:
    value = value.strip()
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    match = re.match(r"(.+?)\s*<[^>]+>$", value)
    if match:
        return match.group(1).strip()
    return value


def normalize_text(text: str) -> str:
    text = unescape(text)
    text = text.replace(r"\"", '"')
    text = re.sub(r"\\([\\`*{}\[\]()#+\-.!_<>])", r"\1", text)

    text = re.sub(r"\{[A-Za-z0-9_.-]+\}`([^`]+)`", lambda m: role_text(m.group(1)), text)
    text = re.sub(r":[A-Za-z0-9_.-]+:`([^`]+)`", lambda m: role_text(m.group(1)), text)
    text = re.sub(r"`([^`<>]+?)\s*<[^`>]+>`_", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    text = text.replace("**", "").replace("*", "")
    text = text.replace("\\-", "-").replace("\\_", "_")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def looks_like_prose(text: str, *, allow_indented: bool = False, markdown: bool = True) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if LABEL_RE.match(stripped) or LINK_DEF_RE.match(stripped):
        return False
    if markdown and stripped.startswith(MARKDOWN_SKIP_PREFIXES):
        return False
    if stripped.endswith("::"):
        return False
    if not allow_indented and stripped.startswith(CODEISH_PREFIXES):
        if not stripped.startswith("["):
            return False
    if allow_indented and (stripped.startswith(CODEISH_PREFIXES) or stripped.startswith(CODEISH_WORDS)):
        return False
    if stripped.startswith(CODEISH_WORDS) and sum(ch.isalpha() for ch in stripped) < 40:
        return False
    if re.search(r"^[A-Za-z_<>%@!.$-]+\s*::?=", stripped):
        return False
    if stripped.count(" ") < 3 or sum(ch.isalpha() for ch in stripped) < 18:
        return False
    return True


def markdown_line_item(line: str, *, raw_html: bool) -> LineItem | None:
    if raw_html or "\t" in line or line.rstrip() != line:
        return None
    stripped = line.strip()
    if not looks_like_prose(line, markdown=True):
        return None

    for regex in (MD_BLOCKQUOTE_BULLET_RE, MD_BLOCKQUOTE_NUMBER_RE):
        match = regex.match(line)
        if match and looks_like_prose(match.group(2), markdown=True):
            prefix, content = match.groups()
            return LineItem(content, prefix, "> " + " " * (len(prefix) - 2), True, len(content))

    match = MD_BLOCKQUOTE_RE.match(line)
    if match and looks_like_prose(match.group(2), markdown=True):
        prefix, content = match.groups()
        return LineItem(content, prefix, prefix, False, len(content))

    match = MD_DEFINITION_RE.match(line)
    if match and looks_like_prose(match.group(2), markdown=True):
        prefix, content = match.groups()
        base_indent = re.match(r"^\s*", prefix).group(0)
        return LineItem(content, prefix, base_indent + "    ", True, len(content))

    for regex in (MD_BULLET_RE, MD_NUMBER_RE):
        match = regex.match(line)
        if match and looks_like_prose(match.group(2), markdown=True):
            prefix, content = match.groups()
            return LineItem(content, prefix, " " * len(prefix), True, len(content))

    match = MD_INDENT_RE.match(line)
    if match and looks_like_prose(match.group(2), allow_indented=True, markdown=True):
        prefix, content = match.groups()
        return LineItem(content, prefix, prefix, False, len(content))

    if line.startswith(" "):
        return None
    return LineItem(line, "", "", False, len(line))


def unescaped_backtick_count(text: str) -> int:
    count = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "`":
            count += 1
    return count


def markdown_continuation_item(line: str, items: list[LineItem], *, raw_html: bool) -> LineItem | None:
    if not items or raw_html or "\t" in line or line.rstrip() != line:
        return None
    prefix = items[-1].subsequent_prefix
    if not prefix or not line.startswith(prefix):
        return None
    content = line[len(prefix) :]
    stripped = content.strip()
    if not stripped:
        return None
    if stripped.startswith(MARKDOWN_SKIP_PREFIXES) and not stripped.startswith("<"):
        return None

    prior_text = " ".join(item.content for item in items)
    if unescaped_backtick_count(prior_text) % 2 == 1:
        return LineItem(content, prefix, prefix, False, len(content))
    if re.match(r"^<[^>]+>`?\s+\S", stripped):
        return LineItem(content, prefix, prefix, False, len(content))
    return None


def rst_line_item(lines: list[str], index: int) -> LineItem | None:
    line = lines[index]
    if "\t" in line or line.rstrip() != line:
        return None
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(".. ") or stripped.startswith("::"):
        return None
    if is_heading_adornment(line):
        return None
    if index + 1 < len(lines) and is_heading_adornment(lines[index + 1]):
        return None
    if index > 0 and is_heading_adornment(lines[index - 1]):
        return None
    if not looks_like_prose(line, markdown=False):
        return None

    for regex in (RST_BULLET_RE, RST_NUMBER_RE, RST_FIELD_RE):
        match = regex.match(line)
        if match and looks_like_prose(match.group(2), markdown=False):
            prefix, content = match.groups()
            return LineItem(content, prefix, " " * len(prefix), True, len(content))

    match = RST_INDENT_RE.match(line)
    if match and looks_like_prose(match.group(2), allow_indented=True, markdown=False):
        prefix, content = match.groups()
        return LineItem(content, prefix, prefix, False, len(content))

    if line.startswith(" "):
        return None
    return LineItem(line, "", "", False, len(line))


def rst_continuation_item(line: str, items: list[LineItem]) -> LineItem | None:
    if not items or "\t" in line or line.rstrip() != line:
        return None
    prefix = items[-1].subsequent_prefix
    if not prefix or not line.startswith(prefix):
        return None
    content = line[len(prefix) :]
    stripped = content.strip()
    if not stripped:
        return None
    if stripped.startswith(".. ") or stripped.startswith("::") or is_heading_adornment(line):
        return None
    if stripped.startswith(CODEISH_WORDS):
        return None

    prior_text = " ".join(item.content for item in items)
    if unescaped_backtick_count(prior_text) % 2 == 1:
        return LineItem(content, prefix, prefix, False, len(content))
    if looks_like_prose(content, allow_indented=True, markdown=False):
        return LineItem(content, prefix, prefix, False, len(content))
    return None


def append_block(blocks: list[Block], start: int, end: int, items: list[LineItem]) -> None:
    text = " ".join(item.content.strip() for item in items)
    key = normalize_text(text)
    if not key:
        return
    blocks.append(
        Block(
            start=start,
            end=end,
            text=text,
            key=key,
            widths=[max(20, item.source_width) for item in items],
            initial_prefix=items[0].initial_prefix,
            subsequent_prefix=items[0].subsequent_prefix,
        )
    )


def markdown_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    items: list[LineItem] = []
    start = 0
    fence: tuple[str, int] | None = None
    raw_html = False

    def flush(end: int) -> None:
        nonlocal items, start
        if items:
            append_block(blocks, start, end, items)
            items = []

    for index, line in enumerate(lines):
        if fence:
            if is_fence_end(line, fence):
                fence = None
            flush(index)
            continue
        fence_start = is_fence_start(line)
        if fence_start:
            flush(index)
            fence = fence_start
            continue

        lower = line.strip().lower()
        if "<table" in lower:
            raw_html = True

        item = markdown_line_item(line, raw_html=raw_html)
        if item is None:
            item = markdown_continuation_item(line, items, raw_html=raw_html)
        if item is None:
            flush(index)
        else:
            if item.starts_block and items:
                flush(index)
            if not items:
                start = index
            items.append(item)

        if "</table>" in lower:
            raw_html = False

    flush(len(lines))
    return blocks


def rst_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    items: list[LineItem] = []
    start = 0

    def flush(end: int) -> None:
        nonlocal items, start
        if items:
            append_block(blocks, start, end, items)
            items = []

    for index in range(len(lines)):
        item = rst_line_item(lines, index)
        if item is None:
            item = rst_continuation_item(lines[index], items)
        if item is None:
            flush(index)
            continue
        if item.starts_block and items:
            flush(index)
        if not items:
            start = index
        items.append(item)

    flush(len(lines))
    return blocks


def reflow_to_widths(text: str, widths: list[int]) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    line_index = 0

    def current_width() -> int:
        if not widths:
            return DEFAULT_WIDTH
        return max(20, widths[min(line_index, len(widths) - 1)])

    for word in words:
        if not current:
            current.append(word)
            continue
        candidate_len = sum(len(part) for part in current) + len(current) + len(word)
        if candidate_len <= current_width():
            current.append(word)
        else:
            lines.append(" ".join(current))
            line_index += 1
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def render_markdown_block(block: Block, source: Block, fallback_width: int) -> list[str]:
    widths = source.widths or [fallback_width]
    if len(widths) == 1:
        text_width = max(20, fallback_width - len(block.initial_prefix))
        widths = [max(widths[0], text_width)]
    wrapped = reflow_to_widths(block.text, widths)
    if not wrapped:
        return []

    result = [block.initial_prefix + wrapped[0]]
    result.extend(block.subsequent_prefix + line for line in wrapped[1:])
    return result


def anchor(key: str, words: int) -> str:
    return " ".join(key.split()[:words])


def build_indexes(blocks: list[Block], min_key_chars: int, anchor_words: int) -> tuple[dict[str, deque[Block]], dict[str, deque[Block]]]:
    exact: dict[str, deque[Block]] = defaultdict(deque)
    fuzzy: dict[str, deque[Block]] = defaultdict(deque)
    for block in blocks:
        if len(block.key) < min_key_chars:
            continue
        exact[block.key].append(block)
        fuzzy[anchor(block.key, anchor_words)].append(block)
    return exact, fuzzy


def find_source_block(
    block: Block,
    exact: dict[str, deque[Block]],
    fuzzy: dict[str, deque[Block]],
    *,
    use_fuzzy: bool,
    min_score: float,
    anchor_words: int,
) -> tuple[Block | None, bool]:
    queue = exact.get(block.key)
    if queue:
        return queue.popleft(), False

    if not use_fuzzy:
        return None, False

    candidates = fuzzy.get(anchor(block.key, anchor_words))
    if not candidates:
        return None, False
    best_index = -1
    best_score = 0.0
    for index, candidate in enumerate(candidates):
        score = SequenceMatcher(None, block.key, candidate.key, autojunk=False).ratio()
        if score > best_score:
            best_score = score
            best_index = index
    if best_index < 0 or best_score < min_score:
        return None, False
    candidate = candidates[best_index]
    del candidates[best_index]
    return candidate, True


def reflow_file(
    md_text: str,
    rst_text: str,
    *,
    fallback_width: int,
    min_key_chars: int,
    fuzzy: bool,
    min_score: float,
    anchor_words: int,
    only_longer_than: int | None,
) -> tuple[str, FileStats, list[Block]]:
    md_lines = md_text.splitlines()
    source = rst_blocks(rst_text.splitlines())
    target = markdown_blocks(md_lines)
    exact, fuzzy_index = build_indexes(source, min_key_chars, anchor_words)
    stats = FileStats(source_blocks=len(source), markdown_blocks=len(target))
    replacements: list[tuple[int, int, list[str]]] = []
    unmatched: list[Block] = []

    for block in target:
        if len(block.key) < min_key_chars:
            continue
        if only_longer_than is not None:
            if all(len(line) <= only_longer_than for line in md_lines[block.start : block.end]):
                continue
        source_block, fuzzy_match = find_source_block(
            block,
            exact,
            fuzzy_index,
            use_fuzzy=fuzzy,
            min_score=min_score,
            anchor_words=anchor_words,
        )
        if source_block is None:
            stats.unmatched += 1
            unmatched.append(block)
            continue
        stats.matched += 1
        stats.fuzzy_matched += int(fuzzy_match)
        new_lines = render_markdown_block(block, source_block, fallback_width)
        if new_lines != md_lines[block.start : block.end]:
            stats.changed += 1
            replacements.append((block.start, block.end, new_lines))
        else:
            stats.unchanged += 1

    for start, end, new_lines in reversed(replacements):
        md_lines[start:end] = new_lines
    return "\n".join(md_lines) + ("\n" if md_text.endswith("\n") else ""), stats, unmatched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reflow converted MyST Markdown to resemble original reST wrapping."
    )
    parser.add_argument("markdown_files", nargs="+", help="Converted .md files to reflow.")
    parser.add_argument(
        "--rst-ref",
        help="Git ref containing original .rst files. If omitted, scan ancestors.",
    )
    parser.add_argument(
        "--rst-file",
        type=Path,
        help="Use this original .rst file. Only valid with one Markdown input.",
    )
    parser.add_argument(
        "--scan-ancestors",
        type=int,
        default=20,
        help="Ancestor depth to scan for inferred .rst files when --rst-ref is omitted.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write reflowed Markdown back to the input files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if any file would change.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Fallback wrap width when the original block has no useful profile.",
    )
    parser.add_argument(
        "--min-key-chars",
        type=int,
        default=40,
        help="Minimum normalized block length eligible for matching.",
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Allow high-confidence fuzzy matches with the same leading words.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.97,
        help="Minimum SequenceMatcher ratio for --fuzzy matches.",
    )
    parser.add_argument(
        "--anchor-words",
        type=int,
        default=8,
        help="Leading normalized words used to bucket fuzzy matches.",
    )
    parser.add_argument(
        "--only-longer-than",
        type=int,
        help="Only reflow Markdown blocks containing a line longer than this value.",
    )
    parser.add_argument(
        "--report-unmatched",
        action="store_true",
        help="Print line numbers and previews for unmatched Markdown prose blocks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rst_file and len(args.markdown_files) != 1:
        raise SystemExit("error: --rst-file is only valid with one Markdown input")

    repo = git_root()
    any_changed = False
    totals = Counter()

    for raw_path in args.markdown_files:
        md_path = Path(raw_path)
        md_repo_path = repo_relative(md_path, repo)
        source_desc, rst_text = find_rst_source(
            repo,
            md_repo_path,
            args.rst_ref,
            args.rst_file,
            args.scan_ancestors,
        )
        md_text = md_path.read_text(errors="replace")
        new_text, stats, unmatched = reflow_file(
            md_text,
            rst_text,
            fallback_width=args.width,
            min_key_chars=args.min_key_chars,
            fuzzy=args.fuzzy,
            min_score=args.min_score,
            anchor_words=args.anchor_words,
            only_longer_than=args.only_longer_than,
        )
        changed = new_text != md_text
        any_changed |= changed
        totals.update(stats.__dict__)

        action = "updated" if changed and args.in_place else "would update" if changed else "unchanged"
        print(f"{md_repo_path}: {action}")
        print(f"  source: {source_desc}")
        print(
            "  blocks: "
            f"source={stats.source_blocks}, markdown={stats.markdown_blocks}, "
            f"matched={stats.matched}, fuzzy={stats.fuzzy_matched}, "
            f"changed={stats.changed}, unmatched={stats.unmatched}"
        )
        if args.report_unmatched and unmatched:
            for block in unmatched[:50]:
                preview = block.text[:100].replace("\n", " ")
                print(f"    unmatched line {block.start + 1}: {preview}")
            if len(unmatched) > 50:
                print(f"    ... {len(unmatched) - 50} more unmatched blocks")

        if changed and args.in_place:
            md_path.write_text(new_text)

    if len(args.markdown_files) > 1:
        print(
            "total blocks: "
            f"source={totals['source_blocks']}, markdown={totals['markdown_blocks']}, "
            f"matched={totals['matched']}, fuzzy={totals['fuzzy_matched']}, "
            f"changed={totals['changed']}, unmatched={totals['unmatched']}"
        )

    if args.check and any_changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
