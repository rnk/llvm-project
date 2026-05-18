# [RFC] Make MyST Markdown the LLVM docs format, RIP reST

**TL;DR:** Markdown is far and away the most popular way to format plain text
files. We should migrate from [reStructuredText][rest] (reST or rst) to Markedly
Structured Text ([MyST][myst]), make `myst_parser` a hard dependency for
building LLVM documentation with Sphinx, and set an explicit long-term goal of
removing hand-written reST from the tree.

If you've been on the internet in the last 10 years, you know that Markdown is
the ubiquitous, default way to format plain text. reST has a deep, extensible
feature set, but simplicity has won out. The most useful docs are the ones that
exist, and are easy to update with zero friction. Every IDE under the sun, such
as VS Code, IntelliJ, NeoVim, etc, supports rendering Markdown dialects live in
some way. reST has served us well, but I believe that now is the time to revise
that choice and set a long-term goal to migrate our docs to Markdown.

Since 2018 ([D44910](https://reviews.llvm.org/D44910)), LLVM has used a
Markdown dialect called Markedly Structured Text (MyST) for portions of its
documentation. Individual subprojects have effectively been free to choose
between reST and MyST at their own discretion, and there has been no coherent
policy about which is preferred. This has led to [backporting the CIR docs to
reST][cir-rest]. The CIR docs were originally Markdown, but were converted back
to the legacy reST format. The point of this RFC is to declare affirmatively
which format we prefer, update the [Sphinx quickstart template][llvm-sphinx-docs]
to that effect, and make a full migration the desired end state.

I generated the tables below with [a small analysis script][doc-history-script]
over LLVM documentation paths and recent git history:

```bash
python3 llvm/utils/analyze-doc-edit-history.py \
  --since 2025-05-12 --until 2026-05-12 \
  --max-roots 12 --max-examples 6
```

The "current footprint" numbers are from this branch after the sample
conversions of LangRef, DeveloperPolicy, and CMake; the recent-activity numbers
are from one year of upstream git history ending at commit `976195f9d5be` on
2026-05-12.

## Proposal

* MyST Markdown should become the preferred and eventual sole hand-authored
  format for LLVM Sphinx documentation.
* New hand-written documentation should use `.md` unless there is a concrete
  blocker.
* Existing `.rst` files may continue to be edited until they are converted, but
  we should welcome mechanical conversion PRs.
* `myst_parser` should become a hard dependency for Sphinx documentation builds,
  including the man-page builder.
* The Sphinx quickstart template should default to Markdown.

## What the tree is already doing

Raw file counts still show a large reST legacy footprint, but they are not a
good proxy for contributor preference. The tree contains many generated or
template-like reST pages, especially AMDGPU instruction fragments and
clang-tidy's one-page-per-check docs. Once those are excluded, Markdown is
already a material part of the hand-authored docs surface.

| Scope | Format | Files | File share | Lines | Line share |
| --- | --- | --- | --- | --- | --- |
| All docs markup | .rst | 2,269 | 91.0% | 328,004 | 74.5% |
| All docs markup | .md | 224 | 9.0% | 112,456 | 25.5% |
| Excluding AMDGPU reference and clang-tidy check pages | .rst | 614 | 73.3% | 256,182 | 69.5% |
| Excluding AMDGPU reference and clang-tidy check pages | .md | 224 | 26.7% | 112,456 | 30.5% |
| Also excluding release notes | .rst | 604 | 73.2% | 254,012 | 69.4% |
| Also excluding release notes | .md | 221 | 26.8% | 111,984 | 30.6% |

Several subprojects have already effectively chosen Markdown. MLIR is entirely
Markdown in this analysis, Flang is overwhelmingly Markdown, and LLDB has
substantial Markdown usage. The main LLVM docs still carry most of the legacy
reST weight.

| Docs root | .md files | .md lines | .rst files | .rst lines |
| --- | --- | --- | --- | --- |
| llvm/docs | 23 | 40,508 | 1,329 | 185,949 |
| clang-tools-extra/docs | 0 | 0 | 620 | 32,447 |
| clang/docs | 2 | 1,002 | 128 | 75,316 |
| mlir/docs | 85 | 37,040 | 0 | 0 |
| flang/docs | 78 | 24,474 | 1 | 63 |
| lldb/docs | 18 | 5,396 | 32 | 12,330 |

New-file activity is a better signal than total legacy footprint. Over the last
year, after excluding the large generated-ish AMDGPU and clang-tidy buckets, new
Markdown and new reST files are essentially tied.

| Scope | Format | Added files | Authors | Top roots |
| --- | --- | --- | --- | --- |
| All docs markup | .rst | 238 | 64 | llvm/docs: 146, clang-tools-extra/docs: 58, clang/docs: 19 |
| All docs markup | .md | 56 | 26 | lldb/docs: 17, flang/docs: 15, clang/docs: 5 |
| Excluding AMDGPU reference and clang-tidy check pages | .rst | 55 | 35 | clang/docs: 19, llvm/docs: 16, libc/docs: 7 |
| Excluding AMDGPU reference and clang-tidy check pages | .md | 56 | 26 | lldb/docs: 17, flang/docs: 15, clang/docs: 5 |
| Also excluding release notes and Flang meeting notes | .rst | 53 | 34 | clang/docs: 19, llvm/docs: 16, libc/docs: 7 |
| Also excluding release notes and Flang meeting notes | .md | 44 | 25 | lldb/docs: 17, clang/docs: 5, mlir/docs: 5 |

The contributor data also does not support the idea that reST is uniquely
preferred. More people touched reST because more of the tree is still reST and
because some areas require it by convention. Where Markdown is enabled and
accepted, contributors use it.

| Metric | Value |
| --- | --- |
| All docs commits in the last year | 2,788 |
| Unique docs authors | 826 |
| Markdown-only docs commits | 404 |
| RST-only docs commits | 2,211 |
| Commits touching both Markdown and RST docs | 100 |
| Authors touching Markdown docs | 234 |
| Authors touching RST docs | 693 |

| Docs root | Markdown authors | RST authors |
| --- | --- | --- |
| mlir/docs | 61 | 0 |
| flang/docs | 55 | 2 |
| llvm/docs | 97 | 271 |
| clang/docs | 4 | 339 |
| clang-tools-extra/docs | 0 | 106 |
| lldb/docs | 16 | 23 |

## Conversion experiment

I converted three important LLVM documents with `pandoc 3.1.11.1` and light
post-processing:

* [LangRef](https://llvm.org/docs/LangRef.html)
* [DeveloperPolicy](https://llvm.org/docs/DeveloperPolicy.html)
* [CMake](https://llvm.org/docs/CMake.html)

The HTML output before the conversion was captured in `/tmp/llvm-docs-before`.
The converted output was built in `/tmp/llvm-docs-after-review`.

The before and after HTML builds both succeed with the same two warnings:
`ProjectGovernance.rst` and this RFC are not included in any toctree. The
converted tree also succeeds with the Sphinx man-page builder.

| Build | Command | Result |
| --- | --- | --- |
| Before HTML | `sphinx-build -b html llvm/docs /tmp/llvm-docs-before` | succeeds, 2 warnings |
| After HTML | `sphinx-build -E -b html llvm/docs /tmp/llvm-docs-after-review` | succeeds, 2 warnings |
| After man pages | `sphinx-build -E -b man llvm/docs /tmp/llvm-docs-man-after` | succeeds, 2 warnings |

The rendered-page diff is also encouraging. DeveloperPolicy has identical
visible text after conversion. CMake has one visible-text change, and it removes
literal backticks that were leaking into the old rendered page. LangRef is much
larger, but the high-level structure is stable: the converted page has the same
heading counts, the same table count, and the same link count.

| Page | Block comparison | Visible text | Notes |
| --- | --- | --- | --- |
| CMake | 474 blocks before and after, 0.9979 similarity | one small difference | removes visible backticks around one `llvm-mt` mention |
| DeveloperPolicy | 413 blocks before and after, 1.0000 similarity | identical | no visible text changes |
| LangRef | 9,146 to 9,175 blocks, 0.9898 similarity | small differences | mostly field-list/definition-list normalization and code quoting; 20 tables before and after; 5,318 links before and after |

This is not an argument that the whole migration is a one-click `pandoc` job.
There will be local cleanup. It is an argument that even very large, link-heavy
core documents can be converted without changing their substance or breaking the
documentation build.

## Man pages are not a blocker

There is not zero support for generating man pages from Markdown. Sphinx's man
builder consumes a Sphinx/docutils document tree; MyST parses Markdown into that
same world. In this branch, the existing man-page build still succeeds with
`myst_parser` installed.

The real blocker is dependency policy. `llvm/docs/conf.py` currently treats
`myst_parser` as optional for `builder-man`, with a comment saying that the man
build does not use Markdown pages. That assumption stops being true if command
guide pages move to Markdown. The clean bridge is to make `myst_parser` a hard
dependency for all Sphinx documentation builds, including man pages, and keep
using the existing Sphinx man builder.

There are alternatives: `pandoc -t man`, `scdoc`, and `ronn`-style tools can
produce man-page output from Markdown-like input. They are plausible emergency
bridges for simple command docs, but they would create a second documentation
pipeline and would not naturally understand Sphinx roles, cross-references,
toctrees, or LLVM's existing Sphinx configuration. Keeping Sphinx as the only
renderer and feeding it MyST Markdown is the least disruptive path.

## Why full migration matters

A "both formats are fine" policy sounds low-risk, but it has already produced
format churn. Contributors cannot tell which format is preferred, reviewers can
ask for Markdown to be converted back to reST, and subprojects converge on
different conventions. That is worse than either format by itself.

Markdown is the format most contributors already know, most editors preview, and
most external documentation systems understand. MyST gives us the Sphinx features
we still need: cross-references, directives, roles, admonitions, tables,
numbered figures, and multiple builders. A full migration gives contributors one
answer and lets tooling improve around one source format.

[myst]: https://myst-parser.readthedocs.io/en/latest/
[rest]: https://devguide.python.org/documentation/markup/
[mdn]: https://developer.mozilla.org/en-US/
[ai-tool-policy-md]: http://github.com/llvm/llvm-project/blob/main/llvm/docs/AIToolPolicy.md
[dev-policy-md]: http://github.com/llvm/llvm-project/blob/main/llvm/docs/DeveloperPolicy.md
[ai-policy-html]: https://llvm.org/docs/AIToolPolicy.html
[cir-rest]: https://github.com/llvm/llvm-project/issues/191850
[llvm-sphinx-docs]: https://llvm.org/docs/SphinxQuickstartTemplate.html
[doc-history-script]: https://github.com/rnk/llvm-project/blob/llvm-markdown/llvm/utils/analyze-doc-edit-history.py
