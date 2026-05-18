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

Since 2018 ([D44910]), LLVM has used a Markdown dialect called Markedly
Structured Text (MyST) for portions of its documentation. Individual subprojects
have effectively been free to choose between reST and MyST at their own
discretion, and there has been no coherent policy about which is preferred. This
has led to [backporting the CIR docs to reST][cir-rest]. The CIR docs were
originally Markdown, but were converted back to the legacy reST format. The
point of this RFC is to declare affirmatively which format we prefer, update the
[Sphinx quickstart template][llvm-sphinx-docs] to that effect, and make a full
migration the desired end state.

## Proposal

* MyST Markdown should become the preferred and eventual sole hand-authored
  format for LLVM Sphinx documentation.
* New hand-written documentation should use `.md` unless there is a concrete
  blocker.
* Existing `.rst` files may continue to be edited until they are converted, but
  we should welcome mechanical conversion PRs.
* `myst_parser` should become a hard dependency for Sphinx documentation builds,
  including the man-page builder.
* The [Sphinx quickstart template][llvm-sphinx-docs] should recommend Markdown
  for new docs.

I'm willing to commit to migrating some key documents one at a time, but I can't
promise to drive this to completion. This is what I handle:

* [SphinxQuickStartTemplate](llvm-sphinx-docs): Important, because this tells
  people how to write docs.
* [LangRef](https://llvm.org/docs/LangRef.html): The most important doc. The
  edits *must not reflow text* needlessly to avoid conflicts with pending
  patches.
* [DeveloperPolicy](https://llvm.org/docs/DeveloperPolicy.html): Also an
  important doc.
* [CMake](https://llvm.org/docs/CMake.html): Next most important doc.

I've actually already prototyped the migration on GitHub in [rnk:llvm-markdown],
and I am serving up a copy of the generated documents [here (starting with the
quickstart)][staging-llvmdocs], if you want to compare.



## What the tree is doing today

I generated these numbers with [a small analysis script][doc-history-script]
over LLVM documentation paths and recent git history:

```bash
python3 llvm/utils/analyze-doc-edit-history.py \
  --since 2025-05-12 --until 2026-05-12 \
  --max-roots 12 --max-examples 6
```

The footprint numbers are from this branch after converting LangRef,
DeveloperPolicy, and CMake. The recent-activity numbers cover one year of
upstream history ending at `976195f9d5be` on 2026-05-12.

| Scope | Markdown files | Markdown lines | reST files | reST lines |
| --- | --- | --- | --- | --- |
| All docs markup | 224 | 112,456 (25.5%) | 2,269 | 328,004 (74.5%) |
| Excluding AMDGPU reference and clang-tidy check pages | 224 | 112,456 (30.5%) | 614 | 256,182 (69.5%) |
| Also excluding release notes | 221 | 111,984 (30.6%) | 604 | 254,012 (69.4%) |

MLIR is entirely Markdown in this analysis, Flang is almost entirely Markdown,
and LLDB is substantially Markdown. The main LLVM docs still carry most of the
old reST weight.

| Scope | Markdown added files | reST added files |
| --- | --- | --- |
| All docs markup | 56 files by 26 authors | 238 files by 64 authors |
| Excluding AMDGPU reference and clang-tidy check pages | 56 files by 26 authors | 55 files by 35 authors |
| Also excluding release notes and Flang meeting notes | 44 files by 25 authors | 53 files by 34 authors |

## Man pages are not a blocker

Sphinx is able to generate man pages from MyST when the `myst_parser` module is
installed, but it isn't always present.

`llvm/docs/conf.py` currently treats `myst_parser` as optional for
`builder-man`, because the man build does not use Markdown pages today. That
assumption stops being true if command-guide pages move to Markdown. I would
rather make `myst_parser` required than add a second documentation pipeline with
`pandoc -t man`, `scdoc`, or `ronn`.

[myst]: https://myst-parser.readthedocs.io/en/latest/
[rest]: https://devguide.python.org/documentation/markup/
[rnk:llvm-markdown]: https://github.com/rnk/llvm-project/tree/llvm-markdown
[staging-llvmdocs]: https://llvmdocs.staging.reidkleckner.dev/
[ai-tool-policy-md]: http://github.com/llvm/llvm-project/blob/main/llvm/docs/AIToolPolicy.md
[dev-policy-md]: http://github.com/llvm/llvm-project/blob/main/llvm/docs/DeveloperPolicy.md
[ai-policy-html]: https://llvm.org/docs/AIToolPolicy.html
[D44910]: https://reviews.llvm.org/D44910
[cir-rest]: https://github.com/llvm/llvm-project/issues/191850
[llvm-sphinx-docs]: https://llvm.org/docs/SphinxQuickstartTemplate.html
[doc-history-script]: https://github.com/rnk/llvm-project/blob/llvm-markdown/llvm/utils/analyze-doc-edit-history.py
