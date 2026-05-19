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
some way. reST has served us well, but I believe that now is the time set a
long-term goal to migrate our docs to Markdown.

Since 2018 ([D44910]), LLVM has used a Markdown dialect called Markedly
Structured Text (MyST) for portions of its documentation. Individual subprojects
have effectively been free to choose between reST and MyST at their own
discretion, and there has been no coherent policy about which is preferred.

Newer projects have tended to prefer markdown. MLIR is entirely Markdown, Flang
is almost entirely Markdown, and LLDB is substantially Markdown. The main LLVM
docs still carry most of the old reST weight.

This has led to [backporting the CIR docs to reST][cir-rest]. The CIR docs were
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
promise to personally hunt down every last `.rst` file in the monorepo. I'm
hoping that, in true open source fashion, volunteers will pitch in and help
migrate their own docs and help review and approve mechanical conversion PRs.

These are the docs I plan to migrate, in this order:

* [SphinxQuickstartTemplate][llvm-sphinx-docs]: This is effectively our policy
  doc, so it goes first as an obvious demo of how to write new docs.
* [LangRef](https://llvm.org/docs/LangRef.html): The most important doc. The
  edits *must not reflow text* needlessly to avoid conflicts with pending
  patches.
* [DeveloperPolicy](https://llvm.org/docs/DeveloperPolicy.html): Also an
  important doc.
* [CMake](https://llvm.org/docs/CMake.html): Next most important doc.

I've actually already prototyped the migration on GitHub in [rnk:llvm-markdown],
and I am serving up a copy of the generated documents [here (starting with the
quickstart)][staging-llvmdocs], if you want to compare.

## Build impact

LLVM already uses MyST for HTML documentation when `myst_parser` is available.
Making it required also keeps man-page generation in the existing Sphinx
pipeline if command-guide pages eventually move to Markdown. I will be honest, I
am not an expert in our release packaging pipeline, so this is probably going to
require support from packagers.

[myst]: https://myst-parser.readthedocs.io/en/latest/
[rest]: https://devguide.python.org/documentation/markup/
[rnk:llvm-markdown]: https://github.com/rnk/llvm-project/tree/llvm-markdown
[staging-llvmdocs]: https://llvmdocs.staging.reidkleckner.dev/
[D44910]: https://reviews.llvm.org/D44910
[cir-rest]: https://github.com/llvm/llvm-project/issues/191850
[llvm-sphinx-docs]: https://llvm.org/docs/SphinxQuickstartTemplate.html
