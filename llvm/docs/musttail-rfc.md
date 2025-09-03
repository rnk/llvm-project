
[RFC] Relaxing musttail constraints
===================================

Since the `musttail` LLVM IR marker was added [in
2014](https://github.com/llvm/llvm-project/commit/5772b77789728e045ccd8c93d79820632e81c4e6)
and then surfaced as a [frontend attribute in
2021](https://reviews.llvm.org/D99517), we've had a lot more implementation
experience with it. CPython has
[adopted](https://github.com/python/cpython/issues/128563) it, GCC has
[implemented
it](https://gcc.gnu.org/git/gitweb.cgi?p=gcc.git;h=59dd1d7ab21ad9a7ebf641ec9aeea609c003ad2f)
in C++ and C, and the C committee is discussing the possibility of adding a
similar feature under the name `return goto`
([n3266](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3266.htm#user-content-5-tail-call-elimination)).
Over the years, many folks have wondered why the Clang and LLVM `musttail`
marker requires prototype match, and given the feedback we've gotten along the
way, it seems like it's time to revisit that and consider relaxing this
restriction.

Today, users are unhappy with the status quo in both directions. Users are upset
that there are backend errors on other architectures
([arm32](https://github.com/llvm/llvm-project/issues/50758),
[power](https://github.com/llvm/llvm-project/issues/56679), mips,
[riscv](https://github.com/llvm/llvm-project/issues/84090)), and upset that
they get [frontend errors](https://github.com/llvm/llvm-project/issues/54964)
when the backend can do the tail call without the explicit annotation
*requiring* the tail call. These are fundamentally conflicting expectations.
Some users seem to expect that if the backend can tail call something, they
should be able to add an annotation that forces it to do so, and it should
always succeed. Some users don't want hidden compiler ICE portability bombs in
their codebase, which is what the verifier rules were intended to prevent.
However, experience has shown that there are always new targets and
microarchitectures on the horizon, and backend developers generally consider
tail call support to be optional, so these rules haven't eliminated the risk of
backend errors.

Long story short, I think the right tradeoff today is to give up on the
portability goals that were never achieved, and just give the performance
tweakers the low-level control that they want. On some level, this means
mandatory tail calling will always and forever be a low-level, target-specific
feature, like inline assembly and vector intrinsics, but these features are sort
of what make C/C++ unique, and restricting them is a losing battle.

Background
-----------

To start at the beginning, the most common reason for a backend to consider a
call site ineligible for tail call elimination is that the call site requires
additional argument memory beyond what was allocated by the caller, and the
calling convention is not callee-cleanup. Consider the following stack diagram
(ACII art makes everything better) and a call chain of `foo -> bar -musttail->
baz`:

```
              ┌───────────────┐
              │               │
              │ foo locals    │
              │               │
              ┌───────────────┐
  SP before & │ arg1...       │        │
  after call  │ arg0          │        │
  ─────────►  ┌───────────────┐        │ stack growth
              │ return address│        │
              ┌───────────────┐        │
              │               │        │
              │               │        ▼
              │ bar/baz locals│
              │               │
              │               │
              └───────────────┘
```

In the above diagram, if the call from `bar` to `baz` needed to add `arg2` to
the stack, there's no room for it in memory, and there's no way to communicate
to `baz` what the final SP should be after `baz` returns to `foo`.

The vast majority of C calling conventions are *not* callee-cleanup, for a few
reasons. Having a single level stack pointer throughout the entire call frame
simplifies code generation and unwind information. Dynamic stack pointer
adjustments become scheduling barriers. C also has a legacy of unprototyped and
variadic functions, where sometimes functions are called with additional
arguments. If the caller and callee don't precisely agree on the number of
arguments, it's safer to let the caller own the argument memory.

For this reason, most of the LLVM calling conventions that support guaranteed
TCE are callee cleanup (`tailcc`, `swiftcc`, others), but the most common widely
used platform-default C calling conventions are not. For callee-cleanup
conventions, the tail call site simply allocates additional memory and moves the
return address, assuming it lives in memory:

```
            ┌───────────────┐             ┌───────────────┐             ┌───────────────┐
            │               │             │               │             │               │
            │ foo    locals │             │ foo    locals │   SP after  │ foo    locals │
            │               │             │               │   baz return│               │
            ┌───────────────┐             ┌───────────────┐   ────────► ┌───────────────┐
 SP before  │ arg1...       │             │ arg2          │             │ arg2          │
 bar call   │ arg0          │             │ arg1          │             │ arg1          │
 ────────►  ┌───────────────┐             │ arg0          │             │ arg0          │
            │ return address│  SP on baz  ┌───────────────┐             ┌───────────────┐
            ┌───────────────┐  entry      │ return address│             │ return address│
            │               │  ────────►  ┌───────────────┐             ┌───────────────┐
            │               │             │               │             │               │
            │    bar locals │             │               │             │               │
            │               │             │ free          │             │    baz locals │
            │               │             │               │             │               │
            └───────────────┘             │               │             │               │
                                          └───────────────┘             └───────────────┘
```

In our example, after `baz` returns, the stack pointer returns to where `foo`
expects it to be. Moving the return address works

When I was working on the specifications for the LLVM IR feature in 2014, I was
trying to design the feature in a way that would prevent unpredictable backend
errors. Consider that at the time, my job was to make `clang-cl` ABI compatible
with MSVC, so non-portable target-specific features were my biggest obstacle,
and I didn't want to create new non-portable features. The prototype match
requirement means that, when parameter are passed in memory, which they always
are in the general case, the amount of stack memory used for receiving and
passing arguments is always the same. This means you don't have to adjust the
stack in the prologue and epilogue, and crucially, to me, *you don't have to
move the **return address** in memory*. Modern, post-x86 architectures generally
use a link register to track the return address, so this is less of an issue
there, but for myself and many others, x86_64 isn't legacy yet, so this mattered
a lot to me. I could see existing legacy codepaths in the X86 backend for moving
the return address, and they looked like pits of bugs and special cases. In
fact, they were a source of pain and confusion when we were working on improving
Win64 unwind info, we actually dropped support for any call that has a non-zero
"tail call return address delta". The [error path is still there
today](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86FrameLowering.cpp#L1664),
and if we were to relax the prototype match restrictions on musttail today, this
error would immediately increase Windows portability barriers.

The next important consideration is that, if you want musttail calls to be
reliable, you need to ensure that they endure through mid-level transformations.
The prototype match verifier rule was a powerful tool for auditing LLVM for
transforms like argument promotion which can change the prototype in ways that
break the backend's ability to emit the tail call. For example, argument
promotion is a kind of interprocedural scalar-replacement-of-aggregates (SROA),
and it can easily take a tail call that would only use registers, to one that
passes arguments in memory, and would cause the backend to fail to emit the tail
call that the user requested. If you dig through the logs, you can find many
instances of folks powering down IPO transforms in the presence of musttail
calls thanks to this verifier check.

The most obvious way that the middle-end breaks musttail calls is that it
inserts instrumentation between the call and the return, but we have a verifier
rule to catch that. A guaranteed tail call would really be a single operation,
and it seems like there is the opportunity improve our representation further
with a creative generalization of the `callbr` instruction, but that's beyond
the scope of this RFC.

In 2021, Tim Northover relaxed the verifier
([9ff2eb1ea596a52ad2b5cfab826548c3af0a1e6e](http://github.com/llvm/llvm-project/commit/9ff2eb1ea596a52ad2b5cfab826548c3af0a1e6e),
[D102612](https://reviews.llvm.org/D102612)) rule to allow prototype mismatch
with `tailcc` and `swifttailcc`, which was a new calling convention added to
support Swift coroutines. As covered earlier, callee cleanup conventions make
tail calls more reliable, but there are a number of edge cases 


Next Steps
-----------

So where should we go from here?

Before we do anything, we need to improve the backend diagnostic. LLVM has
facilities for reporting backend errors for stack frame size overflow and inline
assembly emission errors, and we should be using those APIs to emit readable
error messages instead of an internal error stack spew. This should reduce the
rate of bug reports over the long run since we'll stop prompting users to file
bugs on impossible musttail calls.

Personally, my view is that we should compromise on portability and just give the performance tweakers what they want, and remove all of the frontend and verifier safety checks from the feature.
