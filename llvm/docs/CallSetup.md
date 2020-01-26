# RFC: Replace inalloca with llvm.call.setup

In order to pass non-trivially copyable objects by value in a way that is
compatible with the Visual C++ compiler on 32-bit x86, Clang has to be able to
separate the allocation of call argument memory from the call site. The
`inalloca` feature was added to LLVM for this purpose. However, this feature
usually results in inefficient code, and is often incompatible with otherwise
straightforward LLVM IR transforms. Therefore, I would like to get rid of it
and replace it with `llvm.call.setup`, which I hope will be more maintainable
in the long run.

These are some of the drawbacks of inalloca that I want to fix with the new IR:
- Blocks most interprocedural prototype changing transforms: dead argument
  elimination, argument promotion, application of fastcc, etc.
- Blocks alias analysis: Unrelated args in memory are hard to analyze
- Blocks function attribute inference for unrelated parameters
- Hard for frontends to use because one argument can affect every other argument

Since inalloca was added, the token type was added to LLVM. Transforms are not
allowed to obscure the definition of a token value, and tokens can be used to
create single-entry, multi-exit regions in LLVM IR. This allows the creation of
a call setup operation that must remain paired with its call site throughout
mid-level optimization. That guarantee allows the backend to look at the call
site when emitting the call setup instructions. If the IR for a call setup
forms a proper region, that can also help the backend perform frame pointer
elimination in many cases.

## New IR Features

Here is the list of new IR features I think this requires:

Intrinsics:
- `token @llvm.call.setup(i32 %numArgs)`
- `i8* @llvm.call.alloc(token %site, i32 %argIdx)`
- `void @llvm.call.teardown(token)`

Attributes:
- `preallocated(<ty>)`, similar to byval, but no copy

Bundles:
- `[ "callsetup"(token %site) ]`

Verifier rules:
- `llvm.call.setup` must have exactly one corresponding call site
- call site must have equal number of preallocated args to `llvm.call.setup`

## Intended Usage

Here is an example of LLVM IR for an unpacked call site:

```llvm
%cs = call token @llvm.call.setup(i32 3)
%m0 = call i8* @llvm.call.alloc(token %cs, i32 0)  ;; allocates {i32}
call void @llvm.memset*(i8* %m0, i32 0, i32 4)
%m1 = call i8* @llvm.call.alloc(token %cs, i32 1)  ;; allocates {i32, i32}
call void @llvm.memset*(i8* %m1, i32 0, i32 8)
%m2 = call i8* @llvm.call.alloc(token %cs, i32 2)  ;; allocates {i32, i32, i32}
call void @llvm.memset*(i8* %m2, i32 0, i32 12)
call void @use_callsetup(
    {i32}* preallocated({i32}) %m1,
    i32 13,
    {i32,i32}* preallocated({i32,i32}) %m2,
    i32 42,
    {i32,i32,i32}* preallocated({i32,i32,i32}) %m3)
    [ "callsetup"(token %cs) ]
```

Many transforms will need to be made aware of the new verifier guarantees, but
they should only block optimizations on preallocated arguments. The goal is
that unrelated arguments, such as the i32 arguments above, remain unaffected.
DAE, for example, is free to eliminate the plain integers, but it cannot
eliminate a preallocated argument without adjusting the call.setup.

The next most important thing to keep in mind is how this interacts with
exception handling. This is where `llvm.call.teardown` comes into play. The
idea is that, in a call region, clang should push an exceptional cleanup onto
the cleanup stack. Here is what the IR for the previous example would look
like, assuming each argument has a constructor that may throw:

```llvm
%cs = call token @llvm.call.setup(i32 3)
%m0 = call i8* @llvm.call.alloc(token %cs, i32 0)
invoke void @ctor0(i8* %m0)
  to label %cont0 unwind label %cleanupCall

cont0:
%m1 = call i8* @llvm.call.alloc(token %cs, i32 1)
invoke void @ctor1(i8* %m1)
  to label %cont1 unwind label %cleanup0

cont1:
%m2 = call i8* @llvm.call.alloc(token %cs, i32 2)
invoke void @ctor2(i8* %m2)
  to label %cont2 unwind label %cleanup1

cont2:
  call void @use_callsetup(i8* preallocated %m1,
                           i32 13,
                           i8* preallocated %m2,
                           i32 42,
                           i8* preallocated %m3)
      [ "callsetup"(token %cs) ]


cleanup1:
  %cl1 = cleanuppad unwind to caller
  call void @dtor(i8* %m1) [ "funclet"(token %cl1) ]
  cleanupret %cl1 to label %cleanup0

cleanup0:
  %cl0 = cleanuppad unwind to caller
  call void @dtor(i8% %m0) [ "funclet"(token %cl2) ]
  cleanupret %cl0 to label %cleanupCall

cleanupCall:
  %clC = cleanuppad unwind to caller
  call void @llvm.call.teardown(token %cs)  ;; Or llvm.call.cleanup?
  cleanupret %clC to caller
```

Generally, cleanups to tear down a call setup region are not needed if control
cannot return to the current function. The cleanupCall block is an example of
such an unnecessary cleanup. However, to make things easy for the inliner, the
frontend is required to emit these cleanups. Prior to code generation, the
WinEHPrepare pass can remove any unneeded argument memory cleanups.

## Inliner Considerations

When inlining a call site with a callsetup bundle, the inliner will want to
turn preallocated arguments into regular static allocas. This should be
straightforward:

- Create a static alloca for each preallocated argument using its type
- Replace all uses of `llvm.call.alloc` with the corresponding new alloca
- Insert `lifetime.start/end`. Start at the alloc site, end at
  `llvm.call.teardown` and after call.
- Remove all setup, alloc, teardown intrinsics.

## GlobalOpt Considerations

Functions which cannot be inlined but are internal should receive the same
treatment described above. Global opt should call the same utility as the
inliner, and then remove the `preallocated` argument attributes from the
function prototype. This will enable downstream optimizations such as dead
argument elimination (DAE) and argument promotion.

## Corner cases

### Catching an exception within a call region

If an exception is thrown and caught within the call setup region, the newly
established SP must be saved into the EH record when a call is setup. Consider
the case below of inlining try / catch into a call region:

```c++
struct Foo {
  Foo(int x);
  Foo(const Foo &o);
  ~Foo();
  int x, y, z;
};
void use_callsetup(int, Foo, Foo);
int maythrow2();
static inline int maythrow() {
  try { return maythrow2(); } catch (int) {}
  return 0;
}
Foo getf();
int main() {
  use_callsetup(maythrow(), getf(), getf());
}
```

The backend should be able to detect whether SP must be saved or not by
checking if there are any invokes reachable along normal paths from the call
setup that do not first reach the call itself, or a normal `llvm.call.teardown`.

### Non-exceptional call region exit

It is possible, using statement expressions, to exit a call setup region with
normal control flow. In this case, the Clang should emit a normal cleanup to
call `llvm.eh.teardown`. Consider:

```c++
#define RETURN_IF_ERR(maybeError) \
  ({                              \
    auto val = maybeError;        \
    if (val.isError())            \
      return val.getError();      \
    val.getValue;                 \
  })
ErrorOr<int> mayFail();
void use_callsetup(int, Foo);
void f(bool cond) {
  use_callsetup(RETURN_IF_ERR(mayFail()), Foo());
}
```

### Inlining non-EH code into EH code

If exceptions are disabled, the frontend should not emit exceptional cleanups
to teardown the call region. However, this code could be inlined into a
function that uses exceptions, and the caller could catch an exception thrown
through the non-EH code. Generally this should be impossible, because the calls
will be marked nounwind, and it is UB to throw an exception through a nounwind
call site. However, SEH allows exceptions to be thrown through nounwind call
sites. Consider:

```c++
int mayThrow();
void use_callsetup(int, Foo);
static inline void inlineMe() {
  void use_callsetup(mayThrow(), Foo());
}
void f(int numRetries) {
  for (int i = 0; i < numRetries; i++) {
    __try {
      inlineMe();
    } __except (1) {
    }
  }
}
```

Unless the compiler inserts stack adjustments along the unwind path, this well
setup call argument memory for every loop iteration, and never clean it up.
This could result in code generation bugs or excessive stack memory use. The
two options are:
1. Refuse to inline functions containing call regions through invoke call sites
1. Teach the inliner to synthesize llvm.call.teardown cleanups inside call
   regions
1. Do nothing, the frontend already disables inlining into `__try`

Given that this corner case seems specific to SEH, the third option seems most
reasonable.

## Implementation Steps

In the spirit of incremental development, I think the implementation could be
broken down into the following patch series:
- IR: LLVM IR intrinsics and attributes
- Clang: IRGen, under -cc1 flag
- X86: backend implementation: inefficient, no EH
- Transforms: Update inliner and globalopt, audit other transforms
- X86: MSVC C++ EH SP management
- ... test it on Chrome with -cc1 flag
- Clang: Remove cc1 flag and inalloca IRGen logic
- ... announce inalloca removal from LLVM IR
- inalloca removal

## Backwards compatibility

It may be possible to upgrade some bitcode from `inalloca` to
`llvm.call.setup`, but in cases with complex control flow where the allocation
site does not dominate the call, it will not be straightforward. Therefore, I
think we should make an exception to LLVM's usual policy of bitcode backwards
compatibility, and drop support for `inalloca`. The `inalloca` attribute was
generally a source of bugs, and was only used for code targetting
`i*86-windows-msvc`. To the best of my knowledge, there are no users of LLVM
who archive bitcode for that platform and expect to be able to upgrade it for
use with future LLVM versions.
