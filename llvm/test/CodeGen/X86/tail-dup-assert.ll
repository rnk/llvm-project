; RUN: llc -relocation-model=pic -mtriple=x86_64-linux < %s | FileCheck %s

; Make sure tail duplication can undo simplifycfg noreturn tail merging when
; appropriate and leave it alone when not profitable.

@.str = private unnamed_addr constant [6 x i8] c"x < y\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"y - x > 7\00", align 1
@.str.2 = private unnamed_addr constant [11 x i8] c"y - x < 40\00", align 1

@__file__ = private unnamed_addr constant [6 x i8] c"t.cpp\00", align 1
@__pretty_function__ = private unnamed_addr constant [35 x i8] c"void f(unsigned int, unsigned int)\00", align 1

declare void @force_stack_setup()

; Function Attrs: noreturn
declare void @__assert_fail_2(i8*, i32)

define void @duplicate_assert_2(i32 %x, i32 %y) {
entry:
  tail call void @force_stack_setup()
  %cmp = icmp ugt i32 %y, %x
  br i1 %cmp, label %cond.end, label %cond.false

cond.end:                                         ; preds = %entry
  %sub = sub i32 %y, %x
  %cmp1 = icmp ugt i32 %sub, 7
  br i1 %cmp1, label %cond.end4, label %cond.false

cond.end4:                                        ; preds = %cond.end
  %cmp6 = icmp ult i32 %sub, 40
  br i1 %cmp6, label %cond.end9, label %cond.false

cond.end9:                                        ; preds = %cond.end4
  ret void

cond.false:                                       ; preds = %cond.end4, %cond.end, %entry
  %noreturntail10 = phi i32 [ 33, %entry ], [ 34, %cond.end ], [ 35, %cond.end4 ]
  %noreturntail = phi i8* [ getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), %entry ], [ getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i64 0, i64 0), %cond.end ], [ getelementptr inbounds ([11 x i8], [11 x i8]* @.str.2, i64 0, i64 0), %cond.end4 ]
  tail call void @__assert_fail_2(i8* nonnull %noreturntail, i32 %noreturntail10)
  unreachable
}

; CHECK-LABEL: duplicate_assert_2:
; CHECK:         retq
; CHECK: .LBB0_{{.*}}:
; CHECK:         leaq    .L.str(%rip), %rdi
; CHECK:         movl    $33, %esi
; CHECK:         callq __assert_fail_2
; CHECK: .LBB0_{{.*}}:
; CHECK:         leaq    .L.str.1(%rip), %rdi
; CHECK:         movl    $34, %esi
; CHECK:         callq __assert_fail_2
; CHECK: .LBB0_{{.*}}:
; CHECK:         leaq    .L.str.2(%rip), %rdi
; CHECK:         movl    $35, %esi
; CHECK:         callq __assert_fail_2

declare void @__assert_fail_4(i8*, i8*, i32, i8*)

define void @leave_assert_4(i32 %x, i32 %y) {
entry:
  tail call void @force_stack_setup()
  %cmp = icmp ugt i32 %y, %x
  br i1 %cmp, label %cond.end, label %cond.false

cond.end:                                         ; preds = %entry
  %sub = sub i32 %y, %x
  %cmp1 = icmp ugt i32 %sub, 7
  br i1 %cmp1, label %cond.end4, label %cond.false

cond.end4:                                        ; preds = %cond.end
  %cmp6 = icmp ult i32 %sub, 40
  br i1 %cmp6, label %cond.end9, label %cond.false

cond.end9:                                        ; preds = %cond.end4
  ret void

cond.false:                                       ; preds = %cond.end4, %cond.end, %entry
  %noreturntail10 = phi i32 [ 33, %entry ], [ 34, %cond.end ], [ 35, %cond.end4 ]
  %noreturntail = phi i8* [ getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), %entry ], [ getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i64 0, i64 0), %cond.end ], [ getelementptr inbounds ([11 x i8], [11 x i8]* @.str.2, i64 0, i64 0), %cond.end4 ]
  tail call void @__assert_fail_4(i8* %noreturntail, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @__file__, i64 0, i64 0), i32 %noreturntail10, i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__pretty_function__, i64 0, i64 0))
  unreachable
}

; CHECK-LABEL: leave_assert_4:
; CHECK:         retq
; CHECK: .LBB1_{{.*}}:
; CHECK:         leaq    .L.str(%rip), %rdi
; CHECK:         movl    $33, %edx
; CHECK:         jmp [[fail_bb:\.LBB1_[0-9]+]]
; CHECK: .LBB1_{{.*}}:
; CHECK:         leaq    .L.str.1(%rip), %rdi
; CHECK:         movl    $34, %edx
; CHECK:         jmp [[fail_bb:\.LBB1_[0-9]+]]
; CHECK: .LBB1_{{.*}}:
; CHECK:         leaq    .L.str.2(%rip), %rdi
; CHECK:         movl    $35, %edx
; CHECK: [[fail_bb]]:
; CHECK:         leaq    .L__file__(%rip), %rsi
; CHECK:         leaq    .L__pretty_function__(%rip), %rcx
; CHECK:         callq   __assert_fail_4@PLT
