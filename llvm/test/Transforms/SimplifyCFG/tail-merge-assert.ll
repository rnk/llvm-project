; RUN: opt -tail-merge-bbs-ending-in-unreachable -simplifycfg -S < %s | FileCheck %s

; Test that we tail merge this kind of code with glibc-style assertion
; failure calls:
;   #include <assert.h>
;   void merge_glibc_asserts(unsigned x, unsigned y) {
;     assert(x < y);
;     assert(y - x > 7);
;     assert(y - x < 40);
;   }
;
; glibc's __assert_fail function takes four parameters, and it is profitable to
; phi two of them.

@.str = private unnamed_addr constant [6 x i8] c"x < y\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"t.cpp\00", align 1
@__PRETTY_FUNCTION__._Z1fjj = private unnamed_addr constant [35 x i8] c"void f(unsigned int, unsigned int)\00", align 1
@.str.2 = private unnamed_addr constant [10 x i8] c"y - x > 7\00", align 1
@.str.3 = private unnamed_addr constant [11 x i8] c"y - x < 40\00", align 1

declare void @glibc_assert_fail(i8*, i8*, i32, i8*)

define void @merge_glibc_asserts(i32 %x, i32 %y) {
entry:
  %cmp = icmp ugt i32 %y, %x
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  tail call void @glibc_assert_fail(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), i32 3, i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__PRETTY_FUNCTION__._Z1fjj, i64 0, i64 0)) #2
  unreachable

cond.end:                                         ; preds = %entry
  %sub = sub i32 %y, %x
  %cmp1 = icmp ugt i32 %sub, 7
  br i1 %cmp1, label %cond.end4, label %cond.false3

cond.false3:                                      ; preds = %cond.end
  tail call void @glibc_assert_fail(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), i32 4, i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__PRETTY_FUNCTION__._Z1fjj, i64 0, i64 0)) #2
  unreachable

cond.end4:                                        ; preds = %cond.end
  %cmp6 = icmp ult i32 %sub, 40
  br i1 %cmp6, label %cond.end9, label %cond.false8

cond.false8:                                      ; preds = %cond.end4
  tail call void @glibc_assert_fail(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), i32 5, i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__PRETTY_FUNCTION__._Z1fjj, i64 0, i64 0)) #2
  unreachable

cond.end9:                                        ; preds = %cond.end4
  ret void
}

; CHECK-LABEL: define void @merge_glibc_asserts(i32 %x, i32 %y)
; CHECK: cond.false:
; CHECK: %[[phi_line:[^ ]*]] = phi i32 [ 3, %entry ], [ 4, %cond.end ], [ 5, %cond.end4 ]
; CHECK: %[[phi_expr:[^ ]*]] = phi i8* [ getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), %entry ], [ getelementptr inbounds ([10 x i8], [10 x i8]* @.str.2, i64 0, i64 0), %cond.end ], [ getelementptr inbounds ([11 x i8], [11 x i8]* @.str.3, i64 0, i64 0), %cond.end4 ]
; CHECK: tail call void @glibc_assert_fail(
; CHECK-SAME: i8* %[[phi_expr]],
; CHECK-SAME: i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0)
; CHECK-SAME: i32 %[[phi_line]],
; CHECK-SAME: i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__PRETTY_FUNCTION__._Z1fjj, i64 0, i64 0))
