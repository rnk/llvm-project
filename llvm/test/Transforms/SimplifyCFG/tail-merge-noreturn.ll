; RUN: opt -tail-merge-bbs-ending-in-unreachable -simplifycfg -S < %s | FileCheck %s

; Test that we tail merge noreturn call blocks and phi constants properly.

declare void @abort()
declare void @assert_fail_1(i32)
declare void @assert_fail_1_alt(i32)

define void @merge_simple() {
  %c1 = call i1 @foo()
  br i1 %c1, label %cont1, label %a1
a1:
  call void @assert_fail_1(i32 0)
  unreachable
cont1:
  %c2 = call i1 @foo()
  br i1 %c2, label %cont2, label %a2
a2:
  call void @assert_fail_1(i32 0)
  unreachable
cont2:
  %c3 = call i1 @foo()
  br i1 %c3, label %cont3, label %a3
a3:
  call void @assert_fail_1(i32 0)
  unreachable
cont3:
  ret void
}

; CHECK-LABEL: define void @merge_simple()
; CHECK: br i1 %c1, label %cont1, label %a1
; CHECK: a1:
; CHECK: call void @assert_fail_1(i32 0)
; CHECK: unreachable
; CHECK: br i1 %c2, label %cont2, label %a1
; CHECK-NOT: assert_fail_1
; CHECK: br i1 %c3, label %cont3, label %a1
; CHECK-NOT: assert_fail_1
; CHECK: ret void

define void @phi_three_constants() {
entry:
  %c1 = call i1 @foo()
  br i1 %c1, label %cont1, label %a1
a1:
  call void @assert_fail_1(i32 0)
  unreachable
cont1:
  %c2 = call i1 @foo()
  br i1 %c2, label %cont2, label %a2
a2:
  call void @assert_fail_1(i32 1)
  unreachable
cont2:
  %c3 = call i1 @foo()
  br i1 %c3, label %cont3, label %a3
a3:
  call void @assert_fail_1(i32 2)
  unreachable
cont3:
  ret void
}

; CHECK-LABEL: define void @phi_three_constants()
; CHECK: br i1 %c1, label %cont1, label %a1
; CHECK: a1:
; CHECK: %[[p:[^ ]*]] = phi i32 [ 0, %entry ], [ 1, %cont1 ], [ 2, %cont2 ]
; CHECK: call void @assert_fail_1(i32 %[[p]])
; CHECK: unreachable
; CHECK: br i1 %c2, label %cont2, label %a1
; CHECK-NOT: assert_fail_1
; CHECK: br i1 %c3, label %cont3, label %a1
; CHECK-NOT: assert_fail_1
; CHECK: ret void


define void @dont_phi_values(i32 %x, i32 %y) {
  %c1 = call i1 @foo()
  br i1 %c1, label %cont1, label %a1
a1:
  call void @assert_fail_1(i32 %x)
  unreachable
cont1:
  %c2 = call i1 @foo()
  br i1 %c2, label %cont2, label %a2
a2:
  call void @assert_fail_1(i32 %y)
  unreachable
cont2:
  ret void
}

; CHECK-LABEL: define void @dont_phi_values(i32 %x, i32 %y)
; CHECK:   call void @assert_fail_1(i32 %x)
; CHECK:   call void @assert_fail_1(i32 %y)
; CHECK:   ret void


define void @dont_phi_callees() {
  %c1 = call i1 @foo()
  br i1 %c1, label %cont1, label %a1
cont1:
  %c2 = call i1 @foo()
  br i1 %c2, label %cont2, label %a2
cont2:
  ret void
a1:
  call void @assert_fail_1(i32 0)
  unreachable
a2:
  call void @assert_fail_1_alt(i32 0)
  unreachable
}

; CHECK-LABEL: define void @dont_phi_callees()
; CHECK:   call void @assert_fail_1(i32 0)
; CHECK:   unreachable
; CHECK:   call void @assert_fail_1_alt(i32 0)
; CHECK:   unreachable


declare i1 @foo()
declare i1 @bar()

define void @unmergeable_phis(i32 %v, i1 %c) {
entry:
  br i1 %c, label %s1, label %s2
s1:
  %c1 = call i1 @foo()
  br i1 %c1, label %a1, label %a2
s2:
  %c2 = call i1 @bar()
  br i1 %c2, label %a1, label %a2
a1:
  %l1 = phi i32 [ 0, %s1 ], [ 1, %s2 ]
  call void @assert_fail_1(i32 %l1)
  unreachable
a2:
  %l2 = phi i32 [ 2, %s1 ], [ 3, %s2 ]
  call void @assert_fail_1(i32 %l2)
  unreachable
}

; CHECK-LABEL: define void @unmergeable_phis(i32 %v, i1 %c)
; CHECK: a1:
; CHECK:   %l1 = phi i32 [ 0, %s1 ], [ 1, %s2 ], [ %l2, %a2 ]
; CHECK:   call void @assert_fail_1(i32 %l1)
; CHECK:   unreachable
; CHECK: a2:
; CHECK:   %l2 = phi i32 [ 2, %s1 ], [ 3, %s2 ]
; CHECK:   br label %a1


define void @tail_merge_switch(i32 %v) {
entry:
  switch i32 %v, label %ret [
    i32 0, label %a1
    i32 13, label %a2
    i32 42, label %a3
  ]
a1:
  call void @assert_fail_1(i32 0)
  unreachable
a2:
  call void @assert_fail_1(i32 1)
  unreachable
a3:
  call void @assert_fail_1(i32 2)
  unreachable
ret:
  ret void
}

; CHECK-LABEL: define void @tail_merge_switch(i32 %v)
; CHECK: a1:
; CHECK:   %[[p:[^ ]*]] = phi i32 [ 0, %entry ], [ 1, %a2 ], [ 2, %a3 ]
; CHECK:   call void @assert_fail_1(i32 %[[p]])
; CHECK:   unreachable
; CHECK: a2:
; CHECK:   br label %a1
; CHECK: a3:
; CHECK:   br label %a1


define void @need_to_add_bb2_preds(i1 %c1) {
bb1:
  br i1 %c1, label %bb2, label %a1
bb2:
  %c2 = call i1 @bar()
  br i1 %c2, label %a2, label %a3

a1:
  call void @assert_fail_1(i32 0)
  unreachable
a2:
  call void @assert_fail_1(i32 1)
  unreachable
a3:
  call void @assert_fail_1(i32 2)
  unreachable
}

; CHECK-LABEL: define void @need_to_add_bb2_preds(i1 %c1)
; CHECK: bb1:
; CHECK:   br i1 %c1, label %bb2, label %a1
; CHECK: bb2:
; CHECK:   %c2 = call i1 @bar()
; CHECK:   %[[sel:[^ ]*]] = select i1 %c2, i32 1, i32 2
; CHECK:   br label %a1
; CHECK: a1:
; CHECK:   %[[p:[^ ]*]] = phi i32 [ 0, %bb1 ], [ %[[sel]], %bb2 ]
; CHECK:   call void @assert_fail_1(i32 %[[p]])
; CHECK:   unreachable


define void @phi_in_bb2() {
entry:
  %c1 = call i1 @foo()
  br i1 %c1, label %cont1, label %a1
a1:
  call void @assert_fail_1(i32 0)
  unreachable
cont1:
  %c2 = call i1 @foo()
  br i1 %c2, label %cont2, label %a2
a2:
  %p2 = phi i32 [ 1, %cont1 ], [ 2, %cont2 ]
  call void @assert_fail_1(i32 %p2)
  unreachable
cont2:
  %c3 = call i1 @foo()
  br i1 %c3, label %cont3, label %a2
cont3:
  ret void
}

; CHECK-LABEL: define void @phi_in_bb2()
; CHECK: a1:
; CHECK:   %[[p:[^ ]*]] = phi i32 [ 0, %entry ], [ 1, %cont1 ], [ 2, %cont2 ]
; CHECK:   call void @assert_fail_1(i32 %[[p]])
; CHECK:   unreachable
; CHECK: cont3:
; CHECK:   ret void


; Don't tail merge these noreturn blocks using lifetime end. It prevents us
; from sharing stack slots for x and y.

declare void @escape_i32_ptr(i32*)
declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)

define void @dont_merge_lifetimes(i32 %c1, i32 %c2) {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  switch i32 %c1, label %if.end9 [
    i32 13, label %if.then
    i32 42, label %if.then3
  ]

if.then:                                          ; preds = %entry
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start(i64 4, i8* nonnull %0)
  store i32 0, i32* %x, align 4
  %tobool = icmp eq i32 %c2, 0
  br i1 %tobool, label %if.end, label %if.then1

if.then1:                                         ; preds = %if.then
  call void @escape_i32_ptr(i32* nonnull %x)
  br label %if.end

if.end:                                           ; preds = %if.then1, %if.then
  call void @llvm.lifetime.end(i64 4, i8* nonnull %0)
  call void @abort()
  unreachable

if.then3:                                         ; preds = %entry
  %1 = bitcast i32* %y to i8*
  call void @llvm.lifetime.start(i64 4, i8* nonnull %1)
  store i32 0, i32* %y, align 4
  %tobool5 = icmp eq i32 %c2, 0
  br i1 %tobool5, label %if.end7, label %if.then6

if.then6:                                         ; preds = %if.then3
  call void @escape_i32_ptr(i32* nonnull %y)
  br label %if.end7

if.end7:                                          ; preds = %if.then6, %if.then3
  call void @llvm.lifetime.end(i64 4, i8* nonnull %1)
  call void @abort()
  unreachable

if.end9:                                          ; preds = %entry
  ret void
}

; CHECK-LABEL: define void @dont_merge_lifetimes(i32 %c1, i32 %c2)
; CHECK: call void @abort()
; CHECK: unreachable
; CHECK: call void @abort()
; CHECK: unreachable


; Dead phis in the block need to be handled.

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

define void @dead_phi() {
entry:
  %c1 = call i1 @foo()
  br i1 %c1, label %cont1, label %a1
a1:
  %dead = phi i32 [ 0, %entry ], [ 1, %cont1 ]
  call void @assert_fail_1(i32 0)
  unreachable
cont1:
  %c2 = call i1 @foo()
  br i1 %c2, label %cont2, label %a1
cont2:
  %c3 = call i1 @foo()
  br i1 %c3, label %cont3, label %a3
a3:
  call void @assert_fail_1(i32 0)
  unreachable
cont3:
  ret void
}

; CHECK-LABEL: define void @dead_phi()
; CHECK: a1:
; CHECK-NEXT: call void @assert_fail_1(i32 0)
; CHECK-NOT: @assert_fail_1
; CHECK: ret void

define void @strip_dbg_value(i32 %c) {
entry:
  call void @llvm.dbg.value(metadata i32 %c, i64 0, metadata !12, metadata !13), !dbg !14
  switch i32 %c, label %sw.epilog [
    i32 13, label %sw.bb
    i32 42, label %sw.bb1
  ]

sw.bb:                                            ; preds = %entry
  call void @llvm.dbg.value(metadata i32 55, i64 0, metadata !12, metadata !13), !dbg !14
  tail call void @abort()
  unreachable

sw.bb1:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 67, i64 0, metadata !12, metadata !13), !dbg !14
  tail call void @abort()
  unreachable

sw.epilog:                                        ; preds = %entry
  ret void
}

; CHECK-LABEL: define void @strip_dbg_value(i32 %c)
; CHECK: entry:
; CHECK:   call void @llvm.dbg.value(metadata i32 %c, {{.*}})
; CHECK:   switch i32 %c, label %sw.epilog [
; CHECK:     i32 13, label %sw.bb
; CHECK:     i32 42, label %sw.bb
; CHECK:   ]
; CHECK: sw.bb:                                            ; preds = %entry
; CHECK-NOT: llvm.dbg.value
; CHECK:   tail call void @abort()
; CHECK:   unreachable

define void @dead_phi_and_dbg(i32 %c) {
entry:
  call void @llvm.dbg.value(metadata i32 %c, i64 0, metadata !12, metadata !13), !dbg !14
  switch i32 %c, label %sw.epilog [
    i32 13, label %sw.bb
    i32 42, label %sw.bb1
    i32 53, label %sw.bb2
  ]

sw.bb:                                            ; preds = %entry
  %c.1 = phi i32 [ 55, %entry], [ 67, %sw.bb1 ]
  call void @llvm.dbg.value(metadata i32 %c.1, i64 0, metadata !12, metadata !13), !dbg !14
  tail call void @abort()
  unreachable

sw.bb1:
  br label %sw.bb

sw.bb2:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 84, i64 0, metadata !12, metadata !13), !dbg !14
  tail call void @abort()
  unreachable

sw.epilog:                                        ; preds = %entry
  ret void
}

; CHECK-LABEL: define void @dead_phi_and_dbg(i32 %c)
; CHECK: entry:
; CHECK:   call void @llvm.dbg.value(metadata i32 %c, {{.*}})
; CHECK:   switch i32 %c, label %sw.epilog [
; CHECK:     i32 13, label %sw.bb
; CHECK:     i32 42, label %sw.bb
; CHECK:     i32 53, label %sw.bb
; CHECK:   ]
; CHECK: sw.bb:                                            ; preds = %entry
; CHECK-NOT: llvm.dbg.value
; CHECK:   tail call void @abort()
; CHECK:   unreachable
; CHECK-NOT: call void @abort()
; CHECK:   ret void



!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "asdf")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!12 = !DILocalVariable(name: "c", scope: !7)
!13 = !DIExpression()
!14 = !DILocation(line: 2, column: 12, scope: !7)
