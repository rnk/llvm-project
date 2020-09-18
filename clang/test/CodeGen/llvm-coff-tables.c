// RUN: %clang_cc1 -triple x86_64-windows-msvc -fllvm-dllexports -fllvm-symbol-roots %s -emit-llvm -o - -fms-extensions | FileCheck %s

void __declspec(dllexport) foo() { }
void __attribute__((used)) bar() { }

// CHECK: !llvm.module.flags = !{!{{[0-9]+}}, ![[DLL:[0-9]+]], ![[ROOT:[0-9]+]]}
// CHECK: ![[DLL]] = !{i32 1, !"LlvmDllexports", i32 1}
// CHECK: ![[ROOT]] = !{i32 1, !"LlvmSymbolRoots", i32 1}
