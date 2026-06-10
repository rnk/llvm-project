#!/usr/bin/env python3
"""Stress harness for PR173586's RemoveFileOnSignal slot reuse fix.

This script expects llvm/lib/Support/Unix/Signals.inc to include the temporary
LLVM_SIGNAL_CLEANUP_DEBUG instrumentation from the artifact branch. It launches
the ASTImporter.cpp compile command from build_nvvmb/compile_commands.json and
sends SIGUSR1 repeatedly while clang writes its temporary output file.

The debug signal cleanup intentionally removes clang's active temporary output,
so the compile is expected to fail. The interesting output is the list-state
counter lines printed by the signal handler.
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time


ROOT = os.environ.get("LLVM_PR173586_ROOT", os.getcwd())
COMPILE_COMMANDS = os.environ.get(
    "LLVM_PR173586_COMPILE_COMMANDS",
    os.path.join(ROOT, "build_nvvmb", "compile_commands.json"),
)
CLANG = os.environ.get(
    "LLVM_PR173586_CLANG", os.path.join(ROOT, "build_nvvmb", "bin", "clang-23")
)
SOURCE = os.environ.get(
    "LLVM_PR173586_SOURCE",
    os.path.join(ROOT, "clang", "lib", "AST", "ASTImporter.cpp"),
)
OUTPUT = os.environ.get("LLVM_PR173586_OUTPUT", "/tmp/pr173586-ASTImporter.o")


def load_command():
    with open(COMPILE_COMMANDS, encoding="utf-8") as f:
        commands = json.load(f)
    for entry in commands:
        if entry.get("file") != SOURCE:
            continue

        argv = shlex.split(entry["command"])
        argv[0] = CLANG
        argv.insert(1, "--driver-mode=g++")
        argv.insert(2, "--target=x86_64-unknown-linux-gnu")
        argv.insert(3, "-fintegrated-cc1")
        argv[4:4] = [
            "-isystem",
            "/usr/include/c++/15",
            "-isystem",
            "/usr/include/x86_64-linux-gnu/c++/15",
            "-isystem",
            "/usr/lib/gcc/x86_64-linux-gnu/15/include",
        ]

        for i, arg in enumerate(argv):
            if arg == "-o" and i + 1 < len(argv):
                argv[i + 1] = OUTPUT
                break
        else:
            argv.extend(["-o", OUTPUT])

        return entry["directory"], argv

    raise RuntimeError(f"could not find compile command for {SOURCE}")


def main():
    cwd, argv = load_command()
    env = os.environ.copy()
    env["LLVM_SIGNAL_CLEANUP_DEBUG"] = "1"
    try:
        os.unlink(OUTPUT)
    except FileNotFoundError:
        pass

    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    signals_sent = 0
    while proc.poll() is None and signals_sent < 300:
        time.sleep(0.1)
        if proc.poll() is not None:
            break
        try:
            proc.send_signal(signal.SIGUSR1)
            signals_sent += 1
        except ProcessLookupError:
            break

    stdout, stderr = proc.communicate(timeout=120)
    debug_lines = [
        line
        for line in stderr.splitlines()
        if line.startswith("LLVM_SIGNAL_CLEANUP_DEBUG ")
    ]
    sentinel_lines = [line for line in debug_lines if "sentinel=0" not in line]
    null_lines = [line for line in debug_lines if " null=0 " not in line]

    print(f"compile_returncode={proc.returncode}")
    print(f"signals_sent={signals_sent}")
    print(f"debug_lines={len(debug_lines)}")
    print(f"sentinel_lines={len(sentinel_lines)}")
    print(f"null_lines={len(null_lines)}")
    print("last_debug_lines:")
    for line in debug_lines[-10:]:
        print(line)

    if stdout:
        print("stdout:")
        print(stdout)

    non_debug_stderr = "\n".join(
        line
        for line in stderr.splitlines()
        if not line.startswith("LLVM_SIGNAL_CLEANUP_DEBUG ")
    )
    if non_debug_stderr:
        print("stderr:")
        print(non_debug_stderr, file=sys.stdout)

    return 0 if debug_lines else 2


if __name__ == "__main__":
    raise SystemExit(main())
