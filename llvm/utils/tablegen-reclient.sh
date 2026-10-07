#!/usr/bin/env bash
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# LLVM_TABLEGEN_LAUNCHER for reclient. TableGen.cmake prefixes every tablegen()
# custom command with it:
#   tablegen-reclient.sh <tblgen> [opts] -I<dir>... <file.td> -o <out> -d <depfile>
# and it runs the command through rewrapper, strictly remote (no local fallback).
# Inputs are derived from the
# command line:
#   inputs  = tblgen binary + the .td file + every *.td under each -I dir
#   outputs = -o file, -d depfile, and for -gen-register-info the four
#             <stem>{Enums,Header,MCDesc,TargetDesc}.inc that tablegen() also declares
# The tblgen binary must run on the workers; use a static binary if they have no libc/loader.
# Env: RBE_ACTIVE, RBE_DIR, RBE_EXEC_ROOT. Unset RBE_ACTIVE => passthrough.
# RBE_REWRAPPER overrides the rewrapper executable (default: rewrapper from PATH).
# Debug: TBLGEN_LAUNCH_LOG=<file> appends "building <td>"; TBLGEN_NOCACHE=1 skips the remote cache.
cmd=("$@")
exe=$1
inc=(); td=; out=; dep=; regi=0
argv=("$@"); i=1
while [ $i -lt ${#argv[@]} ]; do
  a=${argv[$i]}
  case "$a" in
    -I) i=$((i+1)); inc+=("${argv[$i]}");;
    -I*) inc+=("${a#-I}");;
    -o) i=$((i+1)); out=${argv[$i]};;
    -d) i=$((i+1)); dep=${argv[$i]};;
    -gen-register-info) regi=1;;
    *.td) td=$a;;
  esac
  i=$((i+1))
done

[ -n "$TBLGEN_LAUNCH_LOG" ] && echo "building $(basename "$td")" >> "$TBLGEN_LAUNCH_LOG"
[ -n "${RBE_ACTIVE:-}" ] && [ -n "$out" ] || exec "${cmd[@]}"

root=$RBE_EXEC_ROOT
rel() { local p; p=$(realpath -m "$1"); echo "${p#"$root"/}"; }

# .td closure per -I dir, cached for the build (one find per dir, not per action).
cache=$root/.tblgen-td-cache; mkdir -p "$cache"
list=$(mktemp "$cache/inputs.XXXXXX")
{
  rel "$exe"
  rel "$td"
  for d in "${inc[@]}"; do
    f="$cache/$(echo "$d" | tr / _)"
    [ -f "$f" ] || { find "$d" -name '*.td' > "$f.$$" && mv "$f.$$" "$f"; }
    while read -r p; do rel "$p"; done < "$f"
  done
} | sort -u > "$list"

outs=("$(rel "$out")")
[ -n "$dep" ] && outs+=("$(rel "$dep")")
if [ $regi = 1 ]; then
  s=$(rel "${out%.*}")
  for x in Enums Header MCDesc TargetDesc; do outs+=("$s$x.inc"); done
fi
outfiles=$(IFS=,; echo "${outs[*]}")

"${RBE_REWRAPPER:-rewrapper}" \
  -cfg="$RBE_DIR/rewrapper.cfg" \
  -exec_root="$root" \
  -exec_strategy=remote \
  -labels=type=tool \
  ${TBLGEN_NOCACHE:+-remote_accept_cache=false} \
  -input_list_paths="$list" \
  -output_files="$outfiles" \
  -- "${cmd[@]}"
rc=$?
rm -f "$list"
exit $rc
