#!/bin/sh
#
# Run clang-tidy in fix mode and verify the result.
# Usage:
#   check_clang_tidy_fix.sh <source-file> <check-name> <temp-file> \
#     [optional clang-tidy arguments]
#
# Example:
#   // RUN: $(dirname %s)/check_clang_tidy_fix.sh %s llvm-include-order %t -- -isystem $(dirname %s)/Inputs/Headers
#   // REQUIRES: shell

INPUT_FILE=$1
CHECK_TO_RUN=$2
TEMPORARY_FILE=$3.cpp
# Feed the rest arguments to clang-tidy.
shift 3
if [ "$#" -eq 0 ] ; then
  # Default to -- --std=c++11
  set - -- --std=c++11
fi

set -o errexit

# Remove the contents of the CHECK lines to avoid CHECKs matching on themselves.
# We need to keep the comments to preserve line numbers while avoiding empty
# lines which could potentially trigger formatting-related checks.
sed 's#// *CHECK-[A-Z-]*:.*#//#' ${INPUT_FILE} > ${TEMPORARY_FILE}

clang-tidy ${TEMPORARY_FILE} -fix --checks="-*,${CHECK_TO_RUN}" "$@" \
  > ${TEMPORARY_FILE}.msg 2>&1

FileCheck -input-file=${TEMPORARY_FILE} ${INPUT_FILE} \
  -check-prefix=CHECK-FIXES -strict-whitespace

if grep -q CHECK-MESSAGES ${INPUT_FILE}; then
  FileCheck -input-file=${TEMPORARY_FILE}.msg ${INPUT_FILE} \
    -check-prefix=CHECK-MESSAGES -implicit-check-not="{{warning|error}}:"
fi
