set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(COFFOptionsTableGen)

set(LLVM_OPTIONAL_SOURCES
  DebugTypesCUDA.cu
  DebugTypesCUDAStub.cpp
  )

option(LLD_ENABLE_COFF_GHASH_CUDA
  "Enable CUDA acceleration for COFF global hash type merging" OFF)

if(LLD_ENABLE_COFF_GHASH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "LLD_ENABLE_COFF_GHASH_CUDA requires a working CUDA compiler")
  endif()
  find_package(CUDAToolkit REQUIRED)
  set(lld_coff_cuda_sources DebugTypesCUDA.cu)
  set(lld_coff_cuda_libs CUDA::cudart_static)
else()
  set(lld_coff_cuda_sources DebugTypesCUDAStub.cpp)
endif()

add_lld_library(lldCOFF
  CallGraphSort.cpp
  Chunks.cpp
  COFFLinkerContext.cpp
  DebugTypes.cpp
  ${lld_coff_cuda_sources}
  DLL.cpp
  Driver.cpp
  DriverUtils.cpp
  ICF.cpp
  InputFiles.cpp
  LLDMapFile.cpp
  LTO.cpp
  MapFile.cpp
  MarkLive.cpp
  MinGW.cpp
  PDB.cpp
  SymbolTable.cpp
  Symbols.cpp
  Writer.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BinaryFormat
  BitWriter
  Core
  DebugInfoCodeView
  DebugInfoDWARF
  DebugInfoMSF
  DebugInfoPDB
  Demangle
  DTLTO
  LibDriver
  LTO
  MC
  Object
  Option
  Passes
  Support
  TargetParser
  WindowsDriver
  WindowsManifest

  LINK_LIBS
  lldCommon
  ${lld_coff_cuda_libs}
  ${LLVM_PTHREAD_LIB}
  ${LLVM_ATOMIC_LIB}

  DEPENDS
  COFFOptionsTableGen
  intrinsics_gen
  )

if(LLD_ENABLE_COFF_GHASH_CUDA)
  set_target_properties(lldCOFF PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(lldCOFF PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-exceptions>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-funwind-tables>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-rtti>)
  elseif(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    target_compile_options(lldCOFF PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-fno-exceptions>
      $<$<COMPILE_LANGUAGE:CUDA>:-funwind-tables>
      $<$<COMPILE_LANGUAGE:CUDA>:-fno-rtti>)
  endif()

  if(CMAKE_CXX_STANDARD)
    set_target_properties(lldCOFF PROPERTIES
      CUDA_STANDARD ${CMAKE_CXX_STANDARD}
      CUDA_STANDARD_REQUIRED ON)
  endif()
endif()
