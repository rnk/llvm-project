set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(COFFOptionsTableGen)

set(LLVM_OPTIONAL_SOURCES
  DebugTypesCUDA.cu
  DebugTypesCUDAStub.cpp
  PDBSymbolRemapCUDA.cu
  PDBSymbolRemapCUDAStub.cpp
  )

option(LLD_ENABLE_COFF_GHASH_CUDA
  "Enable CUDA acceleration for COFF global hash type merging" OFF)
option(LLD_ENABLE_COFF_GHASH_HIP
  "Enable AMD HIP acceleration for COFF global hash type merging" OFF)
set(LLD_COFF_GHASH_HIP_ARCHITECTURES "" CACHE STRING
  "AMDGPU architectures for COFF HIP global hash acceleration, for example gfx1151")
set(LLD_COFF_GHASH_HIP_ROCM_ROOT "" CACHE PATH
  "ROCm installation root for COFF HIP global hash acceleration")

if(LLD_ENABLE_COFF_GHASH_CUDA AND LLD_ENABLE_COFF_GHASH_HIP)
  message(SEND_ERROR
    "LLD_ENABLE_COFF_GHASH_CUDA and LLD_ENABLE_COFF_GHASH_HIP are mutually exclusive")
endif()

set(lld_coff_gpu_backend "")
set(lld_coff_gpu_sources)
set(lld_coff_gpu_libs)

if(LLD_ENABLE_COFF_GHASH_CUDA)
  set(lld_coff_gpu_backend CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "LLD_ENABLE_COFF_GHASH_CUDA requires a working CUDA compiler")
  endif()
  find_package(CUDAToolkit REQUIRED)
  set(lld_coff_cuda_sources
    DebugTypesCUDA.cu
    PDBSymbolRemapCUDA.cu
    )
  set(lld_coff_gpu_sources ${lld_coff_cuda_sources})
  set(lld_coff_gpu_libs CUDA::cudart_static)
elseif(LLD_ENABLE_COFF_GHASH_HIP)
  set(lld_coff_gpu_backend HIP)
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(SEND_ERROR
      "LLD_ENABLE_COFF_GHASH_HIP requires CMake 3.21 or newer for HIP language support")
  endif()
  if(LLD_COFF_GHASH_HIP_ROCM_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${LLD_COFF_GHASH_HIP_ROCM_ROOT}")
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${LLD_COFF_GHASH_HIP_ROCM_ROOT}"
      CACHE PATH "ROCm root for the HIP compiler" FORCE)
  endif()
  if(NOT CMAKE_HIP_COMPILER AND
     CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_HIP_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
      "HIP compiler for COFF global hash acceleration" FORCE)
  endif()
  if(LLD_COFF_GHASH_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${LLD_COFF_GHASH_HIP_ARCHITECTURES}"
      CACHE STRING "HIP GPU architectures" FORCE)
  elseif(NOT CMAKE_HIP_ARCHITECTURES)
    find_program(LLD_AMDGPU_ARCH_EXECUTABLE amdgpu-arch)
    if(LLD_AMDGPU_ARCH_EXECUTABLE)
      execute_process(
        COMMAND "${LLD_AMDGPU_ARCH_EXECUTABLE}"
        RESULT_VARIABLE LLD_AMDGPU_ARCH_RESULT
        OUTPUT_VARIABLE LLD_AMDGPU_ARCH_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(LLD_AMDGPU_ARCH_RESULT EQUAL 0 AND LLD_AMDGPU_ARCH_OUTPUT)
        string(REGEX REPLACE "[ \t\r\n]+" ";"
          LLD_DETECTED_HIP_ARCHITECTURES "${LLD_AMDGPU_ARCH_OUTPUT}")
        list(REMOVE_DUPLICATES LLD_DETECTED_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "${LLD_DETECTED_HIP_ARCHITECTURES}"
          CACHE STRING "HIP GPU architectures" FORCE)
      endif()
    endif()
  endif()
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
  else()
    message(SEND_ERROR
      "LLD_ENABLE_COFF_GHASH_HIP requires a working HIP compiler")
  endif()
  find_package(hip REQUIRED CONFIG)
  find_package(rocthrust QUIET CONFIG)
  set(lld_coff_gpu_sources
    DebugTypesCUDA.cu
    PDBSymbolRemapCUDA.cu
    )
  set_source_files_properties(${lld_coff_gpu_sources} PROPERTIES LANGUAGE HIP)
  set(lld_coff_gpu_libs hip::amdhip64)
  set(lld_coff_gpu_include_dirs ${hip_INCLUDE_DIRS})
  if(TARGET roc::rocthrust)
    list(APPEND lld_coff_gpu_include_dirs
      $<TARGET_PROPERTY:roc::rocthrust,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
else()
  set(lld_coff_gpu_sources
    DebugTypesCUDAStub.cpp
    PDBSymbolRemapCUDAStub.cpp
    )
endif()

add_lld_library(lldCOFF
  CallGraphSort.cpp
  Chunks.cpp
  COFFLinkerContext.cpp
  DebugTypes.cpp
  ${lld_coff_gpu_sources}
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
  ${lld_coff_gpu_libs}
  ${LLVM_PTHREAD_LIB}
  ${LLVM_ATOMIC_LIB}

  DEPENDS
  COFFOptionsTableGen
  intrinsics_gen
  )

if(lld_coff_gpu_backend)
  target_compile_definitions(lldCOFF PRIVATE LLD_ENABLE_COFF_GHASH_GPU=1)
endif()

if(lld_coff_gpu_backend STREQUAL "CUDA")
  set_target_properties(lldCOFF PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_compile_definitions(lldCOFF PRIVATE
    LLD_ENABLE_COFF_GHASH_CUDA=1
    LLD_COFF_GHASH_USE_CUDA=1)
  foreach(cuda_include_dir ${CUDAToolkit_INCLUDE_DIRS})
    if(EXISTS "${cuda_include_dir}/cccl/thrust/execution_policy.h")
      target_include_directories(lldCOFF SYSTEM PRIVATE
        "${cuda_include_dir}/cccl")
    endif()
  endforeach()
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(lldCOFF PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
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
elseif(lld_coff_gpu_backend STREQUAL "HIP")
  target_compile_definitions(lldCOFF PRIVATE
    LLD_ENABLE_COFF_GHASH_HIP=1
    LLD_COFF_GHASH_USE_HIP=1
    __HIP_PLATFORM_AMD__=1)
  target_include_directories(lldCOFF SYSTEM PRIVATE
    ${lld_coff_gpu_include_dirs})
  target_compile_options(lldCOFF PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-fPIC>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:HIP>:-funwind-tables>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-rtti>)

  if(CMAKE_CXX_STANDARD)
    set_target_properties(lldCOFF PROPERTIES
      HIP_STANDARD ${CMAKE_CXX_STANDARD}
      HIP_STANDARD_REQUIRED ON)
  endif()
  if(CMAKE_HIP_ARCHITECTURES)
    set_target_properties(lldCOFF PROPERTIES
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  endif()
endif()
