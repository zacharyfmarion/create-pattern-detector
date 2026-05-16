cmake_minimum_required(VERSION 3.15)
project(TreeMakerHeadless LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TREEMAKER_SOURCE_DIR "" CACHE PATH "Path to an external TreeMaker checkout's src directory")
if(NOT TREEMAKER_SOURCE_DIR)
  message(FATAL_ERROR "Set -DTREEMAKER_SOURCE_DIR=/path/to/treemaker/src")
endif()

set(TM_SOURCE "${TREEMAKER_SOURCE_DIR}/Source")
if(NOT EXISTS "${TM_SOURCE}/tmModel/tmModel.h")
  message(FATAL_ERROR "TREEMAKER_SOURCE_DIR must point at the legacy checkout's src directory")
endif()

file(GLOB_RECURSE MODEL_CPP "${TM_SOURCE}/tmModel/*.cpp")
file(GLOB_RECURSE MODEL_C "${TM_SOURCE}/tmModel/wnlib/*.c")

add_executable(
  treemaker-json-cli
  treemaker-json-cli.cpp
  compat/wn_compat.c
  ${MODEL_CPP}
  ${MODEL_C}
  "${TM_SOURCE}/tmHeader.cpp"
)

target_include_directories(treemaker-json-cli PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${TM_SOURCE}"
  "${TM_SOURCE}/tmModel"
  "${TM_SOURCE}/tmModel/tmNLCO"
  "${TM_SOURCE}/tmModel/tmSolvers"
  "${TM_SOURCE}/tmModel/tmOptimizers"
  "${TM_SOURCE}/tmModel/tmTreeClasses"
  "${TM_SOURCE}/tmModel/tmPtrClasses"
  "${TM_SOURCE}/tmModel/wnlib"
  "${TM_SOURCE}/tmModel/wnlib/cmp"
  "${TM_SOURCE}/tmModel/wnlib/conjdir"
  "${TM_SOURCE}/tmModel/wnlib/cpy"
  "${TM_SOURCE}/tmModel/wnlib/list"
  "${TM_SOURCE}/tmModel/wnlib/low"
  "${TM_SOURCE}/tmModel/wnlib/mat"
  "${TM_SOURCE}/tmModel/wnlib/mem"
  "${TM_SOURCE}/tmModel/wnlib/random"
  "${TM_SOURCE}/tmModel/wnlib/vect"
)

target_compile_options(treemaker-json-cli PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-Wno-error=incompatible-function-pointer-types>
)
