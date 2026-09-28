cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# One .cpp (the MLX primitive and its dispatch) and one .metal (the kernels) per op.
set(OPS rms_norm swiglu rope decode_attention flash_prefill paged_attention kv_quantize)

# ----------------------------- Library -----------------------------
add_library(mini_vllm_ext)
list(TRANSFORM OPS PREPEND ${CMAKE_CURRENT_LIST_DIR}/src/ OUTPUT_VARIABLE OP_SOURCES)
list(TRANSFORM OP_SOURCES APPEND .cpp)
target_sources(mini_vllm_ext PRIVATE ${OP_SOURCES} ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp)
target_include_directories(mini_vllm_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(mini_vllm_ext PUBLIC mlx)

# ----------------------------- Metal -----------------------------
if(MLX_BUILD_METAL)
  list(TRANSFORM OPS PREPEND ${CMAKE_CURRENT_LIST_DIR}/src/ OUTPUT_VARIABLE METAL_SOURCES)
  list(TRANSFORM METAL_SOURCES APPEND .metal)
  mlx_build_metallib(
    TARGET mini_vllm_ext_metallib
    TITLE mini_vllm_ext
    SOURCES ${METAL_SOURCES}
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(mini_vllm_ext mini_vllm_ext_metallib)
endif()

# ----------------------------- Python bindings -----------------------------
nanobind_add_module(_ext NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mini_vllm_ext)
if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
