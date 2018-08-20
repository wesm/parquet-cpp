# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/.parquetcppversion" PARQUET_VERSION)
string(REPLACE "\n" "" PARQUET_VERSION "${PARQUET_VERSION}")
string(REGEX MATCH "^([0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?)" VERSION ${PARQUET_VERSION})
if(NOT VERSION)
  message(FATAL_ERROR "invalid .parquetcppversion")
endif()

function(ADD_PARQUET_TEST REL_TEST_NAME)
  set(options NO_VALGRIND)
  set(one_value_args)
  set(multi_value_args EXTRA_DEPENDENCIES LABELS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Error: unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  ADD_ARROW_TEST({REL_TEST_NAME}
    STATIC_LINK_LIBS parquet_static
    LABELS "parquet")
endfunction()

# TODO(wesm): Handling of ABI/SO version

if (ARROW_BOOST_USE_SHARED)
  set(PARQUET_BOOST_LINK_LIBS
    boost_shared_regex)
  if(MSVC)
    set(PARQUET_BOOST_LINK_LIBS ${PARQUET_BOOST_LINK_LIBS}
      boost_shared_system)
  endif()
else()
  set(PARQUET_BOOST_LINK_LIBS
    boost_static_regex)
  if(MSVC)
    set(PARQUET_BOOST_LINK_LIBS ${PARQUET_BOOST_LINK_LIBS}
      boost_static_system boost_static_filesystem)
  endif()
endif()

set(PARQUET_TEST_LINK_LIBS
  ${ARROW_TEST_LINK_LIBS}
  parquet_static)

set(PARQUET_TEST_SHARED_LINK_LIBS
  ${ARROW_TEST_LINK_LIBS}
  parquet_shared)

#############################################################
# Benchmark linking

if (PARQUET_BUILD_STATIC)
  set(PARQUET_BENCHMARK_LINK_LIBS
    parquet_benchmark_main
    parquet_static)
else()
  set(PARQUET_BENCHMARK_LINK_LIBS
    parquet_benchmark_main
    parquet_shared)
endif()

############################################################
# Generated Thrift sources

if (NOT MSVC)
  set_source_files_properties(src/parquet/parquet_types.cpp PROPERTIES
    COMPILE_FLAGS -Wno-unused-variable)
endif()

# List of thrift output targets
set(THRIFT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(THRIFT_OUTPUT_FILES "${THRIFT_OUTPUT_DIR}/parquet_types.cpp")
set(THRIFT_OUTPUT_FILES ${THRIFT_OUTPUT_FILES} "${THRIFT_OUTPUT_DIR}/parquet_types.h")
set(THRIFT_OUTPUT_FILES ${THRIFT_OUTPUT_FILES} "${THRIFT_OUTPUT_DIR}/parquet_constants.cpp")
set(THRIFT_OUTPUT_FILES ${THRIFT_OUTPUT_FILES} "${THRIFT_OUTPUT_DIR}/parquet_constants.h")

set_source_files_properties(${THRIFT_OUTPUT_FILES} PROPERTIES GENERATED TRUE)

get_filename_component(ABS_PARQUET_THRIFT parquet.thrift ABSOLUTE)

add_custom_command(
  OUTPUT ${THRIFT_OUTPUT_FILES}
  COMMAND ${THRIFT_COMPILER} --gen cpp -out ${THRIFT_OUTPUT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/parquet.thrift
  DEPENDS ${ABS_PARQUET_THRIFT} thriftstatic
  COMMENT "Running thrift compiler on parquet.thrift"
  VERBATIM
)

############################################################
# Library config

set(LIBPARQUET_SRCS
  arrow/reader.cc
  arrow/record_reader.cc
  arrow/schema.cc
  arrow/writer.cc
  bloom_filter.cc
  column_reader.cc
  column_scanner.cc
  column_writer.cc
  exception.cc
  file_reader.cc
  file_writer.cc
  metadata.cc
  murmur3.cc
  parquet_constants.cpp
  parquet_types.cpp
  printer.cc
  schema.cc
  statistics.cc
  types.cc
  util/comparison.cc
  util/memory.cc
)

# # Ensure that thrift compilation is done before using its generated headers
# # in parquet code.
add_custom_target(parquet-thrift-deps ALL
  DEPENDS ${THRIFT_OUTPUT_FILES})
set(PARQUET_DEPENDENCIES ${PARQUET_DEPENDENCIES} parquet-thrift-deps)

ADD_ARROW_LIB(parquet
  SOURCES ${LIBPARQUET_SRCS}
  DEPENDENCIES ${PARQUET_DEPENDENCIES}
  SHARED_LINK_FLAGS ${PARQUET_SHARED_LINK_FLAGS}
  SHARED_PRIVATE_LINK_LIBS ${PARQUET_INTERFACE_LINK_LIBS}
  STATIC_LINK_LIBS ${PARQUET_INTERFACE_LINK_LIBS}
)

if (NOT PARQUET_MINIMAL_DEPENDENCY)
# These are libraries that we will link privately with parquet_shared (as they
# do not need to be linked transitively by other linkers), but publicly with
# parquet_static (because internal users need to transitively link all
# dependencies)
  set(LIBPARQUET_INTERFACE_LINK_LIBS
    ${ARROW_LINK_LIBS}
    ${PARQUET_BOOST_LINK_LIBS}
    thriftstatic
  )
# Although we don't link parquet_objlib against anything, we need it to depend
# on these libs as we may generate their headers via ExternalProject_Add
  set(PARQUET_DEPENDENCIES ${PARQUET_DEPENDENCIES} ${LIBPARQUET_INTERFACE_LINK_LIBS})
endif()

if(NOT APPLE AND NOT MSVC)
  # Localize thirdparty symbols using a linker version script. This hides them
  # from the client application. The OS X linker does not support the
  # version-script option.
  set(SHARED_LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/parquet/symbols.map")
endif()

add_subdirectory(api)
add_subdirectory(arrow)
add_subdirectory(util)

# Headers: top level
install(FILES
  bloom_filter.h
  column_reader.h
  column_page.h
  column_scanner.h
  column_writer.h
  encoding.h
  exception.h
  file_reader.h
  file_writer.h
  hasher.h
  metadata.h
  murmur3.h
  printer.h
  properties.h
  schema.h
  statistics.h
  types.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parquet")

configure_file(parquet_version.h.in
  "${CMAKE_CURRENT_BINARY_DIR}/parquet_version.h"
  @ONLY)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/parquet_version.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parquet")

# pkg-config support
configure_file(parquet.pc.in
  "${CMAKE_CURRENT_BINARY_DIR}/parquet.pc"
  @ONLY)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/parquet.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

ADD_PARQUET_TEST(bloom_filter-test)
ADD_PARQUET_TEST(column_reader-test)
ADD_PARQUET_TEST(column_scanner-test)
ADD_PARQUET_TEST(column_writer-test)
ADD_PARQUET_TEST(file-deserialize-test)
ADD_PARQUET_TEST(file-serialize-test)
ADD_PARQUET_TEST(properties-test)
ADD_PARQUET_TEST(statistics-test)
ADD_PARQUET_TEST(encoding-test)
ADD_PARQUET_TEST(metadata-test)
ADD_PARQUET_TEST(public-api-test)
ADD_PARQUET_TEST(types-test)
ADD_PARQUET_TEST(reader-test)
ADD_PARQUET_TEST(schema-test)

ADD_ARROW_BENCHMARK(column-io-benchmark)
ARROW_BENCHMARK_LINK_LIBRARIES(column-io-benchmark
  parquet_static)

ADD_ARROW_BENCHMARK(encoding-benchmark)
ARROW_BENCHMARK_LINK_LIBRARIES(encoding-benchmark
  parquet_static)
