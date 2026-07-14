cmake_minimum_required(VERSION 3.11)

file(READ "VERSION" version)
string(STRIP "${version}" version)
# For dev builds, the build number is appended as the tweak component (e.g. "1.1.4.42")
if(NOT "$ENV{FRANKY_DEV_BUILD_NUMBER}" STREQUAL "")
    set(version "${version}.$ENV{FRANKY_DEV_BUILD_NUMBER}")
endif()
project(franky VERSION "${version}" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include(GNUInstallDirs)

option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON_MODULE "Build python module" ON)
option(BUILD_PYTHON_STUBS "Build stubs" ON)
option(BUILD_TESTS "Build tests" ON)

find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package(Franka REQUIRED)

set(FRANKA_VERSION_DEFINES "")
foreach (franka_min_version 0.8 0.9 0.10 0.15 0.18)
    if (${Franka_VERSION} VERSION_GREATER_EQUAL "${franka_min_version}.0")
        string(REPLACE "." "_" franka_min_version_str "${franka_min_version}")
        list(APPEND FRANKA_VERSION_DEFINES "FRANKA_${franka_min_version_str}")
    endif ()
endforeach ()
add_compile_definitions(${FRANKA_VERSION_DEFINES})

message("Found Eigen Version: ${Eigen3_VERSION}")
message("Found Franka Version: ${Franka_VERSION}")

find_package(ruckig 0.17.3 REQUIRED)

set(FRANKY_PREBUILT_LIB "" CACHE FILEPATH "Path to a pre-built libfranky.so; skips rebuilding the core library")

if (FRANKY_PREBUILT_LIB)
    add_library(franky SHARED IMPORTED GLOBAL)
    set_target_properties(franky PROPERTIES
        IMPORTED_LOCATION "${FRANKY_PREBUILT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
        INTERFACE_LINK_LIBRARIES "Franka::Franka;Eigen3::Eigen;ruckig::ruckig"
    )
else ()
    file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
    add_library(franky SHARED ${SRC})
    target_include_directories(franky PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(franky PUBLIC Franka::Franka Eigen3::Eigen ruckig::ruckig)
endif ()


if (BUILD_PYTHON_MODULE)
    if (NOT DEFINED PYTHON_EXECUTABLE)
        find_package(Python3 COMPONENTS Interpreter Development)
        set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    endif ()

    # Check if pybind11 exists as a subdirectory
    if (EXISTS pybind11)
        add_subdirectory(pybind11)
    else ()
        find_package(pybind11 2.6 REQUIRED)
    endif ()

    file(GLOB_RECURSE PYTHON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp")
    pybind11_add_module(_franky-lib ${PYTHON_SRC})
    target_link_libraries(_franky-lib PUBLIC franky)
    set_target_properties(_franky-lib PROPERTIES OUTPUT_NAME "_franky")

    # Regenerates python/docstrings.hpp (which is checked in) from the doxygen comments in the C++
    # headers. Not part of the regular build, as it requires pybind11_mkdoc and libclang; run it
    # after changing doc comments in the headers. It uses the include directories and version
    # defines of the actual build, so the extracted libfranka docstrings match the libfranka
    # version franky is built against.
    set(DOCSTRING_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    foreach (dependency Franka::Franka Eigen3::Eigen ruckig::ruckig)
        get_target_property(dependency_include_dirs ${dependency} INTERFACE_INCLUDE_DIRECTORIES)
        if (dependency_include_dirs)
            list(APPEND DOCSTRING_INCLUDE_DIRS ${dependency_include_dirs})
        endif ()
    endforeach ()
    add_custom_target(franky-docstrings
            ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_docstrings.py"
            --output "${CMAKE_CURRENT_SOURCE_DIR}/python/docstrings.hpp"
            --include-dirs ${DOCSTRING_INCLUDE_DIRS}
            --defines ${FRANKA_VERSION_DEFINES}
            --std "c++${CMAKE_CXX_STANDARD}"
            COMMENT "Generating Python docstrings from the C++ headers..."
            COMMAND_EXPAND_LISTS
            VERBATIM)

    if (BUILD_PYTHON_STUBS)
        # Build stubs
        add_custom_target(
                franky-stubs
                ${CMAKE_SOURCE_DIR}/generate_stubs.bash "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}" "$<TARGET_FILE:_franky-lib>"
                COMMENT "Generating Python stubs..."
                DEPENDS _franky-lib
        )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/stubs/_franky-stubs/__init__.pyi DESTINATION lib OPTIONAL)

        add_custom_target(_franky
                DEPENDS _franky-lib
                DEPENDS franky-stubs)
    else ()
        add_custom_target(_franky
                DEPENDS _franky-lib)
    endif ()
endif ()


if (BUILD_EXAMPLES)
    foreach (example IN ITEMS linear)
        add_executable(${example} "examples/${example}.cpp")
        target_link_libraries(${example} PRIVATE franky)
    endforeach ()
endif ()

if (BUILD_TESTS)
    enable_testing()

    find_package(Catch2 REQUIRED)

    file(GLOB TEST_FILES "test/test-*.cpp")

    foreach (test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE franky Catch2::Catch2)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach ()
endif ()


if (NOT FRANKY_PREBUILT_LIB)
    install(TARGETS franky
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
    )
endif ()
