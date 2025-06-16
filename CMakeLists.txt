cmake_minimum_required(VERSION 3.11)

file(READ "VERSION" version)
string(STRIP "${version}" version)
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

if (${Franka_VERSION} VERSION_GREATER_EQUAL "0.8.0")
    add_compile_definitions(FRANKA_0_8)
endif ()

if (${Franka_VERSION} VERSION_GREATER_EQUAL "0.9.0")
    add_compile_definitions(FRANKA_0_9)
endif ()

if (${Franka_VERSION} VERSION_GREATER_EQUAL "0.10.0")
    add_compile_definitions(FRANKA_0_10)
endif ()

if (${Franka_VERSION} VERSION_GREATER_EQUAL "0.15.0")
    add_compile_definitions(FRANKA_0_15)
endif ()

message("Found Eigen Version: ${Eigen3_VERSION}")
message("Found Franka Version: ${Franka_VERSION}")

# Disabling examples and python modules of ruckig
set(BUILD_EXAMPLES_TMP ${BUILD_EXAMPLES})
set(BUILD_EXAMPLES OFF)
set(BUILD_PYTHON_MODULE_TMP ${BUILD_PYTHON_MODULE})
set(BUILD_PYTHON_MODULE OFF)
add_subdirectory(ruckig)
set(BUILD_EXAMPLES ${BUILD_EXAMPLES_TMP})
set(BUILD_PYTHON_MODULE ${BUILD_PYTHON_MODULE_TMP})

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(franky ${SRC})
target_include_directories(franky PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(franky PUBLIC Franka::Franka Eigen3::Eigen ruckig)


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


install(TARGETS franky
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)
