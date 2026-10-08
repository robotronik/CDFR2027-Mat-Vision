cmake_minimum_required(VERSION 3.16)
project(matvision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Base de compilation pour clangd / clang-tools (build/compile_commands.json).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui calib3d objdetect aruco)
find_package(Threads REQUIRED)

add_library(matvision_core STATIC
  src/logger.cpp
  src/geometry.cpp
  src/config.cpp
  src/camera.cpp
  src/aruco.cpp
  src/calibration.cpp
  src/table.cpp
  src/overlay.cpp
  src/vision.cpp
  src/fleet.cpp
  src/api.cpp
)
target_include_directories(matvision_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(matvision_core PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_definitions(matvision_core PUBLIC MATVISION_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(matvision_core PRIVATE -Wall -Wextra)

add_executable(matvision-server src/main_server.cpp)
target_link_libraries(matvision-server PRIVATE matvision_core)

add_executable(matvision-calibrate src/main_calibrate.cpp)
target_link_libraries(matvision-calibrate PRIVATE matvision_core)

enable_testing()
add_executable(matvision-tests tests/test_main.cpp)
target_link_libraries(matvision-tests PRIVATE matvision_core)
add_test(NAME smoke COMMAND matvision-tests)
