cmake_minimum_required(VERSION 3.16)
project(kmpc_cpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
option(KMPC_PIN_VERSIONS
  "Eigen/Pinocchio/ProxSuite 版本对不上则 FATAL（Thor 镜像可 OFF）" ON)
include(cmake/versions.cmake)

set(CMAKE_CXX_STANDARD ${KMPC_REQUIRED_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(KMPC_REQUIRE_PINOCCHIO "C2 起必须找到 Pinocchio" ON)
option(KMPC_REQUIRE_PROXSUITE "C4 起必须找到 ProxSuite" ON)
if(NOT DEFINED BUILD_TESTING)
  set(BUILD_TESTING ON)
endif()

if(EXISTS "/opt/ros/humble")
  list(APPEND CMAKE_PREFIX_PATH "/opt/ros/humble")
endif()
# 不要把笔记本 x86 的 install/xoip_ros2 拷上 Thor。容器里 source aarch64 overlay 即可。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../install/xoip_ros2")
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../install/xoip_ros2")
endif()

find_package(Eigen3 ${KMPC_PINNED_EIGEN_VERSION} REQUIRED NO_MODULE)
kmpc_check_eigen_version()

find_package(yaml-cpp REQUIRED)
kmpc_check_yaml_cpp_version()
if(TARGET yaml-cpp::yaml-cpp)
  set(KMPC_YAML_TARGET yaml-cpp::yaml-cpp)
else()
  set(KMPC_YAML_TARGET yaml-cpp)
endif()

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  enable_testing()
  include(GoogleTest)
endif()

if(KMPC_PIN_VERSIONS)
  find_package(pinocchio ${KMPC_PINNED_PINOCCHIO_VERSION} REQUIRED)
else()
  # Thor 镜像是 3.9.x；-DKMPC_PIN_VERSIONS=OFF 时不要卡死在 4.0.0。
  find_package(pinocchio REQUIRED)
endif()
kmpc_check_pinocchio_version()
if(KMPC_REQUIRE_PINOCCHIO AND NOT pinocchio_FOUND)
  message(FATAL_ERROR
    "Pinocchio not found. C2 需要对拍 FK/J。安装：sudo apt install ros-humble-pinocchio")
endif()

find_package(ProxQP REQUIRED)
kmpc_check_proxsuite_version()
if(KMPC_REQUIRE_PROXSUITE AND NOT ProxQP_FOUND)
  message(FATAL_ERROR
    "ProxSuite not found. C4 需要 ProxQP。安装：sudo apt install ros-humble-proxsuite")
endif()

set(KMPC_PY_SHARE "${CMAKE_CURRENT_SOURCE_DIR}/../whole_body_kmpc/src/whole_body_kmpc")
if(NOT EXISTS "${KMPC_PY_SHARE}/config/joint_limits.yaml")
  # 源码在 bags/kmpc_cpp 时，金标在仓库根的 whole_body_kmpc。
  set(KMPC_PY_SHARE "${CMAKE_CURRENT_SOURCE_DIR}/../../whole_body_kmpc/src/whole_body_kmpc")
endif()
if(NOT EXISTS "${KMPC_PY_SHARE}/config/joint_limits.yaml")
  message(FATAL_ERROR "Python 金标配置不存在: ${KMPC_PY_SHARE}/config")
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  file(CREATE_LINK "${KMPC_PY_SHARE}/config" "${CMAKE_CURRENT_SOURCE_DIR}/config" SYMBOLIC)
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/urdf")
  file(CREATE_LINK "${KMPC_PY_SHARE}/urdf" "${CMAKE_CURRENT_SOURCE_DIR}/urdf" SYMBOLIC)
endif()

add_library(kmpc_core
  src/core/config.cpp
  src/core/constraints.cpp
  src/core/horizon_qp.cpp
  src/core/kinematics.cpp
  src/core/safety.cpp
  src/core/whole_body_guards.cpp
  src/core/joint_space_tracker.cpp
  src/solver/qp_solver.cpp
  src/tracker/cartesian_tracker.cpp
  src/tracker/line_path.cpp
  src/tracker/line_qp.cpp
  src/planner/upper_planner.cpp
)
target_include_directories(kmpc_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(kmpc_core PUBLIC
  Eigen3::Eigen
  ${KMPC_YAML_TARGET}
  pinocchio::pinocchio
  kmpc::proxqp
)
target_compile_options(kmpc_core PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(kmpc_core PUBLIC
  KMPC_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
  KMPC_URDF_DIR="${CMAKE_CURRENT_SOURCE_DIR}/urdf"
  KMPC_GOLD_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/gold"
  KMPC_REPLAY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/replay"
)

function(kmpc_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE kmpc_core GTest::gtest GTest::gtest_main)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Werror)
  gtest_discover_tests(${name} TEST_PREFIX "${name}.")
endfunction()

# 插补器不进 kmpc_core，单测不链 Pinocchio。
function(kmpc_add_interp_test)
  add_executable(test_stream_interpolator
    interpolator/test_stream_interpolator.cpp)
  target_include_directories(test_stream_interpolator PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/interpolator/include")
  target_link_libraries(test_stream_interpolator PRIVATE
    Eigen3::Eigen GTest::gtest GTest::gtest_main)
  target_compile_options(test_stream_interpolator PRIVATE -Wall -Wextra -Werror)
  gtest_discover_tests(test_stream_interpolator TEST_PREFIX "test_stream_interpolator.")
endfunction()

if(BUILD_TESTING)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_sanity.cpp")
    kmpc_add_test(test_sanity tests/unit/test_sanity.cpp)
    kmpc_add_test(test_types tests/unit/test_types.cpp)
    kmpc_add_test(test_config tests/unit/test_config.cpp)
    kmpc_add_test(test_constraints tests/unit/test_constraints.cpp)
    kmpc_add_test(test_safety tests/unit/test_safety.cpp)
    kmpc_add_test(test_kinematics tests/unit/test_kinematics.cpp)
    kmpc_add_test(test_kinematics_gold tests/gold/test_kinematics_gold.cpp)
    kmpc_add_test(test_horizon_qp tests/unit/test_horizon_qp.cpp)
    kmpc_add_test(test_qp_solver tests/unit/test_qp_solver.cpp)
    kmpc_add_test(test_qp_gold tests/gold/test_qp_gold.cpp)
    kmpc_add_test(test_tracker tests/unit/test_tracker.cpp)
    kmpc_add_test(test_tracker_gold tests/gold/test_tracker_gold.cpp)
    kmpc_add_test(test_planner tests/unit/test_planner.cpp)
    kmpc_add_test(test_planner_gold tests/gold/test_planner_gold.cpp)
    kmpc_add_test(test_replay tests/replay/test_replay.cpp)
    kmpc_add_test(test_node_cache tests/unit/test_node_cache.cpp)
    target_include_directories(test_node_cache PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ros2")
  endif()
  kmpc_add_interp_test()
  kmpc_add_test(test_line_path tests/unit/test_line_path.cpp)
  kmpc_add_test(test_layer_diag tests/unit/test_layer_diag.cpp)
  target_include_directories(test_layer_diag PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ros2")
  kmpc_add_test(test_line_segment tests/unit/test_line_segment.cpp)
  target_include_directories(test_line_segment PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ros2")
  kmpc_add_test(test_alarm_resume tests/unit/test_alarm_resume.cpp)
  target_include_directories(test_alarm_resume PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ros2")
  add_executable(replay_recent tests/replay/replay_recent.cpp)
  target_link_libraries(replay_recent PRIVATE kmpc_core)
  add_executable(probe_field_line tests/replay/probe_field_line.cpp)
  target_link_libraries(probe_field_line PRIVATE kmpc_core)
endif()

# Thor colcon 常关 BUILD_TESTING；插补单测仍要能编、能 ctest。
if(NOT TARGET test_stream_interpolator)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    include(GoogleTest)
    kmpc_add_interp_test()
  endif()
endif()

option(KMPC_BUILD_ROS2 "C8：编 rclcpp kmpc_node" ON)

if(KMPC_BUILD_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(xoip_ros2 QUIET)

  add_executable(kmpc_node ros2/kmpc_node.cpp)
  target_include_directories(kmpc_node PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/ros2"
    "${CMAKE_CURRENT_SOURCE_DIR}/interpolator/include")
  target_compile_options(kmpc_node PRIVATE -Wall -Wextra -Werror)
  ament_target_dependencies(kmpc_node
    rclcpp std_msgs sensor_msgs geometry_msgs visualization_msgs)
  target_link_libraries(kmpc_node kmpc_core)
  if(xoip_ros2_FOUND)
    ament_target_dependencies(kmpc_node xoip_ros2)
    target_compile_definitions(kmpc_node PRIVATE KMPC_HAVE_XOIP=1)
  else()
    message(WARNING "xoip_ros2 not found; kmpc_node 只发 /kmpc/status，不发 stream")
  endif()

  install(TARGETS kmpc_node DESTINATION lib/${PROJECT_NAME})
  install(FILES ros2/launch_common.py DESTINATION share/${PROJECT_NAME}/launch)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sim/launch")
    install(DIRECTORY sim/launch/ DESTINATION share/${PROJECT_NAME}/launch)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/real/launch")
    install(DIRECTORY real/launch/ DESTINATION share/${PROJECT_NAME}/launch)
  endif()
  ament_package()
endif()

message(STATUS "kmpc_cpp Eigen=${Eigen3_VERSION} yaml-cpp target=${KMPC_YAML_TARGET}")
message(STATUS "kmpc_cpp Pinocchio=${pinocchio_VERSION} ProxQP found=${ProxQP_FOUND}")
message(STATUS "kmpc_cpp ROS2 node=${KMPC_BUILD_ROS2} xoip=${xoip_ros2_FOUND}")
