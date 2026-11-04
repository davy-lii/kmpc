# 生产求解器走 ProxSuite（ProxQP）。系统无 config 时由本文件找头文件。
# qpmad 仅当 ProxQP 编不过时备选（同一 solve_box_ineq 接口，C4 再接）。

set(ProxQP_FOUND FALSE)
set(ProxQP_VERSION "")

find_package(proxsuite QUIET CONFIG)
if(proxsuite_FOUND)
  set(ProxQP_FOUND TRUE)
  if(DEFINED proxsuite_VERSION)
    set(ProxQP_VERSION "${proxsuite_VERSION}")
  endif()
  if(NOT TARGET kmpc::proxqp)
    if(TARGET proxsuite::proxsuite)
      add_library(kmpc::proxqp INTERFACE IMPORTED)
      target_link_libraries(kmpc::proxqp INTERFACE proxsuite::proxsuite)
    elseif(TARGET proxsuite)
      add_library(kmpc::proxqp INTERFACE IMPORTED)
      target_link_libraries(kmpc::proxqp INTERFACE proxsuite)
    endif()
  endif()
else()
  find_path(PROXSUITE_INCLUDE_DIR
    NAMES proxsuite/proxqp/dense/solver.hpp
    PATHS
      /usr/include
      /usr/local/include
      /opt/ros/humble/include
  )
  if(PROXSUITE_INCLUDE_DIR)
    set(ProxQP_FOUND TRUE)
    add_library(kmpc::proxqp INTERFACE IMPORTED)
    target_include_directories(kmpc::proxqp INTERFACE "${PROXSUITE_INCLUDE_DIR}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ProxQP
  REQUIRED_VARS ProxQP_FOUND
)
