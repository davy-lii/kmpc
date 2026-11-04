# 依赖版本就是金标环境的一部分。换机器导致 J 漂 1e-12 不算算法错。
# 主版本对不上 → FATAL_ERROR。Pinocchio / ProxSuite 在本机安装后再钉 x.y.z。
# Thor 镜像若次版本对不上：-DKMPC_PIN_VERSIONS=OFF（只警告，仍用板上的库）。

set(KMPC_REQUIRED_CXX_STANDARD 17)

if(NOT DEFINED KMPC_PIN_VERSIONS)
  set(KMPC_PIN_VERSIONS ON)
endif()

function(kmpc_version_mismatch pkg found pinned)
  if(KMPC_PIN_VERSIONS)
    message(FATAL_ERROR
      "${pkg} ${found} != pinned ${pinned}. "
      "金标环境不允许换主/次版本。Thor 镜像差次版本可 -DKMPC_PIN_VERSIONS=OFF")
  else()
    message(WARNING
      "${pkg} ${found} != pinned ${pinned}（KMPC_PIN_VERSIONS=OFF，继续编）")
  endif()
endfunction()

# 开工机 2026-09：Ubuntu 22.04.5 / g++ 11.4.0
set(KMPC_PINNED_EIGEN_WORLD 3)
set(KMPC_PINNED_EIGEN_MAJOR 4)
set(KMPC_PINNED_EIGEN_VERSION "3.4.0")
set(KMPC_PINNED_YAML_CPP_VERSION "0.7.0")
set(KMPC_PINNED_GTEST_VERSION "1.11.0")
set(KMPC_PINNED_PINOCCHIO_VERSION "4.0.0")
set(KMPC_PINNED_PROXSUITE_VERSION "0.7.3")

function(kmpc_check_eigen_version)
  if(NOT DEFINED EIGEN3_VERSION_NUMBER AND DEFINED Eigen3_VERSION)
    set(EIGEN3_VERSION_NUMBER "${Eigen3_VERSION}")
  endif()
  if(DEFINED Eigen3_VERSION)
    if(NOT Eigen3_VERSION VERSION_EQUAL KMPC_PINNED_EIGEN_VERSION)
      kmpc_version_mismatch("Eigen" "${Eigen3_VERSION}" "${KMPC_PINNED_EIGEN_VERSION}")
    endif()
  endif()
endfunction()

function(kmpc_check_yaml_cpp_version)
  if(DEFINED yaml-cpp_VERSION AND NOT yaml-cpp_VERSION STREQUAL "")
    if(NOT yaml-cpp_VERSION VERSION_EQUAL KMPC_PINNED_YAML_CPP_VERSION)
      # Debian 包常带 +dfsg 后缀；只比 x.y.z 前三段
      string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _yaml_xyz "${yaml-cpp_VERSION}")
      if(_yaml_xyz AND NOT _yaml_xyz VERSION_EQUAL KMPC_PINNED_YAML_CPP_VERSION)
        kmpc_version_mismatch("yaml-cpp" "${yaml-cpp_VERSION}" "${KMPC_PINNED_YAML_CPP_VERSION}")
      endif()
    endif()
  endif()
endfunction()

function(kmpc_check_pinocchio_version)
  if(DEFINED pinocchio_VERSION AND NOT pinocchio_VERSION STREQUAL "")
    if(NOT pinocchio_VERSION VERSION_EQUAL KMPC_PINNED_PINOCCHIO_VERSION)
      kmpc_version_mismatch("Pinocchio" "${pinocchio_VERSION}" "${KMPC_PINNED_PINOCCHIO_VERSION}")
    endif()
  endif()
endfunction()

function(kmpc_check_proxsuite_version)
  if(DEFINED proxsuite_VERSION AND NOT proxsuite_VERSION STREQUAL "")
    if(NOT proxsuite_VERSION VERSION_EQUAL KMPC_PINNED_PROXSUITE_VERSION)
      kmpc_version_mismatch("ProxSuite" "${proxsuite_VERSION}" "${KMPC_PINNED_PROXSUITE_VERSION}")
    endif()
  endif()
endfunction()
