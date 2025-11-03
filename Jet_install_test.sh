export BASIC_PROXY="tencent.ssslab.cn:7893"
export HTTP_PROXY="http://$BASIC_PROXY"
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="socks5://$BASIC_PROXY"

current_path=$(pwd)
echo "current_path:${current_path}."

# # GKlib
# git clone https://github.com/KarypisLab/GKlib.git
# cd GKlib
echo "make config prefix="${current_path}""
# make config prefix="${current_path}"
# make -j4
# make install
# cd ..

# # Metis
# git clone https://github.com/KarypisLab/METIS.git
# cd METIS
echo "make config prefix="${current_path}""
# make config prefix="${current_path}"
# make -j4
# make install
# cd ..

# # Kokkos
# unzip kokkos-develop.zip
# cd kokkos-develop
# mkdir build
# cd build 
# echo "cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_EXTENSIONS=On -DCMAKE_INSTALL_PREFIX="${current_path}" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_BLACKWELL120=on .."
# cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_EXTENSIONS=On -DCMAKE_INSTALL_PREFIX="${current_path}" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_BLACKWELL120=on ..
# make -j4
# make install
# cd ..
# echo "35:$(pwd)"
# cd ..
# echo "37:$(pwd)"

# # Kokkos Kernels
unzip kokkos-kernels-develop.zip
cd kokkos-kernels-develop
mkdir build
cd build 
echo "cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_EXTENSIONS=On -DCMAKE_INSTALL_PREFIX="${current_path}" -DKokkos_ENABLE_CUDA=ON .."
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_EXTENSIONS=On -DCMAKE_INSTALL_PREFIX="${current_path}" -DKokkos_ENABLE_CUDA=ON ..
make -j4
make install
cd ..
echo "49:$(pwd)"
cd ..
echo "51:$(pwd)"

#jet
# unzip Jet-Partitioner-main.zip
# cd Jet-Partitioner-main

# # 覆盖写入cmakelists.txt文件
# cat << 'EOF' > cmakelists.txt
# #IMPORTANT: Use the cmake flag -DCMAKE_CXX_COMPILER=/path/to/your/nvcc_wrapper
# cmake_minimum_required(VERSION 3.18)
# project(jetpartition CXX)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# # 核心修改：将路径变量声明为CACHE变量----------------------------------------------
# set(GKLIB_ROOT "/usr/local" CACHE PATH "Path to GKlib installation")
# set(METIS_ROOT "/usr/local" CACHE PATH "Path to Metis installation")
# set(Kokkos_ROOT "/home/lyj" CACHE PATH "Path to Kokkos installation")
# set(KokkosKernels_ROOT "/home/lyj" CACHE PATH "Path to KokkosKernels installation")

# # 验证变量传递（新增调试信息）----------------------------------------------------
# message(STATUS "========================================================")
# message(STATUS "Build Configuration:")
# message(STATUS "  GKlib root:       ${GKLIB_ROOT}")
# message(STATUS "  Metis root:       ${METIS_ROOT}")
# message(STATUS "  Kokkos root:      ${Kokkos_ROOT}")
# message(STATUS "  KokkosKernels:    ${KokkosKernels_ROOT}")
# message(STATUS "========================================================")

# # 配置GKlib库（优化路径处理）------------------------------------------------------
# find_library(GKLIB_LIB GKlib
#     NAMES libGKlib.a libGKlib.so
#     PATHS "${GKLIB_ROOT}/lib"  # 直接使用变量路径
#     NO_DEFAULT_PATH
#     DOC "GKlib library path"
# )
# if(NOT GKLIB_LIB)
#     message(FATAL_ERROR "GKlib not found in: ${GKLIB_ROOT}/lib\n"
#         "Specify correct path with: cmake -DGKLIB_ROOT=/your/gklib/path")
# endif()
# message(STATUS "Found GKlib: ${GKLIB_LIB}")

# # 配置Metis库（增强路径检查）------------------------------------------------------
# find_library(METIS_LIB metis
#     NAMES libmetis.a libmetis.so
#     PATHS "${METIS_ROOT}/lib"  # 直接使用变量路径
#     NO_DEFAULT_PATH
#     DOC "Metis library path"
# )
# if(NOT METIS_LIB)
#     message(FATAL_ERROR "Metis not found in: ${METIS_ROOT}/lib\n"
#         "Specify correct path with: cmake -DMETIS_ROOT=/your/metis/path")
# endif()
# message(STATUS "Found Metis: ${METIS_LIB}")

# # 配置头文件包含路径（简化配置）----------------------------------------------------
# include_directories(
#     "${GKLIB_ROOT}/include"  # 直接引用路径
#     "${METIS_ROOT}/include"
# )

# # 增强Kokkos配置（添加版本检查）---------------------------------------------------
# find_package(Kokkos REQUIRED 
#     HINTS "${Kokkos_ROOT}"
#     PATH_SUFFIXES "lib64/cmake/Kokkos" "lib/cmake/Kokkos"
#     NO_DEFAULT_PATH
# )
# if(NOT Kokkos_FOUND)
#     message(FATAL_ERROR "Kokkos not found in: ${Kokkos_ROOT}\n"
#         "Check path with: cmake -DKokkos_ROOT=/kokkos/install/path")
# endif()

# # 增强KokkosKernels配置（添加组件支持）---------------------------------------------
# find_package(KokkosKernels REQUIRED
#     HINTS "${KokkosKernels_ROOT}"
#     PATH_SUFFIXES "lib64/cmake/KokkosKernels" "lib/cmake/KokkosKernels"
#     NO_DEFAULT_PATH
#     COMPONENTS BLAS
# )
# message(STATUS "Kokkos Kernels Includes: ${KokkosKernels_INCLUDE_DIRS}")

# # 编译选项配置（保持原样）---------------------------------------------------------
# add_compile_options(-O3 -Wall -Wextra -Wshadow)

# # 可执行文件配置（统一管理）-------------------------------------------------------
# set(EXECUTABLES
#     jet
#     jet4
#     jet2
#     jet_host
#     jet_import
#     jet_export
#     jet_serial
#     pstat
# )

# foreach(exe ${EXECUTABLES})
#     add_executable(${exe} ${exe}.cpp)
# endforeach()

# # 统一链接配置（优化依赖顺序）-----------------------------------------------------
# foreach(target ${EXECUTABLES})
#     target_link_libraries(${target}
#         Kokkos::kokkoskernels  # 顺序敏感，kernels在前
#         Kokkos::kokkos
#         ${GKLIB_LIB}
#         ${METIS_LIB}
#         pthread
#     )
#     # 添加头文件包含路径
#     target_include_directories(${target} PRIVATE
#         "${GKLIB_ROOT}/include"
#         "${METIS_ROOT}/include"
#     )
# endforeach()
# EOF

# echo "cmakelists.txt has been updated successfully!"

# mkdir build
# cd build

# echo "cmake \
#   -DGKLIB_ROOT="${current_path}" \
#   -DMETIS_ROOT="${current_path}" \
#   -DKokkos_ROOT="${current_path}" \
#   -DKokkosKernels_ROOT="${current_path}" .."
# cmake \
#   -DGKLIB_ROOT="${current_path}" \
#   -DMETIS_ROOT="${current_path}" \
#   -DKokkos_ROOT="${current_path}" \
#   -DKokkosKernels_ROOT="${current_path}" ..
# make -j4
# cd ..
# cd ..
