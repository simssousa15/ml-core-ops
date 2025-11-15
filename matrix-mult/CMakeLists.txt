cmake_minimum_required(VERSION 3.10)

project(Matrix_Multiplication C)

# Set C standard and optimization flags
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mavx2 -mfma -funroll-loops")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native -mfma")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")

# Find OpenBLAS
find_package(BLAS REQUIRED)

add_executable(m_mul main.c)

# Link OpenBLAS
target_link_libraries(m_mul ${BLAS_LIBRARIES})

# Print found BLAS library (helpful for debugging)
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
