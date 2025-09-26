# Minimum required version of CMake
cmake_minimum_required(VERSION 3.15)

# Project name and language
project(YeagerChessEngine CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Protobuf library. This will be found automatically if
# you installed it with pacman in MSYS2/MinGW.
find_package(Protobuf REQUIRED)

# Find the gRPC library.
find_package(gRPC REQUIRED)

# === Generate C++ sources from your .proto file ===
# Get the absolute path to the proto file
get_filename_component(PROTO_FILE "inference.proto" ABSOLUTE)
# Define where the generated files will go (in the build directory)
set(PROTO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# This command tells CMake how to run protoc
protobuf_generate(
    TARGET ${PROJECT_NAME}
    LANGUAGE cpp
    PROTOS ${PROTO_FILE}
    EXPORT_MACRO YOUR_LIB_EXPORT_SYMBOL # Avoids a Windows warning
    OUT_VAR PROTO_SRCS
    PLUGIN_OUT_VAR GRPC_SRCS
    PLUGINS "grpc"
)

# === Define the executable and its source files ===
add_executable(chess.exe
    chess.cpp
    mcts.cpp
    remote_evaluator.cpp
    position.cpp
    types.cpp
    tables.cpp
    fen_to_tensor.cpp
    # Add the generated source files to the compilation
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# === Configure Include Directories ===
# This is the part that fixes your "cannot open source file" error!
target_include_directories(chess.exe PRIVATE
    # Directory where you have your own headers (mcts.h, etc.)
    ${CMAKE_CURRENT_SOURCE_DIR}
    # Directory where generated proto headers are placed
    ${PROTO_GENERATED_DIR}
    # Let CMake add the include paths for gRPC and Protobuf automatically
    ${gRPC_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIRS}
)

# === Link Libraries ===
# This fixes the "undefined reference" linker errors
target_link_libraries(chess.exe PRIVATE
    gRPC::grpc++
    protobuf::libprotobuf
)