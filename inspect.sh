#!/bin/bash

echo "=== INSPECTING GENERATED PROTO FILES ==="

echo "1. Checking C++ generated service definition..."
if [ -f "inference.grpc.pb.h" ]; then
    echo "Service classes in C++ header:"
    grep -A5 -B2 "class.*Service" inference.grpc.pb.h
    echo ""
    echo "Method definitions:"
    grep -A3 -B1 "Evaluate" inference.grpc.pb.h
else
    echo "✗ inference.grpc.pb.h not found!"
fi

echo -e "\n2. Checking the actual .proto file used..."
echo "Current inference.proto content:"
cat inference.proto 2>/dev/null || echo "✗ inference.proto not found!"

echo -e "\n3. Checking if files were generated from same proto..."
echo "Modification times:"
ls -la inference.proto inference.pb.* inference.grpc.pb.* 2>/dev/null

echo -e "\n4. Testing proto compilation..."
echo "Recompiling proto files..."
protoc --cpp_out=. inference.proto 2>&1
protoc --grpc_out=. --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) inference.proto 2>&1

echo -e "\n5. Checking namespace and service name..."
if [ -f "inference.grpc.pb.h" ]; then
    echo "Namespace:"
    grep -n "namespace" inference.grpc.pb.h | head -5
    echo "Service name:"
    grep -n "InferenceService" inference.grpc.pb.h | head -5
fi