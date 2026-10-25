/*
 * Adapted from attention-mps-torch, Copyright 2026 Jason Hurt.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Changes: inference-only unmasked operation; ARC-safe buffer access without
 * PyTorch's MRC-only internal headers; explicit zero-offset checks; feed order
 * follows the compiled graph; optional FP16-intermediate rewrite is omitted.
 */
#include <ATen/ATen.h>
#include <torch/mps.h>
#include <torch/library.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>

static MPSDataType scalarType(at::ScalarType type) {
    switch (type) {
        case at::kFloat: return MPSDataTypeFloat32;
        case at::kHalf: return MPSDataTypeFloat16;
        case at::kBFloat16: return MPSDataTypeBFloat16;
        default: TORCH_CHECK(false, "Unsupported attention dtype");
    }
}

static id<MTLBuffer> buffer(const at::Tensor& tensor) {
    return (__bridge id<MTLBuffer>)tensor.storage().data();
}

static at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    for (const auto& tensor : {q, k, v}) {
        TORCH_CHECK(tensor.device().is_mps() && tensor.dim() == 4,
                    "Expected four-dimensional MPS tensors");
        TORCH_CHECK(tensor.is_contiguous() && tensor.storage_offset() == 0,
                    "Expected contiguous, zero-offset attention buffers");
        TORCH_CHECK(tensor.scalar_type() == q.scalar_type(), "Attention dtypes must match");
    }
    TORCH_CHECK(k.sizes() == v.sizes() && q.size(0) == k.size(0) &&
                q.size(1) == k.size(1) && q.size(3) == k.size(3),
                "Expected matching batch, heads, and head dimensions");
    auto output = at::empty(q.sizes(), q.options());
    @autoreleasepool {
        static NSMutableDictionary<NSString*, NSArray*>* cache = nil;
        static dispatch_once_t once;
        dispatch_once(&once, ^{ cache = [NSMutableDictionary new]; });
        id<MTLCommandBuffer> command = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            NSString* key = [NSString stringWithFormat:@"%lld_%lld_%lld_%lld_%lld_%d",
                q.size(0), q.size(1), q.size(2), k.size(2), q.size(3), int(q.scalar_type())];
            NSArray<NSNumber*>* qs = @[@(q.size(0)), @(q.size(1)), @(q.size(2)), @(q.size(3))];
            NSArray<NSNumber*>* ks = @[@(k.size(0)), @(k.size(1)), @(k.size(2)), @(k.size(3))];
            MPSDataType type = scalarType(q.scalar_type());
            NSArray* entry = cache[key];
            if (!entry) {
                MPSGraph* graph = [MPSGraph new];
                MPSGraphTensor* qt = [graph placeholderWithShape:qs dataType:type name:@"q"];
                MPSGraphTensor* kt = [graph placeholderWithShape:ks dataType:type name:@"k"];
                MPSGraphTensor* vt = [graph placeholderWithShape:ks dataType:type name:@"v"];
                MPSGraphTensor* result = [graph scaledDotProductAttentionWithQueryTensor:qt
                    keyTensor:kt valueTensor:vt maskTensor:nil
                    scale:1.0f / std::sqrt(float(q.size(3))) name:@"attention"];
                NSDictionary* feeds = @{
                    qt: [[MPSGraphShapedType alloc] initWithShape:qs dataType:type],
                    kt: [[MPSGraphShapedType alloc] initWithShape:ks dataType:type],
                    vt: [[MPSGraphShapedType alloc] initWithShape:ks dataType:type]};
                MPSGraphCompilationDescriptor* descriptor = [MPSGraphCompilationDescriptor new];
                descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
                MPSGraphExecutable* executable = [graph compileWithDevice:
                    [MPSGraphDevice deviceWithMTLDevice:command.device] feeds:feeds
                    targetTensors:@[result] targetOperations:nil compilationDescriptor:descriptor];
                NSMutableArray<NSNumber*>* order = [NSMutableArray new];
                NSArray* placeholders = @[qt, kt, vt];
                for (MPSGraphTensor* input in executable.feedTensors) {
                    NSUInteger index = [placeholders indexOfObjectIdenticalTo:input];
                    TORCH_CHECK(index != NSNotFound, "Unexpected MPSGraph feed");
                    [order addObject:@(index)];
                }
                TORCH_CHECK(order.count == 3, "Missing MPSGraph feed order");
                entry = @[executable, order];
                cache[key] = entry;
            }
            NSArray* inputs = @[
                [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer(q) shape:qs dataType:type],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer(k) shape:ks dataType:type],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer(v) shape:ks dataType:type]];
            NSMutableArray* ordered = [NSMutableArray new];
            for (NSNumber* index in entry[1]) [ordered addObject:inputs[index.unsignedIntegerValue]];
            MPSGraphTensorData* target = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buffer(output) shape:qs dataType:type];
            [(MPSGraphExecutable*)entry[0] encodeToCommandBuffer:command inputsArray:ordered
                resultsArray:@[target] executionDescriptor:nil];
            torch::mps::commit();
        });
    }
    // PyTorch owns the buffers. Complete their native use before returning.
    torch::mps::synchronize();
    return output;
}

TORCH_LIBRARY(corridorkey_mps, m) {
    m.def("attention(Tensor q, Tensor k, Tensor v) -> Tensor");
    m.impl("attention", c10::DispatchKey::MPS, TORCH_FN(attention));
}
