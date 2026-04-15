// GPU Kernels for LiquifyStretch - Metal Implementation (macOS)
// Metal shader for GPU-accelerated warp effects

#ifdef __METAL_KERNEL__

#include <metal.math>

using namespace metal;

struct Vertex {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct RasterizerData {
    float4 position [[position]];
    float2 texCoord;
};

struct WarpParams {
    int warpType;
    float centerX;
    float centerY;
    float strength;
    float radius;
    float angle;
    int imageWidth;
    int imageHeight;
};

float interpBilinear(texture2d<float, access::sample> img, sampler s, float2 coord) {
    return img.sample(s, coord).r;
}

kernel void warpPushKernel(texture2d<float, access::read> inTex [[texture(0)]],
                           texture2d<float, access::write> outTex [[texture(1)]],
                           constant WarpParams& params [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.imageWidth || gid.y >= params.imageHeight) return;
    
    float px = (float)gid.x;
    float py = (float)gid.y;
    float cx = params.centerX;
    float cy = params.centerY;
    float strength = params.strength;
    float radius = params.radius;
    float angle = params.angle;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax((float)params.imageWidth, (float)params.imageHeight) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) {
        outTex.write(inTex.read(gid), gid);
        return;
    }
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float a = angle * M_PI_F / 180.0f;
    float ox = cos(a) * strength * 0.01f * falloff * 20.0f;
    float oy = sin(a) * strength * 0.01f * falloff * 20.0f;
    
    float2 srcCoord = (float2)(px + ox, py + oy) / float2(params.imageWidth, params.imageHeight);
    auto color = inTex.sample(sampler(coord::normalized, address::clamp_to_edge), srcCoord);
    outTex.write(color, gid);
}

kernel void warpTwirlKernel(texture2d<float, access::read> inTex [[texture(0)]],
                           texture2d<float, access::write> outTex [[texture(1)]],
                           constant WarpParams& params [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.imageWidth || gid.y >= params.imageHeight) return;
    
    float px = (float)gid.x;
    float py = (float)gid.y;
    float cx = params.centerX;
    float cy = params.centerY;
    float strength = params.strength;
    float radius = params.radius;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax((float)params.imageWidth, (float)params.imageHeight) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) {
        outTex.write(inTex.read(gid), gid);
        return;
    }
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float t = strength * 0.001f * falloff;
    float c = cos(t);
    float s = sin(t);
    float ox = dx * c - dy * s - dx;
    float oy = dx * s + dy * c - dy;
    
    float2 srcCoord = (float2)(px + ox, py + oy) / float2(params.imageWidth, params.imageHeight);
    auto color = inTex.sample(sampler(coord::normalized, address::clamp_to_edge), srcCoord);
    outTex.write(color, gid);
}

kernel void warpPinchKernel(texture2d<float, access::read> inTex [[texture(0)]],
                           texture2d<float, access::write> outTex [[texture(1)]],
                           constant WarpParams& params [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.imageWidth || gid.y >= params.imageHeight) return;
    
    float px = (float)gid.x;
    float py = (float)gid.y;
    float cx = params.centerX;
    float cy = params.centerY;
    float strength = params.strength;
    float radius = params.radius;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax((float)params.imageWidth, (float)params.imageHeight) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) {
        outTex.write(inTex.read(gid), gid);
        return;
    }
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float p = -strength * 0.001f * falloff;
    float ox = dx * p;
    float oy = dy * p;
    
    float2 srcCoord = (float2)(px + ox, py + oy) / float2(params.imageWidth, params.imageHeight);
    auto color = inTex.sample(sampler(coord::normalized, address::clamp_to_edge), srcCoord);
    outTex.write(color, gid);
}

kernel void warpBloatKernel(texture2d<float, access::read> inTex [[texture(0)]],
                            texture2d<float, access::write> outTex [[texture(1)]],
                            constant WarpParams& params [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.imageWidth || gid.y >= params.imageHeight) return;
    
    float px = (float)gid.x;
    float py = (float)gid.y;
    float cx = params.centerX;
    float cy = params.centerY;
    float strength = params.strength;
    float radius = params.radius;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax((float)params.imageWidth, (float)params.imageHeight) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) {
        outTex.write(inTex.read(gid), gid);
        return;
    }
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float b = strength * 0.001f * falloff;
    float ox = -dx * b;
    float oy = -dy * b;
    
    float2 srcCoord = (float2)(px + ox, py + oy) / float2(params.imageWidth, params.imageHeight);
    auto color = inTex.sample(sampler(coord::normalized, address::clamp_to_edge), srcCoord);
    outTex.write(color, gid);
}

kernel void warpTurbulenceKernel(texture2d<float, access::read> inTex [[texture(0)]],
                                texture2d<float, access::write> outTex [[texture(1)]],
                                constant WarpParams& params [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.imageWidth || gid.y >= params.imageHeight) return;
    
    float px = (float)gid.x;
    float py = (float)gid.y;
    float cx = params.centerX;
    float cy = params.centerY;
    float strength = params.strength;
    float radius = params.radius;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax((float)params.imageWidth, (float)params.imageHeight) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) {
        outTex.write(inTex.read(gid), gid);
        return;
    }
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float f = 0.1f;
    float n = strength * 0.01f;
    float ox = sin(py * f + n * 10.0f) * cos(px * f) * n * falloff * 10.0f;
    float oy = sin(py * f + n * 10.0f) * cos(px * f) * n * falloff * 10.0f;
    
    float2 srcCoord = (float2)(px + ox, py + oy) / float2(params.imageWidth, params.imageHeight);
    auto color = inTex.sample(sampler(coord::normalized, address::clamp_to_edge), srcCoord);
    outTex.write(color, gid);
}

kernel void warpSwirlKernel(texture2d<float, access::read> inTex [[texture(0)]],
                           texture2d<float, access::write> outTex [[texture(1)]],
                           constant WarpParams& params [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.imageWidth || gid.y >= params.imageHeight) return;
    
    float px = (float)gid.x;
    float py = (float)gid.y;
    float cx = params.centerX;
    float cy = params.centerY;
    float strength = params.strength;
    float radius = params.radius;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax((float)params.imageWidth, (float)params.imageHeight) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) {
        outTex.write(inTex.read(gid), gid);
        return;
    }
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float t = strength * 0.002f * falloff;
    float c = cos(t);
    float s = sin(t);
    float ox = dx * c - dy * s - dx;
    float oy = dx * s + dy * c - dy;
    
    float2 srcCoord = (float2)(px + ox, py + oy) / float2(params.imageWidth, params.imageHeight);
    auto color = inTex.sample(sampler(coord::normalized, address::clamp_to_edge), srcCoord);
    outTex.write(color, gid);
}

kernel void warpShiftKernel(texture2d<float, access::read> inTex [[texture(0)]],
                           texture2d<float, access::write> outTex [[texture(1)]],
                           constant WarpParams& params [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.imageWidth || gid.y >= params.imageHeight) return;
    
    float px = (float)gid.x;
    float py = (float)gid.y;
    float cx = params.centerX;
    float cy = params.centerY;
    float strength = params.strength;
    float radius = params.radius;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax((float)params.imageWidth, (float)params.imageHeight) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx * 0.5f) {
        outTex.write(inTex.read(gid), gid);
        return;
    }
    
    float falloff = 1.0f - (dist / (radiusPx * 0.5f));
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float shift = strength * 0.02f * falloff;
    int shiftPx = (int)(shift * (float)params.imageWidth / 100.0f);
    
    if (shiftPx > 0 && gid.x + shiftPx < params.imageWidth) {
        uint2 srcPos = uint2(gid.x + shiftPx, gid.y);
        outTex.write(inTex.read(srcPos), gid);
    } else {
        outTex.write(inTex.read(gid), gid);
    }
}

#else

// C++ wrapper for Metal device-side code
#import <Metal/Metal.h>
#import <MetalComputeFoundation.h>

extern "C" void RunMetalWarpKernel(void* cmdQueue, int w, int h, int warpType,
                                    float cx, float cy, float strength, float radius, float angle,
                                    id<MTLBuffer> imageBuffer, id<MTLComputePipelineState> pipeline) {
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)cmdQueue;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    WarpParams params;
    params.warpType = warpType;
    params.centerX = cx;
    params.centerY = cy;
    params.strength = strength;
    params.radius = radius;
    params.angle = angle;
    params.imageWidth = w;
    params.imageHeight = h;
    
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:imageBuffer offset:0 atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(w, h, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    [encoder dispatchThreads(gridSize threadgroupsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

#endif