// GPU Kernels for LiquifyStretch - OpenCL Implementation
// OpenCL 2.0 kernel for GPU-accelerated warp effects

#pragma once

#ifdef __OPENCL_KERNEL__

__kernel void warpPushKernel(__global float* img, int w, int h, float cx, float cy, float strength, float radius, float angle) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= w || py >= h) return;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) return;
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float a = angle * M_PI_F / 180.0f;
    float ox = cos(a) * strength * 0.01f * falloff * 20.0f;
    float oy = sin(a) * strength * 0.01f * falloff * 20.0f;
    
    int idx = (py * w + px) * 4;
    for (int ch = 0; ch < 4; ch++) {
        img[idx + ch] = read_imagef(img, (float2)(px + ox, py + ch)).s0;
    }
}

__kernel void warpTwirlKernel(__global float* img, int w, int h, float cx, float cy, float strength, float radius) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= w || py >= h) return;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) return;
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float t = strength * 0.001f * falloff;
    float c = cos(t);
    float s = sin(t);
    float ox = dx * c - dy * s - dx;
    float oy = dx * s + dy * c - dy;
    
    int idx = (py * w + px) * 4;
    for (int ch = 0; ch < 4; ch++) {
        img[idx + ch] = read_imagef(img, (float2)(px + ox, py + oy)).s0;
    }
}

__kernel void warpPinchKernel(__global float* img, int w, int h, float cx, float cy, float strength, float radius) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= w || py >= h) return;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) return;
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float p = -strength * 0.001f * falloff;
    float ox = dx * p;
    float oy = dy * p;
    
    int idx = (py * w + px) * 4;
    for (int ch = 0; ch < 4; ch++) {
        img[idx + ch] = read_imagef(img, (float2)(px + ox, py + oy)).s0;
    }
}

__kernel void warpBloatKernel(__global float* img, int w, int h, float cx, float cy, float strength, float radius) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= w || py >= h) return;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) return;
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float b = strength * 0.001f * falloff;
    float ox = -dx * b;
    float oy = -dy * b;
    
    int idx = (py * w + px) * 4;
    for (int ch = 0; ch < 4; ch++) {
        img[idx + ch] = read_imagef(img, (float2)(px + ox, py + oy)).s0;
    }
}

__kernel void warpTurbulenceKernel(__global float* img, int w, int h, float cx, float cy, float strength, float radius) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= w || py >= h) return;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) return;
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float f = 0.1f;
    float n = strength * 0.01f;
    float ox = sin(py * f + n * 10.0f) * cos(px * f) * n * falloff * 10.0f;
    float oy = sin(py * f + n * 10.0f) * cos(px * f) * n * falloff * 10.0f;
    
    int idx = (py * w + px) * 4;
    for (int ch = 0; ch < 4; ch++) {
        img[idx + ch] = read_imagef(img, (float2)(px + ox, py + oy)).s0;
    }
}

__kernel void warpSwirlKernel(__global float* img, int w, int h, float cx, float cy, float strength, float radius) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= w || py >= h) return;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx) return;
    
    float falloff = 1.0f - (dist / radiusPx);
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float t = strength * 0.002f * falloff;
    float c = cos(t);
    float s = sin(t);
    float ox = dx * c - dy * s - dx;
    float oy = dx * s + dy * c - dy;
    
    int idx = (py * w + px) * 4;
    for (int ch = 0; ch < 4; ch++) {
        img[idx + ch] = read_imagef(img, (float2)(px + ox, py + oy)).s0;
    }
}

__kernel void warpShiftKernel(__global float* img, int w, int h, float cx, float cy, float strength, float radius) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= w || py >= h) return;
    
    float dx = px - cx;
    float dy = py - cy;
    float dist = sqrt(dx * dx + dy * dy);
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    if (dist > radiusPx * 0.5f) return;
    
    float falloff = 1.0f - (dist / (radiusPx * 0.5f));
    falloff = falloff * falloff * (3.0f - 2.0f * falloff);
    
    float shift = strength * 0.02f * falloff;
    int shiftPx = (int)(shift * (float)w / 100.0f);
    
    if (shiftPx > 0 && px + shiftPx < w) {
        int idx = (py * w + px) * 4;
        int srcIdx = (py * w + px + shiftPx) * 4;
        for (int ch = 0; ch < 4; ch++) {
            img[idx + ch] = img[srcIdx + ch];
        }
    }
}

#else

// C++ wrapper for OpenCL - this is the host-side code
#include <CL/cl.h>

extern "C" void RunOpenCLWarpKernel(void* cmdQueue, int w, int h, int warpType,
                                      float cx, float cy, float strength, float radius, float angle,
                                      cl_mem imageBuffer, cl_program program) {
    cl_kernel kernel = NULL;
    cl_int err;
    
    const char* kernelNames[] = {
        "warpPushKernel",
        "warpTwirlKernel", 
        "warpPinchKernel",
        "warpBloatKernel",
        "warpTurbulenceKernel",
        "warpSwirlKernel",
        "warpShiftKernel"
    };
    
    if (warpType < 0 || warpType > 6) return;
    
    kernel = clCreateKernel(program, kernelNames[warpType], &err);
    if (err != CL_SUCCESS || kernel == NULL) return;
    
    int radiusPx = (int)(radius * fmax(w, h) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
    clSetKernelArg(kernel, 1, sizeof(int), &w);
    clSetKernelArg(kernel, 2, sizeof(int), &h);
    clSetKernelArg(kernel, 3, sizeof(float), &cx);
    clSetKernelArg(kernel, 4, sizeof(float), &cy);
    clSetKernelArg(kernel, 5, sizeof(float), &strength);
    clSetKernelArg(kernel, 6, sizeof(float), &radius);
    clSetKernelArg(kernel, 7, sizeof(float), &angle);
    
    size_t globalSize[2] = {(size_t)w, (size_t)h};
    size_t localSize[2] = {16, 16};
    
    err = clEnqueueNDRangeKernel((cl_command_queue)cmdQueue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    
    if (kernel) clReleaseKernel(kernel);
}

#endif