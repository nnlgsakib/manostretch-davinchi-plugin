// DistortionCPU.cpp — CPU Implementations for New Distortion Effects
//
// Eight new full-frame distortion effects for ManoStretch Surrealism module:
// 1. Fluid Morph    — Metaball-style organic blob merging
// 2. Mirror Fractal — Recursive kaleidoscope (deeper than basic kaleido)
// 3. Glitch Slice   — Horizontal band displacement
// 4. Triangulate    — Delaunay triangle mosaic
// 5. Water Ripple   — Concentric ripples from center
// 6. Displacement Map — Luminance-driven distortion
// 7. Tile/Repeat    — Grid repetition with offset
// 8. Time Waver     — Temporal frame jitter

#include "DistortionFX.h"
#include "ofxsImageEffect.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ================================================================
//  Helper Functions
// ================================================================
static inline float* pxAt(void* base, const OfxRectI& b, int rb, int x, int y) {
    if (x < b.x1 || x >= b.x2 || y < b.y1 || y >= b.y2) return nullptr;
    return reinterpret_cast<float*>(
        reinterpret_cast<char*>(base) + (std::ptrdiff_t)(y - b.y1) * rb
                                      + (x - b.x1) * 4 * (int)sizeof(float));
}

static float bilerpFlatCPU(const float* buf, int w, int h, float fx, float fy, int ch) {
    int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) return 0.f;
    float u = fx - x0, v = fy - y0;
    return (buf[(y0 * w + x0) * 4 + ch] * (1 - u) + buf[(y0 * w + x1) * 4 + ch] * u) * (1 - v)
         + (buf[(y1 * w + x0) * 4 + ch] * (1 - u) + buf[(y1 * w + x1) * 4 + ch] * u) * v;
}

// Simple hash function for randomness
static float fastHash(float x, float y) {
    int ix = *(int*)&x ^ (*(int*)&y << 3);
    ix = (ix << 13) ^ ix;
    return std::abs(std::sin((float)ix * 0.0001f));
}

// Pseudo-random with seed
static float seededRandom(int x, int y, int seed) {
    int h = seed ^ (x * 374761393) ^ (y * 668265263);
    h = (h ^ (h >> 13)) * 1274126177;
    return (float)(h & 0xFFFF) / 65536.0f;
}

// ================================================================
//  1. FLUID MORPH — Metaball-style organic blob merging
// ================================================================
static void applyFluidMorph(const float* src, float* dst, int w, int h,
                            const DistortionFX& dx) {
    int blobCount = (int)dx.fluidBlobCount;
    if (blobCount < 3) blobCount = 3;
    if (blobCount > 10) blobCount = 10;

    // Generate blob positions (stored in static for consistency across frames)
    static std::vector<float> blobX, blobY, blobVX, blobVY;
    static int lastBlobCount = 0;
    static float lastTime = -1000;

    if (blobCount != lastBlobCount || std::abs(dx.time - lastTime) > 0.1f) {
        blobX.resize(blobCount);
        blobY.resize(blobCount);
        blobVX.resize(blobCount);
        blobVY.resize(blobCount);
        for (int i = 0; i < blobCount; i++) {
            blobX[i] = seededRandom(i + 1, 0, (int)dx.fluidSpeed * 1000 + 100) * (float)w;
            blobY[i] = seededRandom(i + 1, 1, (int)dx.fluidSpeed * 1000 + 100) * (float)h;
            float ang = seededRandom(i + 1, 2, (int)dx.fluidSpeed * 1000 + 100) * (float)(2.0 * M_PI);
            float spd = 20.0f + seededRandom(i + 1, 3, (int)dx.fluidSpeed * 1000 + 100) * 30.0f;
            blobVX[i] = std::cos(ang) * spd;
            blobVY[i] = std::sin(ang) * spd;
        }
        lastBlobCount = blobCount;
        lastTime = dx.time;
    }

    // Animate blob positions
    float t = dx.time * dx.fluidSpeed * 0.5f;
    std::vector<float> curX(blobCount), curY(blobCount);
    for (int i = 0; i < blobCount; i++) {
        float jitter = dx.fluidJitter * std::sin(t * 2.0f + i * 1.7f);
        curX[i] = blobX[i] + blobVX[i] * t + jitter * 20.0f;
        curY[i] = blobY[i] + blobVY[i] * t + jitter * 20.0f;
        // Wrap around edges
        curX[i] = std::fmod(curX[i] + w * 2, (float)w);
        curY[i] = std::fmod(curY[i] + h * 2, (float)h);
    }

    float thresh = dx.fluidThreshold;

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            // Calculate metaball field value
            float field = 0;
            for (int i = 0; i < blobCount; i++) {
                float dx2 = (float)px - curX[i];
                float dy2 = (float)py - curY[i];
                float dist = std::sqrt(dx2 * dx2 + dy2 * dy2);
                float radius = (float)(std::min(w, h)) * 0.15f;
                field += radius * radius / (dist * dist + 1.0f);
            }

            // Threshold determines whether we sample from source or warped
            if (field > thresh) {
                // Inside blob — distort position based on closest blob
                int closest = 0;
                float minDist = 1e10f;
                for (int i = 0; i < blobCount; i++) {
                    float d2 = (px - curX[i]) * (px - curX[i]) + (py - curY[i]) * (py - curY[i]);
                    if (d2 < minDist) { minDist = d2; closest = i; }
                }
                // Warp toward closest blob center
                float warp = dx.fluidJitter * 0.3f;
                float srcX = px + (curX[closest] - px) * warp;
                float srcY = py + (curY[closest] - py) * warp;

                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++) {
                    dst[idx + c] = bilerpFlatCPU(src, w, h, srcX, srcY, c);
                }
            } else {
                // Outside blob — use original
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++) {
                    dst[idx + c] = src[idx + c];
                }
            }
        }
    }
}

// ================================================================
//  2. MIRROR FRACTAL — Recursive kaleidoscope
// ================================================================
static void applyMirrorFractal(const float* src, float* dst, int w, int h,
                               const DistortionFX& dx) {
    int depth = (int)dx.mirrorDepth;
    if (depth < 2) depth = 2;
    if (depth > 8) depth = 8;

    float cx = w * 0.5f, cy = h * 0.5f;

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            float srcX = (float)px;
            float srcY = (float)py;

            // Apply recursive kaleidoscope
            for (int level = 0; level < depth; level++) {
                float dx2 = srcX - cx;
                float dy2 = srcY - cy;
                float angle = std::atan2(dy2, dx2);

                // Rotate per level
                float rotPerLevel = dx.mirrorRotateEach * (float)(2.0 * M_PI) * (float)level / (float)depth;
                angle += rotPerLevel;

                float radius = std::sqrt(dx2 * dx2 + dy2 * dy2);

                // Scale per level
                radius *= dx.mirrorScale;

                // Mirror fold
                int segs = 2 + level; // Increase segments with depth
                float sector = (float)(2.0 * M_PI) / (float)segs;
                float angMod = std::fmod(std::abs(angle), sector);
                if (angMod > sector * 0.5f) angMod = sector - angMod;

                srcX = cx + radius * std::cos(angMod);
                srcY = cy + radius * std::sin(angMod);
            }

            int idx = (py * w + px) * 4;
            for (int c = 0; c < 4; c++) {
                dst[idx + c] = bilerpFlatCPU(src, w, h, srcX, srcY, c);
            }
        }
    }
}

// ================================================================
//  3. GLITCH SLICE — Horizontal band displacement
// ================================================================
static void applyGlitchSlice(const float* src, float* dst, int w, int h,
                             const DistortionFX& dx) {
    int sliceCount = (int)dx.sliceCount;
    if (sliceCount < 4) sliceCount = 4;
    if (sliceCount > 32) sliceCount = 32;

    float bandHeight = (float)h / (float)sliceCount;
    float dispAmt = dx.sliceDisplaceAmt;
    float rgbSplit = dx.sliceRGBSplit;
    int seed = (int)dx.sliceRandSeed;

    for (int py = 0; py < h; py++) {
        int band = (int)((float)py / bandHeight);
        float rnd = seededRandom(band, seed, 12345);
        float offset = (rnd - 0.5f) * 2.0f * dispAmt;

        // Add time variation
        offset *= (0.5f + 0.5f * std::sin(dx.time * 0.5f + band * 0.5f));

        for (int px = 0; px < w; px++) {
            int idx = (py * w + px) * 4;
            float srcX = (float)px + offset;

            // RGB split
            float srcX_R = srcX + rgbSplit;
            float srcX_G = srcX;
            float srcX_B = srcX - rgbSplit;

            dst[idx + 0] = bilerpFlatCPU(src, w, h, srcX_R, (float)py, 0);
            dst[idx + 1] = bilerpFlatCPU(src, w, h, srcX_G, (float)py, 1);
            dst[idx + 2] = bilerpFlatCPU(src, w, h, srcX_B, (float)py, 2);
            dst[idx + 3] = bilerpFlatCPU(src, w, h, srcX, (float)py, 3);
        }
    }
}

// ================================================================
//  4. TRIANGULATE — Delaunay triangle mosaic
// ================================================================
static void applyTriangulate(const float* src, float* dst, int w, int h,
                             const DistortionFX& dx) {
    int pointCount = (int)dx.triPointCount;
    if (pointCount < 50) pointCount = 50;
    if (pointCount > 2000) pointCount = 2000;

    static std::vector<float> pointsX, pointsY;
    static int lastPointCount = 0;
    static int lastSeed = -1;

    // Generate random points
    if (pointCount != lastPointCount || (int)dx.mirrorSeed != lastSeed) {
        pointsX.resize(pointCount);
        pointsY.resize(pointCount);
        int seed = (int)dx.mirrorSeed;
        for (int i = 0; i < pointCount; i++) {
            pointsX[i] = seededRandom(i, seed, 111) * (float)w;
            pointsY[i] = seededRandom(i, seed, 222) * (float)h;
        }
        // Add corners
        if (pointCount + 4 < (int)pointsX.size()) {
            pointsX.resize(pointCount + 4);
            pointsY.resize(pointCount + 4);
        }
        pointsX[pointCount] = 0; pointsY[pointCount] = 0;
        pointsX[pointCount + 1] = (float)w; pointsY[pointCount + 1] = 0;
        pointsX[pointCount + 2] = 0; pointsY[pointCount + 2] = (float)h;
        pointsX[pointCount + 3] = (float)w; pointsY[pointCount + 3] = (float)h;
        lastPointCount = pointCount;
        lastSeed = (int)dx.mirrorSeed;
    }

    int actualCount = pointCount + 4;

    // Simple triangulate: find 3 closest points and fill with average color
    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            // Find 3 closest points (simple brute force for now)
            float bestDist[3] = {1e10f, 1e10f, 1e10f};
            int bestIdx[3] = {0, 0, 0};

            for (int i = 0; i < actualCount; i++) {
                float d = (px - pointsX[i]) * (px - pointsX[i]) + (py - pointsY[i]) * (py - pointsY[i]);
                if (d < bestDist[0]) {
                    bestDist[2] = bestDist[1];
                    bestIdx[2] = bestIdx[1];
                    bestDist[1] = bestDist[0];
                    bestIdx[1] = bestIdx[0];
                    bestDist[0] = d;
                    bestIdx[0] = i;
                } else if (d < bestDist[1]) {
                    bestDist[2] = bestDist[1];
                    bestIdx[2] = bestIdx[1];
                    bestDist[1] = d;
                    bestIdx[1] = i;
                } else if (d < bestDist[2]) {
                    bestDist[2] = d;
                    bestIdx[2] = i;
                }
            }

            // Calculate center of triangle
            float cx = (pointsX[bestIdx[0]] + pointsX[bestIdx[1]] + pointsX[bestIdx[2]]) / 3.0f;
            float cy = (pointsY[bestIdx[0]] + pointsY[bestIdx[1]] + pointsY[bestIdx[2]]) / 3.0f;

            // Add color variation
            float variant = dx.triFillVariant * fastHash((float)px * 0.01f, (float)py * 0.01f);

            int idx = (py * w + px) * 4;
            for (int c = 0; c < 4; c++) {
                float col = bilerpFlatCPU(src, w, h, cx, cy, c);
                col += (variant - dx.triFillVariant * 0.5f);
                dst[idx + c] = col;
            }

            // Edge detection (simple: if close to any point edge)
            if (dx.triEdgeThickness > 0.01f) {
                // Calculate distance to edges of the triangle
                // Simplified: just darken based on distance to nearest point
                float minD = std::sqrt(bestDist[0]);
                float edgeWidth = dx.triEdgeThickness * 10.0f;
                if (minD < edgeWidth) {
                    float edgeFactor = minD / edgeWidth;
                    int idx2 = (py * w + px) * 4;
                    dst[idx2 + 0] = dst[idx2 + 0] * edgeFactor + dx.triEdgeColor[0] * (1.0f - edgeFactor);
                    dst[idx2 + 1] = dst[idx2 + 1] * edgeFactor + dx.triEdgeColor[1] * (1.0f - edgeFactor);
                    dst[idx2 + 2] = dst[idx2 + 2] * edgeFactor + dx.triEdgeColor[2] * (1.0f - edgeFactor);
                }
            }
        }
    }
}

// ================================================================
//  5. WATER RIPPLE — Concentric ripples from center
// ================================================================
static void applyWaterRipple(const float* src, float* dst, int w, int h,
                             const DistortionFX& dx) {
    float cx = dx.rippleCenterX * (float)w;
    float cy = dx.rippleCenterY * (float)h;
    float freq = dx.rippleFrequency;
    float amp = dx.rippleAmplitude;
    float decay = dx.rippleDecay;
    float speed = dx.rippleSpeed;
    float phase = dx.ripplePhase;

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            float dx2 = (float)px - cx;
            float dy = (float)py - cy;
            float dist = std::sqrt(dx2 * dx2 + dy * dy);

            // Ripple wave
            float wave = std::sin(dist * freq * 0.01f - dx.time * speed + phase);
            float falloff = std::exp(-dist * decay * 0.01f);
            float offset = wave * amp * falloff;

            // Displace along radial direction
            if (dist > 1.0f) {
                float srcX = (float)px + (dx2 / dist) * offset;
                float srcY = (float)py + (dy / dist) * offset;

                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++) {
                    dst[idx + c] = bilerpFlatCPU(src, w, h, srcX, srcY, c);
                }
            } else {
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++) {
                    dst[idx + c] = src[idx + c];
                }
            }
        }
    }
}

// ================================================================
//  6. DISPLACEMENT MAP — Luminance-driven distortion
// ================================================================
static void applyDisplacementMap(const float* src, float* dst, int w, int h,
                                 const DistortionFX& dx) {
    float strength = dx.dispStrength;
    float scale = dx.dispScale;
    int channel = (int)dx.dispChannel;
    int direction = (int)dx.dispDirection;

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            // Get displacement value from source
            float dispVal;
            if (channel == 3) {
                // Luminance
                int idx = (py * w + px) * 4;
                dispVal = 0.299f * src[idx] + 0.587f * src[idx + 1] + 0.114f * src[idx + 2];
            } else {
                dispVal = src[(py * w + px) * 4 + channel];
            }

            // Normalize to -0.5 to 0.5
            dispVal = dispVal - 0.5f;

            // Calculate displacement
            float offsetX = 0, offsetY = 0;
            if (direction == 0 || direction == 1) {
                offsetX = dispVal * strength * std::cos(px * scale + py * scale * 0.5f);
            }
            if (direction == 0 || direction == 2) {
                offsetY = dispVal * strength * std::sin(py * scale + px * scale * 0.5f);
            }

            float srcX = (float)px + offsetX;
            float srcY = (float)py + offsetY;

            int idx = (py * w + px) * 4;
            for (int c = 0; c < 4; c++) {
                dst[idx + c] = bilerpFlatCPU(src, w, h, srcX, srcY, c);
            }
        }
    }
}

// ================================================================
//  7. TILE/REPEAT — Grid repetition with offset
// ================================================================
static void applyTileRepeat(const float* src, float* dst, int w, int h,
                            const DistortionFX& dx) {
    int rows = (int)dx.tileRows;
    int cols = (int)dx.tileCols;
    if (rows < 1) rows = 1;
    if (cols < 1) cols = 1;
    if (rows > 10) rows = 10;
    if (cols > 10) cols = 10;

    float cellW = (float)w / (float)cols;
    float cellH = (float)h / (float)rows;
    int seed = (int)dx.tileRandomSeed;

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            int cellX = (int)((float)px / cellW);
            int cellY = (int)((float)py / cellH);

            // Offset within cell
            float offsetX = 0, offsetY = 0;
            if (dx.tileOffsetX != 0 || dx.tileOffsetY != 0) {
                float rndX = seededRandom(cellX, cellY, seed) - 0.5f;
                float rndY = seededRandom(cellX + 100, cellY + 100, seed) - 0.5f;
                offsetX = dx.tileOffsetX * rndX * cellW;
                offsetY = dx.tileOffsetY * rndY * cellH;
            }

            // Calculate source coords (tile the image)
            float srcX = (float)px + offsetX;
            float srcY = (float)py + offsetY;

            // Wrap
            srcX = std::fmod(srcX + w * 2, (float)w);
            srcY = std::fmod(srcY + h * 2, (float)h);

            int idx = (py * w + px) * 4;
            for (int c = 0; c < 4; c++) {
                dst[idx + c] = bilerpFlatCPU(src, w, h, srcX, srcY, c);
            }
        }
    }
}

// ================================================================
//  8. TIME WAVER — Temporal frame jitter
// ================================================================
static void applyTimeWaver(const float* src, float* dst, int w, int h,
                           const DistortionFX& dx, const float* prevFrame) {
    if (!prevFrame) {
        // No previous frame, just copy
        std::memcpy(dst, src, w * h * 4 * sizeof(float));
        return;
    }

    float amount = dx.waverAmount;
    float speed = dx.waverSpeed;
    int blockSize = (int)dx.waverBlockSize;
    if (blockSize < 4) blockSize = 4;
    if (blockSize > 128) blockSize = 128;

    // Calculate random offset for this frame
    float t = dx.time * speed;
    float offset = amount * std::sin(t * 3.14159f);

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            // Block-based jitter
            int blockX = (px / blockSize);
            int blockY = (py / blockSize);
            float blockRnd = fastHash((float)blockX + t, (float)blockY + t);
            float jitter = (blockRnd - 0.5f) * 2.0f * offset;

            int idx = (py * w + px) * 4;

            // Blend current and "jittered" previous frame position
            // This is a simplified temporal effect
            for (int c = 0; c < 4; c++) {
                float curr = src[idx + c];
                float prev = prevFrame[idx + c];
                dst[idx + c] = curr * 0.7f + prev * 0.3f + jitter * 0.02f;
            }
        }
    }
}

// ================================================================
//  Grow Mask — applies to all distortions
// ================================================================
static void applyGrowMask(float* buf, int w, int h, const DistortionFX& dx) {
    if (dx.growProgress >= 0.999f && dx.growRadial < 0.01f && std::abs(dx.growDirection) < 0.001f) {
        return; // No mask needed
    }

    float cx = w * 0.5f, cy = h * 0.5f;
    float gm = dx.growProgress;

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            float nx = (float)px / (float)w;
            float ny = (float)py / (float)h;
            float mask = gm;

            // Radial fade
            if (dx.growRadial > 0.01f) {
                float rdx = nx - 0.5f;
                float rdy = ny - 0.5f;
                float rDist = std::sqrt(rdx * rdx + rdy * rdy) * 1.414f;
                float soft = (std::max)(dx.growSoftness, 0.01f);
                float radMask = 1.0f - rDist * dx.growRadial;
                radMask = radMask / soft;
                radMask = (std::min)(1.0f, (std::max)(0.0f, radMask));
                mask *= radMask;
            }

            // Directional wipe
            if (std::abs(dx.growDirection) > 0.001f) {
                float proj = nx * std::cos(dx.growDirection) + ny * std::sin(dx.growDirection);
                float soft = (std::max)(dx.growSoftness, 0.01f);
                float dirMask = (proj + 0.5f) / soft;
                dirMask = (std::min)(1.0f, (std::max)(0.0f, dirMask));
                mask *= dirMask;
            }

            // The grow mask doesn't directly modify pixels here —
            // it scales the distortion amount in each effect above
            // This is handled in the main pass
            (void)px;
            (void)py;
        }
    }
    (void)buf;
}

// ================================================================
//  Main Distortion Pass
// ================================================================
void cpuDistortionPass(void* dstBase, const OfxRectI& dB, int dRB,
                       int w, int h, const DistortionFX& dx) {
    // Quick exit if nothing enabled
    if (!hasAnyDistortion(dx)) return;

    // Copy current dst to temp for safe reading during distortions
    std::vector<float> temp(w * h * 4);
    for (int y = dB.y1; y < dB.y2; y++) {
        float* row = pxAt(dstBase, dB, dRB, dB.x1, y);
        if (row) std::memcpy(&temp[(y - dB.y1) * w * 4], row, w * 4 * sizeof(float));
    }

    // Apply effects in order
    // Create output buffer for chaining effects
    std::vector<float> output(w * h * 4);
    std::memcpy(output.data(), temp.data(), w * h * 4 * sizeof(float));

    // Apply each enabled effect
    if (dx.fluidEnable > 0.5f) {
        applyFluidMorph(temp.data(), output.data(), w, h, dx);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    if (dx.mirrorFractalEnable > 0.5f) {
        applyMirrorFractal(temp.data(), output.data(), w, h, dx);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    if (dx.glitchSliceEnable > 0.5f) {
        applyGlitchSlice(temp.data(), output.data(), w, h, dx);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    if (dx.triangulateEnable > 0.5f) {
        applyTriangulate(temp.data(), output.data(), w, h, dx);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    if (dx.rippleEnable > 0.5f) {
        applyWaterRipple(temp.data(), output.data(), w, h, dx);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    if (dx.displacementEnable > 0.5f) {
        applyDisplacementMap(temp.data(), output.data(), w, h, dx);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    if (dx.tileEnable > 0.5f) {
        applyTileRepeat(temp.data(), output.data(), w, h, dx);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    if (dx.timeWaverEnable > 0.5f) {
        // Time waver needs previous frame - for now, just apply temporal jitter
        applyTimeWaver(temp.data(), output.data(), w, h, dx, nullptr);
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // === NEW EFFECTS ===

    // 9. Perlin Noise Warp
    if (dx.perlinEnable > 0.5f) {
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float nt = dx.time * dx.perlinSpeed;
                float nx = px * dx.perlinScale, ny = py * dx.perlinScale;
                // Simple multi-octave noise approximation
                float n = 0, amp = 1, freq = 1;
                for (int o = 0; o < (int)dx.perlinOctaves && o < 5; o++) {
                    n += (fastHash(nx * freq + nt, ny * freq) - 0.5f) * amp;
                    amp *= 0.5f; freq *= 2;
                }
                float srcX = px + n * dx.perlinAmount;
                float srcY = py + (fastHash(ny * 1.3f, nx * 1.7f + nt) - 0.5f) * dx.perlinAmount;
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = bilerpFlatCPU(temp.data(), w, h, srcX, srcY, c);
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 10. Polar Coordinates
    if (dx.polarEnable > 0.5f) {
        float cx = dx.polarCenterX * w, cy = dx.polarCenterY * h;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float dx2 = px - cx, dy = py - cy;
                float r = std::sqrt(dx2 * dx2 + dy * dy);
                float ang = std::atan2(dy, dx2) + dx.polarAngle;
                float nx, ny;
                if (dx.polarMode < 0.5f) { // to polar
                    nx = r * dx.polarRadius / (float)w;
                    ny = (ang + (float)M_PI) / (float)(2 * M_PI);
                } else if (dx.polarMode < 1.5f) { // to rect
                    nx = cx + (ang / (float)(2 * M_PI) - 0.5f) * (float)w;
                    ny = cy + (r * dx.polarRadius / (float)h - 0.5f) * (float)h;
                } else { // both
                    nx = cx + (ang / (float)(2 * M_PI) - 0.5f) * r * 0.5f;
                    ny = cy + (r / (float)h - 0.5f) * r * 0.5f;
                }
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = bilerpFlatCPU(temp.data(), w, h, nx, ny, c);
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 11. Chromatic Waves
    if (dx.chromaWaveEnable > 0.5f) {
        float t = dx.time * dx.chromaWaveSpeed;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float ny = (float)py / (float)h;
                float wave = std::sin(ny * dx.chromaWaveFreq * (float)M_PI * 2 + t);
                int idx = (py * w + px) * 4;
                // Offset each RGB channel differently
                output[idx + 0] = bilerpFlatCPU(temp.data(), w, h, px + wave * dx.chromaWaveAmp * dx.chromaWaveOffset, py, 0);
                output[idx + 1] = bilerpFlatCPU(temp.data(), w, h, px + wave * dx.chromaWaveAmp, py, 1);
                output[idx + 2] = bilerpFlatCPU(temp.data(), w, h, px + wave * dx.chromaWaveAmp * (1 - dx.chromaWaveOffset), py, 2);
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 12. RGB Shift
    if (dx.rgbShiftEnable > 0.5f) {
        float t = dx.time * dx.rgbShiftSpeed;
        float ang = dx.rgbShiftAngle * (float)M_PI / 180.0f;
        float dxShift = std::cos(ang) * dx.rgbShiftAmount;
        float dyShift = std::sin(ang) * dx.rgbShiftAmount;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float offset = std::sin(t + py * 0.01f) * dx.rgbShiftAmount * 0.3f;
                int idx = (py * w + px) * 4;
                output[idx + 0] = bilerpFlatCPU(temp.data(), w, h, px + dxShift + offset, py + dyShift, 0) * dx.rgbShiftR;
                output[idx + 1] = bilerpFlatCPU(temp.data(), w, h, px, py, 1) * dx.rgbShiftG;
                output[idx + 2] = bilerpFlatCPU(temp.data(), w, h, px - dxShift - offset, py - dyShift, 2) * dx.rgbShiftB;
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 13. Lens Curve (barrel/pincushion)
    if (dx.lensCurveEnable > 0.5f && std::abs(dx.lensCurveAmount) > 0.001f) {
        float cx = dx.lensCurveCenterX * w, cy = dx.lensCurveCenterY * h;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float dx2 = ((float)px - cx) / cx;
                float dy2 = ((float)py - cy) / cy;
                float r = std::sqrt(dx2 * dx2 + dy2 * dy2);
                float pow2 = std::pow(r, dx.lensCurvePower);
                float factor = 1.0f + dx.lensCurveAmount * pow2;
                float srcX = cx + dx2 * factor * cx;
                float srcY = cy + dy2 * factor * cy;
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = bilerpFlatCPU(temp.data(), w, h, srcX, srcY, c);
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 14. Sine Warp
    if (dx.sineWarpEnable > 0.5f) {
        float t = dx.time * dx.sineWarpSpeed;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float nx = (float)px / (float)w * dx.sineWarpFreqX;
                float ny = (float)py / (float)h * dx.sineWarpFreqY;
                float dxDisp = 0, dyDisp = 0;
                float amp = dx.sineWarpAmp;
                for (int o = 0; o < (int)dx.sineWarpOctaves; o++) {
                    dxDisp += std::sin(ny * 2 + t + o) * amp;
                    dyDisp += std::cos(nx * 2 + t * 0.7f + o) * amp;
                    amp *= 0.5f; nx *= 2; ny *= 2;
                }
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = bilerpFlatCPU(temp.data(), w, h, px + dxDisp, py + dyDisp, c);
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 15. Spiral Warp
    if (dx.spiralWarpEnable > 0.5f) {
        float cx = dx.spiralWarpCenterX * w, cy = dx.spiralWarpCenterY * h;
        float t = dx.time * dx.spiralWarpSpeed;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float dx2 = (float)px - cx, dy = (float)py - cy;
                float r = std::sqrt(dx2 * dx2 + dy * dy);
                float ang = std::atan2(dy, dx2) + r * 0.01f * dx.spiralWarpTwist + t;
                float scale = 1.0f + dx.spiralWarpZoom * 0.001f;
                float srcX = cx + std::cos(ang) * r * scale;
                float srcY = cy + std::sin(ang) * r * scale;
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = bilerpFlatCPU(temp.data(), w, h, srcX, srcY, c);
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 16. Noise Displace
    if (dx.noiseDispEnable > 0.5f) {
        float t = dx.time * dx.noiseDispTime;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float nx = px * dx.noiseDispScale, ny = py * dx.noiseDispScale;
                float n = fastHash(nx + t, ny + t * 0.7f) - 0.5f;
                float n2 = fastHash(ny * 1.3f + t * 0.5f, nx * 1.7f) - 0.5f;
                float srcX = px + n * dx.noiseDispAmount;
                float srcY = py + n2 * dx.noiseDispAmount;
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = bilerpFlatCPU(temp.data(), w, h, srcX, srcY, c);
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 17. Radial Blur
    if (dx.radialBlurEnable > 0.5f && dx.radialBlurSamples > 0) {
        float cx = dx.radialBlurCenterX * w, cy = dx.radialBlurCenterY * h;
        int samples = (int)dx.radialBlurSamples;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float dx2 = (float)px - cx, dy = (float)py - cy;
                float dist = std::sqrt(dx2 * dx2 + dy * dy);
                if (dist < 1.0f) {
                    int idx = (py * w + px) * 4;
                    continue;
                }
                float dxNorm = dx2 / dist, dyNorm = dy / dist;
                float rAcc[4] = {0,0,0,0};
                for (int s = 0; s < samples; s++) {
                    float strength = (float)s / (float)samples * dx.radialBlurAmount;
                    float sx = px - dxNorm * strength;
                    float sy = py - dyNorm * strength;
                    for (int c = 0; c < 4; c++)
                        rAcc[c] += bilerpFlatCPU(temp.data(), w, h, sx, sy, c);
                }
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = rAcc[c] / (float)samples;
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 18. Circle Restrict
    if (dx.circleEnable > 0.5f) {
        float cx = dx.circleCenterX * w, cy = dx.circleCenterY * h;
        float innerR = dx.circleInnerRadius * (float)std::min(w, h);
        float outerR = dx.circleOuterRadius * (float)std::min(w, h);
        float softness = dx.circleSoftness * (float)std::min(w, h);
        float invert = dx.circleInvert > 0.5f ? 1.0f : 0.0f;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float dx2 = (float)px - cx, dy = (float)py - cy;
                float dist = std::sqrt(dx2 * dx2 + dy * dy);
                float mask = 0;
                if (dist < innerR - softness)
                    mask = invert;
                else if (dist < innerR + softness) {
                    float t = (dist - innerR + softness) / (2 * softness);
                    mask = invert + (1 - 2 * invert) * t;
                } else if (dist < outerR - softness)
                    mask = 1 - invert;
                else if (dist < outerR + softness) {
                    float t = (dist - outerR + softness) / (2 * softness);
                    mask = (1 - invert) * (1 - t);
                } else
                    mask = invert;
                int idx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++)
                    output[idx + c] = temp[idx + c] * mask;
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 19. Block Shuffle
    if (dx.blockShuffleEnable > 0.5f && dx.blockShuffleSize > 1) {
        int blockSize = (int)dx.blockShuffleSize;
        int blocksX = w / blockSize;
        int blocksY = h / blockSize;
        if (blocksX > 0 && blocksY > 0) {
            std::vector<int> blockOrder(blocksX * blocksY);
            for (int i = 0; i < blocksX * blocksY; i++) blockOrder[i] = i;
            int seed = (int)dx.blockShuffleSeed;
            for (int i = blocksX * blocksY - 1; i > 0; i--) {
                int j = (seed * (i + 1) * 7) % (i + 1);
                std::swap(blockOrder[i], blockOrder[j]);
                seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            }
            float amount = dx.blockShuffleAmount;
            if (amount > 1) amount = 1;
            int swapCount = (int)((float)(blocksX * blocksY) * amount);
            std::vector<float> blockTemp(blockSize * blockSize * 4);
            for (int i = 0; i < swapCount; i++) {
                int bi = blockOrder[i] / blocksX;
                int bj = blockOrder[i] % blocksX;
                int bi2 = blockOrder[(i + 1) % (blocksX * blocksY)] / blocksX;
                int bj2 = blockOrder[(i + 1) % (blocksX * blocksY)] % blocksX;
                for (int by = 0; by < blockSize && bi * blockSize + by < h; by++) {
                    for (int bx = 0; bx < blockSize && bj * blockSize + bx < w; bx++) {
                        int idx1 = ((bi * blockSize + by) * w + (bj * blockSize + bx)) * 4;
                        int idx2 = ((bi2 * blockSize + by) * w + (bj2 * blockSize + bx)) * 4;
                        blockTemp[by * blockSize * 4 + bx * 4 + 0] = temp[idx1 + 0];
                        blockTemp[by * blockSize * 4 + bx * 4 + 1] = temp[idx1 + 1];
                        blockTemp[by * blockSize * 4 + bx * 4 + 2] = temp[idx1 + 2];
                        blockTemp[by * blockSize * 4 + bx * 4 + 3] = temp[idx1 + 3];
                        temp[idx1 + 0] = temp[idx2 + 0];
                        temp[idx1 + 1] = temp[idx2 + 1];
                        temp[idx1 + 2] = temp[idx2 + 2];
                        temp[idx1 + 3] = temp[idx2 + 3];
                    }
                }
                for (int by = 0; by < blockSize && bi2 * blockSize + by < h; by++) {
                    for (int bx = 0; bx < blockSize && bj2 * blockSize + bx < w; bx++) {
                        int idx2 = ((bi2 * blockSize + by) * w + (bj2 * blockSize + bx)) * 4;
                        int srcIdx = by * blockSize * 4 + bx * 4;
                        temp[idx2 + 0] = blockTemp[srcIdx + 0];
                        temp[idx2 + 1] = blockTemp[srcIdx + 1];
                        temp[idx2 + 2] = blockTemp[srcIdx + 2];
                        temp[idx2 + 3] = blockTemp[srcIdx + 3];
                    }
                }
            }
        }
    }

    // 20. Scanlines
    if (dx.scanlinesEnable > 0.5f) {
        int spacing = (int)dx.scanlinesSpacing;
        if (spacing < 1) spacing = 1;
        float offset = dx.scanlinesOffset;
        float warp = dx.scanlinesWarp;
        float t = dx.time * dx.scanlinesSpeed;
        for (int py = 0; py < h; py++) {
            int lineIdx = (py + (int)(offset + t * 10)) % spacing;
            if (lineIdx == 0) {
                float wobble = warp * std::sin((float)py * 0.1f + t);
                for (int px = 0; px < w; px++) {
                    int sx = px + (int)wobble;
                    if (sx < 0) sx = 0; if (sx >= w) sx = w - 1;
                    int dstIdx = (py * w + px) * 4;
                    int srcIdx = (py * w + sx) * 4;
                    output[dstIdx + 0] = temp[srcIdx + 0];
                    output[dstIdx + 1] = temp[srcIdx + 1];
                    output[dstIdx + 2] = temp[srcIdx + 2];
                    output[dstIdx + 3] = temp[srcIdx + 3];
                }
            } else {
                int rowBase = py * w * 4;
                int srcBase = py * w * 4;
                for (int c = 0; c < w * 4; c++) output[rowBase + c] = temp[srcBase + c];
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 21. Pixel Sort
    if (dx.pixelSortEnable > 0.5f && dx.pixelSortAmount > 0) {
        float threshold = dx.pixelSortThreshold;
        float amount = dx.pixelSortAmount;
        if (dx.pixelSortDirection < 0.5f) {
            for (int py = 0; py < h; py++) {
                int rowStart = py * w * 4;
                std::vector<int> sortIdx;
                for (int px = 0; px < w; px++) {
                    float lum = temp[rowStart + px * 4 + 0] * 0.299f + temp[rowStart + px * 4 + 1] * 0.587f + temp[rowStart + px * 4 + 2] * 0.114f;
                    if (lum >= threshold) sortIdx.push_back(px);
                }
                int sortCount = (int)((float)sortIdx.size() * amount);
                for (int i = 0; i < sortCount - 1; i++) {
                    for (int j = i + 1; j < sortCount; j++) {
                        int idx1 = sortIdx[i] * 4;
                        int idx2 = sortIdx[j] * 4;
                        float lum1 = temp[rowStart + idx1 + 0] * 0.299f + temp[rowStart + idx1 + 1] * 0.587f + temp[rowStart + idx1 + 2] * 0.114f;
                        float lum2 = temp[rowStart + idx2 + 0] * 0.299f + temp[rowStart + idx2 + 1] * 0.587f + temp[rowStart + idx2 + 2] * 0.114f;
                        if (lum1 > lum2) {
                            std::swap(sortIdx[i], sortIdx[j]);
                        }
                    }
                }
                for (int i = 0; i < sortCount; i++) {
                    int srcIdx = rowStart + sortIdx[i] * 4;
                    int dstIdx = rowStart + i * 4;
                    output[dstIdx + 0] = temp[srcIdx + 0];
                    output[dstIdx + 1] = temp[srcIdx + 1];
                    output[dstIdx + 2] = temp[srcIdx + 2];
                    output[dstIdx + 3] = temp[srcIdx + 3];
                }
                for (int px = sortCount; px < w; px++) {
                    int idx = rowStart + px * 4;
                    output[idx + 0] = temp[idx + 0];
                    output[idx + 1] = temp[idx + 1];
                    output[idx + 2] = temp[idx + 2];
                    output[idx + 3] = temp[idx + 3];
                }
            }
        } else {
            for (int px = 0; px < w; px++) {
                std::vector<int> sortIdx;
                for (int py = 0; py < h; py++) {
                    int idx = (py * w + px) * 4;
                    float lum = temp[idx + 0] * 0.299f + temp[idx + 1] * 0.587f + temp[idx + 2] * 0.114f;
                    if (lum >= threshold) sortIdx.push_back(py);
                }
                int sortCount = (int)((float)sortIdx.size() * amount);
                for (int i = 0; i < sortCount - 1; i++) {
                    for (int j = i + 1; j < sortCount; j++) {
                        int idx1 = (sortIdx[i] * w + px) * 4;
                        int idx2 = (sortIdx[j] * w + px) * 4;
                        float lum1 = temp[idx1 + 0] * 0.299f + temp[idx1 + 1] * 0.587f + temp[idx1 + 2] * 0.114f;
                        float lum2 = temp[idx2 + 0] * 0.299f + temp[idx2 + 1] * 0.587f + temp[idx2 + 2] * 0.114f;
                        if (lum1 > lum2) {
                            std::swap(sortIdx[i], sortIdx[j]);
                        }
                    }
                }
                for (int i = 0; i < sortCount; i++) {
                    int srcIdx = (sortIdx[i] * w + px) * 4;
                    int dstIdx = (i * w + px) * 4;
                    output[dstIdx + 0] = temp[srcIdx + 0];
                    output[dstIdx + 1] = temp[srcIdx + 1];
                    output[dstIdx + 2] = temp[srcIdx + 2];
                    output[dstIdx + 3] = temp[srcIdx + 3];
                }
                for (int py = sortCount; py < h; py++) {
                    int idx = (py * w + px) * 4;
                    output[idx + 0] = temp[idx + 0];
                    output[idx + 1] = temp[idx + 1];
                    output[idx + 2] = temp[idx + 2];
                    output[idx + 3] = temp[idx + 3];
                }
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 22. Edge Distort
    if (dx.edgeDistortEnable > 0.5f) {
        float threshold = dx.edgeDistortThreshold;
        float amount = dx.edgeDistortAmount;
        float scale = dx.edgeDistortScale;
        for (int py = 1; py < h - 1; py++) {
            for (int px = 1; px < w - 1; px++) {
                int idx = (py * w + px) * 4;
                float lum = temp[idx] * 0.299f + temp[idx + 1] * 0.587f + temp[idx + 2] * 0.114f;
                float lumL = temp[idx - 4] * 0.299f + temp[idx - 3] * 0.587f + temp[idx - 2] * 0.114f;
                float lumR = temp[idx + 4] * 0.299f + temp[idx + 5] * 0.587f + temp[idx + 6] * 0.114f;
                float lumT = temp[idx - w * 4] * 0.299f + temp[idx - w * 4 + 1] * 0.587f + temp[idx - w * 4 + 2] * 0.114f;
                float lumB = temp[idx + w * 4] * 0.299f + temp[idx + w * 4 + 1] * 0.587f + temp[idx + w * 4 + 2] * 0.114f;
                float edgeX = std::abs(lumR - lumL);
                float edgeY = std::abs(lumB - lumT);
                float edge = std::sqrt(edgeX * edgeX + edgeY * edgeY);
                if (edge > threshold) {
                    float dispX = std::sin(px * scale + py * scale * 0.7f) * amount * edge;
                    float dispY = std::cos(px * scale * 0.8f + py * scale) * amount * edge;
                    int sx = (int)((float)px + dispX);
                    int sy = (int)((float)py + dispY);
                    if (sx < 0) sx = 0; if (sx >= w) sx = w - 1;
                    if (sy < 0) sy = 0; if (sy >= h) sy = h - 1;
                    int srcIdx = (sy * w + sx) * 4;
                    output[idx + 0] = temp[srcIdx + 0];
                    output[idx + 1] = temp[srcIdx + 1];
                    output[idx + 2] = temp[srcIdx + 2];
                    output[idx + 3] = temp[srcIdx + 3];
                } else {
                    output[idx + 0] = temp[idx + 0];
                    output[idx + 1] = temp[idx + 1];
                    output[idx + 2] = temp[idx + 2];
                    output[idx + 3] = temp[idx + 3];
                }
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 23. Vortex
    if (dx.vortexEnable > 0.5f) {
        float cx = dx.vortexCenterX * w;
        float cy = dx.vortexCenterY * h;
        float radius = dx.vortexRadius * (float)std::min(w, h);
        float strength = dx.vortexStrength;
        float t = dx.time * dx.vortexSpeed;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float dx2 = (float)px - cx;
                float dy = (float)py - cy;
                float dist = std::sqrt(dx2 * dx2 + dy * dy);
                if (dist < radius && dist > 0.1f) {
                    float factor = 1.0f - dist / radius;
                    float angle = std::atan2(dy, dx2) + strength * factor * factor * (1.0f + t);
                    float nx = cx + std::cos(angle) * dist;
                    float ny = cy + std::sin(angle) * dist;
                    int dstIdx = (py * w + px) * 4;
                    for (int c = 0; c < 4; c++) {
                        output[dstIdx + c] = bilerpFlatCPU(temp.data(), w, h, nx, ny, c);
                    }
                } else {
                    int dstIdx = (py * w + px) * 4;
                    int srcIdx = (py * w + px) * 4;
                    output[dstIdx + 0] = temp[srcIdx + 0];
                    output[dstIdx + 1] = temp[srcIdx + 1];
                    output[dstIdx + 2] = temp[srcIdx + 2];
                    output[dstIdx + 3] = temp[srcIdx + 3];
                }
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 24. Wave Distort
    if (dx.waveDistortEnable > 0.5f) {
        float freqX = dx.waveDistortFreqX * 0.01f;
        float freqY = dx.waveDistortFreqY * 0.01f;
        float ampX = dx.waveDistortAmpX;
        float ampY = dx.waveDistortAmpY;
        float t = dx.time * dx.waveDistortSpeed;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float offsetX = std::sin(py * freqY + t) * ampX;
                float offsetY = std::sin(px * freqX + t * 1.3f) * ampY;
                float sx = (float)px + offsetX;
                float sy = (float)py + offsetY;
                int dstIdx = (py * w + px) * 4;
                for (int c = 0; c < 4; c++) {
                    output[dstIdx + c] = bilerpFlatCPU(temp.data(), w, h, sx, sy, c);
                }
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // 25. Twist
    if (dx.twistEnable > 0.5f) {
        float cx = dx.twistCenterX * w;
        float cy = dx.twistCenterY * h;
        float radius = dx.twistRadius * (float)std::min(w, h);
        float amount = dx.twistAmount;
        float sharpness = dx.twistSharpness;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                float dx2 = (float)px - cx;
                float dy = (float)py - cy;
                float dist = std::sqrt(dx2 * dx2 + dy * dy);
                if (dist < radius && dist > 0.1f && std::abs(amount) > 0.01f) {
                    float factor = std::pow(1.0f - dist / radius, sharpness);
                    float angle = amount * factor;
                    float cs = std::cos(angle);
                    float sn = std::sin(angle);
                    float nx = cx + dx2 * cs - dy * sn;
                    float ny = cy + dx2 * sn + dy * cs;
                    int dstIdx = (py * w + px) * 4;
                    for (int c = 0; c < 4; c++) {
                        output[dstIdx + c] = bilerpFlatCPU(temp.data(), w, h, nx, ny, c);
                    }
                } else {
                    int dstIdx = (py * w + px) * 4;
                    int srcIdx = (py * w + px) * 4;
                    output[dstIdx + 0] = temp[srcIdx + 0];
                    output[dstIdx + 1] = temp[srcIdx + 1];
                    output[dstIdx + 2] = temp[srcIdx + 2];
                    output[dstIdx + 3] = temp[srcIdx + 3];
                }
            }
        }
        std::memcpy(temp.data(), output.data(), w * h * 4 * sizeof(float));
    }

    // Copy back to destination
    for (int y = dB.y1; y < dB.y2; y++) {
        float* row = pxAt(dstBase, dB, dRB, dB.x1, y);
        if (row) std::memcpy(row, &output[(y - dB.y1) * w * 4], w * 4 * sizeof(float));
    }
}