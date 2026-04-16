// DistortionFX.h — New Distortion Effects for ManoStretch
//
// 18 new full-frame distortion effects:
// 1. Fluid Morph    — Metaball-style organic blob merging
// 2. Mirror Fractal — Recursive kaleidoscope (deeper than basic kaleido)
// 3. Glitch Slice   — Horizontal band displacement
// 4. Triangulate    — Delaunay triangle mosaic
// 5. Water Ripple   — Concentric ripples from center
// 6. Displacement Map — Luminance-driven distortion
// 7. Tile/Repeat    — Grid repetition with offset
// 8. Time Waver     — Temporal frame jitter
// 9. Perlin Noise Warp — Organic noise-based distortion
// 10. Polar Coordinates — Rectangular <-> Polar conversion
// 11. Chromatic Waves — Animated RGB wave layers
// 12. RGB Shift — Separate RGB channel offset
// 13. Lens Curve — Barrel/pincushion distortion
// 14. Sine Warp — Multi-layer sinusoidal distortion
// 15. Spiral Warp — Spiral swirl from center
// 16. Noise Displace — Fractal noise displacement
// 17. Radial Blur — Blur emanating from center
// 18. Circle Restrict — Mask to circle/ring shapes
// 19. Block Shuffle — Shuffle blocks of pixels
// 20. Scanlines — CRT-style scanline distortion
// 21. Pixel Sort — Sort pixels by brightness
// 22. Edge Distort — Edge-based displacement
// 23. Vortex — Swirling vortex effect
// 24. Wave Distort — Horizontal/vertical wave distortion
// 25. Twist — Pinch/twist distortion
//
// All effects are keyframeable and work with CPU + CUDA.

#ifndef DISTORTION_FX_H
#define DISTORTION_FX_H

#include <cstddef>

// Forward declaration for OfxRectI
struct OfxRectI;

// ================================================================
//  DistortionFX struct — all parameters for new effects
// ================================================================
struct DistortionFX {
    // === ORIGINAL 8 EFFECTS ===

    // Fluid Morph
    float fluidEnable;
    float fluidBlobCount;
    float fluidThreshold;
    float fluidJitter;
    float fluidSpeed;

    // Mirror Fractal (recursive kaleidoscope)
    float mirrorFractalEnable; float mirrorDepth; float mirrorRotateEach; float mirrorScale; float mirrorSeed;

    // Glitch Slice
    float glitchSliceEnable; float sliceCount; float sliceDisplaceAmt; float sliceRandSeed; float sliceRGBSplit;

    // Triangulate
    float triangulateEnable; float triPointCount; float triEdgeThickness; float triEdgeColor[3]; float triFillVariant;

    // Water Ripple
    float rippleEnable; float rippleCenterX; float rippleCenterY; float rippleFrequency;
    float rippleAmplitude; float rippleDecay; float rippleSpeed; float ripplePhase;

    // Displacement Map
    float displacementEnable; float dispStrength; float dispScale; float dispChannel; float dispDirection;

    // Tile/Repeat
    float tileEnable; float tileRows; float tileCols; float tileOffsetX; float tileOffsetY; float tileRandomSeed;

    // Time Waver
    float timeWaverEnable; float waverAmount; float waverSpeed; float waverBlockSize;

    // === NEW EFFECTS ===

    // 9. Perlin Noise Warp
    float perlinEnable;
    float perlinScale;      // noise scale
    float perlinAmount;     // distortion amount
    float perlinOctaves;    // detail layers
    float perlinSpeed;      // animation speed
    float perlinSeed;       // random seed

    // 10. Polar Coordinates
    float polarEnable;
    float polarMode;        // 0=to polar, 1=to rect, 2=both
    float polarCenterX;     // center X (0-1)
    float polarCenterY;     // center Y (0-1)
    float polarRadius;      // radius scale
    float polarAngle;       // angle offset

    // 11. Chromatic Waves
    float chromaWaveEnable;
    float chromaWaveFreq;   // wave frequency
    float chromaWaveAmp;    // amplitude
    float chromaWaveSpeed;  // animation speed
    float chromaWaveOffset; // phase offset between RGB

    // 12. RGB Shift
    float rgbShiftEnable;
    float rgbShiftAmount;   // shift amount
    float rgbShiftAngle;    // shift direction (0-360)
    float rgbShiftSpeed;    // animation speed
    float rgbShiftR;        // red channel boost
    float rgbShiftG;        // green channel boost
    float rgbShiftB;        // blue channel boost

    // 13. Lens Curve (barrel/pincushion)
    float lensCurveEnable;
    float lensCurveAmount;  // distortion amount (-1=pincushion, 1=barrel)
    float lensCurvePower;   // curve exponent
    float lensCurveCenterX; float lensCurveCenterY;

    // 14. Sine Warp
    float sineWarpEnable;
    float sineWarpFreqX;    // X frequency
    float sineWarpFreqY;    // Y frequency
    float sineWarpAmp;      // amplitude
    float sineWarpOctaves;  // layers
    float sineWarpSpeed;    // animation speed

    // 15. Spiral Warp
    float spiralWarpEnable;
    float spiralWarpTwist;  // twist amount
    float spiralWarpZoom;   // zoom in/out
    float spiralWarpCenterX; float spiralWarpCenterY;
    float spiralWarpSpeed;  // animation speed

    // 16. Noise Displace
    float noiseDispEnable;
    float noiseDispScale;   // noise scale
    float noiseDispAmount;  // displacement amount
    float noiseDispSeed;    // random seed
    float noiseDispChannel; // which channel to use
    float noiseDispTime;    // animate time

    // 17. Radial Blur
    float radialBlurEnable;
    float radialBlurAmount; // blur strength
    float radialBlurCenterX; float radialBlurCenterY;
    float radialBlurSamples; // number of samples
    float radialBlurTime;    // animate

    // 18. Circle Restrict
    float circleEnable;
    float circleCenterX; float circleCenterY;
    float circleInnerRadius;  // inner ring radius (0-1)
    float circleOuterRadius;  // outer ring radius (0-1)
    float circleSoftness;     // edge softness
    float circleInvert;       // invert mask

    // Animation
    float time;

    // Grow mask
    float growProgress;
    float growRadial;
    float growDirection;
    float growSoftness;

    // 19. Block Shuffle
    float blockShuffleEnable;
    float blockShuffleSize;    // block size in pixels
    float blockShuffleAmount; // shuffle strength
    float blockShuffleSeed;   // random seed

    // 20. Scanlines
    float scanlinesEnable;
    float scanlinesSpacing;   // line spacing
    float scanlinesOffset;    // phase offset
    float scanlinesWarp;      // line wobble amount
    float scanlinesSpeed;    // animation

    // 21. Pixel Sort
    float pixelSortEnable;
    float pixelSortThreshold; // brightness threshold
    float pixelSortDirection; // 0=horizontal, 1=vertical
    float pixelSortAmount;    // how much to sort

    // 22. Edge Distort
    float edgeDistortEnable;
    float edgeDistortThreshold; // edge detection threshold
    float edgeDistortAmount;    // displacement amount
    float edgeDistortScale;    // noise scale

    // 23. Vortex
    float vortexEnable;
    float vortexCenterX; float vortexCenterY;
    float vortexStrength;     // rotation strength
    float vortexRadius;        // effect radius
    float vortexSpeed;        // animation

    // 24. Wave Distort
    float waveDistortEnable;
    float waveDistortFreqX;    // horizontal frequency
    float waveDistortFreqY;    // vertical frequency
    float waveDistortAmpX;     // horizontal amplitude
    float waveDistortAmpY;     // vertical amplitude
    float waveDistortSpeed;    // animation

    // 25. Twist
    float twistEnable;
    float twistCenterX; float twistCenterY;
    float twistAmount;        // twist amount
    float twistRadius;        // effect radius
    float twistSharpness;     // edge falloff
};

// Initialize to defaults
static inline void initDistortionFX(DistortionFX& d) {
    // Original 8
    d.fluidEnable = 0; d.fluidBlobCount = 5; d.fluidThreshold = 0.5f; d.fluidJitter = 0.3f; d.fluidSpeed = 1.0f;
    d.mirrorFractalEnable = 0; d.mirrorDepth = 3; d.mirrorRotateEach = 0.2f; d.mirrorScale = 0.9f; d.mirrorSeed = 12345;
    d.glitchSliceEnable = 0; d.sliceCount = 8; d.sliceDisplaceAmt = 50; d.sliceRandSeed = 42; d.sliceRGBSplit = 5;
    d.triangulateEnable = 0; d.triPointCount = 500; d.triEdgeThickness = 0; d.triEdgeColor[0]=0; d.triEdgeColor[1]=0; d.triEdgeColor[2]=0; d.triFillVariant = 0.1f;
    d.rippleEnable = 0; d.rippleCenterX = 0.5f; d.rippleCenterY = 0.5f; d.rippleFrequency = 10; d.rippleAmplitude = 20; d.rippleDecay = 0.8f; d.rippleSpeed = 1.0f; d.ripplePhase = 0;
    d.displacementEnable = 0; d.dispStrength = 20; d.dispScale = 0.02f; d.dispChannel = 3; d.dispDirection = 0;
    d.tileEnable = 0; d.tileRows = 2; d.tileCols = 2; d.tileOffsetX = 0; d.tileOffsetY = 0; d.tileRandomSeed = 777;
    d.timeWaverEnable = 0; d.waverAmount = 1; d.waverSpeed = 1; d.waverBlockSize = 32;

    // New effects defaults
    d.perlinEnable = 0; d.perlinScale = 0.01f; d.perlinAmount = 30; d.perlinOctaves = 3; d.perlinSpeed = 1; d.perlinSeed = 1234;
    d.polarEnable = 0; d.polarMode = 0; d.polarCenterX = 0.5f; d.polarCenterY = 0.5f; d.polarRadius = 1; d.polarAngle = 0;
    d.chromaWaveEnable = 0; d.chromaWaveFreq = 5; d.chromaWaveAmp = 20; d.chromaWaveSpeed = 1; d.chromaWaveOffset = 0.5f;
    d.rgbShiftEnable = 0; d.rgbShiftAmount = 20; d.rgbShiftAngle = 0; d.rgbShiftSpeed = 0; d.rgbShiftR = 1; d.rgbShiftG = 1; d.rgbShiftB = 1;
    d.lensCurveEnable = 0; d.lensCurveAmount = 0; d.lensCurvePower = 2; d.lensCurveCenterX = 0.5f; d.lensCurveCenterY = 0.5f;
    d.sineWarpEnable = 0; d.sineWarpFreqX = 5; d.sineWarpFreqY = 5; d.sineWarpAmp = 20; d.sineWarpOctaves = 2; d.sineWarpSpeed = 0;
    d.spiralWarpEnable = 0; d.spiralWarpTwist = 1; d.spiralWarpZoom = 0; d.spiralWarpCenterX = 0.5f; d.spiralWarpCenterY = 0.5f; d.spiralWarpSpeed = 0;
    d.noiseDispEnable = 0; d.noiseDispScale = 0.02f; d.noiseDispAmount = 30; d.noiseDispSeed = 5678; d.noiseDispChannel = 3; d.noiseDispTime = 0;
    d.radialBlurEnable = 0; d.radialBlurAmount = 10; d.radialBlurCenterX = 0.5f; d.radialBlurCenterY = 0.5f; d.radialBlurSamples = 10; d.radialBlurTime = 0;
    d.circleEnable = 0; d.circleCenterX = 0.5f; d.circleCenterY = 0.5f; d.circleInnerRadius = 0; d.circleOuterRadius = 0.5f; d.circleSoftness = 0.1f; d.circleInvert = 0;

    d.time = 0;
    d.growProgress = 1.0f; d.growRadial = 0; d.growDirection = 0; d.growSoftness = 0.5f;

    // New effects (19-25)
    d.blockShuffleEnable = 0; d.blockShuffleSize = 32; d.blockShuffleAmount = 1; d.blockShuffleSeed = 1234;
    d.scanlinesEnable = 0; d.scanlinesSpacing = 4; d.scanlinesOffset = 0; d.scanlinesWarp = 0; d.scanlinesSpeed = 0;
    d.pixelSortEnable = 0; d.pixelSortThreshold = 0.5f; d.pixelSortDirection = 0; d.pixelSortAmount = 0.5f;
    d.edgeDistortEnable = 0; d.edgeDistortThreshold = 0.1f; d.edgeDistortAmount = 50; d.edgeDistortScale = 0.05f;
    d.vortexEnable = 0; d.vortexCenterX = 0.5f; d.vortexCenterY = 0.5f; d.vortexStrength = 1; d.vortexRadius = 0.5f; d.vortexSpeed = 0;
    d.waveDistortEnable = 0; d.waveDistortFreqX = 5; d.waveDistortFreqY = 5; d.waveDistortAmpX = 50; d.waveDistortAmpY = 50; d.waveDistortSpeed = 0;
    d.twistEnable = 0; d.twistCenterX = 0.5f; d.twistCenterY = 0.5f; d.twistAmount = 1; d.twistRadius = 0.5f; d.twistSharpness = 1;
}

// CPU Implementation
void cpuDistortionPass(void* dstBase, const OfxRectI& dB, int dRB, int w, int h, const DistortionFX& dx);

// Check if any distortion is enabled
static inline bool hasAnyDistortion(const DistortionFX& d) {
    return d.fluidEnable > 0.5f || d.mirrorFractalEnable > 0.5f || d.glitchSliceEnable > 0.5f
        || d.triangulateEnable > 0.5f || d.rippleEnable > 0.5f || d.displacementEnable > 0.5f
        || d.tileEnable > 0.5f || d.timeWaverEnable > 0.5f
        || d.perlinEnable > 0.5f || d.polarEnable > 0.5f || d.chromaWaveEnable > 0.5f
        || d.rgbShiftEnable > 0.5f || d.lensCurveEnable > 0.5f || d.sineWarpEnable > 0.5f
        || d.spiralWarpEnable > 0.5f || d.noiseDispEnable > 0.5f || d.radialBlurEnable > 0.5f
        || d.circleEnable > 0.5f
        || d.blockShuffleEnable > 0.5f || d.scanlinesEnable > 0.5f || d.pixelSortEnable > 0.5f
        || d.edgeDistortEnable > 0.5f || d.vortexEnable > 0.5f || d.waveDistortEnable > 0.5f
        || d.twistEnable > 0.5f;
}

// CUDA Implementation
#ifdef HAS_CUDA
extern "C" void RunCudaDistortionPass(void* stream, float* dst, int w, int h,
    // Original 8
    float fluidEnable, float fluidBlobCount, float fluidThreshold, float fluidJitter, float fluidSpeed,
    float mirrorFractalEnable, float mirrorDepth, float mirrorRotateEach, float mirrorScale, float mirrorSeed,
    float glitchSliceEnable, float sliceCount, float sliceDisplaceAmt, float sliceRandSeed, float sliceRGBSplit,
    float triangulateEnable, float triPointCount, float triEdgeThickness, float triEdgeColorR, float triEdgeColorG, float triEdgeColorB, float triFillVariant,
    float rippleEnable, float rippleCenterX, float rippleCenterY, float rippleFrequency, float rippleAmplitude, float rippleDecay, float rippleSpeed, float ripplePhase,
    float displacementEnable, float dispStrength, float dispScale, float dispChannel, float dispDirection,
    float tileEnable, float tileRows, float tileCols, float tileOffsetX, float tileOffsetY, float tileRandomSeed,
    float timeWaverEnable, float waverAmount, float waverSpeed, float waverBlockSize,
    // New effects (10 more)
    float perlinEnable, float perlinScale, float perlinAmount, float perlinOctaves, float perlinSpeed, float perlinSeed,
    float polarEnable, float polarMode, float polarCenterX, float polarCenterY, float polarRadius, float polarAngle,
    float chromaWaveEnable, float chromaWaveFreq, float chromaWaveAmp, float chromaWaveSpeed, float chromaWaveOffset,
    float rgbShiftEnable, float rgbShiftAmount, float rgbShiftAngle, float rgbShiftSpeed, float rgbShiftR, float rgbShiftG, float rgbShiftB,
    float lensCurveEnable, float lensCurveAmount, float lensCurvePower, float lensCurveCenterX, float lensCurveCenterY,
    float sineWarpEnable, float sineWarpFreqX, float sineWarpFreqY, float sineWarpAmp, float sineWarpOctaves, float sineWarpSpeed,
    float spiralWarpEnable, float spiralWarpTwist, float spiralWarpZoom, float spiralWarpCenterX, float spiralWarpCenterY, float spiralWarpSpeed,
    float noiseDispEnable, float noiseDispScale, float noiseDispAmount, float noiseDispSeed, float noiseDispChannel, float noiseDispTime,
    float radialBlurEnable, float radialBlurAmount, float radialBlurCenterX, float radialBlurCenterY, float radialBlurSamples, float radialBlurTime,
    float circleEnable, float circleCenterX, float circleCenterY, float circleInnerRadius, float circleOuterRadius, float circleSoftness, float circleInvert,
    // New effects (19-25)
    float blockShuffleEnable, float blockShuffleSize, float blockShuffleAmount, float blockShuffleSeed,
    float scanlinesEnable, float scanlinesSpacing, float scanlinesOffset, float scanlinesWarp, float scanlinesSpeed,
    float pixelSortEnable, float pixelSortThreshold, float pixelSortDirection, float pixelSortAmount,
    float edgeDistortEnable, float edgeDistortThreshold, float edgeDistortAmount, float edgeDistortScale,
    float vortexEnable, float vortexCenterX, float vortexCenterY, float vortexStrength, float vortexRadius, float vortexSpeed,
    float waveDistortEnable, float waveDistortFreqX, float waveDistortFreqY, float waveDistortAmpX, float waveDistortAmpY, float waveDistortSpeed,
    float twistEnable, float twistCenterX, float twistCenterY, float twistAmount, float twistRadius, float twistSharpness,
    // Animation
    float time,
    // Grow mask
    float growProgress, float growRadial, float growDirection, float growSoftness);
#endif

#endif // DISTORTION_FX_H