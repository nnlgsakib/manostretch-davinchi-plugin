// ManoStretch CUDA Kernels v5.0
// Soft start, post-FX (brightness/saturation), liquidify distortion

#include <cuda_runtime.h>
#include <math.h>

__global__ void copyKernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
extern "C" void RunCudaCopyKernel(void* stream, const float* src, float* dst, int numPx) {
    int n = numPx * 4, bs = 256;
    copyKernel<<<(n+bs-1)/bs, bs, 0, static_cast<cudaStream_t>(stream)>>>(src, dst, n);
}

__device__ float bilerp(const float* img, int w, int h, float x, float y, int ch) {
    int x0=(int)floorf(x), y0=(int)floorf(y), x1=x0+1, y1=y0+1;
    if(x0<0||y0<0||x1>=w||y1>=h) return 0.f;
    float u=x-x0, v=y-y0;
    return (img[(y0*w+x0)*4+ch]*(1-u)+img[(y0*w+x1)*4+ch]*u)*(1-v)
          +(img[(y1*w+x0)*4+ch]*(1-u)+img[(y1*w+x1)*4+ch]*u)*v;
}

__device__ float phash(float x, float y) {
    int ix = __float_as_int(x * 127.1f + y * 311.7f);
    ix = (ix << 13) ^ ix;
    return fabsf(sinf((float)ix * 0.0001f));
}

__device__ float smoothstep(float t) {
    t = fminf(fmaxf(t, 0.f), 1.f);
    return t * t * (3.f - 2.f * t);
}

// Modes: 0=Linear 1=Spiral 2=Wave 3=Taper 4=Smear 5=Shatter
//
// DISPLACEMENT-BASED STRETCH:  pixels are displaced BACKWARD along
// the drag vector.  Near the start → small displacement (face stays).
// Near the end → large displacement (content stretched from start).
// This produces the organic "face pull" effect.
__global__ void stretchKernel(
    const float* src, float* dst, int w, int h,
    float sx, float sy, float ex, float ey,
    float radius, float strength, int mode,
    float tR, float tG, float tB, float tintAmt,
    float fade, float param1, float param2,
    float startBlend, float postOpacity, float postBright, float postSat,
    float postColorR, float postColorG, float postColorB, float postColorAmt,
    float liqAmount, float liqScale,
    float animProgress, float animGrow, float animEvolve, float animEvolveSpeed,
    float animPulse, float animPulseFreq, float animWobble, float time,
    float colorLock, float colorTolerance)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float ddx=ex-sx, ddy=ey-sy;
    float dLen = sqrtf(ddx*ddx + ddy*ddy);
    if (dLen < 1.f) return;

    float ux=ddx/dLen, uy=ddy/dLen;
    float perpX=-uy, perpY=ux;
    float vx=(float)px-sx, vy=(float)py-sy;
    float t = vx*ux + vy*uy;
    float d = vx*perpX + vy*perpY;

    // Soft end caps: extend valid zone beyond stroke endpoints
    float capLen = radius * 0.5f;
    if (t < -capLen || t > dLen + capLen) return;

    // Clamp tNorm to [0,1] for displacement and mode calculations
    float tNorm = fmaxf(0.f, fminf(t / dLen, 1.f));

    float localR = radius;
    if (mode == 3) localR = radius * fmaxf(0.1f, 1.f - 0.9f * tNorm);

    if (fabsf(d) > localR || localR < 0.5f) return;

    // Perpendicular falloff (smooth)
    float fo = smoothstep(1.f - fabsf(d) / localR);

    // Soft start — smooth ramp-up at the beginning
    float sf = 1.f;
    if (startBlend > 0.001f && tNorm < startBlend)
        sf = smoothstep(tNorm / startBlend);

    // Length fade at end
    float lf = 1.f - tNorm * fade;
    if (lf < 0.f) lf = 0.f;

    // Effect strength = how much displacement at this pixel
    float effect = fo * sf * lf;

    // Smooth end caps: fade to zero at both stroke endpoints
    if (t < 0.f) effect *= smoothstep(1.f + t / capLen);
    if (t > dLen) effect *= smoothstep(1.f - (t - dLen) / capLen);

    if (effect < 0.001f) return;

    // Animation: Growth mask — reveal stroke from start to end
    if (animGrow < 0.999f) {
        if (tNorm > animGrow) return;
        float growEdge = 0.05f;
        if (tNorm > animGrow - growEdge)
            effect *= smoothstep((animGrow - tNorm) / growEdge);
    }

    // Animation: Master progress
    effect *= animProgress;
    if (effect < 0.001f) return;

    // Color Lock: reduce effect for pixels whose color differs from stroke start
    if (colorLock > 0.01f) {
        float refR = bilerp(src, w, h, sx, sy, 0);
        float refG = bilerp(src, w, h, sx, sy, 1);
        float refB = bilerp(src, w, h, sx, sy, 2);
        float cr = bilerp(src, w, h, (float)px, (float)py, 0);
        float cg = bilerp(src, w, h, (float)px, (float)py, 1);
        float cb = bilerp(src, w, h, (float)px, (float)py, 2);
        float dr=cr-refR, dg=cg-refG, db=cb-refB;
        float dist = sqrtf(dr*dr + dg*dg + db*db);
        float tol = fmaxf(0.01f, colorTolerance);
        float colorMask = smoothstep(1.f - dist / tol);
        effect *= 1.f - colorLock * (1.f - colorMask);
        if (effect < 0.001f) return;
    }

    // Displacement backward along drag (use clamped tNorm to avoid reverse stretch in cap zones)
    float disp = tNorm * dLen * strength * effect;
    float srcFx = (float)px - disp * ux;
    float srcFy = (float)py - disp * uy;

    // Animation: Pulse — periodic displacement oscillation
    if (animPulse > 0.001f) {
        float pulse = 1.f + animPulse * sinf(time * animPulseFreq * 6.2832f);
        float newDisp = disp * pulse;
        srcFx = (float)px - newDisp * ux;
        srcFy = (float)py - newDisp * uy;
    }

    // Mode-specific source modifications (on top of displacement)
    switch (mode) {
        default: case 0: case 3:
            // Linear / Taper — pure displacement, no extra offset
            break;
        case 1: { // Spiral — rotate the perpendicular offset
            float angle = tNorm * param1 * 6.2832f;
            float cs = cosf(angle), sn = sinf(angle);
            // Rotate d around the drag axis
            srcFx += (d * (cs - 1.f)) * perpX * effect + (d * sn) * ux * effect;
            srcFy += (d * (cs - 1.f)) * perpY * effect + (d * sn) * uy * effect;
            break;
        }
        case 2: { // Wave — sinusoidal perpendicular wobble
            float waveOff = sinf(tNorm * param1 * 6.2832f) * param2 * localR;
            srcFx += waveOff * perpX * effect;
            srcFy += waveOff * perpY * effect;
            break;
        }
        case 4: { // Smear — non-linear displacement (power curve)
            float smearPow = fmaxf(0.1f, param1);
            float smearT = powf(tNorm, smearPow);
            float smDisp = smearT * dLen * strength * effect;
            srcFx = (float)px - smDisp * ux;
            srcFy = (float)py - smDisp * uy;
            break;
        }
        case 5: { // Shatter — random perpendicular scatter
            float rnd = phash((float)px * 0.1f, (float)py * 0.1f);
            float scatter = param1 * localR * (rnd - 0.5f) * 2.f;
            srcFx += scatter * perpX * effect;
            srcFy += scatter * perpY * effect;
            break;
        }
    }

    // Animation: Time evolve — organic turbulence per frame
    if (animEvolve > 0.01f) {
        float et = time * animEvolveSpeed;
        srcFx += sinf(srcFy * 0.03f + et * 2.7183f) * animEvolve * effect;
        srcFy += cosf(srcFx * 0.03f + et * 3.1416f) * animEvolve * effect;
    }

    // Animation: Wobble — time-varying perpendicular oscillation
    if (animWobble > 0.01f) {
        float wob = sinf(time * 3.0f + tNorm * 6.2832f) * animWobble * effect;
        srcFx += wob * perpX;
        srcFy += wob * perpY;
    }

    // Liquidify: sinusoidal offset on source coords
    if (liqAmount > 0.01f) {
        float freq = liqScale * 0.05f;
        srcFx += sinf(srcFy * freq) * liqAmount * effect;
        srcFy += cosf(srcFx * freq) * liqAmount * effect;
    }

    // Sample all 4 channels from ORIGINAL source
    float rgba[4];
    for (int ch = 0; ch < 4; ch++)
        rgba[ch] = bilerp(src, w, h, srcFx, srcFy, ch);

    // Tint
    if (tintAmt > 0.001f) {
        rgba[0] = rgba[0] * (1.f - tintAmt) + rgba[0] * tR * tintAmt;
        rgba[1] = rgba[1] * (1.f - tintAmt) + rgba[1] * tG * tintAmt;
        rgba[2] = rgba[2] * (1.f - tintAmt) + rgba[2] * tB * tintAmt;
    }

    // Post brightness
    if (fabsf(postBright - 1.f) > 0.001f) {
        rgba[0] *= postBright; rgba[1] *= postBright; rgba[2] *= postBright;
    }

    // Post saturation
    if (fabsf(postSat - 1.f) > 0.001f) {
        float gray = 0.299f*rgba[0] + 0.587f*rgba[1] + 0.114f*rgba[2];
        rgba[0] = gray + (rgba[0] - gray) * postSat;
        rgba[1] = gray + (rgba[1] - gray) * postSat;
        rgba[2] = gray + (rgba[2] - gray) * postSat;
    }

    // Post color overlay
    if (postColorAmt > 0.001f) {
        rgba[0] = rgba[0] * (1.f - postColorAmt) + postColorR * postColorAmt;
        rgba[1] = rgba[1] * (1.f - postColorAmt) + postColorG * postColorAmt;
        rgba[2] = rgba[2] * (1.f - postColorAmt) + postColorB * postColorAmt;
    }

    // Blend: effect controls both displacement AND opacity
    float blend = effect * postOpacity;
    int idx = (py * w + px) * 4;
    for (int ch = 0; ch < 4; ch++)
        dst[idx+ch] = dst[idx+ch] * (1.f - blend) + rgba[ch] * blend;
}

extern "C" void RunCudaStretch(
    void* stream, const float* src, float* dst, int w, int h,
    float sx, float sy, float ex, float ey,
    float radius, float strength, int mode,
    float tR, float tG, float tB, float tintAmt,
    float fade, float p1, float p2,
    float startBlend, float postOpacity, float postBright, float postSat,
    float postColorR, float postColorG, float postColorB, float postColorAmt,
    float liqAmount, float liqScale,
    float animProgress, float animGrow, float animEvolve, float animEvolveSpeed,
    float animPulse, float animPulseFreq, float animWobble, float time,
    float colorLock, float colorTolerance)
{
    dim3 blk(16,16), grd((w+15)/16,(h+15)/16);
    stretchKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        src, dst, w, h, sx, sy, ex, ey, radius, strength, mode,
        tR, tG, tB, tintAmt, fade, p1, p2,
        startBlend, postOpacity, postBright, postSat,
        postColorR, postColorG, postColorB, postColorAmt,
        liqAmount, liqScale,
        animProgress, animGrow, animEvolve, animEvolveSpeed,
        animPulse, animPulseFreq, animWobble, time,
        colorLock, colorTolerance);
}