// ManoStretch CUDA Kernels v6.0
// Modular surrealism art tool — Stretch, Surrealism, Dreamcore passes

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
//        6=Mirror 7=Melt 8=Vortex 9=Fractal 10=Glitch 11=Dream
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
    float colorLock, float colorTolerance,
    float chromaAb, float posterize, float hueShift, float solarize, float colorInvert)
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
        case 6: { // Mirror — kaleidoscopic
            float segs = fmaxf(1.f, floorf(param1));
            float angOrig = atan2f(d, fmaxf(0.001f, t));
            float sector = 6.2832f / segs;
            float angMir = fmodf(fabsf(angOrig), sector);
            if (angMir > sector*0.5f) angMir = sector - angMir;
            float rr = sqrtf(t*t + d*d);
            srcFx = sx + rr * cosf(angMir) * ux - rr * sinf(angMir) * perpX;
            srcFy = sy + rr * cosf(angMir) * uy - rr * sinf(angMir) * perpY;
            srcFx -= disp * ux * 0.5f;
            srcFy -= disp * uy * 0.5f;
            break;
        }
        case 7: { // Melt — downward drip
            float meltPow = fmaxf(0.5f, param1);
            float meltAmt = powf(tNorm, meltPow) * param2 * localR * effect;
            srcFy -= meltAmt;
            break;
        }
        case 8: { // Vortex — spiral swirl
            float cx2 = (sx + ex) * 0.5f, cy2 = (sy + ey) * 0.5f;
            float dvx = (float)px - cx2, dvy = (float)py - cy2;
            float dist2 = sqrtf(dvx*dvx + dvy*dvy);
            float maxR = dLen * 0.5f + localR;
            float falloff2 = (dist2 < maxR) ? (1.f - dist2/maxR) : 0.f;
            float angle2 = param1 * effect * falloff2 * 6.2832f;
            float cs2 = cosf(angle2), sn2 = sinf(angle2);
            srcFx = cx2 + dvx * cs2 - dvy * sn2;
            srcFy = cy2 + dvx * sn2 + dvy * cs2;
            break;
        }
        case 9: { // Fractal — iterative warp
            float scale2 = fmaxf(0.1f, param1) * 0.02f;
            int iters = max(1, min((int)(param2 * 4.f), 8));
            float fx2 = srcFx, fy2 = srcFy;
            for (int it = 0; it < iters; it++) {
                fx2 += sinf(fy2 * scale2) * localR * 0.15f * effect;
                fy2 += cosf(fx2 * scale2) * localR * 0.15f * effect;
            }
            srcFx = fx2; srcFy = fy2;
            break;
        }
        case 10: { // Glitch — horizontal band displacement
            float bandH = fmaxf(2.f, param1 * 10.f);
            float band = floorf((float)py / bandH);
            float rnd2 = phash(band * 0.1f, band * 0.3f);
            float offset2 = (rnd2 - 0.5f) * 2.f * param2 * localR * effect;
            srcFx += offset2;
            break;
        }
        case 11: { // Dream — multi-layer warp
            float freq1 = param1 * 0.02f, freq2 = param1 * 0.05f;
            float amp = param2 * localR * 0.3f * effect;
            srcFx += sinf(srcFy * freq1 + time * 0.5f) * amp;
            srcFy += cosf(srcFx * freq2 + time * 0.7f) * amp;
            srcFx += sinf(srcFy * freq2 * 1.5f + time * 0.3f) * amp * 0.5f;
            srcFy += cosf(srcFx * freq1 * 1.3f + time * 0.4f) * amp * 0.5f;
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

    // Chromatic Aberration: offset R and B channels along stroke axis
    if (chromaAb > 0.01f) {
        float caOff = chromaAb * effect;
        rgba[0] = bilerp(src, w, h, srcFx + caOff*ux, srcFy + caOff*uy, 0);
        rgba[2] = bilerp(src, w, h, srcFx - caOff*ux, srcFy - caOff*uy, 2);
    }

    // Posterize: reduce color levels
    if (posterize > 1.5f) {
        float lvl = floorf(posterize);
        for (int c=0; c<3; c++)
            rgba[c] = floorf(rgba[c] * lvl) / lvl;
    }

    // Hue Shift: rotate hue using color matrix
    if (fabsf(hueShift) > 0.001f) {
        float cosA = cosf(hueShift), sinA = sinf(hueShift);
        float s3 = 0.57735f, omc = 1.f - cosA, th = 1.f/3.f;
        float r=rgba[0], g=rgba[1], b=rgba[2];
        rgba[0] = r*(cosA+omc*th) + g*(omc*th-s3*sinA) + b*(omc*th+s3*sinA);
        rgba[1] = r*(omc*th+s3*sinA) + g*(cosA+omc*th) + b*(omc*th-s3*sinA);
        rgba[2] = r*(omc*th-s3*sinA) + g*(omc*th+s3*sinA) + b*(cosA+omc*th);
        for (int c=0; c<3; c++) rgba[c] = fmaxf(0.f, rgba[c]);
    }

    // Solarize: invert pixels above threshold
    if (solarize > 0.01f) {
        float thresh = 1.f - solarize;
        for (int c=0; c<3; c++) {
            if (rgba[c] > thresh)
                rgba[c] = rgba[c]*(1.f-solarize) + (1.f-rgba[c])*solarize;
        }
    }

    // Color Invert: blend toward negative
    if (colorInvert > 0.01f) {
        for (int c=0; c<3; c++)
            rgba[c] = rgba[c]*(1.f-colorInvert) + (1.f-rgba[c])*colorInvert;
    }

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
    float colorLock, float colorTolerance,
    float chromaAb, float posterize, float hueShift, float solarize, float colorInvert)
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
        colorLock, colorTolerance,
        chromaAb, posterize, hueShift, solarize, colorInvert);
}

// ================================================================
//  Global atmosphere FX kernel — full-frame pass
// ================================================================
__global__ void globalFXKernel(float* dst, int w, int h,
    float vignette, float grain, float scanlines, float dreamHaze,
    float globalHueShift, float pixelate, float mirrorGlobal, float time)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int idx = (py * w + px) * 4;

    // Pixelate
    if (pixelate > 1.5f) {
        int blk = (int)pixelate;
        int bx = (px / blk) * blk + blk/2;
        int by = (py / blk) * blk + blk/2;
        bx = min(bx, w-1); by = min(by, h-1);
        int bidx = (by * w + bx) * 4;
        dst[idx+0]=dst[bidx+0]; dst[idx+1]=dst[bidx+1]; dst[idx+2]=dst[bidx+2];
    }

    // Global mirror
    if (mirrorGlobal > 0.5f && px > w/2) {
        int mx = w - 1 - px;
        int midx = (py * w + mx) * 4;
        dst[idx+0]=dst[midx+0]; dst[idx+1]=dst[midx+1]; dst[idx+2]=dst[midx+2];
    }

    float nx = (float)px / (float)w;
    float ny = (float)py / (float)h;

    // Vignette
    if (vignette > 0.01f) {
        float vx2 = nx - 0.5f, vy2 = ny - 0.5f;
        float vDist = sqrtf(vx2*vx2 + vy2*vy2) * 1.414f;
        float vFade = 1.f - vignette * vDist * vDist;
        if (vFade < 0.f) vFade = 0.f;
        dst[idx+0] *= vFade; dst[idx+1] *= vFade; dst[idx+2] *= vFade;
    }

    // Dream Haze
    if (dreamHaze > 0.01f) {
        float lum = 0.299f*dst[idx+0] + 0.587f*dst[idx+1] + 0.114f*dst[idx+2];
        float glow = lum * lum * dreamHaze * 2.f;
        dst[idx+0] += glow; dst[idx+1] += glow; dst[idx+2] += glow;
    }

    // Global Hue Shift
    if (fabsf(globalHueShift) > 0.001f) {
        float cosH = cosf(globalHueShift), sinH = sinf(globalHueShift);
        float s3 = 0.57735f, omc = 1.f - cosH, th = 1.f/3.f;
        float r=dst[idx+0], g=dst[idx+1], b=dst[idx+2];
        dst[idx+0] = r*(cosH+omc*th) + g*(omc*th-s3*sinH) + b*(omc*th+s3*sinH);
        dst[idx+1] = r*(omc*th+s3*sinH) + g*(cosH+omc*th) + b*(omc*th-s3*sinH);
        dst[idx+2] = r*(omc*th-s3*sinH) + g*(omc*th+s3*sinH) + b*(cosH+omc*th);
        for (int c=0; c<3; c++) dst[idx+c] = fmaxf(0.f, dst[idx+c]);
    }

    // Scanlines
    if (scanlines > 0.01f) {
        int lineH = max(2, (int)(4.f / (scanlines + 0.01f)));
        if ((py % lineH) < lineH/2) {
            float dim = 1.f - scanlines * 0.6f;
            dst[idx+0] *= dim; dst[idx+1] *= dim; dst[idx+2] *= dim;
        }
    }

    // Film Grain
    if (grain > 0.01f) {
        float rnd = phash((float)px + time * 100.f, (float)py + time * 73.f);
        float noise = (rnd - 0.5f) * 2.f * grain * 0.15f;
        dst[idx+0] += noise; dst[idx+1] += noise; dst[idx+2] += noise;
    }
}

extern "C" void RunCudaGlobalFX(
    void* stream, float* dst, int w, int h,
    float vignette, float grain, float scanlines, float dreamHaze,
    float globalHueShift, float pixelate, float mirrorGlobal, float time)
{
    bool hasAny = vignette > 0.01f || grain > 0.01f || scanlines > 0.01f
               || dreamHaze > 0.01f || fabsf(globalHueShift) > 0.001f
               || pixelate > 1.5f || mirrorGlobal > 0.5f;
    if (!hasAny) return;
    dim3 blk(16,16), grd((w+15)/16,(h+15)/16);
    globalFXKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        dst, w, h, vignette, grain, scanlines, dreamHaze,
        globalHueShift, pixelate, mirrorGlobal, time);
}

// ================================================================
//  Surrealism module — full-frame distortions & color FX (CUDA)
// ================================================================
__global__ void surrealismKernel(const float* srcCopy, float* dst, int w, int h,
    float fractalAmt, float fractalScale, float kaleidoSegs,
    float vortexAmt, float meltAmt, float glitchAmt, float glitchBand,
    float waveFreq, float waveAmp, float chromaAb, float posterize,
    float hueShift, float solarize, float colorInvert, float time,
    float spatialSpinSpeed, float spatialKaleidoRotate,
    float spatialMeltPulse, float spatialMeltPulseFreq, float spatialEvolveSpeed,
    float growProgress, float growRadial, float growDirection, float growSoftness)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float lx = (float)px, ly = (float)py;
    float srcX = lx, srcY = ly;
    float cx = w * 0.5f, cy = h * 0.5f;

    // Kaleidoscope (+ animated rotation)
    if (kaleidoSegs > 1.5f) {
        float dx = srcX - cx, dy = srcY - cy;
        float ang = atan2f(dy, dx);
        if (fabsf(spatialKaleidoRotate) > 0.001f)
            ang -= spatialKaleidoRotate * time * 0.15f;
        float segs = floorf(kaleidoSegs);
        float sector = 6.2832f / segs;
        float angM = fmodf(fabsf(ang), sector);
        if (angM > sector * 0.5f) angM = sector - angM;
        float rr = sqrtf(dx*dx + dy*dy);
        srcX = cx + rr * cosf(angM);
        srcY = cy + rr * sinf(angM);
    }

    // Vortex swirl (+ animated spin)
    if (vortexAmt > 0.01f || fabsf(spatialSpinSpeed) > 0.001f) {
        float dx = srcX - cx, dy = srcY - cy;
        float dist = sqrtf(dx*dx + dy*dy);
        float maxR = fmaxf((float)w, (float)h) * 0.5f;
        float falloff = (dist < maxR) ? (1.f - dist/maxR) : 0.f;
        float angle = vortexAmt * falloff * falloff * 6.2832f;
        angle += spatialSpinSpeed * time * 0.15f * falloff;
        float cs = cosf(angle), sn = sinf(angle);
        srcX = cx + dx*cs - dy*sn;
        srcY = cy + dx*sn + dy*cs;
    }

    // Fractal warp (evolve speeds up time)
    if (fractalAmt > 0.01f) {
        float scale = fractalScale * 0.02f;
        float et = time * (1.f + spatialEvolveSpeed);
        float ffx = srcX, ffy = srcY;
        for (int it = 0; it < 4; it++) {
            ffx += sinf(ffy * scale + et * 0.3f) * fractalAmt;
            ffy += cosf(ffx * scale + et * 0.2f) * fractalAmt;
        }
        srcX = ffx; srcY = ffy;
    }

    // General evolve displacement
    if (spatialEvolveSpeed > 0.01f) {
        float evT = time * spatialEvolveSpeed * 0.7f;
        float evAmt = spatialEvolveSpeed * 3.f;
        srcX += sinf(ly * 0.05f + evT) * evAmt;
        srcY += cosf(lx * 0.05f + evT * 0.7f) * evAmt;
    }

    // Melt (+ animated pulse)
    { float melt = meltAmt;
      if (spatialMeltPulse > 0.01f)
          melt += spatialMeltPulse * sinf(time * spatialMeltPulseFreq * 0.3f);
      if (fabsf(melt) > 0.01f) {
        float ny = ly / (float)h;
        srcY -= melt * ny * ny * (float)h * 0.1f;
      }
    }

    // Glitch
    if (glitchAmt > 0.01f) {
        float bandH = fmaxf(2.f, glitchBand * 10.f);
        float band = floorf(ly / bandH);
        float rnd = phash(band * 0.1f + time * 0.01f, band * 0.3f);
        srcX += (rnd - 0.5f) * 2.f * glitchAmt * (float)w * 0.05f;
    }

    // Wave
    if (waveAmp > 0.01f) {
        float nx = lx / (float)w, ny = ly / (float)h;
        srcX += sinf(ny * waveFreq * 6.2832f + time) * waveAmp;
        srcY += cosf(nx * waveFreq * 6.2832f + time*0.7f) * waveAmp;
    }

    // Distortion Grow mask
    if (growProgress < 0.999f || growRadial > 0.01f) {
        float gm = growProgress;
        if (growRadial > 0.01f) {
            float rdx = (lx - cx) / (cx + 0.001f);
            float rdy = (ly - cy) / (cy + 0.001f);
            float rDist = sqrtf(rdx*rdx + rdy*rdy);
            float soft = fmaxf(growSoftness, 0.01f);
            float radMask = 1.f - rDist * growRadial;
            radMask = radMask / soft;
            radMask = fminf(1.f, fmaxf(0.f, radMask));
            gm *= radMask;
        }
        if (fabsf(growDirection) > 0.001f) {
            float nx = (lx / (float)w) - 0.5f;
            float ny = (ly / (float)h) - 0.5f;
            float proj = nx * cosf(growDirection) + ny * sinf(growDirection);
            float soft = fmaxf(growSoftness, 0.01f);
            float dirMask = (proj + 0.5f) / soft;
            dirMask = fminf(1.f, fmaxf(0.f, dirMask));
            gm *= dirMask;
        }
        srcX = lx + (srcX - lx) * gm;
        srcY = ly + (srcY - ly) * gm;
    }

    // Sample from source copy
    float rgba[4];
    for (int c = 0; c < 4; c++)
        rgba[c] = bilerp(srcCopy, w, h, srcX, srcY, c);

    // Chromatic Aberration
    if (chromaAb > 0.01f) {
        rgba[0] = bilerp(srcCopy, w, h, srcX + chromaAb, srcY, 0);
        rgba[2] = bilerp(srcCopy, w, h, srcX - chromaAb, srcY, 2);
    }

    // Posterize
    if (posterize > 1.5f) {
        float lvl = floorf(posterize);
        for (int c=0; c<3; c++)
            rgba[c] = floorf(rgba[c] * lvl) / lvl;
    }

    // Hue Shift
    if (fabsf(hueShift) > 0.001f) {
        float cosA = cosf(hueShift), sinA = sinf(hueShift);
        float s3 = 0.57735f, omc = 1.f - cosA, th = 1.f/3.f;
        float r=rgba[0], g=rgba[1], b=rgba[2];
        rgba[0] = r*(cosA+omc*th) + g*(omc*th-s3*sinA) + b*(omc*th+s3*sinA);
        rgba[1] = r*(omc*th+s3*sinA) + g*(cosA+omc*th) + b*(omc*th-s3*sinA);
        rgba[2] = r*(omc*th-s3*sinA) + g*(omc*th+s3*sinA) + b*(cosA+omc*th);
        for (int c=0; c<3; c++) rgba[c] = fmaxf(0.f, rgba[c]);
    }

    // Solarize
    if (solarize > 0.01f) {
        float thresh = 1.f - solarize;
        for (int c=0; c<3; c++) {
            if (rgba[c] > thresh)
                rgba[c] = rgba[c]*(1.f-solarize) + (1.f-rgba[c])*solarize;
        }
    }

    // Color Invert
    if (colorInvert > 0.01f) {
        for (int c=0; c<3; c++)
            rgba[c] = rgba[c]*(1.f-colorInvert) + (1.f-rgba[c])*colorInvert;
    }

    int idx = (py * w + px) * 4;
    for (int c = 0; c < 4; c++) dst[idx+c] = rgba[c];
}

extern "C" void RunCudaSurrealismPass(
    void* stream, float* dst, int w, int h,
    float fractalAmt, float fractalScale, float kaleidoSegs,
    float vortexAmt, float meltAmt, float glitchAmt, float glitchBand,
    float waveFreq, float waveAmp, float chromaAb, float posterize,
    float hueShift, float solarize, float colorInvert, float time,
    float spatialSpinSpeed, float spatialKaleidoRotate,
    float spatialMeltPulse, float spatialMeltPulseFreq, float spatialEvolveSpeed,
    float growProgress, float growRadial, float growDirection, float growSoftness)
{
    bool hasDist = fractalAmt > 0.01f || kaleidoSegs > 1.5f
                || vortexAmt > 0.01f || fabsf(meltAmt) > 0.01f
                || glitchAmt > 0.01f || waveAmp > 0.01f
                || fabsf(spatialSpinSpeed) > 0.001f
                || spatialMeltPulse > 0.01f
                || spatialEvolveSpeed > 0.01f;
    bool hasColor = chromaAb > 0.01f || posterize > 1.5f
                 || fabsf(hueShift) > 0.001f || solarize > 0.01f
                 || colorInvert > 0.01f;
    if (!hasDist && !hasColor) return;

    // Allocate temp copy of dst for safe reading during distortions
    int n = w * h * 4;
    float* tmp = nullptr;
    cudaMalloc(&tmp, n * sizeof(float));
    cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                    static_cast<cudaStream_t>(stream));

    dim3 blk(16,16), grd((w+15)/16,(h+15)/16);
    surrealismKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        tmp, dst, w, h,
        fractalAmt, fractalScale, kaleidoSegs,
        vortexAmt, meltAmt, glitchAmt, glitchBand,
        waveFreq, waveAmp, chromaAb, posterize,
        hueShift, solarize, colorInvert, time,
        spatialSpinSpeed, spatialKaleidoRotate,
        spatialMeltPulse, spatialMeltPulseFreq, spatialEvolveSpeed,
        growProgress, growRadial, growDirection, growSoftness);

    cudaFree(tmp);
}

// ================================================================
//  NEW DISTORTION EFFECTS - CUDA KERNELS
// ================================================================

// 1. Fluid Morph - Metaball-style organic blob merging
__global__ void fluidMorphKernel(const float* src, float* dst, int w, int h,
    float fluidBlobCount, float fluidThreshold, float fluidJitter, float fluidSpeed, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int blobCount = (int)fminf(fmaxf(fluidBlobCount, 3.0f), 10.0f);

    // Generate blob positions on GPU (deterministic based on frame)
    float cx = w * 0.5f, cy = h * 0.5f;
    float maxR = fminf((float)w, (float)h);
    float field = 0;

    for (int i = 0; i < blobCount; i++) {
        float seed = (float)(i + 1) * 1000.0f + fluidSpeed * 100.0f;
        float bx = phash(seed, time * 0.1f) * (float)w;
        float by = phash(seed + 1.0f, time * 0.1f) * (float)h;
        float ang = phash(seed + 2.0f, time * 0.05f) * 6.28318f;
        float speed = 20.0f + phash(seed + 3.0f, time * 0.03f) * 30.0f;
        bx += cosf(ang) * speed * time * fluidSpeed;
        by += sinf(ang) * speed * time * fluidSpeed;
        bx = fmodf(bx + w * 2, (float)w);
        by = fmodf(by + h * 2, (float)h);

        float dx2 = (float)px - bx;
        float dy2 = (float)py - by;
        float dist = sqrtf(dx2 * dx2 + dy2 * dy2);
        float radius = maxR * 0.15f;
        field += radius * radius / (dist * dist + 1.0f);
    }

    int idx = (py * w + px) * 4;
    if (field > fluidThreshold) {
        // Find closest blob center for warp
        float minDist = 1e10f, closestX = 0, closestY = 0;
        for (int i = 0; i < blobCount; i++) {
            float seed = (float)(i + 1) * 1000.0f + fluidSpeed * 100.0f;
            float bx = phash(seed, time * 0.1f) * (float)w;
            float by = phash(seed + 1.0f, time * 0.1f) * (float)h;
            float ang = phash(seed + 2.0f, time * 0.05f) * 6.28318f;
            float speed = 20.0f + phash(seed + 3.0f, time * 0.03f) * 30.0f;
            bx += cosf(ang) * speed * time * fluidSpeed;
            by += sinf(ang) * speed * time * fluidSpeed;
            bx = fmodf(bx + w * 2, (float)w);
            by = fmodf(by + h * 2, (float)h);
            float d = (px - bx) * (px - bx) + (py - by) * (py - by);
            if (d < minDist) { minDist = d; closestX = bx; closestY = by; }
        }
        float warp = fluidJitter * 0.3f;
        float srcX = (float)px + (closestX - (float)px) * warp;
        float srcY = (float)py + (closestY - (float)py) * warp;
        for (int c = 0; c < 4; c++)
            dst[idx + c] = bilerp(src, w, h, srcX, srcY, c);
    } else {
        dst[idx + 0] = src[idx + 0];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = src[idx + 2];
        dst[idx + 3] = src[idx + 3];
    }
}

// 2. Mirror Fractal - Recursive kaleidoscope
__global__ void mirrorFractalKernel(const float* src, float* dst, int w, int h,
    float mirrorDepth, float mirrorRotateEach, float mirrorScale) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int depth = (int)fminf(fmaxf(mirrorDepth, 2.0f), 8.0f);
    float cx = w * 0.5f, cy = h * 0.5f;

    float srcX = (float)px, srcY = (float)py;

    for (int level = 0; level < depth; level++) {
        float dx2 = srcX - cx;
        float dy2 = srcY - cy;
        float angle = atan2f(dy2, dx2);
        float rotPerLevel = mirrorRotateEach * 6.28318f * (float)level / (float)depth;
        angle += rotPerLevel;
        float radius = sqrtf(dx2 * dx2 + dy2 * dy2);
        radius *= mirrorScale;
        int segs = 2 + level;
        float sector = 6.28318f / (float)segs;
        float angMod = fmodf(fabsf(angle), sector);
        if (angMod > sector * 0.5f) angMod = sector - angMod;
        srcX = cx + radius * cosf(angMod);
        srcY = cy + radius * sinf(angMod);
    }

    int idx = (py * w + px) * 4;
    for (int c = 0; c < 4; c++)
        dst[idx + c] = bilerp(src, w, h, srcX, srcY, c);
}

// 3. Glitch Slice - Horizontal band displacement
__global__ void glitchSliceKernel(const float* src, float* dst, int w, int h,
    float sliceCount, float sliceDisplaceAmt, float sliceRandSeed, float sliceRGBSplit, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int nSlices = (int)fminf(fmaxf(sliceCount, 4.0f), 32.0f);
    float bandHeight = (float)h / (float)nSlices;
    int band = (int)((float)py / bandHeight);
    float rnd = phash((float)band + sliceRandSeed, time * 0.1f);
    float offset = (rnd - 0.5f) * 2.0f * sliceDisplaceAmt;
    offset *= (0.5f + 0.5f * sinf(time * 0.5f + band * 0.5f));

    int idx = (py * w + px) * 4;
    float srcX = (float)px + offset;
    dst[idx + 0] = bilerp(src, w, h, srcX + sliceRGBSplit, (float)py, 0);
    dst[idx + 1] = bilerp(src, w, h, srcX, (float)py, 1);
    dst[idx + 2] = bilerp(src, w, h, srcX - sliceRGBSplit, (float)py, 2);
    dst[idx + 3] = bilerp(src, w, h, srcX, (float)py, 3);
}

// 4. Triangulate - Delaunay mosaic (simplified GPU version)
__global__ void triangulateKernel(const float* src, float* dst, int w, int h,
    float triPointCount, float triEdgeThickness, float triEdgeColorR, float triEdgeColorG, float triEdgeColorB,
    float triFillVariant, float mirrorSeed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int pointCount = (int)fminf(fmaxf(triPointCount, 50.0f), 2000.0f);

    // Find 3 closest points (simplified - uses pseudo-random sampling)
    float best1 = 1e10f, best2 = 1e10f, best3 = 1e10f;
    float cx1 = 0, cy1 = 0, cx2 = 0, cy2 = 0, cx3 = 0, cy3 = 0;

    // Sample a subset of points for performance
    for (int i = 0; i < fminf(pointCount, 200); i++) {
        float seed = (float)i + mirrorSeed;
        float sx = phash(seed, 111.0f) * (float)w;
        float sy = phash(seed, 222.0f) * (float)h;
        float d = (px - sx) * (px - sx) + (py - sy) * (py - sy);
        if (d < best1) { best3 = best2; cx3 = cx2; cy3 = cy2; best2 = best1; cx2 = cx1; cy2 = cy1; best1 = d; cx1 = sx; cy1 = sy; }
        else if (d < best2) { best3 = best2; cx3 = cx2; cy2 = cy2; best2 = d; cx2 = sx; cy2 = sy; }
        else if (d < best3) { best3 = d; cx3 = sx; cy3 = sy; }
    }

    float tcx = (cx1 + cx2 + cx3) / 3.0f;
    float tcy = (cy1 + cy2 + cy3) / 3.0f;
    float variant = triFillVariant * phash((float)px * 0.01f, (float)py * 0.01f);

    int idx = (py * w + px) * 4;
    for (int c = 0; c < 4; c++) {
        float col = bilerp(src, w, h, tcx, tcy, c);
        col += (variant - triFillVariant * 0.5f);
        dst[idx + c] = col;
    }

    // Edge detection
    if (triEdgeThickness > 0.01f) {
        float minD = sqrtf(best1);
        float edgeWidth = triEdgeThickness * 10.0f;
        if (minD < edgeWidth) {
            float edgeFactor = minD / edgeWidth;
            dst[idx + 0] = dst[idx + 0] * edgeFactor + triEdgeColorR * (1.0f - edgeFactor);
            dst[idx + 1] = dst[idx + 1] * edgeFactor + triEdgeColorG * (1.0f - edgeFactor);
            dst[idx + 2] = dst[idx + 2] * edgeFactor + triEdgeColorB * (1.0f - edgeFactor);
        }
    }
}

// 5. Water Ripple - Concentric ripples
__global__ void waterRippleKernel(const float* src, float* dst, int w, int h,
    float rippleCenterX, float rippleCenterY, float rippleFrequency, float rippleAmplitude,
    float rippleDecay, float rippleSpeed, float ripplePhase) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float cx = rippleCenterX * (float)w;
    float cy = rippleCenterY * (float)h;
    float dx2 = (float)px - cx;
    float dy = (float)py - cy;
    float dist = sqrtf(dx2 * dx2 + dy * dy);

    float wave = sinf(dist * rippleFrequency * 0.01f - rippleSpeed * 10.0f + ripplePhase);
    float falloff = expf(-dist * rippleDecay * 0.01f);
    float offset = wave * rippleAmplitude * falloff;

    int idx = (py * w + px) * 4;
    if (dist > 1.0f) {
        float srcX = (float)px + (dx2 / dist) * offset;
        float srcY = (float)py + (dy / dist) * offset;
        for (int c = 0; c < 4; c++)
            dst[idx + c] = bilerp(src, w, h, srcX, srcY, c);
    } else {
        dst[idx + 0] = src[idx + 0];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = src[idx + 2];
        dst[idx + 3] = src[idx + 3];
    }
}

// 6. Displacement Map
__global__ void displacementMapKernel(const float* src, float* dst, int w, int h,
    float dispStrength, float dispScale, float dispChannel, float dispDirection) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int idx = (py * w + px) * 4;
    float dispVal;
    int ch = (int)dispChannel;
    if (ch == 3) // Luminance
        dispVal = 0.299f * src[idx] + 0.587f * src[idx + 1] + 0.114f * src[idx + 2];
    else
        dispVal = src[idx + ch];

    dispVal = dispVal - 0.5f;
    float offsetX = 0, offsetY = 0;
    int dir = (int)dispDirection;

    if (dir == 0 || dir == 1)
        offsetX = dispVal * dispStrength * cosf(px * dispScale + py * dispScale * 0.5f);
    if (dir == 0 || dir == 2)
        offsetY = dispVal * dispStrength * sinf(py * dispScale + px * dispScale * 0.5f);

    float srcX = (float)px + offsetX;
    float srcY = (float)py + offsetY;

    for (int c = 0; c < 4; c++)
        dst[idx + c] = bilerp(src, w, h, srcX, srcY, c);
}

// 7. Tile/Repeat
__global__ void tileRepeatKernel(const float* src, float* dst, int w, int h,
    float tileRows, float tileCols, float tileOffsetX, float tileOffsetY, float tileRandomSeed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int rows = (int)fminf(fmaxf(tileRows, 1.0f), 10.0f);
    int cols = (int)fminf(fmaxf(tileCols, 1.0f), 10.0f);
    float cellW = (float)w / (float)cols;
    float cellH = (float)h / (float)rows;
    int cellX = (int)((float)px / cellW);
    int cellY = (int)((float)py / cellH);

    float offsetX = 0, offsetY = 0;
    if (tileOffsetX != 0 || tileOffsetY != 0) {
        float rndX = phash((float)cellX + tileRandomSeed, (float)cellY) - 0.5f;
        float rndY = phash((float)cellX + 100.0f + tileRandomSeed, (float)cellY + 100.0f) - 0.5f;
        offsetX = tileOffsetX * rndX * cellW;
        offsetY = tileOffsetY * rndY * cellH;
    }

    float srcX = fmodf((float)px + offsetX + w * 2, (float)w);
    float srcY = fmodf((float)py + offsetY + h * 2, (float)h);

    int idx = (py * w + px) * 4;
    for (int c = 0; c < 4; c++)
        dst[idx + c] = bilerp(src, w, h, srcX, srcY, c);
}

// 8. Time Waver
__global__ void timeWaverKernel(const float* src, float* dst, int w, int h,
    float waverAmount, float waverSpeed, float waverBlockSize, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float t = time * waverSpeed;
    float offset = waverAmount * sinf(t * 3.14159f);
    int blockSize = (int)fminf(fmaxf(waverBlockSize, 4.0f), 128.0f);

    int blockX = px / blockSize;
    int blockY = py / blockSize;
    float blockRnd = phash((float)blockX + t, (float)blockY + t);
    float jitter = (blockRnd - 0.5f) * 2.0f * offset;

    int idx = (py * w + px) * 4;
    for (int c = 0; c < 4; c++) {
        float curr = src[idx + c];
        float jitterColor = phash((float)px + time, (float)py + time * 0.5f) * jitter * 0.1f;
        dst[idx + c] = curr * 0.85f + jitterColor;
    }
}

// 19. Block Shuffle
__global__ void blockShuffleKernel(const float* src, float* dst, int w, int h,
    float blockSize, float amount, float seed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int bs = (int)fmaxf(blockSize, 8.0f);
    int blocksX = w / bs;
    int blocksY = h / bs;
    if (blocksX <= 0 || blocksY <= 0) {
        int idx = (py * w + px) * 4;
        dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
        return;
    }

    int bi = py / bs;
    int bj = px / bs;
    int blockIdx = bi * blocksX + bj;
    int swapIdx = (blockIdx + 1) % (blocksX * blocksY);
    int targetBi = swapIdx / blocksX;
    int targetBj = swapIdx % blocksX;

    float swapAmount = fminf(amount, 1.0f);
    if (phash((float)blockIdx + seed, (float)blockIdx * 0.5f) > swapAmount) {
        int idx = (py * w + px) * 4;
        dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
        return;
    }

    int srcX = targetBj * bs + (px % bs);
    int srcY = targetBi * bs + (py % bs);
    if (srcX >= w) srcX = w - 1;
    if (srcY >= h) srcY = h - 1;
    int srcIdx = (srcY * w + srcX) * 4;
    int dstIdx = (py * w + px) * 4;
    dst[dstIdx] = src[srcIdx]; dst[dstIdx+1] = src[srcIdx+1]; dst[dstIdx+2] = src[srcIdx+2]; dst[dstIdx+3] = src[srcIdx+3];
}

// 20. Scanlines
__global__ void scanlinesKernel(const float* src, float* dst, int w, int h,
    float spacing, float offset, float warp, float speed, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int sp = (int)fmaxf(spacing, 1.0f);
    float t = time * speed;
    int lineIdx = (py + (int)(offset + t * 10.0f)) % sp;

    int idx = (py * w + px) * 4;
    if (lineIdx == 0) {
        float wobble = warp * sinf((float)py * 0.1f + t);
        int sx = px + (int)wobble;
        if (sx < 0) sx = 0; if (sx >= w) sx = w - 1;
        int srcIdx = (py * w + sx) * 4;
        dst[idx] = src[srcIdx]; dst[idx+1] = src[srcIdx+1]; dst[idx+2] = src[srcIdx+2]; dst[idx+3] = src[srcIdx+3];
    } else {
        dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
    }
}

// 23. Vortex
__global__ void vortexKernel(const float* src, float* dst, int w, int h,
    float centerX, float centerY, float strength, float radius, float speed, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float cx = centerX * (float)w;
    float cy = centerY * (float)h;
    float r = radius * (float)fminf(w, h);
    float dx = (float)px - cx;
    float dy = (float)py - cy;
    float dist = sqrtf(dx * dx + dy * dy);

    int idx = (py * w + px) * 4;
    if (dist < r && dist > 0.1f) {
        float factor = 1.0f - dist / r;
        float t = time * speed;
        float angle = atan2f(dy, dx) + strength * factor * factor * (1.0f + t);
        float nx = cx + cosf(angle) * dist;
        float ny = cy + sinf(angle) * dist;
        for (int c = 0; c < 4; c++)
            dst[idx + c] = bilerp(src, w, h, nx, ny, c);
    } else {
        dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
    }
}

// 24. Wave Distort
__global__ void waveDistortKernel(const float* src, float* dst, int w, int h,
    float freqX, float freqY, float ampX, float ampY, float speed, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float t = time * speed;
    float offsetX = sinf(py * freqX * 0.01f + t) * ampX;
    float offsetY = sinf(px * freqY * 0.01f + t * 1.3f) * ampY;
    float sx = (float)px + offsetX;
    float sy = (float)py + offsetY;

    int idx = (py * w + px) * 4;
    for (int c = 0; c < 4; c++)
        dst[idx + c] = bilerp(src, w, h, sx, sy, c);
}

// 25. Twist
__global__ void twistKernel(const float* src, float* dst, int w, int h,
    float centerX, float centerY, float amount, float radius, float sharpness, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float cx = centerX * (float)w;
    float cy = centerY * (float)h;
    float r = radius * (float)fminf(w, h);
    float dx = (float)px - cx;
    float dy = (float)py - cy;
    float dist = sqrtf(dx * dx + dy * dy);

    int idx = (py * w + px) * 4;
    if (dist < r && dist > 0.1f && fabsf(amount) > 0.01f) {
        float factor = powf(1.0f - dist / r, sharpness);
        float angle = amount * factor;
        float cs = cosf(angle);
        float sn = sinf(angle);
        float nx = cx + dx * cs - dy * sn;
        float ny = cy + dx * sn + dy * cs;
        for (int c = 0; c < 4; c++)
            dst[idx + c] = bilerp(src, w, h, nx, ny, c);
    } else {
        dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
    }
}

// 21. Pixel Sort (horizontal) - simplified version
__global__ void pixelSortKernel(const float* src, float* dst, int w, int h,
    float threshold, float direction, float amount) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int idx = (py * w + px) * 4;
    float lum = src[idx] * 0.299f + src[idx + 1] * 0.587f + src[idx + 2] * 0.114f;

    if (direction < 0.5f) {
        // Horizontal - simpler version: just shift based on brightness
        if (lum > threshold) {
            int offset = (int)((lum - threshold) * amount * w);
            int srcX = px + offset;
            if (srcX >= w) srcX = w - 1;
            int srcIdx = (py * w + srcX) * 4;
            dst[idx] = src[srcIdx]; dst[idx+1] = src[srcIdx+1]; dst[idx+2] = src[srcIdx+2]; dst[idx+3] = src[srcIdx+3];
        } else {
            dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
        }
    } else {
        // Vertical
        if (lum > threshold) {
            int offset = (int)((lum - threshold) * amount * h);
            int srcY = py + offset;
            if (srcY >= h) srcY = h - 1;
            int srcIdx = (srcY * w + px) * 4;
            dst[idx] = src[srcIdx]; dst[idx+1] = src[srcIdx+1]; dst[idx+2] = src[srcIdx+2]; dst[idx+3] = src[srcIdx+3];
        } else {
            dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
        }
    }
}

// 22. Edge Distort
__global__ void edgeDistortKernel(const float* src, float* dst, int w, int h,
    float threshold, float amount, float scale) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < 1 || px >= w-1 || py < 1 || py >= h-1) {
        int idx = (py * w + px) * 4;
        dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
        return;
    }

    int idx = (py * w + px) * 4;
    float lum = src[idx] * 0.299f + src[idx + 1] * 0.587f + src[idx + 2] * 0.114f;
    float lumL = src[idx - 4] * 0.299f + src[idx - 3] * 0.587f + src[idx - 2] * 0.114f;
    float lumR = src[idx + 4] * 0.299f + src[idx + 5] * 0.587f + src[idx + 6] * 0.114f;
    float lumT = src[idx - w*4] * 0.299f + src[idx - w*4 + 1] * 0.587f + src[idx - w*4 + 2] * 0.114f;
    float lumB = src[idx + w*4] * 0.299f + src[idx + w*4 + 1] * 0.587f + src[idx + w*4 + 2] * 0.114f;
    float edgeX = fabsf(lumR - lumL);
    float edgeY = fabsf(lumB - lumT);
    float edge = sqrtf(edgeX * edgeX + edgeY * edgeY);

    if (edge > threshold) {
        float dispX = sinf((float)px * scale + (float)py * scale * 0.7f) * amount * edge;
        float dispY = cosf((float)px * scale * 0.8f + (float)py * scale) * amount * edge;
        int sx = px + (int)dispX;
        int sy = py + (int)dispY;
        if (sx < 0) sx = 0; if (sx >= w) sx = w - 1;
        if (sy < 0) sy = 0; if (sy >= h) sy = h - 1;
        int srcIdx = (sy * w + sx) * 4;
        dst[idx] = src[srcIdx]; dst[idx+1] = src[srcIdx+1]; dst[idx+2] = src[srcIdx+2]; dst[idx+3] = src[srcIdx+3];
    } else {
        dst[idx] = src[idx]; dst[idx+1] = src[idx+1]; dst[idx+2] = src[idx+2]; dst[idx+3] = src[idx+3];
    }
}

// Main distortion launcher
extern "C" void RunCudaDistortionPass(void* stream, float* dst, int w, int h,
    // Fluid Morph
    float fluidEnable, float fluidBlobCount, float fluidThreshold,
    float fluidJitter, float fluidSpeed,
    // Mirror Fractal
    float mirrorFractalEnable, float mirrorDepth, float mirrorRotateEach,
    float mirrorScale, float mirrorSeed,
    // Glitch Slice
    float glitchSliceEnable, float sliceCount, float sliceDisplaceAmt,
    float sliceRandSeed, float sliceRGBSplit,
    // Triangulate
    float triangulateEnable, float triPointCount, float triEdgeThickness,
    float triEdgeColorR, float triEdgeColorG, float triEdgeColorB,
    float triFillVariant,
    // Water Ripple
    float rippleEnable, float rippleCenterX, float rippleCenterY,
    float rippleFrequency, float rippleAmplitude, float rippleDecay,
    float rippleSpeed, float ripplePhase,
    // Displacement
    float displacementEnable, float dispStrength, float dispScale,
    float dispChannel, float dispDirection,
    // Tile Repeat
    float tileEnable, float tileRows, float tileCols,
    float tileOffsetX, float tileOffsetY, float tileRandomSeed,
    // Time Waver
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
    float growProgress, float growRadial, float growDirection, float growSoftness)
{
    // Check if any effect enabled
    bool hasAny = (fluidEnable > 0.5f || mirrorFractalEnable > 0.5f || glitchSliceEnable > 0.5f ||
                   triangulateEnable > 0.5f || rippleEnable > 0.5f || displacementEnable > 0.5f ||
                   tileEnable > 0.5f || timeWaverEnable > 0.5f ||
                   perlinEnable > 0.5f || polarEnable > 0.5f || chromaWaveEnable > 0.5f ||
                   rgbShiftEnable > 0.5f || lensCurveEnable > 0.5f || sineWarpEnable > 0.5f ||
                   spiralWarpEnable > 0.5f || noiseDispEnable > 0.5f || radialBlurEnable > 0.5f ||
                   circleEnable > 0.5f ||
                   blockShuffleEnable > 0.5f || scanlinesEnable > 0.5f || pixelSortEnable > 0.5f ||
                   edgeDistortEnable > 0.5f || vortexEnable > 0.5f || waveDistortEnable > 0.5f ||
                   twistEnable > 0.5f);
    if (!hasAny) return;

    // Allocate temp buffer
    int n = w * h * 4;
    float* tmp = nullptr;
    cudaMalloc(&tmp, n * sizeof(float));
    cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                    static_cast<cudaStream_t>(stream));

    dim3 blk(16, 16), grd((w + 15) / 16, (h + 15) / 16);

    // Apply each enabled effect
    if (fluidEnable > 0.5f) {
        fluidMorphKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, fluidBlobCount, fluidThreshold, fluidJitter, fluidSpeed, time);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (mirrorFractalEnable > 0.5f) {
        mirrorFractalKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, mirrorDepth, mirrorRotateEach, mirrorScale);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (glitchSliceEnable > 0.5f) {
        glitchSliceKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, sliceCount, sliceDisplaceAmt, sliceRandSeed, sliceRGBSplit, time);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (triangulateEnable > 0.5f) {
        triangulateKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, triPointCount, triEdgeThickness, triEdgeColorR, triEdgeColorG, triEdgeColorB,
            triFillVariant, mirrorSeed);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (rippleEnable > 0.5f) {
        waterRippleKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, rippleCenterX, rippleCenterY, rippleFrequency, rippleAmplitude,
            rippleDecay, rippleSpeed, ripplePhase);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (displacementEnable > 0.5f) {
        displacementMapKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, dispStrength, dispScale, dispChannel, dispDirection);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (tileEnable > 0.5f) {
        tileRepeatKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, tileRows, tileCols, tileOffsetX, tileOffsetY, tileRandomSeed);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (timeWaverEnable > 0.5f) {
        timeWaverKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, waverAmount, waverSpeed, waverBlockSize, time);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    // New CUDA effects
    if (blockShuffleEnable > 0.5f) {
        blockShuffleKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, blockShuffleSize, blockShuffleAmount, blockShuffleSeed);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (scanlinesEnable > 0.5f) {
        scanlinesKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, scanlinesSpacing, scanlinesOffset, scanlinesWarp, scanlinesSpeed, time);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (pixelSortEnable > 0.5f) {
        pixelSortKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, pixelSortThreshold, pixelSortDirection, pixelSortAmount);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (edgeDistortEnable > 0.5f) {
        edgeDistortKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, edgeDistortThreshold, edgeDistortAmount, edgeDistortScale);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (vortexEnable > 0.5f) {
        vortexKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, vortexCenterX, vortexCenterY, vortexStrength, vortexRadius, vortexSpeed, time);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (waveDistortEnable > 0.5f) {
        waveDistortKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, waveDistortFreqX, waveDistortFreqY, waveDistortAmpX, waveDistortAmpY, waveDistortSpeed, time);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    if (twistEnable > 0.5f) {
        twistKernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            tmp, dst, w, h, twistCenterX, twistCenterY, twistAmount, twistRadius, twistSharpness, time);
        cudaMemcpyAsync(tmp, dst, n * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream));
    }

    cudaFree(tmp);
}