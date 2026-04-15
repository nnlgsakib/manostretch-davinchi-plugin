// ManoStretch.cpp v6.0 — Surrealism Art Tool for DaVinci Resolve
//
// Three independent modules: Stretch, Surrealism, Dreamcore.
// Each module has its own effects, distortions, animations, configs.
// All params keyframeable. Stroke serialization for undo/redo.

#include "ofxsImageEffect.h"
#include "ofxsParam.h"
#include "ofxsInteract.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define kPluginName        "ManoStretch"
#define kPluginGrouping    "Distort"
#define kPluginDescription \
    "ManoStretch v6.0 — Surrealism Art Tool\n" \
    "Three modules: Stretch · Surrealism · Dreamcore\n" \
    "Stretch: click & drag pixel warp (12 modes, post-FX, animation)\n" \
    "Surrealism: full-frame warps & color (kaleidoscope, vortex, fractal...)\n" \
    "Dreamcore: atmosphere (vignette, grain, haze, scanlines...)\n" \
    "Z undo stroke. [ ] resize brush. R reset. All keyframeable."
#define kPluginIdentifier  "com.mano.stretch"
#define kPluginVersionMajor 6
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false

using namespace OFX;

// ================================================================
//  CUDA
// ================================================================
#ifdef HAS_CUDA
extern "C" void RunCudaCopyKernel(void* stream, const float* src, float* dst, int numPx);
extern "C" void RunCudaStretch(void* stream, const float* src, float* dst,
    int w, int h, float sx, float sy, float ex, float ey,
    float radius, float strength, int mode,
    float tR, float tG, float tB, float tintAmt,
    float fade, float p1, float p2,
    float startBlend, float postOpacity, float postBright, float postSat,
    float postColorR, float postColorG, float postColorB, float postColorAmt,
    float liqAmount, float liqScale,
    float animProgress, float animGrow, float animEvolve, float animEvolveSpeed,
    float animPulse, float animPulseFreq, float animWobble, float time,
    float colorLock, float colorTolerance,
    float chromaAb, float posterize, float hueShift, float solarize, float colorInvert);
extern "C" void RunCudaGlobalFX(void* stream, float* dst,
    int w, int h,
    float vignette, float grain, float scanlines, float dreamHaze,
    float globalHueShift, float pixelate, float mirrorGlobal, float time);
extern "C" void RunCudaSurrealismPass(void* stream, float* dst,
    int w, int h,
    float fractalAmt, float fractalScale,
    float kaleidoSegs, float vortexAmt, float meltAmt,
    float glitchAmt, float glitchBand,
    float waveFreq, float waveAmp,
    float chromaAb, float posterize, float hueShift,
    float solarize, float colorInvert, float time);
#endif

enum StretchMode { eLinear=0, eSpiral=1, eWave=2, eTaper=3, eSmear=4, eShatter=5,
                   eMirror=6, eMelt=7, eVortex=8, eFractal=9, eGlitch=10, eDream=11 };

// ================================================================
//  Stroke — per-drag gesture, captures brush params at creation
// ================================================================
struct StretchStroke {
    double startX, startY, endX, endY;
    double radius, strength;
    int    mode;
    double tintR, tintG, tintB, tintAmt;
    double fade, param1, param2;
};

// Strokes are per-interact-instance (see ManoStretchInteract::m_strokes)

// ================================================================
//  Stroke serialization — stored in OFX StringParam for undo/redo
// ================================================================
static std::string serializeStrokes(const std::vector<StretchStroke>& v) {
    std::string r;
    char buf[400];
    for (const auto& s : v) {
        std::snprintf(buf, sizeof(buf),
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%d,%.4f,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f|",
            s.startX, s.startY, s.endX, s.endY, s.radius, s.strength, s.mode,
            s.tintR, s.tintG, s.tintB, s.tintAmt, s.fade, s.param1, s.param2);
        r += buf;
    }
    return r;
}

static std::vector<StretchStroke> deserializeStrokes(const std::string& data) {
    std::vector<StretchStroke> r;
    if (data.empty()) return r;
    size_t pos = 0;
    while (pos < data.size()) {
        size_t end = data.find('|', pos);
        if (end == std::string::npos) break;
        std::string tok = data.substr(pos, end - pos);
        StretchStroke s = {};
        if (std::sscanf(tok.c_str(),
                "%lf,%lf,%lf,%lf,%lf,%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                &s.startX, &s.startY, &s.endX, &s.endY,
                &s.radius, &s.strength, &s.mode,
                &s.tintR, &s.tintG, &s.tintB, &s.tintAmt,
                &s.fade, &s.param1, &s.param2) == 14) {
            r.push_back(s);
        }
        pos = end + 1;
    }
    return r;
}

// ================================================================
//  CPU helpers
// ================================================================
static inline float* pxAt(void* base, const OfxRectI& b, int rb, int x, int y) {
    if (x < b.x1 || x >= b.x2 || y < b.y1 || y >= b.y2) return nullptr;
    return reinterpret_cast<float*>(
        reinterpret_cast<char*>(base) + (std::ptrdiff_t)(y - b.y1) * rb
                                      + (x - b.x1) * 4 * (int)sizeof(float));
}

static float bilerpCPU(void* base, const OfxRectI& b, int rb, float fx, float fy, int ch) {
    int x0=(int)std::floor(fx), y0=(int)std::floor(fy), x1=x0+1, y1=y0+1;
    float *p00=pxAt(base,b,rb,x0,y0), *p10=pxAt(base,b,rb,x1,y0);
    float *p01=pxAt(base,b,rb,x0,y1), *p11=pxAt(base,b,rb,x1,y1);
    if (!p00||!p10||!p01||!p11) return 0.f;
    float u=fx-x0, v=fy-y0;
    return (p00[ch]*(1-u)+p10[ch]*u)*(1-v) + (p01[ch]*(1-u)+p11[ch]*u)*v;
}

static float cpuHash(float x, float y) {
    int ix = *(int*)&x ^ (*(int*)&y << 3);
    ix = (ix << 13) ^ ix;
    return std::abs(std::sin((float)ix * 0.0001f));
}

static inline float smoothstep(float t) {
    t = (std::max)(0.f, (std::min)(1.f, t));
    return t * t * (3.f - 2.f * t);
}

// ================================================================
//  CPU stretch — matches CUDA kernel exactly
// ================================================================
struct PostFX {
    float startBlend, postOpacity, postBright, postSat;
    float postColorR, postColorG, postColorB, postColorAmt;
    float liqAmount, liqScale;
    float time;
    float animProgress, animGrow;
    float animEvolve, animEvolveSpeed;
    float animPulse, animPulseFreq;
    float animWobble;
    float colorLock, colorTolerance;
    // Surreal per-stroke FX
    float chromaAb;       // chromatic aberration offset
    float posterize;      // color level reduction (0=off)
    float hueShift;       // hue rotation in radians
    float solarize;       // solarization amount
    float colorInvert;    // color inversion amount
    // Global atmosphere FX
    float vignette;       // edge darkening
    float grain;          // film grain amount
    float scanlines;      // retro scanline intensity
    float dreamHaze;      // luminance-based glow
    float globalHueShift; // animated hue cycle
    float pixelate;       // pixelation block size (0=off)
    float mirrorGlobal;   // global mirror axis (0=off)
};

// DISPLACEMENT-BASED STRETCH — matches CUDA kernel.
// Pixels displaced backward along drag = organic face-pull effect.
static void cpuStretch(
    void* srcBase, const OfxRectI& sB, int sRB,
    void* dstBase, const OfxRectI& dB, int dRB,
    const StretchStroke& st, int maxDim, const PostFX& fx)
{
    float sx=(float)st.startX, sy=(float)st.startY;
    float ex=(float)st.endX,   ey=(float)st.endY;
    float ddx=ex-sx, ddy=ey-sy;
    float dLen = std::sqrt(ddx*ddx + ddy*ddy);
    if (dLen < 1.f) return;

    float rPx = (float)(st.radius * maxDim / 100.0);
    float ux=ddx/dLen, uy=ddy/dLen, perpX=-uy, perpY=ux;
    float strength=(float)st.strength, fade=(float)st.fade;
    float p1=(float)st.param1, p2=(float)st.param2;
    float tR=(float)st.tintR, tG=(float)st.tintG, tB=(float)st.tintB;
    float tAmt=(float)st.tintAmt;
    int mode = st.mode;

    // Color Lock: sample reference color at stroke start
    bool useColorLock = fx.colorLock > 0.01f;
    float refR=0.f, refG=0.f, refB=0.f;
    if (useColorLock) {
        refR = bilerpCPU(srcBase, sB, sRB, sx, sy, 0);
        refG = bilerpCPU(srcBase, sB, sRB, sx, sy, 1);
        refB = bilerpCPU(srcBase, sB, sRB, sx, sy, 2);
    }

    float bx0=(std::min)(sx,ex)-rPx, bx1=(std::max)(sx,ex)+rPx;
    float by0=(std::min)(sy,ey)-rPx, by1=(std::max)(sy,ey)+rPx;
    int ix0=(std::max)((int)std::floor(bx0), dB.x1);
    int ix1=(std::min)((int)std::ceil(bx1),  dB.x2-1);
    int iy0=(std::max)((int)std::floor(by0), dB.y1);
    int iy1=(std::min)((int)std::ceil(by1),  dB.y2-1);

    for (int py=iy0; py<=iy1; py++) {
      for (int px=ix0; px<=ix1; px++) {
        float vx=(float)px-sx, vy=(float)py-sy;
        float t=vx*ux+vy*uy, d=vx*perpX+vy*perpY;

        // Soft end caps: extend valid zone beyond stroke endpoints
        float capLen = rPx * 0.5f;
        if (t < -capLen || t > dLen + capLen) continue;

        // Clamp tN to [0,1] for displacement and mode calculations
        float tN = (std::max)(0.f, (std::min)(t / dLen, 1.f));

        float localR = rPx;
        if (mode==eTaper) localR = rPx * (std::max)(0.1f, 1.f-0.9f*tN);

        if (std::abs(d)>localR||localR<0.5f) continue;

        float fo = smoothstep(1.f - std::abs(d)/localR);

        float sf = 1.f;
        if (fx.startBlend>0.001f && tN<fx.startBlend)
            sf = smoothstep(tN / fx.startBlend);

        float lf = 1.f - tN * fade;
        if (lf<0.f) lf=0.f;

        float effect = fo * sf * lf;

        // Smooth end caps: fade to zero at both stroke endpoints
        if (t < 0.f) effect *= smoothstep(1.f + t / capLen);
        if (t > dLen) effect *= smoothstep(1.f - (t - dLen) / capLen);

        if (effect<0.001f) continue;

        // Animation: Growth mask — reveal stroke from start to end
        if (fx.animGrow < 0.999f) {
            if (tN > fx.animGrow) continue;
            float growEdge = 0.05f;
            if (tN > fx.animGrow - growEdge)
                effect *= smoothstep((fx.animGrow - tN) / growEdge);
        }

        // Animation: Master progress
        effect *= fx.animProgress;
        if (effect < 0.001f) continue;

        // Color Lock: reduce effect for pixels whose color differs from stroke start
        if (useColorLock) {
            float cr = bilerpCPU(srcBase, sB, sRB, (float)px, (float)py, 0);
            float cg = bilerpCPU(srcBase, sB, sRB, (float)px, (float)py, 1);
            float cb = bilerpCPU(srcBase, sB, sRB, (float)px, (float)py, 2);
            float dr=cr-refR, dg=cg-refG, db=cb-refB;
            float dist = std::sqrt(dr*dr + dg*dg + db*db);
            float tol = (std::max)(0.01f, fx.colorTolerance);
            float colorMask = smoothstep(1.f - dist / tol);
            effect *= 1.f - fx.colorLock * (1.f - colorMask);
            if (effect < 0.001f) continue;
        }

        // Displacement backward along drag (use clamped tN to avoid reverse stretch in cap zones)
        float disp = tN * dLen * strength * effect;
        float srcFx = (float)px - disp * ux;
        float srcFy = (float)py - disp * uy;

        // Animation: Pulse — periodic displacement oscillation
        if (fx.animPulse > 0.001f) {
            float pulse = 1.f + fx.animPulse * std::sin(fx.time * fx.animPulseFreq * (float)(2.0*M_PI));
            float newDisp = disp * pulse;
            srcFx = (float)px - newDisp * ux;
            srcFy = (float)py - newDisp * uy;
        }

        // Mode-specific modifications
        switch (mode) {
          default: case eLinear: case eTaper: break;
          case eSpiral: {
            float ang=tN*p1*(float)(2.0*M_PI);
            float cs=std::cos(ang), sn=std::sin(ang);
            srcFx += (d*(cs-1.f))*perpX*effect + (d*sn)*ux*effect;
            srcFy += (d*(cs-1.f))*perpY*effect + (d*sn)*uy*effect;
            break;
          }
          case eWave: {
            float wo=std::sin(tN*p1*(float)(2.0*M_PI))*p2*localR;
            srcFx += wo*perpX*effect;
            srcFy += wo*perpY*effect;
            break;
          }
          case eSmear: {
            float smPow = (std::max)(0.1f, p1);
            float smT = std::pow(tN, smPow);
            float smDisp = smT * dLen * strength * effect;
            srcFx = (float)px - smDisp*ux;
            srcFy = (float)py - smDisp*uy;
            break;
          }
          case eShatter: {
            float rnd=cpuHash((float)px*0.1f,(float)py*0.1f);
            float sc2=p1*localR*(rnd-0.5f)*2.f;
            srcFx += sc2*perpX*effect;
            srcFy += sc2*perpY*effect;
            break;
          }
          case eMirror: {
            float mirD = (d < 0.f) ? -d : d;
            float segs = (std::max)(1.f, std::floor(p1));
            float angOrig = std::atan2(d, (std::max)(0.001f, t));
            float sector = (float)(2.0*M_PI) / segs;
            float angMir = std::fmod(std::abs(angOrig), sector);
            if (angMir > sector*0.5f) angMir = sector - angMir;
            float rr = std::sqrt(t*t + d*d);
            srcFx = sx + rr * std::cos(angMir) * ux - rr * std::sin(angMir) * perpX;
            srcFy = sy + rr * std::cos(angMir) * uy - rr * std::sin(angMir) * perpY;
            srcFx -= disp * ux * 0.5f;
            srcFy -= disp * uy * 0.5f;
            break;
          }
          case eMelt: {
            float meltPow = (std::max)(0.5f, p1);
            float meltAmt = std::pow(tN, meltPow) * p2 * localR * effect;
            srcFy -= meltAmt;
            break;
          }
          case eVortex: {
            float cx2 = (sx + ex) * 0.5f, cy2 = (sy + ey) * 0.5f;
            float dvx = (float)px - cx2, dvy = (float)py - cy2;
            float dist2 = std::sqrt(dvx*dvx + dvy*dvy);
            float maxR = dLen * 0.5f + localR;
            float falloff2 = (dist2 < maxR) ? (1.f - dist2/maxR) : 0.f;
            float angle2 = p1 * effect * falloff2 * (float)(2.0*M_PI);
            float cs2 = std::cos(angle2), sn2 = std::sin(angle2);
            srcFx = cx2 + dvx * cs2 - dvy * sn2;
            srcFy = cy2 + dvx * sn2 + dvy * cs2;
            break;
          }
          case eFractal: {
            float scale2 = (std::max)(0.1f, p1) * 0.02f;
            int iters = (std::max)(1, (std::min)((int)(p2 * 4.f), 8));
            float fx2 = srcFx, fy2 = srcFy;
            for (int it = 0; it < iters; it++) {
                fx2 += std::sin(fy2 * scale2) * localR * 0.15f * effect;
                fy2 += std::cos(fx2 * scale2) * localR * 0.15f * effect;
            }
            srcFx = fx2; srcFy = fy2;
            break;
          }
          case eGlitch: {
            float bandH = (std::max)(2.f, p1 * 10.f);
            float band = std::floor((float)py / bandH);
            float rnd2 = cpuHash(band * 0.1f, band * 0.3f);
            float offset2 = (rnd2 - 0.5f) * 2.f * p2 * localR * effect;
            srcFx += offset2;
            break;
          }
          case eDream: {
            float freq1 = p1 * 0.02f, freq2 = p1 * 0.05f;
            float amp = p2 * localR * 0.3f * effect;
            float tm = fx.time;
            srcFx += std::sin(srcFy * freq1 + tm * 0.5f) * amp;
            srcFy += std::cos(srcFx * freq2 + tm * 0.7f) * amp;
            srcFx += std::sin(srcFy * freq2 * 1.5f + tm * 0.3f) * amp * 0.5f;
            srcFy += std::cos(srcFx * freq1 * 1.3f + tm * 0.4f) * amp * 0.5f;
            break;
          }
        }

        // Animation: Time evolve — organic turbulence per frame
        if (fx.animEvolve > 0.01f) {
            float et = fx.time * fx.animEvolveSpeed;
            srcFx += std::sin(srcFy * 0.03f + et * 2.7183f) * fx.animEvolve * effect;
            srcFy += std::cos(srcFx * 0.03f + et * 3.1416f) * fx.animEvolve * effect;
        }

        // Animation: Wobble — time-varying perpendicular oscillation
        if (fx.animWobble > 0.01f) {
            float wob = std::sin(fx.time * 3.0f + tN * (float)(2.0*M_PI)) * fx.animWobble * effect;
            srcFx += wob * perpX;
            srcFy += wob * perpY;
        }

        // Liquidify
        if (fx.liqAmount > 0.01f) {
            float freq = fx.liqScale * 0.05f;
            srcFx += std::sin(srcFy*freq) * fx.liqAmount * effect;
            srcFy += std::cos(srcFx*freq) * fx.liqAmount * effect;
        }

        // Sample from original source
        float rgba[4];
        for (int c=0; c<4; c++)
            rgba[c] = bilerpCPU(srcBase, sB, sRB, srcFx, srcFy, c);

        // Chromatic Aberration: offset R and B channels along stroke axis
        if (fx.chromaAb > 0.01f) {
            float caOff = fx.chromaAb * effect;
            rgba[0] = bilerpCPU(srcBase, sB, sRB, srcFx + caOff*ux, srcFy + caOff*uy, 0);
            rgba[2] = bilerpCPU(srcBase, sB, sRB, srcFx - caOff*ux, srcFy - caOff*uy, 2);
        }

        // Posterize: reduce color levels
        if (fx.posterize > 1.5f) {
            float lvl = std::floor(fx.posterize);
            for (int c=0; c<3; c++)
                rgba[c] = std::floor(rgba[c] * lvl) / lvl;
        }

        // Hue Shift: rotate hue using color matrix
        if (std::abs(fx.hueShift) > 0.001f) {
            float cosA = std::cos(fx.hueShift), sinA = std::sin(fx.hueShift);
            float s3 = 0.57735f, omc = 1.f - cosA, th = 1.f/3.f;
            float r=rgba[0], g=rgba[1], b=rgba[2];
            rgba[0] = r*(cosA+omc*th) + g*(omc*th-s3*sinA) + b*(omc*th+s3*sinA);
            rgba[1] = r*(omc*th+s3*sinA) + g*(cosA+omc*th) + b*(omc*th-s3*sinA);
            rgba[2] = r*(omc*th-s3*sinA) + g*(omc*th+s3*sinA) + b*(cosA+omc*th);
            for (int c=0; c<3; c++) rgba[c] = (std::max)(0.f, rgba[c]);
        }

        // Solarize: invert pixels above threshold
        if (fx.solarize > 0.01f) {
            float thresh = 1.f - fx.solarize;
            for (int c=0; c<3; c++) {
                if (rgba[c] > thresh)
                    rgba[c] = rgba[c]*(1.f-fx.solarize) + (1.f-rgba[c])*fx.solarize;
            }
        }

        // Color Invert: blend toward negative
        if (fx.colorInvert > 0.01f) {
            for (int c=0; c<3; c++)
                rgba[c] = rgba[c]*(1.f-fx.colorInvert) + (1.f-rgba[c])*fx.colorInvert;
        }

        if (tAmt > 0.001f) {
            rgba[0]=rgba[0]*(1.f-tAmt)+rgba[0]*tR*tAmt;
            rgba[1]=rgba[1]*(1.f-tAmt)+rgba[1]*tG*tAmt;
            rgba[2]=rgba[2]*(1.f-tAmt)+rgba[2]*tB*tAmt;
        }
        if (std::abs(fx.postBright-1.f)>0.001f) {
            rgba[0]*=fx.postBright; rgba[1]*=fx.postBright; rgba[2]*=fx.postBright;
        }
        if (std::abs(fx.postSat-1.f)>0.001f) {
            float gray=0.299f*rgba[0]+0.587f*rgba[1]+0.114f*rgba[2];
            rgba[0]=gray+(rgba[0]-gray)*fx.postSat;
            rgba[1]=gray+(rgba[1]-gray)*fx.postSat;
            rgba[2]=gray+(rgba[2]-gray)*fx.postSat;
        }
        if (fx.postColorAmt > 0.001f) {
            rgba[0]=rgba[0]*(1.f-fx.postColorAmt)+fx.postColorR*fx.postColorAmt;
            rgba[1]=rgba[1]*(1.f-fx.postColorAmt)+fx.postColorG*fx.postColorAmt;
            rgba[2]=rgba[2]*(1.f-fx.postColorAmt)+fx.postColorB*fx.postColorAmt;
        }

        float blend = effect * fx.postOpacity;
        float* dp = pxAt(dstBase, dB, dRB, px, py);
        if (!dp) continue;
        for (int c=0; c<4; c++)
            dp[c] = dp[c]*(1.f-blend) + rgba[c]*blend;
      }
    }
}

// ================================================================
//  Global atmosphere FX — full-frame pass after all strokes
// ================================================================
static void cpuGlobalFX(void* dstBase, const OfxRectI& dB, int dRB,
                         int w, int h, const PostFX& fx)
{
    bool hasAny = fx.vignette > 0.01f || fx.grain > 0.01f || fx.scanlines > 0.01f
               || fx.dreamHaze > 0.01f || std::abs(fx.globalHueShift) > 0.001f
               || fx.pixelate > 1.5f || fx.mirrorGlobal > 0.5f;
    if (!hasAny) return;

    float cosH=0,sinH=0,s3H=0,omcH=0,thH=0;
    if (std::abs(fx.globalHueShift) > 0.001f) {
        cosH = std::cos(fx.globalHueShift); sinH = std::sin(fx.globalHueShift);
        s3H = 0.57735f; omcH = 1.f - cosH; thH = 1.f/3.f;
    }

    for (int py=dB.y1; py<dB.y2; py++) {
      for (int px=dB.x1; px<dB.x2; px++) {
        float* dp = pxAt(dstBase, dB, dRB, px, py);
        if (!dp) continue;

        if (fx.pixelate > 1.5f) {
            int blk = (int)fx.pixelate;
            int bx = (px / blk) * blk + blk/2;
            int by = (py / blk) * blk + blk/2;
            float* bp = pxAt(dstBase, dB, dRB,
                (std::min)(bx, dB.x2-1), (std::min)(by, dB.y2-1));
            if (bp) { dp[0]=bp[0]; dp[1]=bp[1]; dp[2]=bp[2]; }
        }

        if (fx.mirrorGlobal > 0.5f) {
            int mx = dB.x1 + (dB.x2 - 1 - px);
            float* mp = pxAt(dstBase, dB, dRB, mx, py);
            if (mp && px > (dB.x1 + dB.x2)/2) {
                dp[0]=mp[0]; dp[1]=mp[1]; dp[2]=mp[2];
            }
        }

        float nx = (float)(px - dB.x1) / (float)w;
        float ny = (float)(py - dB.y1) / (float)h;

        if (fx.vignette > 0.01f) {
            float vx2 = nx - 0.5f, vy2 = ny - 0.5f;
            float vDist = std::sqrt(vx2*vx2 + vy2*vy2) * 1.414f;
            float vFade = 1.f - fx.vignette * vDist * vDist;
            if (vFade < 0.f) vFade = 0.f;
            dp[0] *= vFade; dp[1] *= vFade; dp[2] *= vFade;
        }

        if (fx.dreamHaze > 0.01f) {
            float lum = 0.299f*dp[0] + 0.587f*dp[1] + 0.114f*dp[2];
            float glow = lum * lum * fx.dreamHaze * 2.f;
            dp[0] += glow; dp[1] += glow; dp[2] += glow;
        }

        if (std::abs(fx.globalHueShift) > 0.001f) {
            float r=dp[0], g=dp[1], b=dp[2];
            dp[0] = r*(cosH+omcH*thH) + g*(omcH*thH-s3H*sinH) + b*(omcH*thH+s3H*sinH);
            dp[1] = r*(omcH*thH+s3H*sinH) + g*(cosH+omcH*thH) + b*(omcH*thH-s3H*sinH);
            dp[2] = r*(omcH*thH-s3H*sinH) + g*(omcH*thH+s3H*sinH) + b*(cosH+omcH*thH);
            for (int c=0; c<3; c++) dp[c] = (std::max)(0.f, dp[c]);
        }

        if (fx.scanlines > 0.01f) {
            int lineH = (std::max)(2, (int)(4.f / (fx.scanlines + 0.01f)));
            if ((py % lineH) < lineH/2) {
                float dim = 1.f - fx.scanlines * 0.6f;
                dp[0] *= dim; dp[1] *= dim; dp[2] *= dim;
            }
        }

        if (fx.grain > 0.01f) {
            float rnd = cpuHash((float)px + fx.time * 100.f, (float)py + fx.time * 73.f);
            float noise = (rnd - 0.5f) * 2.f * fx.grain * 0.15f;
            dp[0] += noise; dp[1] += noise; dp[2] += noise;
        }
      }
    }
}

// ================================================================
//  Flat-buffer bilinear sampler (for temp copy in surrealism pass)
// ================================================================
static float bilerpFlat(const float* buf, int w, int h, float fx, float fy, int ch) {
    int x0=(int)std::floor(fx), y0=(int)std::floor(fy), x1=x0+1, y1=y0+1;
    if (x0<0||y0<0||x1>=w||y1>=h) return 0.f;
    float u=fx-x0, v=fy-y0;
    return (buf[(y0*w+x0)*4+ch]*(1-u)+buf[(y0*w+x1)*4+ch]*u)*(1-v)
          +(buf[(y1*w+x0)*4+ch]*(1-u)+buf[(y1*w+x1)*4+ch]*u)*v;
}

// ================================================================
//  Module 2: Surrealism — full-frame distortions & color FX
// ================================================================
struct SurrealismFX {
    float fractalAmt, fractalScale;
    float kaleidoSegs;
    float vortexAmt;
    float meltAmt;
    float glitchAmt, glitchBand;
    float waveFreq, waveAmp;
    float chromaAb;
    float posterize;
    float hueShift;
    float solarize;
    float colorInvert;
    float time;
};

static void cpuSurrealismPass(void* dstBase, const OfxRectI& dB, int dRB,
                               int w, int h, const SurrealismFX& sfx)
{
    bool hasDist = sfx.fractalAmt > 0.01f || sfx.kaleidoSegs > 1.5f
                || sfx.vortexAmt > 0.01f || std::abs(sfx.meltAmt) > 0.01f
                || sfx.glitchAmt > 0.01f || sfx.waveAmp > 0.01f;
    bool hasColor = sfx.chromaAb > 0.01f || sfx.posterize > 1.5f
                 || std::abs(sfx.hueShift) > 0.001f || sfx.solarize > 0.01f
                 || sfx.colorInvert > 0.01f;
    if (!hasDist && !hasColor) return;

    // Copy current dst to temp for safe reading during distortions
    std::vector<float> temp(w * h * 4);
    for (int y = dB.y1; y < dB.y2; y++) {
        float* row = pxAt(dstBase, dB, dRB, dB.x1, y);
        if (row) std::memcpy(&temp[(y - dB.y1) * w * 4], row, w * 4 * sizeof(float));
    }

    float cx = w * 0.5f, cy = h * 0.5f;

    for (int py = dB.y1; py < dB.y2; py++) {
      for (int px = dB.x1; px < dB.x2; px++) {
        float* dp = pxAt(dstBase, dB, dRB, px, py);
        if (!dp) continue;

        float lx = (float)(px - dB.x1), ly = (float)(py - dB.y1);
        float srcX = lx, srcY = ly;

        // Kaleidoscope — mirror fold around center
        if (sfx.kaleidoSegs > 1.5f) {
            float dx = srcX - cx, dy = srcY - cy;
            float ang = std::atan2(dy, dx);
            float segs = std::floor(sfx.kaleidoSegs);
            float sector = (float)(2.0*M_PI) / segs;
            float angM = std::fmod(std::abs(ang), sector);
            if (angM > sector * 0.5f) angM = sector - angM;
            float rr = std::sqrt(dx*dx + dy*dy);
            srcX = cx + rr * std::cos(angM);
            srcY = cy + rr * std::sin(angM);
        }

        // Vortex swirl — radial rotation from center
        if (sfx.vortexAmt > 0.01f) {
            float dx = srcX - cx, dy = srcY - cy;
            float dist = std::sqrt(dx*dx + dy*dy);
            float maxR = (std::max)((float)w, (float)h) * 0.5f;
            float falloff = (dist < maxR) ? (1.f - dist/maxR) : 0.f;
            float angle = sfx.vortexAmt * falloff * falloff * (float)(2.0*M_PI);
            float cs = std::cos(angle), sn = std::sin(angle);
            srcX = cx + dx*cs - dy*sn;
            srcY = cy + dx*sn + dy*cs;
        }

        // Fractal warp — iterative sinusoidal displacement
        if (sfx.fractalAmt > 0.01f) {
            float scale = sfx.fractalScale * 0.02f;
            float ffx = srcX, ffy = srcY;
            for (int it = 0; it < 4; it++) {
                ffx += std::sin(ffy * scale + sfx.time * 0.3f) * sfx.fractalAmt;
                ffy += std::cos(ffx * scale + sfx.time * 0.2f) * sfx.fractalAmt;
            }
            srcX = ffx; srcY = ffy;
        }

        // Melt — vertical drip based on position
        if (std::abs(sfx.meltAmt) > 0.01f) {
            float ny = ly / (float)h;
            srcY -= sfx.meltAmt * ny * ny * (float)h * 0.1f;
        }

        // Glitch — horizontal band displacement
        if (sfx.glitchAmt > 0.01f) {
            float bandH = (std::max)(2.f, sfx.glitchBand * 10.f);
            float band = std::floor(ly / bandH);
            float rnd = cpuHash(band * 0.1f + sfx.time * 0.01f, band * 0.3f);
            srcX += (rnd - 0.5f) * 2.f * sfx.glitchAmt * (float)w * 0.05f;
        }

        // Wave — sinusoidal distortion
        if (sfx.waveAmp > 0.01f) {
            float nx = lx / (float)w, ny = ly / (float)h;
            srcX += std::sin(ny * sfx.waveFreq * (float)(2.0*M_PI) + sfx.time) * sfx.waveAmp;
            srcY += std::cos(nx * sfx.waveFreq * (float)(2.0*M_PI) + sfx.time*0.7f) * sfx.waveAmp;
        }

        // Sample from temp copy
        float rgba[4];
        for (int c = 0; c < 4; c++)
            rgba[c] = bilerpFlat(temp.data(), w, h, srcX, srcY, c);

        // Chromatic Aberration — full-frame RGB split
        if (sfx.chromaAb > 0.01f) {
            rgba[0] = bilerpFlat(temp.data(), w, h, srcX + sfx.chromaAb, srcY, 0);
            rgba[2] = bilerpFlat(temp.data(), w, h, srcX - sfx.chromaAb, srcY, 2);
        }

        // Posterize
        if (sfx.posterize > 1.5f) {
            float lvl = std::floor(sfx.posterize);
            for (int c=0; c<3; c++)
                rgba[c] = std::floor(rgba[c] * lvl) / lvl;
        }

        // Hue Shift
        if (std::abs(sfx.hueShift) > 0.001f) {
            float cosA = std::cos(sfx.hueShift), sinA = std::sin(sfx.hueShift);
            float s3 = 0.57735f, omc = 1.f - cosA, th = 1.f/3.f;
            float r=rgba[0], g=rgba[1], b=rgba[2];
            rgba[0] = r*(cosA+omc*th) + g*(omc*th-s3*sinA) + b*(omc*th+s3*sinA);
            rgba[1] = r*(omc*th+s3*sinA) + g*(cosA+omc*th) + b*(omc*th-s3*sinA);
            rgba[2] = r*(omc*th-s3*sinA) + g*(omc*th+s3*sinA) + b*(cosA+omc*th);
            for (int c=0; c<3; c++) rgba[c] = (std::max)(0.f, rgba[c]);
        }

        // Solarize
        if (sfx.solarize > 0.01f) {
            float thresh = 1.f - sfx.solarize;
            for (int c=0; c<3; c++) {
                if (rgba[c] > thresh)
                    rgba[c] = rgba[c]*(1.f-sfx.solarize) + (1.f-rgba[c])*sfx.solarize;
            }
        }

        // Color Invert
        if (sfx.colorInvert > 0.01f) {
            for (int c=0; c<3; c++)
                rgba[c] = rgba[c]*(1.f-sfx.colorInvert) + (1.f-rgba[c])*sfx.colorInvert;
        }

        for (int c = 0; c < 4; c++) dp[c] = rgba[c];
      }
    }
}

// ================================================================
//  Plugin
// ================================================================
class ManoStretchPlugin : public ImageEffect {
public:
    explicit ManoStretchPlugin(OfxImageEffectHandle h) : ImageEffect(h) {
        m_Dst = fetchClip(kOfxImageEffectOutputClipName);
        m_Src = fetchClip(kOfxImageEffectSimpleSourceClipName);
        m_StrokeData = fetchStringParam("_strokeData");
        m_StartBlend = fetchDoubleParam("startBlend");
        m_PostOpacity = fetchDoubleParam("postOpacity");
        m_PostBright  = fetchDoubleParam("postBright");
        m_PostSat     = fetchDoubleParam("postSat");
        m_PostColor   = fetchRGBParam("postColor");
        m_PostColorAmt = fetchDoubleParam("postColorAmt");
        m_LiqAmount   = fetchDoubleParam("liqAmount");
        m_LiqScale       = fetchDoubleParam("liqScale");
        m_AnimProgress   = fetchDoubleParam("animProgress");
        m_AnimGrow       = fetchDoubleParam("animGrow");
        m_AnimEvolve     = fetchDoubleParam("animEvolve");
        m_AnimEvolveSpeed = fetchDoubleParam("animEvolveSpeed");
        m_AnimPulse      = fetchDoubleParam("animPulse");
        m_AnimPulseFreq  = fetchDoubleParam("animPulseFreq");
        m_AnimWobble     = fetchDoubleParam("animWobble");
        m_ColorLock      = fetchDoubleParam("colorLock");
        m_ColorTolerance = fetchDoubleParam("colorTolerance");
        m_ChromaAb       = fetchDoubleParam("chromaAb");
        m_Posterize      = fetchDoubleParam("posterize");
        m_HueShift       = fetchDoubleParam("hueShift");
        m_Solarize       = fetchDoubleParam("solarize");
        m_ColorInvert    = fetchDoubleParam("colorInvert");
        m_Vignette       = fetchDoubleParam("vignette");
        m_Grain          = fetchDoubleParam("grain");
        m_Scanlines      = fetchDoubleParam("scanlines");
        m_DreamHaze      = fetchDoubleParam("dreamHaze");
        m_GlobalHueShift = fetchDoubleParam("globalHueShift");
        m_Pixelate       = fetchDoubleParam("pixelate");
        m_MirrorGlobal   = fetchDoubleParam("mirrorGlobal");
        // Module enables
        m_EnableStretch    = fetchBooleanParam("enableStretch");
        m_EnableSurrealism = fetchBooleanParam("enableSurrealism");
        m_EnableDreamcore  = fetchBooleanParam("enableDreamcore");
        // Surrealism module params
        m_SurrFractalAmt   = fetchDoubleParam("surrFractalAmt");
        m_SurrFractalScale = fetchDoubleParam("surrFractalScale");
        m_SurrKaleidoSegs  = fetchDoubleParam("surrKaleidoSegs");
        m_SurrVortexAmt    = fetchDoubleParam("surrVortexAmt");
        m_SurrMeltAmt      = fetchDoubleParam("surrMeltAmt");
        m_SurrGlitchAmt    = fetchDoubleParam("surrGlitchAmt");
        m_SurrGlitchBand   = fetchDoubleParam("surrGlitchBand");
        m_SurrWaveFreq     = fetchDoubleParam("surrWaveFreq");
        m_SurrWaveAmp      = fetchDoubleParam("surrWaveAmp");
        m_SurrChromaAb     = fetchDoubleParam("surrChromaAb");
        m_SurrPosterize    = fetchDoubleParam("surrPosterize");
        m_SurrHueShift     = fetchDoubleParam("surrHueShift");
        m_SurrSolarize     = fetchDoubleParam("surrSolarize");
        m_SurrColorInvert  = fetchDoubleParam("surrColorInvert");
    }

    virtual void render(const RenderArguments& a) {
        std::unique_ptr<Image> src(m_Src->fetchImage(a.time));
        std::unique_ptr<Image> dst(m_Dst->fetchImage(a.time));
        if (!src.get() || !dst.get()) return;

        const OfxRectI& dB = dst->getBounds();
        const OfxRectI& sB = src->getBounds();
        int w=dB.x2-dB.x1, h=dB.y2-dB.y1;
        if (w<=0||h<=0) return;

        // Read strokes from the STRING PARAM (source of truth for undo/redo)
        std::string strokeStr;
        m_StrokeData->getValue(strokeStr);
        std::vector<StretchStroke> strokes = deserializeStrokes(strokeStr);

        // Read post-processing params (editable after stretching!)
        PostFX fx;
        double v;
        m_StartBlend->getValue(v);  fx.startBlend  = (float)v;
        m_PostOpacity->getValue(v); fx.postOpacity  = (float)(v / 100.0);
        m_PostBright->getValue(v);  fx.postBright   = (float)v;
        m_PostSat->getValue(v);     fx.postSat      = (float)v;
        { double cr,cg,cb;
          m_PostColor->getValueAtTime(a.time, cr, cg, cb);
          fx.postColorR=(float)cr; fx.postColorG=(float)cg; fx.postColorB=(float)cb; }
        m_PostColorAmt->getValueAtTime(a.time, v); fx.postColorAmt = (float)(v / 100.0);
        m_LiqAmount->getValue(v);   fx.liqAmount    = (float)v;
        m_LiqScale->getValue(v);    fx.liqScale     = (float)v;
        fx.time = (float)a.time;
        m_AnimProgress->getValueAtTime(a.time, v);    fx.animProgress    = (float)(v / 100.0);
        m_AnimGrow->getValueAtTime(a.time, v);         fx.animGrow        = (float)(v / 100.0);
        m_AnimEvolve->getValueAtTime(a.time, v);       fx.animEvolve      = (float)v;
        m_AnimEvolveSpeed->getValueAtTime(a.time, v);  fx.animEvolveSpeed = (float)v;
        m_AnimPulse->getValueAtTime(a.time, v);        fx.animPulse       = (float)(v / 100.0);
        m_AnimPulseFreq->getValueAtTime(a.time, v);    fx.animPulseFreq   = (float)v;
        m_AnimWobble->getValueAtTime(a.time, v);       fx.animWobble      = (float)v;
        m_ColorLock->getValue(v);       fx.colorLock      = (float)(v / 100.0);
        m_ColorTolerance->getValue(v);  fx.colorTolerance = (float)(v / 100.0);
        m_ChromaAb->getValueAtTime(a.time, v);       fx.chromaAb       = (float)v;
        m_Posterize->getValueAtTime(a.time, v);      fx.posterize      = (float)v;
        m_HueShift->getValueAtTime(a.time, v);       fx.hueShift       = (float)(v * M_PI / 180.0);
        m_Solarize->getValueAtTime(a.time, v);       fx.solarize       = (float)(v / 100.0);
        m_ColorInvert->getValueAtTime(a.time, v);    fx.colorInvert    = (float)(v / 100.0);
        m_Vignette->getValueAtTime(a.time, v);       fx.vignette       = (float)(v / 100.0);
        m_Grain->getValueAtTime(a.time, v);          fx.grain          = (float)(v / 100.0);
        m_Scanlines->getValueAtTime(a.time, v);      fx.scanlines      = (float)(v / 100.0);
        m_DreamHaze->getValueAtTime(a.time, v);      fx.dreamHaze      = (float)(v / 100.0);
        m_GlobalHueShift->getValueAtTime(a.time, v); fx.globalHueShift = (float)(v * M_PI / 180.0);
        m_Pixelate->getValueAtTime(a.time, v);       fx.pixelate       = (float)v;
        m_MirrorGlobal->getValueAtTime(a.time, v);   fx.mirrorGlobal   = (float)v;

        // Module enable flags
        bool bStretch = false, bSurrealism = false, bDreamcore = false;
        m_EnableStretch->getValue(bStretch);
        m_EnableSurrealism->getValue(bSurrealism);
        m_EnableDreamcore->getValue(bDreamcore);

        // Surrealism FX params
        SurrealismFX sfx = {};
        sfx.time = (float)a.time;
        if (bSurrealism) {
            m_SurrFractalAmt->getValueAtTime(a.time, v);   sfx.fractalAmt   = (float)v;
            m_SurrFractalScale->getValueAtTime(a.time, v);  sfx.fractalScale = (float)v;
            m_SurrKaleidoSegs->getValueAtTime(a.time, v);   sfx.kaleidoSegs  = (float)v;
            m_SurrVortexAmt->getValueAtTime(a.time, v);     sfx.vortexAmt    = (float)v;
            m_SurrMeltAmt->getValueAtTime(a.time, v);       sfx.meltAmt      = (float)v;
            m_SurrGlitchAmt->getValueAtTime(a.time, v);     sfx.glitchAmt    = (float)v;
            m_SurrGlitchBand->getValueAtTime(a.time, v);    sfx.glitchBand   = (float)v;
            m_SurrWaveFreq->getValueAtTime(a.time, v);      sfx.waveFreq     = (float)v;
            m_SurrWaveAmp->getValueAtTime(a.time, v);       sfx.waveAmp      = (float)v;
            m_SurrChromaAb->getValueAtTime(a.time, v);      sfx.chromaAb     = (float)v;
            m_SurrPosterize->getValueAtTime(a.time, v);     sfx.posterize    = (float)v;
            m_SurrHueShift->getValueAtTime(a.time, v);      sfx.hueShift     = (float)(v * M_PI / 180.0);
            m_SurrSolarize->getValueAtTime(a.time, v);      sfx.solarize     = (float)(v / 100.0);
            m_SurrColorInvert->getValueAtTime(a.time, v);   sfx.colorInvert  = (float)(v / 100.0);
        }

        double rsx=a.renderScale.x, rsy=a.renderScale.y;
        int maxD = (std::max)(w, h);

#ifdef HAS_CUDA
        if (a.isEnabledCudaRender) {
            const float* sD = (const float*)src->getPixelData();
            float* dD = (float*)dst->getPixelData();
            RunCudaCopyKernel(a.pCudaStream, sD, dD, w*h);
            if (bStretch) {
                for (auto& s : strokes) {
                    float lsx=(float)(s.startX*rsx)-dB.x1, lsy=(float)(s.startY*rsy)-dB.y1;
                    float lex=(float)(s.endX*rsx)-dB.x1,   ley=(float)(s.endY*rsy)-dB.y1;
                    float rPx=(float)(s.radius*maxD/100.0);
                    RunCudaStretch(a.pCudaStream, sD, dD, w, h, lsx, lsy, lex, ley,
                        rPx, (float)s.strength, s.mode,
                        (float)s.tintR,(float)s.tintG,(float)s.tintB,(float)s.tintAmt,
                        (float)s.fade, (float)s.param1, (float)s.param2,
                        fx.startBlend, fx.postOpacity, fx.postBright, fx.postSat,
                        fx.postColorR, fx.postColorG, fx.postColorB, fx.postColorAmt,
                        fx.liqAmount, fx.liqScale,
                        fx.animProgress, fx.animGrow, fx.animEvolve, fx.animEvolveSpeed,
                        fx.animPulse, fx.animPulseFreq, fx.animWobble, fx.time,
                        fx.colorLock, fx.colorTolerance,
                        fx.chromaAb, fx.posterize, fx.hueShift, fx.solarize, fx.colorInvert);
                }
            }
            if (bSurrealism) {
                RunCudaSurrealismPass(a.pCudaStream, dD, w, h,
                    sfx.fractalAmt, sfx.fractalScale, sfx.kaleidoSegs,
                    sfx.vortexAmt, sfx.meltAmt, sfx.glitchAmt, sfx.glitchBand,
                    sfx.waveFreq, sfx.waveAmp, sfx.chromaAb, sfx.posterize,
                    sfx.hueShift, sfx.solarize, sfx.colorInvert, sfx.time);
            }
            if (bDreamcore) {
                RunCudaGlobalFX(a.pCudaStream, dD, w, h,
                    fx.vignette, fx.grain, fx.scanlines, fx.dreamHaze,
                    fx.globalHueShift, fx.pixelate, fx.mirrorGlobal, fx.time);
            }
            return;
        }
#endif
        void* dBase = dst->getPixelData();
        void* sBase = src->getPixelData();
        int dRB=dst->getRowBytes(), sRB=src->getRowBytes();

        for (int y=dB.y1; y<dB.y2; y++) {
            float* dr=pxAt(dBase,dB,dRB,dB.x1,y);
            float* sr=pxAt(sBase,sB,sRB,dB.x1,y);
            if(dr&&sr) std::memcpy(dr,sr,w*4*sizeof(float));
            else if(dr) std::memset(dr,0,w*4*sizeof(float));
        }

        if (bStretch) {
            for (auto& s : strokes) {
                StretchStroke ls = s;
                ls.startX*=rsx; ls.startY*=rsy;
                ls.endX*=rsx;   ls.endY*=rsy;
                cpuStretch(sBase, sB, sRB, dBase, dB, dRB, ls, maxD, fx);
            }
        }

        if (bSurrealism)
            cpuSurrealismPass(dBase, dB, dRB, w, h, sfx);

        if (bDreamcore)
            cpuGlobalFX(dBase, dB, dRB, w, h, fx);
    }

    virtual bool isIdentity(const IsIdentityArguments& a,
                            Clip*& c, double& t) {
        bool bS=false, bR=false, bD=false;
        m_EnableStretch->getValue(bS);
        m_EnableSurrealism->getValue(bR);
        m_EnableDreamcore->getValue(bD);
        if (!bR && !bD) {
            if (!bS) { c = m_Src; t = a.time; return true; }
            std::string strokeStr;
            m_StrokeData->getValue(strokeStr);
            if (strokeStr.empty()) { c = m_Src; t = a.time; return true; }
        }
        return false;
    }

private:
    Clip *m_Dst, *m_Src;
    StringParam *m_StrokeData;
    DoubleParam *m_StartBlend, *m_PostOpacity, *m_PostBright, *m_PostSat;
    RGBParam *m_PostColor; DoubleParam *m_PostColorAmt;
    DoubleParam *m_LiqAmount, *m_LiqScale;
    DoubleParam *m_AnimProgress, *m_AnimGrow, *m_AnimEvolve, *m_AnimEvolveSpeed;
    DoubleParam *m_AnimPulse, *m_AnimPulseFreq, *m_AnimWobble;
    DoubleParam *m_ColorLock, *m_ColorTolerance;
    DoubleParam *m_ChromaAb, *m_Posterize, *m_HueShift, *m_Solarize, *m_ColorInvert;
    DoubleParam *m_Vignette, *m_Grain, *m_Scanlines, *m_DreamHaze;
    DoubleParam *m_GlobalHueShift, *m_Pixelate, *m_MirrorGlobal;
    BooleanParam *m_EnableStretch, *m_EnableSurrealism, *m_EnableDreamcore;
    DoubleParam *m_SurrFractalAmt, *m_SurrFractalScale, *m_SurrKaleidoSegs;
    DoubleParam *m_SurrVortexAmt, *m_SurrMeltAmt;
    DoubleParam *m_SurrGlitchAmt, *m_SurrGlitchBand;
    DoubleParam *m_SurrWaveFreq, *m_SurrWaveAmp;
    DoubleParam *m_SurrChromaAb, *m_SurrPosterize, *m_SurrHueShift;
    DoubleParam *m_SurrSolarize, *m_SurrColorInvert;
};

//  Overlay Interact
// ================================================================
class ManoStretchInteract : public OverlayInteract {
public:
    ManoStretchInteract(OfxInteractHandle h, ImageEffect* e)
        : OverlayInteract(h), m_drag(false), m_cx(0),m_cy(0),
          m_sx(0),m_sy(0), m_has(false) {}

    virtual bool draw(const DrawArgs& args) {
        // Re-sync from param when not dragging (catches undo/redo & host changes)
        if (!m_drag) syncFromParam();

        double rv; _effect->fetchDoubleParam("radius")->getValue(rv);
        double rc = 100.0;
        try {
            OfxRectD rod = _effect->fetchClip(kOfxImageEffectSimpleSourceClipName)
                                 ->getRegionOfDefinition(args.time);
            double iw=rod.x2-rod.x1, ih=rod.y2-rod.y1;
            if (iw>0&&ih>0) rc = rv * (std::max)(iw,ih) / 100.0;
        } catch(...) {}
        if (!m_has) return false;

        double tR,tG,tB,tAmt;
        _effect->fetchRGBParam("tintColor")->getValue(tR, tG, tB);
        _effect->fetchDoubleParam("tintAmount")->getValue(tAmt);

#ifdef _WIN32
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);

        float cr=1.f,cg=1.f,cb=1.f;
        if (tAmt>0.01) {
            cr=(float)(tR*0.5+0.5); cg=(float)(tG*0.5+0.5); cb=(float)(tB*0.5+0.5);
        }

        // Brush circle
        glColor4f(cr,cg,cb,0.8f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_LOOP);
        for (int i=0;i<64;i++) {
            double a=2.0*M_PI*i/64.0;
            glVertex2d(m_cx+rc*std::cos(a), m_cy+rc*std::sin(a));
        }
        glEnd();

        // Crosshair
        double cs=(std::max)(3.0,rc*0.08);
        glColor4f(1.f,1.f,1.f,0.5f);
        glBegin(GL_LINES);
        glVertex2d(m_cx-cs,m_cy); glVertex2d(m_cx+cs,m_cy);
        glVertex2d(m_cx,m_cy-cs); glVertex2d(m_cx,m_cy+cs);
        glEnd();

        // Stroke count indicator
        {
            int n = (int)m_strokes.size();
            if (n > 0) {
                // Small dots for each stroke at top-left
                glColor4f(1.f,0.8f,0.f,0.7f);
                glPointSize(4.0f);
                glBegin(GL_POINTS);
                for (int i = 0; i < (std::min)(n, 20); i++)
                    glVertex2d(m_cx - rc + i * 5.0, m_cy + rc + 10.0);
                glEnd();
            }
        }

        if (m_drag) {
            double ddx=m_cx-m_sx, ddy=m_cy-m_sy;
            double len=std::sqrt(ddx*ddx+ddy*ddy);
            if (len>2.0) {
                double ux=ddx/len, uy=ddy/len;
                double px=-uy*rc, py=ux*rc;

                // Direction line
                glColor4f(0.f,1.f,0.4f,0.9f);
                glLineWidth(2.0f);
                glBegin(GL_LINES);
                glVertex2d(m_sx,m_sy); glVertex2d(m_cx,m_cy);
                glEnd();

                // Soft-start indicator: gradient zone at beginning
                double sb; _effect->fetchDoubleParam("startBlend")->getValue(sb);
                double sbLen = len * sb;
                if (sbLen > 2.0) {
                    glColor4f(0.3f,0.6f,1.f,0.25f);
                    double sbx=m_sx+ddx*sb, sby=m_sy+ddy*sb;
                    double spx=-uy*rc*0.5, spy=ux*rc*0.5;
                    glBegin(GL_QUADS);
                    glVertex2d(m_sx+spx,m_sy+spy); glVertex2d(m_sx-spx,m_sy-spy);
                    glVertex2d(sbx-px,sby-py); glVertex2d(sbx+px,sby+py);
                    glEnd();
                }

                // Stretch zone
                glColor4f(cr*0.5f,cg*0.8f,cb*0.5f,0.12f);
                glBegin(GL_QUADS);
                glVertex2d(m_sx+px,m_sy+py); glVertex2d(m_sx-px,m_sy-py);
                glVertex2d(m_cx-px,m_cy-py); glVertex2d(m_cx+px,m_cy+py);
                glEnd();
                glColor4f(0.f,1.f,0.4f,0.5f);
                glLineWidth(1.0f);
                glBegin(GL_LINE_LOOP);
                glVertex2d(m_sx+px,m_sy+py); glVertex2d(m_sx-px,m_sy-py);
                glVertex2d(m_cx-px,m_cy-py); glVertex2d(m_cx+px,m_cy+py);
                glEnd();

                // Anchor
                glColor4f(1.f,0.6f,0.f,0.9f);
                glLineWidth(2.0f);
                glBegin(GL_LINE_LOOP);
                for (int i=0;i<32;i++){
                    double a=2.0*M_PI*i/32.0;
                    glVertex2d(m_sx+7.0*std::cos(a),m_sy+7.0*std::sin(a));
                }
                glEnd();

                // Mode preview
                int mode; _effect->fetchChoiceParam("mode")->getValue(mode);
                if (mode==eSpiral) {
                    double p1; _effect->fetchDoubleParam("param1")->getValue(p1);
                    glColor4f(1.f,0.8f,0.2f,0.6f); glLineWidth(1.5f);
                    glBegin(GL_LINE_STRIP);
                    for(int i=0;i<=40;i++){
                        double tN=(double)i/40.0;
                        double ang=tN*p1*2.0*M_PI;
                        double r2=rc*(1.0-tN*0.7);
                        glVertex2d(m_sx+ddx*tN+r2*std::cos(ang)*(-uy)+r2*std::sin(ang)*ux,
                                   m_sy+ddy*tN+r2*std::cos(ang)*ux+r2*std::sin(ang)*uy);
                    }
                    glEnd();
                } else if (mode==eWave) {
                    double p1,p2;
                    _effect->fetchDoubleParam("param1")->getValue(p1);
                    _effect->fetchDoubleParam("param2")->getValue(p2);
                    glColor4f(0.2f,0.8f,1.f,0.6f); glLineWidth(1.5f);
                    glBegin(GL_LINE_STRIP);
                    for(int i=0;i<=40;i++){
                        double tN=(double)i/40.0;
                        double wo=std::sin(tN*p1*2.0*M_PI)*p2*rc;
                        glVertex2d(m_sx+ddx*tN+wo*(-uy), m_sy+ddy*tN+wo*ux);
                    }
                    glEnd();
                } else if (mode==eTaper) {
                    glColor4f(1.f,0.5f,0.f,0.4f);
                    glBegin(GL_LINE_STRIP);
                    for(int i=0;i<=20;i++){
                        double tN=(double)i/20.0;
                        double r2=rc*(1.0-0.9*tN);
                        glVertex2d(m_sx+ddx*tN+(-uy)*r2,m_sy+ddy*tN+ux*r2);
                    }
                    for(int i=20;i>=0;i--){
                        double tN=(double)i/20.0;
                        double r2=rc*(1.0-0.9*tN);
                        glVertex2d(m_sx+ddx*tN-(-uy)*r2,m_sy+ddy*tN-ux*r2);
                    }
                    glEnd();
                }
            }
        }
        glPopAttrib();
#endif
        return true;
    }

    virtual bool penDown(const PenArgs& args) {
        syncFromParam();  // Always get latest state (catches undo/redo)
        m_drag = true;
        m_sx = m_cx = args.penPosition.x;
        m_sy = m_cy = args.penPosition.y;
        m_has = true;

        StretchStroke s = {};
        s.startX=m_sx; s.startY=m_sy; s.endX=m_sx; s.endY=m_sy;
        double v;
        _effect->fetchDoubleParam("strength")->getValue(v); s.strength=v/100.0;
        _effect->fetchDoubleParam("radius")->getValue(v);   s.radius=v;
        _effect->fetchDoubleParam("fade")->getValue(v);     s.fade=v;
        { double tr,tg,tb;
          _effect->fetchRGBParam("tintColor")->getValue(tr, tg, tb);
          s.tintR=tr; s.tintG=tg; s.tintB=tb; }
        _effect->fetchDoubleParam("tintAmount")->getValue(v); s.tintAmt=v/100.0;
        _effect->fetchDoubleParam("param1")->getValue(v);   s.param1=v;
        _effect->fetchDoubleParam("param2")->getValue(v);   s.param2=v;
        int m; _effect->fetchChoiceParam("mode")->getValue(m); s.mode=m;

        m_strokes.push_back(s);
        requestRedraw();
        return true;
    }

    virtual bool penUp(const PenArgs&) {
        m_drag = false;
        // Finalize: serialize all strokes to param
        syncToParam();
        requestRedraw();
        return true;
    }

    virtual bool penMotion(const PenArgs& args) {
        m_cx=args.penPosition.x; m_cy=args.penPosition.y;
        m_has = true;
        if (m_drag) {
            if (!m_strokes.empty()) {
                m_strokes.back().endX = m_cx;
                m_strokes.back().endY = m_cy;
            }
            syncToParam();
        }
        requestRedraw();
        return true;
    }

    virtual bool keyDown(const KeyArgs& args) {
        syncFromParam();  // Always get latest state
        if (args.keyString=="r"||args.keyString=="R") {
            m_strokes.clear();
            _effect->beginEditBlock("msReset");
            _effect->fetchStringParam("_strokeData")->setValue("");
            _effect->endEditBlock();
            requestRedraw(); return true;
        }
        if (args.keyString=="z"||args.keyString=="Z") {
            if (!m_strokes.empty()) {
                m_strokes.pop_back();
                _effect->beginEditBlock("msUndo");
                _effect->fetchStringParam("_strokeData")->setValue(serializeStrokes(m_strokes));
                _effect->endEditBlock();
                requestRedraw();
            }
            return true;
        }
        if (args.keyString=="[") {
            double r; _effect->fetchDoubleParam("radius")->getValue(r);
            _effect->fetchDoubleParam("radius")->setValue((std::max)(0.5, r-1.0));
            requestRedraw(); return true;
        }
        if (args.keyString=="]") {
            double r; _effect->fetchDoubleParam("radius")->getValue(r);
            _effect->fetchDoubleParam("radius")->setValue((std::min)(50.0, r+1.0));
            requestRedraw(); return true;
        }
        return false;
    }

private:
    bool m_drag; double m_cx,m_cy,m_sx,m_sy; bool m_has;
    std::vector<StretchStroke> m_strokes;

    void syncFromParam() {
        std::string sd;
        _effect->fetchStringParam("_strokeData")->getValue(sd);
        m_strokes = deserializeStrokes(sd);
    }

    void syncToParam() {
        std::string data = serializeStrokes(m_strokes);
        _effect->beginEditBlock("msStroke");
        _effect->fetchStringParam("_strokeData")->setValue(data);
        _effect->endEditBlock();
    }
};

class ManoStretchOD : public DefaultEffectOverlayDescriptor<ManoStretchOD, ManoStretchInteract> {};

// ================================================================
//  Factory
// ================================================================
class ManoStretchFactory : public PluginFactoryHelper<ManoStretchFactory> {
public:
    ManoStretchFactory()
        : PluginFactoryHelper(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}

    virtual void describe(ImageEffectDescriptor& d) {
        d.setLabels(kPluginName, kPluginName, kPluginName);
        d.setPluginDescription(kPluginDescription);
        d.addSupportedContext(eContextFilter);
        d.addSupportedContext(eContextGeneral);
        d.addSupportedBitDepth(eBitDepthFloat);
        d.setSupportsMultiResolution(kSupportsMultiResolution);
        d.setSupportsTiles(kSupportsTiles);
#ifdef HAS_CUDA
        d.setSupportsCudaRender(true);
        d.setSupportsCudaStream(true);
#else
        d.setSupportsCudaRender(false);
        d.setSupportsCudaStream(false);
#endif
        d.setSupportsOpenCLRender(false);
        d.setSupportsMetalRender(false);
        d.setOverlayInteractDescriptor(new ManoStretchOD);
    }

    virtual void describeInContext(ImageEffectDescriptor& d, ContextEnum) {
        ClipDescriptor* sc = d.defineClip(kOfxImageEffectSimpleSourceClipName);
        sc->addSupportedComponent(ePixelComponentRGBA);
        sc->setSupportsTiles(false);
        ClipDescriptor* dc = d.defineClip(kOfxImageEffectOutputClipName);
        dc->addSupportedComponent(ePixelComponentRGBA);
        dc->addSupportedComponent(ePixelComponentAlpha);
        dc->setSupportsTiles(false);

        PageParamDescriptor* pg = d.definePageParam("Controls");

        // ╔═══ STRETCH MODULE ═══╗
        GroupParamDescriptor* gS = d.defineGroupParam("grpStretch");
        gS->setLabels("Stretch","Stretch","Stretch");
        gS->setOpen(true); pg->addChild(*gS);

        // ╔═══ SURREALISM MODULE ═══╗
        GroupParamDescriptor* gR = d.defineGroupParam("grpSurrealism");
        gR->setLabels("Surrealism","Surrealism","Surrealism");
        gR->setOpen(false); pg->addChild(*gR);

        // ╔═══ DREAMCORE MODULE ═══╗
        GroupParamDescriptor* gD = d.defineGroupParam("grpDreamcore");
        gD->setLabels("Dreamcore","Dreamcore","Dreamcore");
        gD->setOpen(false); pg->addChild(*gD);

        // ---- Stretch > Brush Settings ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpBrush");
          g->setLabels("Brush Settings","Brush Settings","Brush Settings");
          g->setOpen(true); g->setParent(*gS);
          pg->addChild(*g);

          ChoiceParamDescriptor* p = d.defineChoiceParam("mode");
          p->setLabels("Mode","Mode","Mode");
          p->appendOption("Linear",  "Straight pixel smear");
          p->appendOption("Spiral",  "Twisted spiral stretch");
          p->appendOption("Wave",    "Wavy/liquid stretch");
          p->appendOption("Taper",   "Narrows toward the end");
          p->appendOption("Smear",   "Motion-blur-like smear");
          p->appendOption("Shatter", "Shattered/glitchy scatter");
          p->appendOption("Mirror",  "Kaleidoscopic mirror reflections");
          p->appendOption("Melt",    "Downward melting/dripping");
          p->appendOption("Vortex",  "Spiral vortex around center");
          p->appendOption("Fractal", "Iterative fractal distortion");
          p->appendOption("Glitch",  "Horizontal band displacement");
          p->appendOption("Dream",   "Multi-layer dreamy warp");
          p->setDefault(0); p->setAnimates(true); p->setParent(*g);
          pg->addChild(*p);

          DoubleParamDescriptor* dp;
          dp = d.defineDoubleParam("strength");
          dp->setLabels("Strength","Strength","Strength");
          dp->setDefault(80); dp->setRange(0,100); dp->setDisplayRange(0,100);
          dp->setHint("How much to stretch (80=organic pull, 100=extreme smear)");
          dp->setAnimates(true); dp->setParent(*g); pg->addChild(*dp);
          dp = d.defineDoubleParam("radius");
          dp->setLabels("Radius","Radius","Radius");
          dp->setDefault(5); dp->setRange(0.5,50); dp->setDisplayRange(0.5,30);
          dp->setHint("[ ] keys to adjust"); dp->setAnimates(true); dp->setParent(*g); pg->addChild(*dp);
          dp = d.defineDoubleParam("startBlend");
          dp->setLabels("Start Blend","Start Blend","Start Blend");
          dp->setDefault(0.15); dp->setRange(0,0.5); dp->setDisplayRange(0,0.5);
          dp->setHint("Soft ramp-up at the stretch origin (0=sharp, 0.5=half-length fade-in)");
          dp->setAnimates(true); dp->setParent(*g); pg->addChild(*dp);
          dp = d.defineDoubleParam("fade");
          dp->setLabels("End Fade","End Fade","End Fade");
          dp->setDefault(0.3); dp->setRange(0,1); dp->setDisplayRange(0,1);
          dp->setHint("Fade-out at the stretch end"); dp->setAnimates(true); dp->setParent(*g); pg->addChild(*dp);
        }

        // ---- Stretch > Color Lock ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpColorLock");
          g->setLabels("Color Lock","Color Lock","Color Lock");
          g->setOpen(false); g->setParent(*gS); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("colorLock");
          p->setLabels("Color Lock","Color Lock","Color Lock");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Lock stretch to colors matching the drag start point (0=stretch all, 100=only matching colors)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("colorTolerance");
          p->setLabels("Color Tolerance","Color Tolerance","Color Tolerance");
          p->setDefault(30); p->setRange(1,100); p->setDisplayRange(1,100);
          p->setHint("How similar colors must be to be stretched (lower=stricter, higher=more inclusive)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Stretch > Colourman ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpColourman");
          g->setLabels("Colourman","Colourman","Colourman");
          g->setOpen(false); g->setParent(*gS); pg->addChild(*g);

          RGBParamDescriptor* c = d.defineRGBParam("tintColor");
          c->setLabels("Tint Color","Tint Color","Tint Color");
          c->setDefault(1, 1, 1);
          c->setHint("Color picker for per-stroke tint");
          c->setAnimates(true); c->setParent(*g); pg->addChild(*c);
          DoubleParamDescriptor* p = d.defineDoubleParam("tintAmount");
          p->setLabels("Tint Amount","Tint Amount","Tint Amount");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Stretch > Mode Detail ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpDetail");
          g->setLabels("Mode Detail","Mode Detail","Mode Detail");
          g->setOpen(false); g->setParent(*gS); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("param1");
          p->setLabels("Detail 1","Detail 1","Detail 1");
          p->setDefault(2); p->setRange(0,20); p->setDisplayRange(0,10);
          p->setHint("Spiral:turns | Wave:freq | Smear:pow | Shatter:scatter | Mirror:segs | Melt:pow | Vortex:turns | Fractal:scale | Glitch:bandH | Dream:freq");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("param2");
          p->setLabels("Detail 2","Detail 2","Detail 2");
          p->setDefault(0.5); p->setRange(0,5); p->setDisplayRange(0,2);
          p->setHint("Wave:amp | Melt:amt | Fractal:iters | Glitch:amt | Dream:amp"); p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Stretch > Post FX ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpPostFX");
          g->setLabels("Post FX","Post FX","Post FX");
          g->setOpen(false); g->setParent(*gS); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("postOpacity");
          p->setLabels("Global Opacity","Global Opacity","Global Opacity");
          p->setDefault(100); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Master opacity for all stretches — change AFTER drawing!");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("postBright");
          p->setLabels("Brightness","Brightness","Brightness");
          p->setDefault(1); p->setRange(0.0,3.0); p->setDisplayRange(0.5,2.0);
          p->setHint("Brighten/darken stretched pixels"); p->setAnimates(true);
          p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("postSat");
          p->setLabels("Saturation","Saturation","Saturation");
          p->setDefault(1); p->setRange(0,3); p->setDisplayRange(0,2);
          p->setHint("Saturate/desaturate stretched pixels"); p->setAnimates(true);
          p->setParent(*g); pg->addChild(*p);
          RGBParamDescriptor* c = d.defineRGBParam("postColor");
          c->setLabels("Color Overlay","Color Overlay","Color Overlay");
          c->setDefault(1, 1, 1);
          c->setHint("Color to overlay on stretched pixels");
          c->setAnimates(true); c->setParent(*g); pg->addChild(*c);
          p = d.defineDoubleParam("postColorAmt");
          p->setLabels("Color Amount","Color Amount","Color Amount");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("How much of the color overlay to apply (0=none, 100=full)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Stretch > Liquidify ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpLiquidify");
          g->setLabels("Liquidify","Liquidify","Liquidify");
          g->setOpen(false); g->setParent(*gS); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("liqAmount");
          p->setLabels("Liquid Amount","Liquid Amount","Liquid Amount");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,50);
          p->setHint("Sinusoidal distortion of stretched pixels");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("liqScale");
          p->setLabels("Liquid Scale","Liquid Scale","Liquid Scale");
          p->setDefault(5); p->setRange(0.5,30); p->setDisplayRange(0.5,20);
          p->setHint("Frequency of the liquid distortion");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Stretch > Animation ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpAnimation");
          g->setLabels("Animation","Animation","Animation");
          g->setOpen(false); g->setParent(*gS); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("animProgress");
          p->setLabels("Effect Progress","Effect Progress","Effect Progress");
          p->setDefault(100); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Master intensity — keyframe 0→100 to animate the stretch in/out");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("animGrow");
          p->setLabels("Growth","Growth","Growth");
          p->setDefault(100); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Stroke reveal — keyframe 0→100 to grow strokes from start to end");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("animEvolve");
          p->setLabels("Time Evolve","Time Evolve","Time Evolve");
          p->setDefault(0); p->setRange(0,50); p->setDisplayRange(0,30);
          p->setHint("Organic turbulence that evolves over time (per-frame variation)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("animEvolveSpeed");
          p->setLabels("Evolve Speed","Evolve Speed","Evolve Speed");
          p->setDefault(1); p->setRange(0.1,10); p->setDisplayRange(0.1,5);
          p->setHint("Speed of the evolving turbulence");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("animPulse");
          p->setLabels("Pulse","Pulse","Pulse");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,50);
          p->setHint("Periodic pulsing of stretch displacement — breathing/pumping effect");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("animPulseFreq");
          p->setLabels("Pulse Speed","Pulse Speed","Pulse Speed");
          p->setDefault(2); p->setRange(0.1,10); p->setDisplayRange(0.1,5);
          p->setHint("Pulse frequency in cycles per second");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("animWobble");
          p->setLabels("Wobble","Wobble","Wobble");
          p->setDefault(0); p->setRange(0,50); p->setDisplayRange(0,30);
          p->setHint("Time-varying perpendicular wobble — creates waving/jelly effect");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Stretch > Per-Stroke FX ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpSurrealFX");
          g->setLabels("Per-Stroke FX","Per-Stroke FX","Per-Stroke FX");
          g->setOpen(false); g->setParent(*gS); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("chromaAb");
          p->setLabels("Chromatic Aberration","Chromatic Aberration","Chromatic Aberration");
          p->setDefault(0); p->setRange(0,50); p->setDisplayRange(0,20);
          p->setHint("RGB channel split along stroke direction");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("posterize");
          p->setLabels("Posterize","Posterize","Posterize");
          p->setDefault(0); p->setRange(0,32); p->setDisplayRange(0,16);
          p->setHint("Reduce color levels for poster/pop-art look (0=off, 4=strong)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("hueShift");
          p->setLabels("Hue Shift","Hue Shift","Hue Shift");
          p->setDefault(0); p->setRange(-180,180); p->setDisplayRange(-180,180);
          p->setHint("Rotate hue of stretched pixels (degrees)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("solarize");
          p->setLabels("Solarize","Solarize","Solarize");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Invert bright pixels for surreal solarization");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("colorInvert");
          p->setLabels("Color Invert","Color Invert","Color Invert");
          p->setHint("Blend toward color negative (0=normal, 100=fully inverted)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Enable Stretch (at bottom) ----
        { BooleanParamDescriptor* b = d.defineBooleanParam("enableStretch");
          b->setLabels("Enable Stretch","Enable Stretch","Enable Stretch");
          b->setDefault(true);
          b->setHint("Enable the stroke-based pixel stretch module");
          b->setAnimates(false); b->setParent(*gS); pg->addChild(*b); }

        // ---- Surrealism > Fractal Warp ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpSurrFractal");
          g->setLabels("Fractal Warp","Fractal Warp","Fractal Warp");
          g->setOpen(false); g->setParent(*gR); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("surrFractalAmt");
          p->setLabels("Amount","Amount","Amount");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,50);
          p->setHint("Iterative sinusoidal warp — organic fractal distortion");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrFractalScale");
          p->setLabels("Scale","Scale","Scale");
          p->setDefault(5); p->setRange(0.1,50); p->setDisplayRange(0.5,20);
          p->setHint("Scale/frequency of fractal warp");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Surrealism > Spatial Distortions ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpSurrSpatial");
          g->setLabels("Spatial Distortions","Spatial Distortions","Spatial Distortions");
          g->setOpen(false); g->setParent(*gR); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("surrKaleidoSegs");
          p->setLabels("Kaleidoscope","Kaleidoscope","Kaleidoscope");
          p->setDefault(0); p->setRange(0,16); p->setDisplayRange(0,12);
          p->setHint("Number of kaleidoscope mirror segments (0=off, 3+=active)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrVortexAmt");
          p->setLabels("Vortex Swirl","Vortex Swirl","Vortex Swirl");
          p->setDefault(0); p->setRange(0,10); p->setDisplayRange(0,5);
          p->setHint("Radial swirl from image center");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrMeltAmt");
          p->setLabels("Melt","Melt","Melt");
          p->setDefault(0); p->setRange(-50,50); p->setDisplayRange(-20,20);
          p->setHint("Vertical drip/melt distortion (negative=up, positive=down)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Surrealism > Glitch ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpSurrGlitch");
          g->setLabels("Glitch","Glitch","Glitch");
          g->setOpen(false); g->setParent(*gR); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("surrGlitchAmt");
          p->setLabels("Amount","Amount","Amount");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,50);
          p->setHint("Horizontal band displacement — digital glitch");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrGlitchBand");
          p->setLabels("Band Height","Band Height","Band Height");
          p->setDefault(3); p->setRange(0.5,20); p->setDisplayRange(1,10);
          p->setHint("Band height for glitch effect");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Surrealism > Wave ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpSurrWave");
          g->setLabels("Wave","Wave","Wave");
          g->setOpen(false); g->setParent(*gR); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("surrWaveFreq");
          p->setLabels("Frequency","Frequency","Frequency");
          p->setDefault(3); p->setRange(0.1,20); p->setDisplayRange(0.5,10);
          p->setHint("Frequency of sinusoidal wave distortion");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrWaveAmp");
          p->setLabels("Amplitude","Amplitude","Amplitude");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,50);
          p->setHint("Amplitude of wave distortion (0=off)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Surrealism > Color FX ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpSurrColor");
          g->setLabels("Color FX","Color FX","Color FX");
          g->setOpen(false); g->setParent(*gR); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("surrChromaAb");
          p->setLabels("Chromatic Aberration","Chromatic Aberration","Chromatic Aberration");
          p->setDefault(0); p->setRange(0,50); p->setDisplayRange(0,20);
          p->setHint("Full-frame RGB channel split");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrPosterize");
          p->setLabels("Posterize","Posterize","Posterize");
          p->setDefault(0); p->setRange(0,32); p->setDisplayRange(0,16);
          p->setHint("Full-frame color level reduction (0=off, 4=strong)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrHueShift");
          p->setLabels("Hue Shift","Hue Shift","Hue Shift");
          p->setDefault(0); p->setRange(-180,180); p->setDisplayRange(-180,180);
          p->setHint("Full-frame hue rotation (degrees)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrSolarize");
          p->setLabels("Solarize","Solarize","Solarize");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Full-frame solarization — invert bright pixels");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("surrColorInvert");
          p->setLabels("Color Invert","Color Invert","Color Invert");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Full-frame blend toward negative");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Enable Surrealism (at bottom of Surrealism group) ----
        { BooleanParamDescriptor* b = d.defineBooleanParam("enableSurrealism");
          b->setLabels("Enable Surrealism","Enable Surrealism","Enable Surrealism");
          b->setDefault(false);
          b->setHint("Enable full-frame surrealism distortions & color FX");
          b->setAnimates(false); b->setParent(*gR); pg->addChild(*b); }

        // ---- Dreamcore > Atmosphere ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpDcAtmos");
          g->setLabels("Atmosphere","Atmosphere","Atmosphere");
          g->setOpen(false); g->setParent(*gD); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("vignette");
          p->setLabels("Vignette","Vignette","Vignette");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Darken image edges for moody atmosphere");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("dreamHaze");
          p->setLabels("Dream Haze","Dream Haze","Dream Haze");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,50);
          p->setHint("Luminance-based glow for dreamy atmosphere");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Dreamcore > Retro FX ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpDcRetro");
          g->setLabels("Retro FX","Retro FX","Retro FX");
          g->setOpen(false); g->setParent(*gD); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("grain");
          p->setLabels("Film Grain","Film Grain","Film Grain");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,50);
          p->setHint("Add noise for analog/VHS feel");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("scanlines");
          p->setLabels("Scanlines","Scanlines","Scanlines");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("Horizontal retro scanline overlay");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("pixelate");
          p->setLabels("Pixelate","Pixelate","Pixelate");
          p->setDefault(0); p->setRange(0,64); p->setDisplayRange(0,32);
          p->setHint("Block size for retro pixelation (0=off)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Dreamcore > Color ----
        { GroupParamDescriptor* g = d.defineGroupParam("grpDcColor");
          g->setLabels("Color","Color","Color");
          g->setOpen(false); g->setParent(*gD); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("globalHueShift");
          p->setLabels("Global Hue Shift","Global Hue Shift","Global Hue Shift");
          p->setDefault(0); p->setRange(-180,180); p->setDisplayRange(-180,180);
          p->setHint("Rotate entire image hue — keyframe for psychedelic cycling");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("mirrorGlobal");
          p->setLabels("Global Mirror","Global Mirror","Global Mirror");
          p->setDefault(0); p->setRange(0,1); p->setDisplayRange(0,1);
          p->setHint("Mirror the image horizontally (0=off, 1=on)");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ---- Enable Dreamcore (at bottom of Dreamcore group) ----
        { BooleanParamDescriptor* b = d.defineBooleanParam("enableDreamcore");
          b->setLabels("Enable Dreamcore","Enable Dreamcore","Enable Dreamcore");
          b->setDefault(false);
          b->setHint("Enable full-frame atmosphere effects (vignette, grain, haze...)");
          b->setAnimates(false); b->setParent(*gD); pg->addChild(*b); }

        // ========== HIDDEN ==========
        { StringParamDescriptor* p = d.defineStringParam("_strokeData");
          p->setLabels("_strokeData","_strokeData","_strokeData");
          p->setDefault(""); p->setIsSecret(true); p->setAnimates(false);
          pg->addChild(*p); }
    }

    virtual ImageEffect* createInstance(OfxImageEffectHandle h, ContextEnum) {
        return new ManoStretchPlugin(h);
    }
};

static ManoStretchFactory g_factory;
void OFX::Plugin::getPluginIDs(PluginFactoryArray& a) { a.push_back(&g_factory); }