// ManoStretch.cpp v4.1 — Advanced Pixel Stretch for DaVinci Resolve
//
// Features: 6 modes, color tint, soft start, post-FX, liquidify,
//           stroke serialization (undo/redo), all params keyframeable.

#include "ofxsImageEffect.h"
#include "ofxsParam.h"
#include "ofxsInteract.h"
#include <cmath>
#include <algorithm>
#include <mutex>
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
    "ManoStretch v4.1 — click & drag to stretch pixels.\n" \
    "6 modes, color tint, post-FX, liquidify.\n" \
    "All params editable after stretching. Ctrl+Z undoes strokes.\n" \
    "[ ] resize brush.  R to reset."
#define kPluginIdentifier  "com.mano.stretch"
#define kPluginVersionMajor 4
#define kPluginVersionMinor 1

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
    float liqAmount, float liqScale);
#endif

enum StretchMode { eLinear=0, eSpiral=1, eWave=2, eTaper=3, eSmear=4, eShatter=5 };

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

// Global stroke list for overlay interact (secondary to string param)
static std::mutex                 g_strokeMutex;
static std::vector<StretchStroke> g_strokes;

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
    float liqAmount, liqScale;
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

        float localR = rPx;
        if (mode==eTaper) localR = rPx * (std::max)(0.1f, 1.f-0.9f*t/dLen);

        if (t<0.f||t>dLen) continue;
        if (std::abs(d)>localR||localR<0.5f) continue;

        float fo = smoothstep(1.f - std::abs(d)/localR);
        float tN = t / dLen;

        float sf = 1.f;
        if (fx.startBlend>0.001f && tN<fx.startBlend)
            sf = smoothstep(tN / fx.startBlend);

        float lf = 1.f - tN * fade;
        if (lf<0.f) lf=0.f;

        float effect = fo * sf * lf;
        if (effect<0.001f) continue;

        // Displacement backward along drag
        float disp = t * strength * effect;
        float srcFx = (float)px - disp * ux;
        float srcFy = (float)py - disp * uy;

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

        float blend = effect * fx.postOpacity;
        float* dp = pxAt(dstBase, dB, dRB, px, py);
        if (!dp) continue;
        for (int c=0; c<4; c++)
            dp[c] = dp[c]*(1.f-blend) + rgba[c]*blend;
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
        m_LiqAmount   = fetchDoubleParam("liqAmount");
        m_LiqScale    = fetchDoubleParam("liqScale");
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
        m_LiqAmount->getValue(v);   fx.liqAmount    = (float)v;
        m_LiqScale->getValue(v);    fx.liqScale     = (float)v;

        double rsx=a.renderScale.x, rsy=a.renderScale.y;
        int maxD = (std::max)(w, h);

#ifdef HAS_CUDA
        if (a.isEnabledCudaRender) {
            const float* sD = (const float*)src->getPixelData();
            float* dD = (float*)dst->getPixelData();
            RunCudaCopyKernel(a.pCudaStream, sD, dD, w*h);
            for (auto& s : strokes) {
                float lsx=(float)(s.startX*rsx)-dB.x1, lsy=(float)(s.startY*rsy)-dB.y1;
                float lex=(float)(s.endX*rsx)-dB.x1,   ley=(float)(s.endY*rsy)-dB.y1;
                float rPx=(float)(s.radius*maxD/100.0);
                RunCudaStretch(a.pCudaStream, sD, dD, w, h, lsx, lsy, lex, ley,
                    rPx, (float)s.strength, s.mode,
                    (float)s.tintR,(float)s.tintG,(float)s.tintB,(float)s.tintAmt,
                    (float)s.fade, (float)s.param1, (float)s.param2,
                    fx.startBlend, fx.postOpacity, fx.postBright, fx.postSat,
                    fx.liqAmount, fx.liqScale);
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

        for (auto& s : strokes) {
            StretchStroke ls = s;
            ls.startX*=rsx; ls.startY*=rsy;
            ls.endX*=rsx;   ls.endY*=rsy;
            cpuStretch(sBase, sB, sRB, dBase, dB, dRB, ls, maxD, fx);
        }
    }

    virtual bool isIdentity(const IsIdentityArguments& a,
                            Clip*& c, double& t) {
        std::string strokeStr;
        m_StrokeData->getValue(strokeStr);
        if (strokeStr.empty()) { c = m_Src; t = a.time; return true; }
        return false;
    }

private:
    Clip *m_Dst, *m_Src;
    StringParam *m_StrokeData;
    DoubleParam *m_StartBlend, *m_PostOpacity, *m_PostBright, *m_PostSat;
    DoubleParam *m_LiqAmount, *m_LiqScale;
};

// ================================================================
//  Overlay Interact
// ================================================================
class ManoStretchInteract : public OverlayInteract {
public:
    ManoStretchInteract(OfxInteractHandle h, ImageEffect* e)
        : OverlayInteract(h), m_drag(false), m_cx(0),m_cy(0),
          m_sx(0),m_sy(0), m_has(false), m_synced(false) {}

    virtual bool draw(const DrawArgs& args) {
        // Sync global strokes from string param on first draw (project reload)
        if (!m_synced) {
            m_synced = true;
            std::string sd;
            _effect->fetchStringParam("_strokeData")->getValue(sd);
            std::lock_guard<std::mutex> lk(g_strokeMutex);
            g_strokes = deserializeStrokes(sd);
        }

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
        _effect->fetchDoubleParam("tintR")->getValue(tR);
        _effect->fetchDoubleParam("tintG")->getValue(tG);
        _effect->fetchDoubleParam("tintB")->getValue(tB);
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
            std::lock_guard<std::mutex> lk(g_strokeMutex);
            int n = (int)g_strokes.size();
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
        _effect->fetchDoubleParam("tintR")->getValue(v);    s.tintR=v;
        _effect->fetchDoubleParam("tintG")->getValue(v);    s.tintG=v;
        _effect->fetchDoubleParam("tintB")->getValue(v);    s.tintB=v;
        _effect->fetchDoubleParam("tintAmount")->getValue(v); s.tintAmt=v/100.0;
        _effect->fetchDoubleParam("param1")->getValue(v);   s.param1=v;
        _effect->fetchDoubleParam("param2")->getValue(v);   s.param2=v;
        int m; _effect->fetchChoiceParam("mode")->getValue(m); s.mode=m;

        { std::lock_guard<std::mutex> lk(g_strokeMutex); g_strokes.push_back(s); }
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
            { std::lock_guard<std::mutex> lk(g_strokeMutex);
              if (!g_strokes.empty()) {
                  g_strokes.back().endX = m_cx;
                  g_strokes.back().endY = m_cy;
              } }
            // Live update: serialize to param for re-render
            syncToParam();
        }
        requestRedraw();
        return true;
    }

    virtual bool keyDown(const KeyArgs& args) {
        if (args.keyString=="r"||args.keyString=="R") {
            { std::lock_guard<std::mutex> lk(g_strokeMutex); g_strokes.clear(); }
            // Clear the string param — this is undoable!
            _effect->beginEditBlock("msReset");
            _effect->fetchStringParam("_strokeData")->setValue("");
            _effect->endEditBlock();
            requestRedraw(); return true;
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
    bool m_drag; double m_cx,m_cy,m_sx,m_sy; bool m_has, m_synced;

    void syncToParam() {
        std::string data;
        { std::lock_guard<std::mutex> lk(g_strokeMutex);
          data = serializeStrokes(g_strokes); }
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

        // ========== BRUSH SETTINGS ==========
        { GroupParamDescriptor* g = d.defineGroupParam("grpBrush");
          g->setLabels("Brush Settings","Brush Settings","Brush Settings");
          g->setOpen(true);
          pg->addChild(*g);

          ChoiceParamDescriptor* p = d.defineChoiceParam("mode");
          p->setLabels("Mode","Mode","Mode");
          p->appendOption("Linear",  "Straight pixel smear");
          p->appendOption("Spiral",  "Twisted spiral stretch");
          p->appendOption("Wave",    "Wavy/liquid stretch");
          p->appendOption("Taper",   "Narrows toward the end");
          p->appendOption("Smear",   "Motion-blur-like smear");
          p->appendOption("Shatter", "Shattered/glitchy scatter");
          p->setDefault(0); p->setAnimates(true); p->setParent(*g);
          pg->addChild(*p);
        }
        { DoubleParamDescriptor* p = d.defineDoubleParam("strength");
          p->setLabels("Strength","Strength","Strength");
          p->setDefault(80); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setHint("How much to stretch (80=organic pull, 100=extreme smear)");
          p->setAnimates(true); pg->addChild(*p); }
        { DoubleParamDescriptor* p = d.defineDoubleParam("radius");
          p->setLabels("Radius","Radius","Radius");
          p->setDefault(5); p->setRange(0.5,50); p->setDisplayRange(0.5,30);
          p->setHint("[ ] keys to adjust"); p->setAnimates(true); pg->addChild(*p); }
        { DoubleParamDescriptor* p = d.defineDoubleParam("startBlend");
          p->setLabels("Start Blend","Start Blend","Start Blend");
          p->setDefault(0.15); p->setRange(0,0.5); p->setDisplayRange(0,0.5);
          p->setHint("Soft ramp-up at the stretch origin (0=sharp, 0.5=half-length fade-in)");
          p->setAnimates(true); pg->addChild(*p); }
        { DoubleParamDescriptor* p = d.defineDoubleParam("fade");
          p->setLabels("End Fade","End Fade","End Fade");
          p->setDefault(0.3); p->setRange(0,1); p->setDisplayRange(0,1);
          p->setHint("Fade-out at the stretch end"); p->setAnimates(true); pg->addChild(*p); }

        // ========== COLOR TINT ==========
        { GroupParamDescriptor* g = d.defineGroupParam("grpTint");
          g->setLabels("Color Tint","Color Tint","Color Tint");
          g->setOpen(false); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("tintR");
          p->setLabels("Tint Red","Tint Red","Tint Red");
          p->setDefault(1); p->setRange(0,2); p->setDisplayRange(0,2);
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("tintG");
          p->setLabels("Tint Green","Tint Green","Tint Green");
          p->setDefault(1); p->setRange(0,2); p->setDisplayRange(0,2);
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("tintB");
          p->setLabels("Tint Blue","Tint Blue","Tint Blue");
          p->setDefault(1); p->setRange(0,2); p->setDisplayRange(0,2);
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("tintAmount");
          p->setLabels("Tint Amount","Tint Amount","Tint Amount");
          p->setDefault(0); p->setRange(0,100); p->setDisplayRange(0,100);
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ========== MODE DETAIL ==========
        { GroupParamDescriptor* g = d.defineGroupParam("grpDetail");
          g->setLabels("Mode Detail","Mode Detail","Mode Detail");
          g->setOpen(false); pg->addChild(*g);

          DoubleParamDescriptor* p;
          p = d.defineDoubleParam("param1");
          p->setLabels("Detail 1","Detail 1","Detail 1");
          p->setDefault(2); p->setRange(0,20); p->setDisplayRange(0,10);
          p->setHint("Spiral: turns | Wave: freq | Smear: amount | Shatter: scatter");
          p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
          p = d.defineDoubleParam("param2");
          p->setLabels("Detail 2","Detail 2","Detail 2");
          p->setDefault(0.5); p->setRange(0,5); p->setDisplayRange(0,2);
          p->setHint("Wave: amplitude"); p->setAnimates(true); p->setParent(*g); pg->addChild(*p);
        }

        // ========== POST FX (editable after stretching) ==========
        { GroupParamDescriptor* g = d.defineGroupParam("grpPostFX");
          g->setLabels("Post FX","Post FX","Post FX");
          g->setOpen(true); pg->addChild(*g);

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
        }

        // ========== LIQUIDIFY ==========
        { GroupParamDescriptor* g = d.defineGroupParam("grpLiquidify");
          g->setLabels("Liquidify","Liquidify","Liquidify");
          g->setOpen(false); pg->addChild(*g);

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