#include "GpuStretchPlugin.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define kPluginName "GpuStretch"
#define kPluginGroupName "Distort"
#define kPluginIdentifier "com.gpustretch.liquify"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

enum WarpType {
    eWarpTypePush = 0,
    eWarpTypeTwirl,
    eWarpTypePinch,
    eWarpTypeBloat,
    eWarpTypeTurbulence,
    eWarpTypeReconstruct
};

typedef struct {
    float x, y;
    float prevX, prevY;
    int warpType;
    float strength;
    float radius;
    float angle;
} WarpStroke;

static WarpStroke g_strokes[100];
static int g_strokeCount = 0;
static int g_clearStrokes = 0;

static float g_params[6] = {1, 50.0f, 10.0f, 0.0f, 0.0f, 1.0f};
static int g_warpType = 0;

static void clearStrokes(void) {
    g_strokeCount = 0;
}

static void addStroke(float x, float y, float prevX, float prevY, int warpType, 
                   float strength, float radius, float angle) {
    if (g_strokeCount < 100) {
        g_strokes[g_strokeCount].x = x;
        g_strokes[g_strokeCount].y = y;
        g_strokes[g_strokeCount].prevX = prevX;
        g_strokes[g_strokeCount].prevY = prevY;
        g_strokes[g_strokeCount].warpType = warpType;
        g_strokes[g_strokeCount].strength = strength;
        g_strokes[g_strokeCount].radius = radius;
        g_strokes[g_strokeCount].angle = angle;
        g_strokeCount++;
    }
}

static float interpolateBilinear(float* img, int w, int h, float x, float y, int ch) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = x - x0;
    float fy = y - y0;
    
    if (x0 < 0 || x0 >= w || y0 < 0 || y0 >= h) return 0;
    if (x1 < 0 || x1 >= w || y1 < 0 || y1 >= h) return 0;
    
    float v00 = img[(y0 * w + x0) * 4 + ch];
    float v10 = img[(y0 * w + x1) * 4 + ch];
    float v01 = img[(y1 * w + x0) * 4 + ch];
    float v11 = img[(y1 * w + x1) * 4 + ch];
    
    float v0 = v00 * (1 - fx) + v10 * fx;
    float v1 = v01 * (1 - fx) + v11 * fx;
    
    return v0 * (1 - fy) + v1 * fy;
}

static void applyWarp(float* dst, int w, int h, float cx, float cy, 
                   int warpType, float strength, float radius, float angle) {
    int radiusPx = (int)(radius * ((float)w / 100.0f));
    if (radiusPx < 1) radiusPx = 1;
    
    for (int py = (int)cy - radiusPx; py <= (int)cy + radiusPx; py++) {
        for (int px = (int)cx - radiusPx; px <= (int)cx + radiusPx; px++) {
            if (px < 0 || px >= w || py < 0 || py >= h) continue;
            
            float dx = (float)px - cx;
            float dy = (float)py - cy;
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist > (float)radiusPx) continue;
            
            float falloff = 1.0f - (dist / (float)radiusPx);
            if (falloff < 0) falloff = 0;
            falloff = falloff * falloff * (3.0f - 2.0f * falloff);
            
            float offX = 0, offY = 0;
            
            switch (warpType) {
                case eWarpTypePush: {
                    float pushAngle = angle * 3.14159265f / 180.0f;
                    float pushStrength = strength * 0.01f;
                    offX = cosf(pushAngle) * pushStrength * falloff * 20.0f;
                    offY = sinf(pushAngle) * pushStrength * falloff * 20.0f;
                    break;
                }
                case eWarpTypeTwirl: {
                    float twist = strength * 0.001f * falloff;
                    float c = cosf(twist);
                    float s = sinf(twist);
                    float newX = dx * c - dy * s;
                    float newY = dx * s + dy * c;
                    offX = newX - dx;
                    offY = newY - dy;
                    break;
                }
                case eWarpTypePinch: {
                    float pinch = -strength * 0.001f * falloff;
                    offX = dx * pinch;
                    offY = dy * pinch;
                    break;
                }
                case eWarpTypeBloat: {
                    float bloat = strength * 0.001f * falloff;
                    offX = -dx * bloat;
                    offY = -dy * bloat;
                    break;
                }
                case eWarpTypeTurbulence: {
                    float freq = 0.1f;
                    float n = strength * 0.01f;
                    offX = sinf(py * freq + n * 10.0f) * cosf(px * freq) * n * falloff * 10.0f;
                    offY = sinf(py * freq + n * 10.0f) * cosf(px * freq) * n * falloff * 10.0f;
                    break;
                }
            }
            
            int idx = (py * w + px) * 4;
            float sx = (float)px + offX;
            float sy = (float)py + offY;
            
            dst[idx + 0] = interpolateBilinear(dst, w, h, sx, sy, 0);
            dst[idx + 1] = interpolateBilinear(dst, w, h, sx, sy, 1);
            dst[idx + 2] = interpolateBilinear(dst, w, h, sx, sy, 2);
        }
    }
}

static void processCPU(float* src, float* dst, int w, int h,
                    int warpType, float strength, float radius, 
                    float angle, float reconstruct) {
    memcpy(dst, src, w * h * 4 * sizeof(float));
    
    for (int i = 0; i < g_strokeCount; i++) {
        applyWarp(dst, w, h, g_strokes[i].x, g_strokes[i].y,
                 g_strokes[i].warpType, g_strokes[i].strength,
                 g_strokes[i].radius, g_strokes[i].angle);
    }
    
    if (reconstruct > 0.0f) {
        for (int i = 0; i < w * h * 4; i++) {
            float orig = src[i];
            float warped = dst[i];
            dst[i] = orig + (warped - orig) * (1.0f - reconstruct);
        }
    }
}

OfxStatus GpuStretchPluginSetPointer(OfxPropertySetHandle props, const char* name, void* ptr, int count) {
    return ofxPropertySetSetPointer(props, name, 0, ptr);
}

OfxStatus GpuStretchPluginGetPointer(OfxPropertySetHandle props, const char* name, void** ptr) {
    return ofxPropertySetGetPointer(props, name, 0, ptr);
}

void GpuStretchProcess(OfxImageClipHandle src, OfxImageClipHandle dst, 
                      double time, int renderW, int renderH,
                      int warpType, float strength, float radius, 
                      float angle, float reconstruct, int gpuMode) {
    OfxPropertySetHandle srcImg = NULL;
    OfxPropertySetHandle dstImg = NULL;
    
    ofxImageClipGetImage(src, time, &srcImg);
    ofxImageClipGetImage(dst, time, &dstImg);
    
    if (!srcImg || !dstImg) return;
    
    float* srcPixels = NULL;
    float* dstPixels = NULL;
    
    ofxPropertySetGetPointer(srcImg, kOfxImagePropData, 0, (void**)&srcPixels);
    ofxPropertySetGetPointer(dstImg, kOfxImagePropData, 0, (void**)&dstPixels);
    
    if (srcPixels && dstPixels) {
        if (gpuMode) {
            OfxPropertySetHandle clQueue = NULL;
            ofxPropertySetGetPointer(dstImg, "OfxImagePropGPUCommandQueue", 0, (void**)&clQueue);
        }
        
        processCPU(srcPixels, dstPixels, renderW, renderH, warpType, strength, radius, angle, reconstruct);
    }
    
    if (srcImg) ofxImageClipReleaseImage(srcImg);
    if (dstImg) ofxImageClipReleaseImage(dstImg);
}

OfxStatus GpuStretchDescribe(OfxImageEffectDescriptor* desc) {
    ofxImageEffectDescSetProperty(desc, kOfxPropName, kPluginName);
    ofxImageEffectDescSetProperty(desc, kOfxPropPluginDescription, "GPU-accelerated Liquify stretch effect");
    ofxImageEffectDescSetProperty(desc, kOfxPropPluginGroup, kPluginGroupName);
    ofxImageEffectDescSetProperty(desc, kOfxPropVersion, "1");
    ofxImageEffectDescSetProperty(desc, kOfxPropVersion, "0");
    
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropSupportsOverlays, 1);
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropSupportsMultiResolution, 1);
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropSupportsTiles, 1);
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropSupportsCudaStream, 1);
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropOpenCLRenderSupported, 1);
    
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropSupportedContext, eContextFilter);
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropSupportedPixelComponents, ePixelComponentRGBA);
    ofxImageEffectDescSetProperty(desc, kOfxImageEffectPropSupportedPixelDepths, eBitDepthFloat);
    
    OfxImageClipDescriptor* srcClip = NULL;
    ofxImageEffectDescGetClip(desc, kOfxSourceClipName, &srcClip);
    ofxImageClipDescSetProperty(srcClip, kOfxImageClipPropSupportsTiles, 1);
    ofxImageClipDescSetProperty(srcClip, kOfxImageClipPropSupportedPixelComponents, ePixelComponentRGBA);
    ofxImageClipDescSetProperty(srcClip, kOfxImageClipPropSupportedPixelComponents, ePixelComponentRGB);
    
    OfxImageClipDescriptor* dstClip = NULL;
    ofxImageEffectDescGetClip(desc, kOfxOutputClipName, &dstClip);
    ofxImageClipDescSetProperty(dstClip, kOfxImageClipPropSupportsTiles, 1);
    ofxImageClipDescSetProperty(dstClip, kOfxImageClipPropSupportedPixelComponents, ePixelComponentRGBA);
    ofxImageClipDescSetProperty(dstClip, kOfxImageClipPropSupportedPixelComponents, ePixelComponentRGB);
    
    return kOfxStatOK;
}

OfxStatus GpuStretchDescribeInContext(OfxImageEffectDescriptor* desc, OfxContextEnum context) {
    OfxParameterDescriptor* param = NULL;
    OfxPageParamDescriptor* page = NULL;
    ofxImageEffectDescGetParamPage(desc, "Controls", &page);
    
    ofxImageEffectDescGetParam(desc, "enabled", &param);
    ofxParamDescSetProperty(param, kOfxParamPropDefault, 1);
    ofxParamDescSetProperty(param, kOfxParamPropType, eParamTypeBoolean);
    ofxParamDescSetProperty(param, kOfxParamPropHint, "Enable the warp effect");
    ofxParamDescSetProperty(param, kOfxParamPropLabel, "Enabled");
    ofxPageParamAddParam(page, param);
    
    ofxImageEffectDescGetParam(desc, "warpType", &param);
    ofxParamDescSetProperty(param, kOfxParamPropDefault, 0);
    ofxParamDescSetProperty(param, kOfxParamPropType, eParamTypeChoice);
    ofxParamDescSetProperty(param, kOfxParamPropHint, "Type of warp operation");
    ofxParamDescSetProperty(param, kOfxParamPropLabel, "Warp Type");
    ofxParamChoiceAppendOption(param, "Push");
    ofxParamChoiceAppendOption(param, "Twirl");
    ofxParamChoiceAppendOption(param, "Pinch");
    ofxParamChoiceAppendOption(param, "Bloat");
    ofxParamChoiceAppendOption(param, "Turbulence");
    ofxParamChoiceAppendOption(param, "Reconstruct");
    ofxPageParamAddParam(page, param);
    
    ofxImageEffectDescGetParam(desc, "strength", &param);
    ofxParamDescSetProperty(param, kOfxParamPropDefault, 50.0);
    ofxParamDescSetProperty(param, kOfxParamPropType, eParamTypeDouble);
    ofxParamDescSetProperty(param, kOfxParamPropScriptName, "strength");
    ofxParamDescSetProperty(param, kOfxParamPropDoubleType, eParamTypePlain);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMin, 0.0);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMax, 100.0);
    ofxParamDescSetProperty(param, kOfxParamPropHint, "Strength of the warp effect");
    ofxParamDescSetProperty(param, kOfxParamPropLabel, "Strength");
    ofxPageParamAddParam(page, param);
    
    ofxImageEffectDescGetParam(desc, "radius", &param);
    ofxParamDescSetProperty(param, kOfxParamPropDefault, 10.0);
    ofxParamDescSetProperty(param, kOfxParamPropType, eParamTypeDouble);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMin, 1.0);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMax, 100.0);
    ofxParamDescSetProperty(param, kOfxParamPropHint, "Brush radius");
    ofxParamDescSetProperty(param, kOfxParamPropLabel, "Radius");
    ofxPageParamAddParam(page, param);
    
    ofxImageEffectDescGetParam(desc, "angle", &param);
    ofxParamDescSetProperty(param, kOfxParamPropDefault, 0.0);
    ofxParamDescSetProperty(param, kOfxParamPropType, eParamTypeDouble);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMin, -180.0);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMax, 180.0);
    ofxParamDescSetProperty(param, kOfxParamPropHint, "Push direction angle");
    ofxParamDescSetProperty(param, kOfxParamPropLabel, "Angle");
    ofxPageParamAddParam(page, param);
    
    ofxImageEffectDescGetParam(desc, "reconstructAmount", &param);
    ofxParamDescSetProperty(param, kOfxParamPropDefault, 0.0);
    ofxParamDescSetProperty(param, kOfxParamPropType, eParamTypeDouble);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMin, 0.0);
    ofxParamDescSetPropertyDouble(param, kOfxParamPropMax, 100.0);
    ofxParamDescSetProperty(param, kOfxParamPropHint, "Amount of reconstruction");
    ofxParamDescSetProperty(param, kOfxParamPropLabel, "Reconstruct");
    ofxPageParamAddParam(page, param);
    
    return kOfxStatOK;
}

static OfxImageEffectDriver* g_effect = NULL;

static OfxPropertySetHandle g_paramEnabled = NULL;
static OfxPropertySetHandle g_paramWarpType = NULL;
static OfxPropertySetHandle g_paramStrength = NULL;
static OfxPropertySetHandle g_paramRadius = NULL;
static OfxPropertySetHandle g_paramAngle = NULL;
static OfxPropertySetHandle g_paramReconstruct = NULL;

static OfxStatus GpuStretchOnInstanceChanged(OfxPropertySetHandle inst, const char* propName, OfxPropertyType propType, int instanceChangedArgsCount, ...) {
    if (g_paramEnabled == NULL) return kOfxStatOK;
    
    int enabled = 0;
    ofxPropertyGetInt(g_paramEnabled, kOfxParamPropValue, 0, &enabled);
    
    if (!enabled) return kOfxStatOK;
    
    return kOfxStatOK;
}

static OfxStatus GpuStretchOnRender(OfxPropertySetHandle inst, OfxRenderArguments* args) {
    OfxImageClipHandle srcClip = NULL;
    OfxImageClipHandle dstClip = NULL;
    
    ofxImageEffectGetSource(inst, &srcClip);
    ofxImageEffectGetDestination(inst, &dstClip);
    
    if (!srcClip || !dstClip) return kOfxStatErrValue;
    
    int enabled = 1;
    double strength = 50.0;
    double radius = 10.0;
    double angle = 0.0;
    double reconstruct = 0.0;
    int warpType = 0;
    
    if (g_paramEnabled) ofxPropertyGetInt(g_paramEnabled, kOfxParamPropValue, 0, &enabled);
    if (g_paramWarpType) ofxPropertyGetInt(g_paramWarpType, kOfxParamPropValue, 0, &warpType);
    if (g_paramStrength) ofxPropertyGetDouble(g_paramStrength, kOfxParamPropValue, 0, &strength);
    if (g_paramRadius) ofxPropertyGetDouble(g_paramRadius, kOfxParamPropValue, 0, &radius);
    if (g_paramAngle) ofxPropertyGetDouble(g_paramAngle, kOfxParamPropValue, 0, &angle);
    if (g_paramReconstruct) ofxPropertyGetDouble(g_paramReconstruct, kOfxParamPropValue, 0, &reconstruct);
    
    if (!enabled) {
        ofxImageEffectCopyClipLine(inst, srcClip, dstClip, args->time, args->renderScaleX, args->renderScaleY);
        return kOfxStatOK;
    }
    
    int renderW = (int)(args->renderScaleX * 1920);
    int renderH = (int)(args->renderScaleY * 1080);
    
    int gpuMode = 0;
    
    GpuStretchProcess(srcClip, dstClip, args->time, renderW, renderH,
                    warpType, (float)strength, (float)radius, 
                    (float)angle, (float)reconstruct, gpuMode);
    
    return kOfxStatOK;
}

static OfxStatus GpuStretchOnInteractPenDown(OfxPropertySetHandle interp, OfxInteractPenArguments* args) {
    float x = args->x;
    float y = args->y;
    
    addStroke(x, y, x, y, g_warpType, g_params[1], g_params[2], g_params[3]);
    
    if (g_effect) ofxImageEffectRequestRender(g_effect);
    
    return kOfxStatOK;
}

static OfxStatus GpuStretchOnInteractPenMotion(OfxPropertySetHandle interp, OfxInteractPenArguments* args) {
    float x = args->x;
    float y = args->y;
    
    if (g_strokeCount > 0) {
        WarpStroke* last = &g_strokes[g_strokeCount - 1];
        addStroke(x, y, last->x, last->y, g_warpType, g_params[1], g_params[2], g_params[3]);
    }
    
    if (g_effect) ofxImageEffectRequestRender(g_effect);
    
    return kOfxStatOK;
}

static OfxStatus GpuStretchOnInteractKeyDown(OfxPropertySetHandle interp, OfxInteractKeyArguments* args) {
    if (args->key == 'r' || args->key == 'R') {
        clearStrokes();
        if (g_effect) ofxImageEffectRequestRender(g_effect);
    }
    return kOfxStatOK;
}

static OfxStatus GpuStretchMain(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
    if (strcmp(action, kOfxActionDescribe) == 0) {
        OfxImageEffectDescriptor* desc = NULL;
        ofxPropertyGetPointer(inArgs, kOfxArgPropDescriptor, 0, (void**)&desc);
        return GpuStretchDescribe(desc);
    }
    else if (strcmp(action, kOfxActionDescribeInContext) == 0) {
        OfxImageEffectDescriptor* desc = NULL;
        OfxContextEnum context = eContextFilter;
        ofxPropertyGetPointer(inArgs, kOfxArgPropDescriptor, 0, (void**)&desc);
        ofxPropertyGetInt(inArgs, kOfxArgPropContext, 0, (int*)&context);
        return GpuStretchDescribeInContext(desc, context);
    }
    else if (strcmp(action, kOfxActionCreateInstance) == 0) {
        OfxImageEffectDriver* effect = NULL;
        ofxPropertyGetPointer(inArgs, kOfxArgPropImageEffect, 0, (void**)&effect);
        g_effect = effect;
        
        ofxImageEffectGetParam(effect, "enabled", &g_paramEnabled);
        ofxImageEffectGetParam(effect, "warpType", &g_paramWarpType);
        ofxImageEffectGetParam(effect, "strength", &g_paramStrength);
        ofxImageEffectGetParam(effect, "radius", &g_paramRadius);
        ofxImageEffectGetParam(effect, "angle", &g_paramAngle);
        ofxImageEffectGetParam(effect, "reconstructAmount", &g_paramReconstruct);
        
        return kOfxStatOK;
    }
    else if (strcmp(action, kOfxActionDestroyInstance) == 0) {
        g_effect = NULL;
        return kOfxStatOK;
    }
    else if (strcmp(action, kOfxActionRender) == 0) {
        OfxRenderArguments args;
        ofxPropertyGetDouble(inArgs, kOfxArgPropTime, 0, &args.time);
        ofxPropertyGetDouble(inArgs, kOfxArgPropRenderScale, 0, &args.renderScaleX);
        ofxPropertyGetDouble(inArgs, kOfxArgPropRenderScale, 1, &args.renderScaleY);
        
        OfxPropertySetHandle inst = NULL;
        ofxPropertyGetPointer(inArgs, kOfxArgPropImageEffect, 0, (void**)&inst);
        
        return GpuStretchOnRender(inst, &args);
    }
    else if (strcmp(action, kOfxInteractActionPenDown) == 0) {
        OfxInteractPenArguments args;
        ofxPropertyGetDouble(inArgs, kOfxArgPropPenX, 0, &args.x);
        ofxPropertyGetDouble(inArgs, kOfxArgPropPenY, 0, &args.y);
        
        OfxPropertySetHandle interp = NULL;
        ofxPropertyGetPointer(inArgs, kOfxArgPropInteract, 0, (void**)&interp);
        
        return GpuStretchOnInteractPenDown(interp, &args);
    }
    else if (strcmp(action, kOfxInteractActionPenMotion) == 0) {
        OfxInteractPenArguments args;
        ofxPropertyGetDouble(inArgs, kOfxArgPropPenX, 0, &args.x);
        ofxPropertyGetDouble(inArgs, kOfxArgPropPenY, 0, &args.y);
        
        OfxPropertySetHandle interp = NULL;
        ofxPropertyGetPointer(inArgs, kOfxArgPropInteract, 0, (void**)&interp);
        
        return GpuStretchOnInteractPenMotion(interp, &args);
    }
    else if (strcmp(action, kOfxInteractActionKeyDown) == 0) {
        OfxInteractKeyArguments args;
        ofxPropertyGetInt(inArgs, kOfxArgPropKey, 0, &args.key);
        
        OfxPropertySetHandle interp = NULL;
        ofxPropertyGetPointer(inArgs, kOfxArgPropInteract, 0, (void**)&interp);
        
        return GpuStretchOnInteractKeyDown(interp, &args);
    }
    
    return kOfxStatOK;
}

static OfxPlugin g_plugin = {
    kOfxImageEffectPluginApi,
    kOfxImageEffectPluginApiVersion,
    kPluginIdentifier,
    kPluginVersionMajor,
    kPluginVersionMinor,
    GpuStretchMain
};

OfxPlugin* getOfxPlugin(int index) {
    if (index == 0) return &g_plugin;
    return NULL;
}

int getNumberOfPlugins(void) {
    return 1;
}