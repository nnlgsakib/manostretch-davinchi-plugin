#include "GpuStretchPlugin.h"
#include "ofxsImageEffect.h"
#include "ofxsParam.h"
#include "ofxsInteract.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace GpuStretch {

#define kPluginName "GpuStretch"
#define kPluginGroupName "GpuStretch"
#define kPluginDescription "GPU-accelerated Liquify stretch effect for DaVinci Resolve"
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

enum { kMaxWarps = 100 };

struct WarpStroke {
    float x;
    float y;
    float prevX;
    float prevY;
    int warpType;
    float strength;
    float radius;
    float angle;
};

static std::vector<WarpStroke> g_warpStrokes;
static bool g_clearStrokes = false;

class GpuStretchPlugin : public OFX::ImageEffect {
public:
    GpuStretchPlugin(OFX::ImageEffect* instance)
        : OFX::ImageEffect(instance)
        , dstClip_(nullptr)
        , srcClip_(nullptr)
        , enabledParam_(nullptr)
        , warpTypeParam_(nullptr)
        , strengthParam_(nullptr)
        , radiusParam_(nullptr)
        , angleParam_(nullptr)
        , reconstructAmountParam_(nullptr)
    {
        dstClip_ = fetchClip(kOfxOutputClipName);
        srcClip_ = fetchClip(kOfxSourceClipName);

        enabledParam_ = fetchBooleanParam("enabled");
        warpTypeParam_ = fetchChoiceParam("warpType");
        strengthParam_ = fetchDoubleParam("strength");
        radiusParam_ = fetchDoubleParam("radius");
        angleParam_ = fetchDoubleParam("angle");
        reconstructAmountParam_ = fetchDoubleParam("reconstructAmount");
    }

    void render(const OFX::RenderArguments& args) override
    {
        if (!enabledParam_->getValueAtTime(args.time)) {
            copySourceToDestination(srcClip_, dstClip_, args);
            return;
        }

        std::unique_ptr<OFX::Image> src(srcClip_->fetchImage(args.time));
        std::unique_ptr<OFX::Image> dst(dstClip_->fetchImage(args.time));

        if (!src.get() || !dst.get()) {
            return;
        }

        OFX::PixelType srcPixelType = src->getPixelType();
        OFX::BitDepthEnum srcBitDepth = src->getBitDepth();
        OFX::ePixelComponent srcComponents = src->getComponent();

        int renderWidth = (int)dst->getRenderScale().x * dst->getWidth();
        int renderHeight = (int)dst->getRenderScale().y * dst->getHeight();

        bool warpTypeVal;
        enabledParam_->getValueAtTime(args.time, warpTypeVal);
        
        int warpTypeVal2;
        warpTypeParam_->getValueAtTime(args.time, warpTypeVal2);
        
        double strengthVal;
        strengthParam_->getValueAtTime(args.time, strengthVal);
        
        double radiusVal;
        radiusParam_->getValueAtTime(args.time, radiusVal);
        
        double angleVal;
        angleParam_->getValueAtTime(args.time, angleVal);
        
        double reconstructVal;
        reconstructAmountParam_->getValueAtTime(args.time, reconstructVal);

        bool doCUDA = false;
        bool doOpenCL = false;
        bool doMetal = false;

#ifdef ENABLE_CUDA
        doCUDA = true;
#endif
#ifdef ENABLE_OPENCL
        doOpenCL = true;
#endif
#ifdef ENABLE_METAL
        doMetal = true;
#endif

        if (doCUDA) {
            processImagesCUDA(src.get(), dst.get(), renderWidth, renderHeight,
                             (WarpType)warpTypeVal2, (float)strengthVal, 
                             (float)radiusVal, (float)angleVal, (float)reconstructVal);
        } else if (doOpenCL) {
            processImagesOpenCL(src.get(), dst.get(), renderWidth, renderHeight,
                               (WarpType)warpTypeVal2, (float)strengthVal,
                               (float)radiusVal, (float)angleVal, (float)reconstructVal);
        } else if (doMetal) {
            processImagesMetal(src.get(), dst.get(), renderWidth, renderHeight,
                              (WarpType)warpTypeVal2, (float)strengthVal,
                              (float)radiusVal, (float)angleVal, (float)reconstructVal);
        } else {
            processImagesCPU(src.get(), dst.get(), renderWidth, renderHeight,
                           (WarpType)warpTypeVal2, (float)strengthVal,
                           (float)radiusVal, (float)angleVal, (float)reconstructVal);
        }
    }

    bool isIdentity(const OFX::IsIdentityArguments& args) override
    {
        bool enabled;
        enabledParam_->getValueAtTime(args.time, enabled);
        return !enabled;
    }

    void getClipComponents(const OFX::ClipComponentsArguments& args,
                        OFX::ClipComponentSet& components) override
    {
        if (srcClip_) {
            components.insert(*srcClip_);
        }
    }

protected:
    OFX::Clip* dstClip_;
    OFX::Clip* srcClip_;
    
    OFX::BooleanParam* enabledParam_;
    OFX::ChoiceParam* warpTypeParam_;
    OFX::DoubleParam* strengthParam_;
    OFX::DoubleParam* radiusParam_;
    OFX::DoubleParam* angleParam_;
    OFX::DoubleParam* reconstructAmountParam_;

    void processImagesCPU(OFX::Image* src, OFX::Image* dst, int width, int height,
                        WarpType warpType, float strength, float radius,
                        float angle, float reconstruct);

    void processImagesCUDA(OFX::Image* src, OFX::Image* dst, int width, int height,
                         WarpType warpType, float strength, float radius,
                         float angle, float reconstruct);

    void processImagesOpenCL(OFX::Image* src, OFX::Image* dst, int width, int height,
                            WarpType warpType, float strength, float radius,
                            float angle, float reconstruct);

    void processImagesMetal(OFX::Image* src, OFX::Image* dst, int width, int height,
                          WarpType warpType, float strength, float radius,
                          float angle, float reconstruct);

    void applyWarp(float* dstPixels, int width, int height,
                 float centerX, float centerY,
                 WarpType warpType, float strength,
                 float radius, float angle);

    float interpolatePixel(float* srcPixels, int width, int height,
                        float x, float y, int channel);
};

void GpuStretchPlugin::processImagesCPU(OFX::Image* src, OFX::Image* dst, int width, int height,
                                     WarpType warpType, float strength, float radius,
                                     float angle, float reconstruct)
{
    float* srcPixels = (float*)src->getPixelData();
    float* dstPixels = (float*)dst->getPixelData();

    for (int i = 0; i < width * height * 4; i++) {
        dstPixels[i] = srcPixels[i];
    }

    for (const auto& stroke : g_warpStrokes) {
        applyWarp(dstPixels, width, height, stroke.x, stroke.y,
                (WarpType)stroke.warpType, stroke.strength,
                stroke.radius, stroke.angle);
    }

    if (reconstruct > 0.0f) {
        for (int i = 0; i < width * height * 4; i++) {
            float srcVal = srcPixels[i];
            float dstVal = dstPixels[i];
            dstPixels[i] = srcVal + (dstVal - srcVal) * (1.0f - reconstruct);
        }
    }
}

void GpuStretchPlugin::applyWarp(float* dstPixels, int width, int height,
                         float centerX, float centerY,
                         WarpType warpType, float strength,
                         float radius, float angle)
{
    int radiusPx = (int)(radius * std::max(width, height) / 100.0f);
    if (radiusPx < 1) radiusPx = 1;

    for (int py = centerY - radiusPx; py <= centerY + radiusPx; py++) {
        for (int px = centerX - radiusPx; px <= centerX + radiusPx; px++) {
            if (px < 0 || px >= width || py < 0 || py >= height) continue;

            float dx = px - centerX;
            float dy = py - centerY;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist > radiusPx) continue;

            float falloff = 1.0f - (dist / radiusPx);
            if (falloff < 0) falloff = 0;
            falloff = falloff * falloff * (3.0f - 2.0f * falloff);

            float offsetX = 0, offsetY = 0;

            switch (warpType) {
                case eWarpTypePush: {
                    float pushAngle = angle * 3.14159f / 180.0f;
                    offsetX = std::cos(pushAngle) * strength * falloff * 20.0f;
                    offsetY = std::sin(pushAngle) * strength * falloff * 20.0f;
                    break;
                }
                case eWarpTypeTwirl: {
                    float twistAmount = strength * falloff * 0.5f;
                    float cosA = std::cos(twistAmount);
                    float sinA = std::sin(twistAmount);
                    float newX = dx * cosA - dy * sinA;
                    float newY = dx * sinA + dy * cosA;
                    offsetX = newX - dx;
                    offsetY = newY - dy;
                    break;
                }
                case eWarpTypePinch: {
                    float pinchAmount = -strength * falloff * 0.3f;
                    offsetX = dx * pinchAmount;
                    offsetY = dy * pinchAmount;
                    break;
                }
                case eWarpTypeBloat: {
                    float bloatAmount = strength * falloff * 0.3f;
                    offsetX = -dx * bloatAmount;
                    offsetY = -dy * bloatAmount;
                    break;
                }
                case eWarpTypeTurbulence: {
                    float turbFreq = 0.1f;
                    float noiseX = std::sin(py * turbFreq + strength * 10.0f) * std::cos(px * turbFreq);
                    float noiseY = std::sin(py * turbFreq + strength * 10.0f) * std::cos(px * turbFreq);
                    offsetX = noiseX * strength * falloff * 10.0f;
                    offsetY = noiseY * strength * falloff * 10.0f;
                    break;
                }
                default:
                    break;
            }

            int idx = (py * width + px) * 4;
            
            float sampleX = px + offsetX;
            float sampleY = py + offsetY;

            dstPixels[idx + 0] = interpolatePixel(dstPixels, width, height, sampleX, sampleY, 0);
            dstPixels[idx + 1] = interpolatePixel(dstPixels, width, height, sampleX, sampleY, 1);
            dstPixels[idx + 2] = interpolatePixel(dstPixels, width, height, sampleX, sampleY, 2);
        }
    }
}

float GpuStretchPlugin::interpolatePixel(float* srcPixels, int width, int height,
                                   float x, float y, int channel)
{
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = x - x0;
    float fy = y - y0;

    if (x0 < 0 || x0 >= width || y0 < 0 || y0 >= height) return 0;
    if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height) return 0;

    float v00 = srcPixels[(y0 * width + x0) * 4 + channel];
    float v10 = srcPixels[(y0 * width + x1) * 4 + channel];
    float v01 = srcPixels[(y1 * width + x0) * 4 + channel];
    float v11 = srcPixels[(y1 * width + x1) * 4 + channel];

    float v0 = v00 * (1 - fx) + v10 * fx;
    float v1 = v01 * (1 - fx) + v11 * fx;

    return v0 * (1 - fy) + v1 * fy;
}

void GpuStretchPlugin::processImagesCUDA(OFX::Image* src, OFX::Image* dst, int width, int height,
                                    WarpType warpType, float strength, float radius,
                                    float angle, float reconstruct)
{
    processImagesCPU(src, dst, width, height, warpType, strength, radius, angle, reconstruct);
}

void GpuStretchPlugin::processImagesOpenCL(OFX::Image* src, OFX::Image* dst, int width, int height,
                                     WarpType warpType, float strength, float radius,
                                     float angle, float reconstruct)
{
    processImagesCPU(src, dst, width, height, warpType, strength, radius, angle, reconstruct);
}

void GpuStretchPlugin::processImagesMetal(OFX::Image* src, OFX::Image* dst, int width, int height,
                                    WarpType warpType, float strength, float radius,
                                    float angle, float reconstruct)
{
    processImagesCPU(src, dst, width, height, warpType, strength, radius, angle, reconstruct);
}

class GpuStretchInteract : public OFX::Interact {
public:
    GpuStretchInteract(OFX::ImageEffect* effect)
        : OFX::Interact(effect)
        , m_effect(effect)
        , m_isDragging(false)
        , m_lastX(0)
        , m_lastY(0)
    {
    }

    void penDown(const OFX::PenArguments& args) override
    {
        m_isDragging = true;
        m_lastX = args.x;
        m_lastY = args.y;

        WarpStroke stroke;
        stroke.x = args.x;
        stroke.y = args.y;
        stroke.prevX = args.x;
        stroke.prevY = args.y;
        
        OFX::ChoiceParam* warpTypeParam = m_effect->fetchChoiceParam("warpType");
        OFX::DoubleParam* strengthParam = m_effect->fetchDoubleParam("strength");
        OFX::DoubleParam* radiusParam = m_effect->fetchDoubleParam("radius");
        OFX::DoubleParam* angleParam = m_effect->fetchDoubleParam("angle");

        int warpType = 0;
        double strength = 50.0;
        double radius = 10.0;
        double angle = 0.0;

        warpTypeParam->getValue(warpType);
        strengthParam->getValue(strength);
        radiusParam->getValue(radius);
        angleParam->getValue(angle);

        stroke.warpType = warpType;
        stroke.strength = (float)strength;
        stroke.radius = (float)radius;
        stroke.angle = (float)angle;

        g_warpStrokes.push_back(stroke);
    }

    void penUp(const OFX::PenArguments& args) override
    {
        m_isDragging = false;
    }

    void penMotion(const OFX::PenArguments& args) override
    {
        if (!m_isDragging) return;

        if (!g_warpStrokes.empty()) {
            WarpStroke& lastStroke = g_warpStrokes.back();
            lastStroke.prevX = lastStroke.x;
            lastStroke.prevY = lastStroke.y;
            lastStroke.x = args.x;
            lastStroke.y = args.y;
        }

        m_lastX = args.x;
        m_lastY = args.y;

        m_effect->requestRender();
    }

    void keyDown(const OFX::KeyArguments& args) override
    {
        if (args.keySym == 'r' || args.keySym == 'R') {
            g_warpStrokes.clear();
            g_clearStrokes = true;
            m_effect->requestRender();
        }
    }

protected:
    void draw(const OFX::DrawArguments& args) override
    {
    }

private:
    OFX::ImageEffect* m_effect;
    bool m_isDragging;
    float m_lastX;
    float m_lastY;
};

class GpuStretchFactory : public OFX::PluginFactory {
public:
    GpuStretchFactory()
        : OFX::PluginFactory(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
    {
    }

    void describe(OFX::ImageEffectDescriptor& desc) override
    {
        desc.setLabel(kPluginName);
        desc.setPluginGroup(kPluginGroupName);
        desc.setPluginDescription(kPluginDescription);
        desc.setVersionMajor(kPluginVersionMajor);
        desc.setVersionMinor(kPluginVersionMinor);

        desc.addSupportedContext(eContextFilter);
        desc.addSupportedContext(eContextGeneral);
        desc.addSupportedPixelComponent(ePixelComponentRGBA);
        desc.addSupportedPixelComponent(ePixelComponentRGB);
        desc.addSupportedBitDepth(eBitDepthFloat);

        desc.setSupportsMultipleClipRenderings(false);
        desc.setSupportsTiles(true);
        desc.setSupportsMultiResolution(true);
        desc.setSupportsCudaStream(true);

        OFX::ClipDescriptor* srcClip = desc.defineClip(kOfxSourceClipName);
        srcClip->addSupportedComponent(ePixelComponentRGBA);
        srcClip->addSupportedComponent(ePixelComponentRGB);
        srcClip->setSupportsTiles(true);

        OFX::ClipDescriptor* dstClip = desc.defineClip(kOfxOutputClipName);
        dstClip->addSupportedComponent(ePixelComponentRGBA);
        dstClip->addSupportedComponent(ePixelComponentRGB);
        dstClip->setSupportsTiles(true);
    }

    void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) override
    {
        OFX::ClipDescriptor* srcClip = desc.getClipByName(kOfxSourceClipName);
        OFX::ClipDescriptor* dstClip = desc.getClipByName(kOfxOutputClipName);

        srcClip->setOptional(false);
        dstClip->setOptional(false);

        OFX::PageParamDescriptor* page = desc.definePageParam("Controls");

        OFX::BooleanParamDescriptor* enabled = desc.defineBooleanParam("enabled");
        enabled->setDefault(true);
        enabled->setHint("Enable the warp effect");
        enabled->setLabel("Enabled");
        page->addChild(*enabled);

        OFX::ChoiceParamDescriptor* warpType = desc.defineChoiceParam("warpType");
        warpType->setDefault(0);
        warpType->setHint("Type of warp operation");
        warpType->setLabel("Warp Type");
        warpType->appendOption("Push");
        warpType->appendOption("Twirl");
        warpType->appendOption("Pinch");
        warpType->appendOption("Bloat");
        warpType->appendOption("Turbulence");
        warpType->appendOption("Reconstruct");
        page->addChild(*warpType);

        OFX::DoubleParamDescriptor* strength = desc.defineDoubleParam("strength");
        strength->setDefault(50.0);
        strength->setRange(0.0, 100.0);
        strength->setHint("Strength of the warp effect");
        strength->setLabel("Strength");
        page->addChild(*strength);

        OFX::DoubleParamDescriptor* radius = desc.defineDoubleParam("radius");
        radius->setDefault(10.0);
        radius->setRange(1.0, 100.0);
        radius->setHint("Brush radius");
        radius->setLabel("Radius");
        page->addChild(*radius);

        OFX::DoubleParamDescriptor* angle = desc.defineDoubleParam("angle");
        angle->setDefault(0.0);
        angle->setRange(-180.0, 180.0);
        angle->setHint("Push direction angle");
        angle->setLabel("Angle");
        page->addChild(*angle);

        OFX::DoubleParamDescriptor* reconstructAmount = desc.defineDoubleParam("reconstructAmount");
        reconstructAmount->setDefault(0.0);
        reconstructAmount->setRange(0.0, 100.0);
        reconstructAmount->setHint("Amount of reconstruction");
        reconstructAmount->setLabel("Reconstruct");
        page->addChild(*reconstructAmount);

        desc.setImageEffectInteractLayouter(
            new OFX::DefaultInteractLayouter(desc, "GPU Liquify", 350, 220));

        desc.setOverlayInteractDescriptorFactory(
            [](OFX::ImageEffect* effect) -> OFX::Interact* {
                return new GpuStretchInteract(effect);
            });
    }

    OFX::ImageEffect* createInstance(OFX::ContextEnum context, OFX::ImageEffect* instance) override
    {
        return new GpuStretchPlugin(instance);
    }
};

static GpuStretchFactory p_GpuStretchFactory;

}

OfxStatus pluginMain(const char* action, const void* handle, OFX::PropertySetType inArgs, OFX::PropertySetType outArgs)
{
    return OFX::ImageEffect::pluginMain(action, handle, inArgs, outArgs, &GpuStretch::p_GpuStretchFactory);
}