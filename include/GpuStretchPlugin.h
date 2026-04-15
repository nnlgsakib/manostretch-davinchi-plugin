#ifndef GPU_STRETCH_PLUGIN_H
#define GPU_STRETCH_PLUGIN_H

#include <string>
#include <vector>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxInteract.h"

namespace GpuStretch {

struct StretchPoint {
    float x;
    float y;
    float strength;
    float radius;
};

struct StretchParams {
    float centerX;
    float centerY;
    float strength;
    float radius;
    float angle;
    float falloff;
    int warpType;
    bool enabled;
    std::vector<StretchPoint> points;
};

class ImageProcessor {
public:
    virtual ~ImageProcessor() = default;
    
    virtual void processImagesCUDA(const float* src, float* dst, 
                                  int width, int height, 
                                  const StretchParams& params) = 0;
    
    virtual void processImagesOpenCL(const float* src, float* dst,
                                     int width, int height,
                                     const StretchParams& params) = 0;
    
    virtual void processImagesMetal(const float* src, float* dst,
                                 int width, int height,
                                 const StretchParams& params) = 0;
    
    virtual void multiThreadProcessImages(const float* src, float* dst,
                                            int width, int height,
                                            const StretchParams& params) = 0;
};

class GpuStretchEffect : public OFX::ImageEffect {
public:
    GpuStretchEffect(OFX::ImageEffect* instance);
    virtual ~GpuStretchEffect();

    void render(const OFX::RenderArguments& args) override;
    bool isIdentity(const OFX::IsIdentityArguments& args) override;
    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args,
                          OFX::RegionOfInterestSet& regions) override;
    void getClipComponents(const OFX::ClipComponentsArguments& args,
                        OFX::ClipComponentSet& components) override;
    bool getFrameBounds(const OFX::FrameBoundsArguments& args,
                   float& boundsMin, float& boundsMax,
                   float& boundsMinY, float& boundsMaxY) override;

    void setSrcClip(OFX::Clip* clip);
    void setDstClip(OFX::Clip* clip);

protected:
    void setupAndProcess(const OFX::RenderArguments& args, OFX::ImageProcessor& processor);

private:
    OFX::Clip* m_srcClip;
    OFX::Clip* m_dstClip;
    StretchParams m_params;
    
    void getParams();
    void updateCursorPosition();
};

class GpuStretchFactory : public OFX::PluginFactory {
public:
    GpuStretchFactory();
    
    void describe(OFX::ImageEffectDescriptor& desc) override;
    void describeInContext(OFX::ImageEffectDescriptor& desc,
                          OFX::ContextEnum context) override;
    OFX::ImageEffect* createInstance(OFX::ContextEnum context,
                                    OFX::ImageEffect* instance) override;

private:
    void defineParameters(OFX::ImageEffectDescriptor& desc);
};

class GpuStretchInteract : public OFX::Interact {
public:
    GpuStretchInteract(OFX::ImageEffect* effect);
    
    voidpenDown(const OFX::PenArguments& args) override;
    void penUp(const OFX::PenArguments& args) override;
    void penMotion(const OFX::PenArguments& args) override;
    void keyDown(const OFX::KeyArguments& args) override;
    void keyUp(const OFX::KeyArguments& args) override;
    void keyFocus(const OFX::KeyFocusArguments& args) override;

protected:
    void draw(const OFX::DrawArguments& args) override;

private:
    OFX::ImageEffect* m_effect;
    int m_selectedPointIndex;
    float m_lastPenX;
    float m_lastPenY;
};

OfxStatus pluginMain(const char* action, const void* handle, OFX::PropertySetType inArgs, OFX::PropertySetType outArgs);

}

#endif