#pragma once
#ifndef _ofxCore_h_
#define _ofxCore_h_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OfxStatus
typedef int OfxStatus;
#endif

#ifndef OfxBool
typedef int OfxBool;
#endif

#define kOfxStatOK                  0
#define kOfxStatErrUnknown           1
#define kOfxStatErrFatal            2
#define kOfxStatErrPluginBadAPI     3
#define kOfxStatErrPluginWrongAPI   4
#define kOfxStatErrHostBadAPI       5
#define kOfxStatErrHostWrongAPI     6
#define kOfxStatErrMissedArgImage    7
#define kOfxStatErrMissedArgUTF8     8
#define kOfxStatErrSetFailed        14
#define kOfxStatErrValue          100
#define kOfxStatErrWrongType       101
#define kOfxStatErrWrongDimension  102
#define kOfxStatErrBadIndex        103
#define kOfxStatErrMemory        200

#define kOfxTrue                   1
#define kOfxFalse                 0

#define kOfxPropName               "OfxPropName"
#define kOfxPropPluginDescription "OfxPropPluginDescription"
#define kOfxPropPluginGroup        "OfxPropPluginGroup"
#define kOfxPropVersion           "OfxPropVersion"

typedef char* OfxPluginID;
typedef struct OfxPluginStruct* OfxPluginHandle;
typedef struct OfxPropertySetStruct* OfxPropertySetHandle;
typedef struct OfxHostStruct* OfxHostHandle;
typedef struct OfxImageClipStruct* OfxImageClipHandle;

typedef struct OfxPluginStruct {
    const char* pluginApi;
    int apiVersion;
    const char* pluginIdentifier;
    int pluginVersionMajor;
    int pluginVersionMinor;
    OfxStatus (*mainEntry)(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out);
} OfxPlugin;

typedef OfxPlugin* (*OfxPluginGetPluginFunc)(int index);
typedef int (*OfxPluginGetNumberOfPluginsFunc)(void);

OfxStatus ofxPropertySetGetPointer(OfxPropertySetHandle props, const char* name, int index, void** value);
OfxStatus ofxPropertySetSetPointer(OfxPropertySetHandle props, const char* name, int index, void* value);
OfxStatus ofxPropertyGetInt(OfxPropertySetHandle props, const char* name, int index, int* value);
OfxStatus ofxPropertySetInt(OfxPropertySetHandle props, const char* name, int index, int value);
OfxStatus ofxPropertyGetDouble(OfxPropertySetHandle props, const char* name, int index, double* value);
OfxStatus ofxPropertySetDouble(OfxPropertySetHandle props, const char* name, int index, double value);
OfxStatus ofxPropertyGetString(OfxPropertySetHandle props, const char* name, int index, char** value);
OfxStatus ofxPropertySetString(OfxPropertySetHandle props, const char* name, int index, const char* value);

typedef struct OfxImageClipStruct* OfxImageClipHandle;
typedef struct OfxImageStruct* OfxImageHandle;

OfxStatus ofxImageClipGetImage(OfxImageClipHandle clip, double time, OfxPropertySetHandle* propertySet);
OfxStatus ofxImageClipReleaseImage(OfxPropertySetHandle propSet);

#define kOfxImagePropData           "OfxImagePropData"
#define kOfxImagePropBounds        "OfxImagePropBounds"
#define kOfxImagePropPixelDepth   "OfxImagePropPixelDepth"
#define kOfxImagePropComponents   "OfxImagePropComponents"
#define kOfxImagePropRowBytes     "OfxImagePropRowBytes"
#define kOfxImagePropPreMultiplied "OfxImagePropPreMultiplied"

typedef void* OfxMemoryHandle;
typedef struct OfxMemorySuiteV1Struct {
    OfxStatus (*flush)(OfxMemoryHandle handle);
    OfxStatus (*alloc)(OfxMemoryHandle handle, int nBytes, OfxMemoryHandle* memory);
    OfxStatus (*lock)(OfxMemoryHandle handle);
    OfxStatus (*unlock)(OfxMemoryHandle handle);
} OfxMemorySuiteV1;

typedef struct OfxMultiThreadSuiteV1Struct {
    OfxStatus (*lockMutex)(void* mutex);
    OfxStatus (*unlockMutex)(void* mutex);
    OfxStatus (*createMutex)(void** mutex);
    OfxStatus (*destroyMutex)(void* mutex);
    OfxStatus (*allocMultiThread)(int nThreads, int nCount, int threadIndex);
    OfxStatus (*waitMultiThread)(int nThreads, int nCount);
} OfxMultiThreadSuiteV1;

typedef struct OfxPropertySuiteV1Struct {
    OfxStatus (*propSetPointer)(OfxPropertySetHandle handle, const char* property, int index, void* value);
    OfxStatus (*propGetPointer)(OfxPropertySetHandle handle, const char* property, int index, void** value);
    OfxStatus (*propSetInt)(OfxPropertySetHandle handle, const char* property, int index, int value);
    OfxStatus (*propGetInt)(OfxPropertySetHandle handle, const char* property, int index, int* value);
    OfxStatus (*propSetDouble)(OfxPropertySetHandle handle, const char* property, int index, double value);
    OfxStatus (*propGetDouble)(OfxPropertySetHandle handle, const char* property, int index, double* value);
    OfxStatus (*propSetString)(OfxPropertySetHandle handle, const char* property, int index, const char* value);
    OfxStatus (*propGetString)(OfxPropertySetHandle handle, const char* property, int index, char** value);
} OfxPropertySuiteV1;

typedef struct OfxImageEffectSuiteV1Struct {
    OfxStatus (*getParamSet)(OfxPropertySetHandle imageEffect, OfxPropertySetHandle* paramSet);
    OfxStatus (*getCLIP)(OfxPropertySetHandle imageEffect, const char* name, OfxImageClipHandle* clip);
    OfxStatus (*clipDefine)(OfxPropertySetHandle desc, const char* name, OfxImageClipHandle* clip);
    OfxStatus (*paramDefine)(OfxPropertySetHandle desc, const char* name, OfxPropertySetHandle* param);
    OfxStatus (*getPropertySet)(OfxPropertySetHandle imageEffect, OfxPropertySetHandle* propSet);
} OfxImageEffectSuiteV1;

#ifdef __cplusplus
}
#endif

#endif