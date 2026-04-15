cmake_minimum_required(VERSION 3.18)
project(ManoStretch VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- CUDA support (optional) ----
# We compile .cu files directly with nvcc via custom command to avoid
# requiring the VS CUDA MSBuild integration.
find_program(NVCC_EXECUTABLE nvcc
    PATHS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"
          "$ENV{CUDA_PATH}/bin"
    NO_DEFAULT_PATH
)
if(NOT NVCC_EXECUTABLE)
    find_program(NVCC_EXECUTABLE nvcc)
endif()

if(NVCC_EXECUTABLE)
    set(HAS_CUDA TRUE)
    # Determine CUDA toolkit root from nvcc location
    get_filename_component(NVCC_BIN_DIR "${NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT "${NVCC_BIN_DIR}" DIRECTORY)
    message(STATUS "CUDA found: ${NVCC_EXECUTABLE} — GPU acceleration enabled")
    message(STATUS "CUDA toolkit root: ${CUDA_TOOLKIT_ROOT}")
else()
    set(HAS_CUDA FALSE)
    message(STATUS "CUDA (nvcc) not found — building CPU-only plugin")
endif()

if(WIN32)
    set(DEFAULT_PLUGIN_PATH "C:/Program Files/Common Files/OFX/Plugins" CACHE PATH "Default OFX plugin install path")
    set(RESOLVE_SDK_DIR "C:/ProgramData/Blackmagic Design/DaVinci Resolve/Support/Developer/OpenFX")
elseif(APPLE)
    set(DEFAULT_PLUGIN_PATH "/Library/OFX/Plugins" CACHE PATH "Default OFX plugin install path")
    set(RESOLVE_SDK_DIR "/Applications/DaVinci Resolve/Support/Developer/OpenFX")
else()
    set(DEFAULT_PLUGIN_PATH "/usr/OFX/Plugins" CACHE PATH "Default OFX plugin install path")
    set(RESOLVE_SDK_DIR "/opt/resolve/Developer/OpenFX")
endif()

# Add BOTH OFX include directories
set(OFX_CORE_INCLUDE "${RESOLVE_SDK_DIR}/OpenFX-1.4/include")
set(OFX_SUPPORT_INCLUDE "${RESOLVE_SDK_DIR}/Support/include")
set(OFX_SUPPORT_LIB "${RESOLVE_SDK_DIR}/Support/Library")

if(EXISTS "${OFX_CORE_INCLUDE}")
    include_directories("${OFX_CORE_INCLUDE}")
    message(STATUS "Using OFX core headers from: ${OFX_CORE_INCLUDE}")
else()
    message(WARNING "OFX core headers not found")
endif()

if(EXISTS "${OFX_SUPPORT_INCLUDE}")
    include_directories("${OFX_SUPPORT_INCLUDE}")
    message(STATUS "Using OFX support headers from: ${OFX_SUPPORT_INCLUDE}")
else()
    message(WARNING "OFX support headers not found")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Plugin source
set(PLUGIN_SOURCES
    src/ManoStretch.cpp
)

# OFX Support library sources
set(OFX_SUPPORT_SOURCES
    "${OFX_SUPPORT_LIB}/ofxsCore.cpp"
    "${OFX_SUPPORT_LIB}/ofxsImageEffect.cpp"
    "${OFX_SUPPORT_LIB}/ofxsParams.cpp"
    "${OFX_SUPPORT_LIB}/ofxsProperty.cpp"
    "${OFX_SUPPORT_LIB}/ofxsPropertyValidation.cpp"
    "${OFX_SUPPORT_LIB}/ofxsInteract.cpp"
    "${OFX_SUPPORT_LIB}/ofxsMultiThread.cpp"
    "${OFX_SUPPORT_LIB}/ofxsLog.cpp"
)

add_library(ManoStretch MODULE ${PLUGIN_SOURCES} ${OFX_SUPPORT_SOURCES})

# CUDA configuration — compile .cu with nvcc directly, link .obj
if(HAS_CUDA)
    set(CUDA_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/CudaKernel.cu")
    set(CUDA_KERNEL_OBJ "${CMAKE_BINARY_DIR}/CudaKernel.obj")

    # Compile the .cu → .obj using nvcc for multiple GPU architectures
    add_custom_command(
        OUTPUT "${CUDA_KERNEL_OBJ}"
        COMMAND "${NVCC_EXECUTABLE}"
            -c "${CUDA_KERNEL_SRC}"
            -o "${CUDA_KERNEL_OBJ}"
            -gencode arch=compute_50,code=sm_50
            -gencode arch=compute_60,code=sm_60
            -gencode arch=compute_70,code=sm_70
            -gencode arch=compute_75,code=sm_75
            -gencode arch=compute_80,code=sm_80
            -gencode arch=compute_86,code=sm_86
            -gencode arch=compute_89,code=sm_89
            --machine 64
            -Xcompiler "/MD"
        DEPENDS "${CUDA_KERNEL_SRC}"
        COMMENT "Compiling CUDA kernel with nvcc"
    )
    add_custom_target(CudaKernelObj DEPENDS "${CUDA_KERNEL_OBJ}")

    add_dependencies(ManoStretch CudaKernelObj)
    target_compile_definitions(ManoStretch PRIVATE HAS_CUDA)
    target_include_directories(ManoStretch PRIVATE "${CUDA_TOOLKIT_ROOT}/include")
    target_link_libraries(ManoStretch PRIVATE
        "${CUDA_KERNEL_OBJ}"
        "${CUDA_TOOLKIT_ROOT}/lib/x64/cudart_static.lib"
    )
endif()

target_compile_definitions(ManoStretch PRIVATE
    OFX_USE_EXCEPTIONS=1
    WIN32
    _WINDOWS
    _USRDLL
)

target_link_libraries(ManoStretch PRIVATE
    kernel32.lib
    user32.lib
    gdi32.lib
    comdlg32.lib
    advapi32.lib
    ole32.lib
    oleaut32.lib
    opengl32.lib
)

set_target_properties(ManoStretch PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    OUTPUT_NAME "ManoStretch"
    MACOSX_BUNDLE_DIRECTORY ""
)

# OFX standard: use Win64 on Windows, MacOS on Mac, Linux-x86-64 on Linux
if(WIN32)
    set(OFX_ARCH "Win64")
elseif(APPLE)
    set(OFX_ARCH "MacOS")
else()
    set(OFX_ARCH "Linux-x86-64")
endif()

install(TARGETS ManoStretch
    RUNTIME DESTINATION "${DEFAULT_PLUGIN_PATH}/ManoStretch.ofx.bundle/Contents/${OFX_ARCH}"
    LIBRARY DESTINATION "${DEFAULT_PLUGIN_PATH}/ManoStretch.ofx.bundle/Contents/${OFX_ARCH}"
)

# Also install to user folder (no admin needed)
install(TARGETS ManoStretch
    RUNTIME DESTINATION "${CMAKE_CURRENT_USER_DOCS}/OFX Plugins/ManoStretch.ofx.bundle/Contents/${OFX_ARCH}"
    LIBRARY DESTINATION "${CMAKE_CURRENT_USER_DOCS}/OFX Plugins/ManoStretch.ofx.bundle/Contents/${OFX_ARCH}"
)

# ================================================================
#  Self-contained Installer (Windows only)
#  Embeds the built .ofx plugin as a Win32 resource so the
#  installer EXE is a single file — no loose .ofx needed.
# ================================================================
if(WIN32)
    # After the plugin is built, copy the .ofx to a staging dir
    # so the resource compiler can find it at a deterministic path.
    set(INSTALLER_BUNDLE_DIR "${CMAKE_BINARY_DIR}/installer_bundle")
    add_custom_command(TARGET ManoStretch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${INSTALLER_BUNDLE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:ManoStretch>"
                                         "${INSTALLER_BUNDLE_DIR}/ManoStretch.ofx"
        COMMENT "Staging plugin binary for installer embedding"
    )

    # Generate the resource script that embeds the .ofx binary.
    # The RC compiler resolves paths relative to the .rc file location,
    # which is CMAKE_BINARY_DIR, so "installer_bundle/ManoStretch.ofx" works.
    set(INSTALLER_RC "${CMAKE_BINARY_DIR}/Installer_generated.rc")
    file(WRITE "${INSTALLER_RC}"
        "#include \"${CMAKE_CURRENT_SOURCE_DIR}/src/resource.h\"\n"
        "IDR_OFX_PLUGIN RCDATA \"installer_bundle\\\\ManoStretch.ofx\"\n"
    )

    add_executable(ManoStretch_Installer WIN32
        src/Installer.cpp
        src/resource.h
        ${INSTALLER_RC}
    )

    # Ensure the plugin is fully built (including the POST_BUILD copy)
    # before the installer target starts compiling its resource script.
    add_dependencies(ManoStretch_Installer ManoStretch)

    target_compile_definitions(ManoStretch_Installer PRIVATE WIN32 _WINDOWS)

    target_link_libraries(ManoStretch_Installer PRIVATE
        shlwapi.lib
        version.lib
        shell32.lib
        advapi32.lib
    )

    set_target_properties(ManoStretch_Installer PROPERTIES
        OUTPUT_NAME "ManoStretch_Installer"
    )

    install(TARGETS ManoStretch_Installer
        RUNTIME DESTINATION "${DEFAULT_PLUGIN_PATH}"
    )
endif()