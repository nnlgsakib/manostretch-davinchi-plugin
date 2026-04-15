// ManoStretch Installer - Self-contained Windows Plugin Installer
// Multi-strategy DaVinci Resolve detection + embedded plugin binary

#include <windows.h>
#include <shlobj.h>
#include <shlwapi.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <winreg.h>
#include <tlhelp32.h>

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "version.lib")
#pragma comment(lib, "shell32.lib")

#define APP_NAME    "ManoStretch"
#define APP_VERSION "4.0.0"
#define APP_VENDOR  "Mano"
#define BUNDLE_ID   "com.mano.stretch"

#ifndef IDR_OFX_PLUGIN
#define IDR_OFX_PLUGIN 101
#endif

#ifndef CSIDL_COMMON_APPDATA
#define CSIDL_COMMON_APPDATA 0x0023
#endif

// ---- Data Structures ----

struct ResolveInfo {
    std::wstring installPath;
    std::wstring version;
    std::wstring edition;   // "Free" or "Studio"
};

struct OFXPluginDir {
    std::wstring path;
    std::wstring label;
    bool requiresAdmin;
};

// ---- Path Utilities ----

static bool PathExists(const std::wstring& p) {
    return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES;
}

static std::wstring WStrToLower(const std::wstring& s) {
    std::wstring r = s;
    for (auto& c : r) c = towlower(c);
    return r;
}

static bool IsDuplicate(const std::vector<ResolveInfo>& results, const std::wstring& path) {
    std::wstring lp = WStrToLower(path);
    for (const auto& r : results)
        if (WStrToLower(r.installPath) == lp) return true;
    return false;
}

static BOOL CreateDirectoryRecursive(LPCWSTR path) {
    WCHAR temp[MAX_PATH];
    wcscpy_s(temp, path);
    for (WCHAR* p = temp + 1; *p; p++) {
        if (*p == L'\\') {
            *p = L'\0';
            CreateDirectoryW(temp, NULL);
            *p = L'\\';
        }
    }
    return CreateDirectoryW(temp, NULL);
}

// ================================================================
//  DaVinci Resolve Detection — 5 independent strategies
// ================================================================

// Strategy 1: Enumerate Windows Uninstall registry keys
static void FindFromUninstallRegistry(std::vector<ResolveInfo>& results) {
    const wchar_t* roots[] = {
        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall",
        L"SOFTWARE\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall"
    };
    HKEY hives[] = { HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER };

    for (HKEY hive : hives) {
        for (const wchar_t* root : roots) {
            HKEY hKey;
            if (RegOpenKeyExW(hive, root, 0,
                    KEY_READ | KEY_ENUMERATE_SUB_KEYS, &hKey) != ERROR_SUCCESS)
                continue;

            DWORD subKeyCount = 0;
            RegQueryInfoKeyW(hKey, NULL, NULL, NULL, &subKeyCount,
                             NULL, NULL, NULL, NULL, NULL, NULL, NULL);

            for (DWORD i = 0; i < subKeyCount; i++) {
                wchar_t subKeyName[256];
                DWORD nameLen = 256;
                if (RegEnumKeyExW(hKey, i, subKeyName, &nameLen,
                                  NULL, NULL, NULL, NULL) != ERROR_SUCCESS)
                    continue;

                HKEY hSubKey;
                if (RegOpenKeyExW(hKey, subKeyName, 0, KEY_READ, &hSubKey) != ERROR_SUCCESS)
                    continue;

                wchar_t displayName[512] = {};
                DWORD displayNameLen = sizeof(displayName);
                DWORD type;

                if (RegQueryValueExW(hSubKey, L"DisplayName", NULL, &type,
                        (BYTE*)displayName, &displayNameLen) == ERROR_SUCCESS) {

                    std::wstring name(displayName);
                    std::wstring nameLow = WStrToLower(name);

                    if (nameLow.find(L"davinci resolve") != std::wstring::npos) {
                        wchar_t installLoc[MAX_PATH] = {};
                        DWORD locLen = sizeof(installLoc);

                        if (RegQueryValueExW(hSubKey, L"InstallLocation", NULL, &type,
                                (BYTE*)installLoc, &locLen) == ERROR_SUCCESS && installLoc[0]) {

                            std::wstring path(installLoc);
                            while (!path.empty() && path.back() == L'\\') path.pop_back();

                            if (!IsDuplicate(results, path) && PathExists(path)) {
                                ResolveInfo info;
                                info.installPath = path;

                                wchar_t ver[64] = {};
                                DWORD verLen = sizeof(ver);
                                if (RegQueryValueExW(hSubKey, L"DisplayVersion", NULL,
                                        &type, (BYTE*)ver, &verLen) == ERROR_SUCCESS)
                                    info.version = ver;

                                info.edition = (name.find(L"Studio") != std::wstring::npos)
                                    ? L"Studio" : L"Free";
                                results.push_back(info);
                            }
                        }
                    }
                }
                RegCloseKey(hSubKey);
            }
            RegCloseKey(hKey);
        }
    }
}

// Strategy 2: Blackmagic Design specific registry keys
static void FindFromBMDRegistry(std::vector<ResolveInfo>& results) {
    struct RegEntry { const wchar_t* key; const wchar_t* edition; };
    RegEntry entries[] = {
        { L"SOFTWARE\\Blackmagic Design\\DaVinci Resolve",                       L"Free"   },
        { L"SOFTWARE\\Blackmagic Design\\DaVinci Resolve Studio",                L"Studio" },
        { L"SOFTWARE\\WOW6432Node\\Blackmagic Design\\DaVinci Resolve",          L"Free"   },
        { L"SOFTWARE\\WOW6432Node\\Blackmagic Design\\DaVinci Resolve Studio",   L"Studio" },
    };

    for (const auto& entry : entries) {
        HKEY hKey;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, entry.key, 0, KEY_READ, &hKey) != ERROR_SUCCESS)
            continue;

        const wchar_t* valueNames[] = {
            L"InstallPath", L"InstallDir", L"InstallLocation", L"Path"
        };
        for (const wchar_t* valName : valueNames) {
            wchar_t buf[MAX_PATH] = {};
            DWORD bufLen = sizeof(buf);
            DWORD type;
            if (RegQueryValueExW(hKey, valName, NULL, &type,
                    (BYTE*)buf, &bufLen) == ERROR_SUCCESS && buf[0]) {

                std::wstring path(buf);
                while (!path.empty() && path.back() == L'\\') path.pop_back();

                if (!IsDuplicate(results, path) && PathExists(path)) {
                    ResolveInfo info;
                    info.installPath = path;
                    info.edition = entry.edition;
                    results.push_back(info);
                }
                break;
            }
        }
        RegCloseKey(hKey);
    }
}

// Strategy 3: Scan every fixed drive for known directory patterns
static void FindFromFileSystem(std::vector<ResolveInfo>& results) {
    DWORD drives = GetLogicalDrives();
    for (int d = 0; d < 26; d++) {
        if (!(drives & (1 << d))) continue;
        wchar_t drive[4] = { (wchar_t)('A' + d), L':', L'\\', 0 };
        if (GetDriveTypeW(drive) != DRIVE_FIXED) continue;

        const wchar_t* subdirs[] = {
            L"Program Files\\Blackmagic Design\\DaVinci Resolve",
            L"Program Files\\Blackmagic Design\\DaVinci Resolve Studio",
            L"Program Files (x86)\\Blackmagic Design\\DaVinci Resolve",
            L"Program Files (x86)\\Blackmagic Design\\DaVinci Resolve Studio",
            L"Blackmagic Design\\DaVinci Resolve",
            L"Blackmagic Design\\DaVinci Resolve Studio",
        };

        for (const wchar_t* subdir : subdirs) {
            std::wstring path = std::wstring(drive) + subdir;
            if (!IsDuplicate(results, path) && PathExists(path + L"\\Resolve.exe")) {
                ResolveInfo info;
                info.installPath = path;
                info.edition = (wcsstr(subdir, L"Studio")) ? L"Studio" : L"Free";
                results.push_back(info);
            }
        }
    }
}

// Strategy 4: Detect from a currently-running Resolve.exe process
static void FindFromRunningProcess(std::vector<ResolveInfo>& results) {
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnap == INVALID_HANDLE_VALUE) return;

    PROCESSENTRY32W pe = {};
    pe.dwSize = sizeof(pe);

    if (Process32FirstW(hSnap, &pe)) {
        do {
            if (_wcsicmp(pe.szExeFile, L"Resolve.exe") == 0) {
                HANDLE hProc = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                           FALSE, pe.th32ProcessID);
                if (hProc) {
                    wchar_t fullPath[MAX_PATH];
                    DWORD pathLen = MAX_PATH;
                    if (QueryFullProcessImageNameW(hProc, 0, fullPath, &pathLen)) {
                        PathRemoveFileSpecW(fullPath);
                        std::wstring path(fullPath);
                        if (!IsDuplicate(results, path)) {
                            ResolveInfo info;
                            info.installPath = path;
                            results.push_back(info);
                        }
                    }
                    CloseHandle(hProc);
                }
            }
        } while (Process32NextW(hSnap, &pe));
    }
    CloseHandle(hSnap);
}

// Strategy 5: Follow environment variables set by Resolve
static void FindFromEnvironment(std::vector<ResolveInfo>& results) {
    const wchar_t* envVars[] = { L"RESOLVE_SCRIPT_LIB", L"RESOLVE_SCRIPT_API" };

    for (const wchar_t* envVar : envVars) {
        wchar_t buf[MAX_PATH];
        if (GetEnvironmentVariableW(envVar, buf, MAX_PATH) == 0) continue;

        std::wstring path(buf);
        for (int i = 0; i < 8 && !path.empty(); i++) {
            if (PathExists(path + L"\\Resolve.exe") && !IsDuplicate(results, path)) {
                ResolveInfo info;
                info.installPath = path;
                results.push_back(info);
                break;
            }
            size_t pos = path.find_last_of(L'\\');
            if (pos == std::wstring::npos) break;
            path = path.substr(0, pos);
        }
    }
}

// Fill in version / edition for any entries still missing them
static void PopulateVersionInfo(std::vector<ResolveInfo>& results) {
    for (auto& info : results) {
        if (info.version.empty()) {
            std::wstring exePath = info.installPath + L"\\Resolve.exe";
            DWORD verSize = GetFileVersionInfoSizeW(exePath.c_str(), NULL);
            if (verSize > 0) {
                std::vector<BYTE> verData(verSize);
                if (GetFileVersionInfoW(exePath.c_str(), 0, verSize, verData.data())) {
                    VS_FIXEDFILEINFO* vffi = NULL;
                    UINT viSize = 0;
                    if (VerQueryValueW(verData.data(), L"\\",
                            (LPVOID*)&vffi, &viSize) && vffi) {
                        wchar_t verStr[48];
                        swprintf_s(verStr, L"%d.%d.%d",
                            HIWORD(vffi->dwFileVersionMS),
                            LOWORD(vffi->dwFileVersionMS),
                            HIWORD(vffi->dwFileVersionLS));
                        info.version = verStr;
                    }
                }
            }
            if (info.version.empty()) info.version = L"unknown";
        }
        if (info.edition.empty()) {
            info.edition = (info.installPath.find(L"Studio") != std::wstring::npos)
                ? L"Studio" : L"Free";
        }
    }
}

// Master: run every strategy, deduplicate, fill blanks
static std::vector<ResolveInfo> FindAllDaVinciResolve() {
    std::vector<ResolveInfo> results;
    FindFromUninstallRegistry(results);
    FindFromBMDRegistry(results);
    FindFromRunningProcess(results);
    FindFromEnvironment(results);
    FindFromFileSystem(results);
    PopulateVersionInfo(results);
    return results;
}

// ================================================================
//  OFX Plugin Directory Discovery
// ================================================================

static std::vector<OFXPluginDir> FindOFXPluginDirs() {
    std::vector<OFXPluginDir> dirs;

    auto addDir = [&](const std::wstring& p, const std::wstring& lbl, bool admin) {
        for (const auto& d : dirs)
            if (WStrToLower(d.path) == WStrToLower(p)) return;
        dirs.push_back({ p, lbl, admin });
    };

    // System-wide OFX (standard path, works with all OFX hosts)
    addDir(L"C:\\Program Files\\Common Files\\OFX\\Plugins",
           L"System OFX (all hosts)", true);

    // ProgramData
    wchar_t programData[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_COMMON_APPDATA, NULL, 0, programData)))
        addDir(std::wstring(programData) + L"\\OFX\\Plugins",
               L"Shared OFX (ProgramData)", true);

    // User Documents (no admin needed)
    wchar_t userDocs[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_PERSONAL, NULL, 0, userDocs)))
        addDir(std::wstring(userDocs) + L"\\OFX Plugins",
               L"User OFX (Documents, no admin)", false);

    return dirs;
}

// ================================================================
//  Embedded Resource Extraction
// ================================================================

static bool HasEmbeddedPlugin() {
    return FindResourceW(NULL, MAKEINTRESOURCEW(IDR_OFX_PLUGIN), MAKEINTRESOURCEW(10)) != NULL;
}

static bool ExtractEmbeddedPlugin(const std::wstring& destPath) {
    HRSRC hRes = FindResourceW(NULL, MAKEINTRESOURCEW(IDR_OFX_PLUGIN), MAKEINTRESOURCEW(10));
    if (!hRes) return false;

    HGLOBAL hMem = LoadResource(NULL, hRes);
    if (!hMem) return false;

    void* data = LockResource(hMem);
    DWORD size = SizeofResource(NULL, hRes);
    if (!data || size == 0) return false;

    HANDLE hFile = CreateFileW(destPath.c_str(), GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return false;

    DWORD written;
    BOOL ok = WriteFile(hFile, data, size, &written, NULL);
    CloseHandle(hFile);
    return ok && (written == size);
}

// Fallback: find .ofx next to installer exe
static std::wstring FindExternalPlugin() {
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    PathRemoveFileSpecW(exePath);

    std::wstring dir(exePath);
    WIN32_FIND_DATAW fd;
    HANDLE hFind = FindFirstFileW((dir + L"\\*.ofx").c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        std::wstring found = dir + L"\\" + fd.cFileName;
        FindClose(hFind);
        return found;
    }
    return L"";
}

// ================================================================
//  Install / Uninstall
// ================================================================

static bool InstallToFolder(const std::wstring& pluginDir) {
    std::wstring bundlePath   = pluginDir + L"\\" APP_NAME L".ofx.bundle";
    std::wstring contentsPath = bundlePath + L"\\Contents";
    std::wstring winPath      = contentsPath + L"\\Win64";
    std::wstring resPath      = contentsPath + L"\\Resources";

    CreateDirectoryRecursive(winPath.c_str());
    CreateDirectoryRecursive(resPath.c_str());

    // Extract from embedded resource, or copy external .ofx
    std::wstring dest = winPath + L"\\" APP_NAME L".ofx";
    bool ok = false;
    if (HasEmbeddedPlugin()) {
        ok = ExtractEmbeddedPlugin(dest);
    } else {
        std::wstring ext = FindExternalPlugin();
        if (!ext.empty())
            ok = (CopyFileW(ext.c_str(), dest.c_str(), FALSE) != 0);
    }
    if (!ok) return false;

    // Write Info.plist
    std::wstring infoPlist = resPath + L"\\Info.plist";
    HANDLE hFile = CreateFileW(infoPlist.c_str(), GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        const char* info =
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\""
            " \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
            "<plist version=\"1.0\">\n"
            "<dict>\n"
            "    <key>CFBundleIdentifier</key>\n"
            "    <string>" BUNDLE_ID "</string>\n"
            "    <key>CFBundleName</key>\n"
            "    <string>" APP_NAME "</string>\n"
            "    <key>CFBundleVersion</key>\n"
            "    <string>" APP_VERSION "</string>\n"
            "    <key>CFBundlePackageType</key>\n"
            "    <string>OFX</string>\n"
            "</dict>\n"
            "</plist>\n";
        DWORD written;
        WriteFile(hFile, info, (DWORD)strlen(info), &written, NULL);
        CloseHandle(hFile);
    }

    // Uninstall batch
    std::wstring uninstBat = pluginDir + L"\\Uninstall_" APP_NAME L".bat";
    hFile = CreateFileW(uninstBat.c_str(), GENERIC_WRITE, 0, NULL,
                        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        const char* batch =
            "@echo off\n"
            "echo Uninstalling " APP_NAME "...\n"
            "cd /d \"%~dp0\"\n"
            "rd /s /q \"" APP_NAME ".ofx.bundle\"\n"
            "del \"Uninstall_" APP_NAME ".bat\"\n"
            "echo Done!\n"
            "pause\n";
        DWORD written;
        WriteFile(hFile, batch, (DWORD)strlen(batch), &written, NULL);
        CloseHandle(hFile);
    }

    // Add/Remove Programs registry entry
    HKEY hKey;
    std::wstring regPath = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\" BUNDLE_ID;
    if (RegCreateKeyW(HKEY_CURRENT_USER, regPath.c_str(), &hKey) == ERROR_SUCCESS) {
        std::wstring wName    = L"" APP_NAME;
        std::wstring wVer     = L"" APP_VERSION;
        std::wstring wVendor  = L"" APP_VENDOR;
        RegSetValueExW(hKey, L"DisplayName",     0, REG_SZ, (const BYTE*)wName.c_str(),     (DWORD)((wName.length()    + 1) * 2));
        RegSetValueExW(hKey, L"DisplayVersion",  0, REG_SZ, (const BYTE*)wVer.c_str(),      (DWORD)((wVer.length()     + 1) * 2));
        RegSetValueExW(hKey, L"Publisher",       0, REG_SZ, (const BYTE*)wVendor.c_str(),    (DWORD)((wVendor.length()  + 1) * 2));
        RegSetValueExW(hKey, L"UninstallString", 0, REG_SZ, (const BYTE*)uninstBat.c_str(),  (DWORD)((uninstBat.length()+ 1) * 2));
        RegCloseKey(hKey);
    }

    return true;
}

static bool UninstallFromFolder(const std::wstring& folder) {
    std::wstring bundlePath = folder + L"\\" APP_NAME L".ofx.bundle";

    // Remove registry
    std::wstring regPath = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\" BUNDLE_ID;
    RegDeleteKeyW(HKEY_CURRENT_USER, regPath.c_str());

    // Delete bundle directory
    SHFILEOPSTRUCTW fs = {};
    fs.wFunc = FO_DELETE;
    wchar_t from[MAX_PATH];
    wcscpy_s(from, bundlePath.c_str());
    from[wcslen(from) + 1] = 0;
    fs.pFrom = from;
    fs.fFlags = FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT;
    SHFileOperationW(&fs);

    DeleteFileW((folder + L"\\Uninstall_" APP_NAME L".bat").c_str());
    return true;
}

// ================================================================
//  UI Helpers
// ================================================================

static std::wstring BuildDetectionReport(
        const std::vector<ResolveInfo>& resolves,
        const std::vector<OFXPluginDir>& dirs) {
    std::wstring msg;

    msg += L"== DaVinci Resolve Detection ==\n\n";
    if (resolves.empty()) {
        msg += L"  (none found — plugin installs to standard OFX dirs)\n";
    } else {
        for (size_t i = 0; i < resolves.size(); i++) {
            msg += L"  ";
            msg += std::to_wstring(i + 1);
            msg += L". ";
            msg += resolves[i].edition;
            msg += L"  v";
            msg += resolves[i].version;
            msg += L"\n     ";
            msg += resolves[i].installPath;
            msg += L"\n";
        }
    }

    msg += L"\n== Install Targets ==\n\n";
    for (size_t i = 0; i < dirs.size(); i++) {
        bool exists = PathExists(dirs[i].path);
        msg += L"  ";
        msg += std::to_wstring(i + 1);
        msg += L". ";
        msg += dirs[i].label;
        if (dirs[i].requiresAdmin) msg += L" [admin]";
        msg += L"\n     ";
        msg += dirs[i].path;
        msg += exists ? L" (exists)\n" : L" (will create)\n";
    }

    msg += L"\nPlugin: ";
    msg += HasEmbeddedPlugin() ? L"bundled in installer" : L"external .ofx file";
    msg += L"\n\nProceed with installation?";

    return msg;
}

// ================================================================
//  Entry Point
// ================================================================

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
    std::wstring cmd = GetCommandLineW();
    bool doUninstall = (cmd.find(L"/uninstall") != std::wstring::npos ||
                        cmd.find(L"-u") != std::wstring::npos);
    bool silent      = (cmd.find(L"/silent") != std::wstring::npos ||
                        cmd.find(L"-s") != std::wstring::npos);

    // Detect Resolve installations & OFX directories
    std::vector<ResolveInfo> resolves = FindAllDaVinciResolve();
    std::vector<OFXPluginDir> pluginDirs = FindOFXPluginDirs();

    // ---- Uninstall flow ----
    if (doUninstall) {
        if (!silent) {
            if (MessageBoxW(NULL,
                    L"Uninstall " APP_NAME L" plugin from all locations?",
                    L"Uninstall " APP_NAME,
                    MB_YESNO | MB_ICONQUESTION) != IDYES)
                return 0;
        }
        for (const auto& dir : pluginDirs)
            UninstallFromFolder(dir.path);

        if (!silent)
            MessageBoxW(NULL, APP_NAME L" has been uninstalled.",
                        L"Uninstall Complete", MB_OK | MB_ICONINFORMATION);
        return 0;
    }

    // ---- Install flow ----

    // Verify we have the plugin binary (embedded or external)
    if (!HasEmbeddedPlugin() && FindExternalPlugin().empty()) {
        if (!silent)
            MessageBoxW(NULL,
                L"Cannot find the " APP_NAME L".ofx plugin!\n\n"
                L"The plugin is not embedded in this installer and no .ofx\n"
                L"file was found next to it.\n\n"
                L"Use the release build or place the .ofx beside the installer.",
                L"Error", MB_OK | MB_ICONERROR);
        return 1;
    }

    // Show detection report & ask to proceed
    if (!silent) {
        std::wstring report = BuildDetectionReport(resolves, pluginDirs);
        if (MessageBoxW(NULL, report.c_str(),
                L"Install " APP_NAME L" v" APP_VERSION,
                MB_YESNO | MB_ICONINFORMATION) != IDYES)
            return 0;
    }

    // Install to every discovered OFX directory
    int successCount = 0;
    int failCount = 0;
    std::wstring failMsg;

    for (const auto& dir : pluginDirs) {
        if (InstallToFolder(dir.path)) {
            successCount++;
        } else {
            failCount++;
            failMsg += L"  - " + dir.label + L"\n";
        }
    }

    // Result dialog
    if (!silent) {
        if (successCount > 0 && failCount == 0) {
            std::wstring msg =
                L"Installation complete!\n\n"
                L"Installed to " + std::to_wstring(successCount) + L" location(s).\n\n"
                L"Restart DaVinci Resolve to use the plugin.\n\n"
                L"To uninstall: run this installer with /uninstall";
            MessageBoxW(NULL, msg.c_str(), L"Success", MB_OK | MB_ICONINFORMATION);
        } else if (successCount > 0) {
            std::wstring msg =
                L"Partially installed (" + std::to_wstring(successCount) +
                L" OK, " + std::to_wstring(failCount) + L" failed).\n\nFailed:\n" +
                failMsg +
                L"\nRun as Administrator for system-wide paths.";
            MessageBoxW(NULL, msg.c_str(), L"Warning", MB_OK | MB_ICONWARNING);
        } else {
            std::wstring msg =
                L"Installation failed!\n\nFailed:\n" + failMsg +
                L"\nTry running as Administrator.";
            MessageBoxW(NULL, msg.c_str(), L"Error", MB_OK | MB_ICONERROR);
            return 1;
        }
    }

    return 0;
}