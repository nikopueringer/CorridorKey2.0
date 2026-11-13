// CorridorKey community Linux bridge. See ../LICENSE for the project licence.
#include <ofxCore.h>
#include <ofxImageEffect.h>
#include <ofxParam.h>
#include <ofxProperty.h>
#include <ofxMessage.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
OfxHost* host = nullptr;
OfxPropertySuiteV1* props = nullptr;
OfxImageEffectSuiteV1* effects = nullptr;
OfxParameterSuiteV1* params = nullptr;
OfxMessageSuiteV2* messages = nullptr;
constexpr size_t maxPixels = 4096 * 4096;

void check(OfxStatus s) { if (s != kOfxStatOK) throw std::runtime_error("Host rejected an OpenFX operation (status " + std::to_string(s) + ")"); }
void str(OfxPropertySetHandle p, const char* key, const char* value, int i = 0) {
    const auto status = props->propSetString(p, key, i, value);
    if (status != kOfxStatOK) throw std::runtime_error(std::string(key) + ": status " + std::to_string(status));
}
void integer(OfxPropertySetHandle p, const char* key, int value, int i = 0) {
    const auto status = props->propSetInt(p, key, i, value);
    if (status != kOfxStatOK) throw std::runtime_error(std::string(key) + ": status " + std::to_string(status));
}
std::string getstr(OfxPropertySetHandle p, const char* key) {
    char* value = nullptr;
    check(props->propGetString(p, key, 0, &value));
    return value ? value : "";
}
std::string defaultSocket() { return "/tmp/corridorkey-" + std::to_string(getuid()) + "/engine.sock"; }
struct Cancelled {};
struct Instance {
    OfxImageEffectHandle effect;
    OfxImageClipHandle source{}, output{};
    OfxParamHandle colour{}, despill{}, mode{}, socket{};
    std::mutex mutex;
    std::vector<float> previous, result;
    std::string previousSocket;
    int width = 0, height = 0, previousColour = -1;
    float previousDespill = -1;
};
Instance* instance(OfxImageEffectHandle h) {
    OfxPropertySetHandle p; check(effects->getPropertySet(h, &p));
    void* data = nullptr; check(props->propGetPointer(p, kOfxPropInstanceData, 0, &data));
    if (!data) throw std::runtime_error("Missing OpenFX instance");
    return static_cast<Instance*>(data);
}

struct Image {
    OfxPropertySetHandle handle{};
    std::array<int, 4> bounds{};
    int rowBytes{};
    char* data{};
    Image(OfxImageClipHandle clip, double time) {
        check(effects->clipGetImage(clip, time, nullptr, &handle));
        try {
            if (getstr(handle, kOfxImageEffectPropPixelDepth) != kOfxBitDepthFloat ||
                getstr(handle, kOfxImageEffectPropComponents) != kOfxImageComponentRGBA)
                throw std::runtime_error("CorridorKey requires float32 RGBA host images");
            check(props->propGetIntN(handle, kOfxImagePropBounds, 4, bounds.data()));
            check(props->propGetInt(handle, kOfxImagePropRowBytes, 0, &rowBytes));
            void* ptr = nullptr; check(props->propGetPointer(handle, kOfxImagePropData, 0, &ptr));
            data = static_cast<char*>(ptr);
            const int64_t width = int64_t(bounds[2]) - bounds[0];
            const int64_t height = int64_t(bounds[3]) - bounds[1];
            if (!data || width < 1 || height < 1 || width > 8192 || height > 8192 ||
                uint64_t(width * height) > maxPixels || std::abs(int64_t(rowBytes)) < width * 16)
                throw std::runtime_error("Invalid host image or frame exceeds 16 megapixels");
        } catch (...) { effects->clipReleaseImage(handle); handle = nullptr; throw; }
    }
    ~Image() { if (handle) effects->clipReleaseImage(handle); }
    Image(const Image&) = delete;
    float* pixel(int x, int y) const {
        if (x < bounds[0] || x >= bounds[2] || y < bounds[1] || y >= bounds[3]) return nullptr;
        return reinterpret_cast<float*>(data + ptrdiff_t(y - bounds[1]) * rowBytes) + ptrdiff_t(x - bounds[0]) * 4;
    }
};

struct Socket {
    int fd = -1;
    ~Socket() { if (fd >= 0) close(fd); }
};
void transfer(int fd, void* buffer, size_t length, bool sending, OfxImageEffectHandle effect) {
    auto* p = static_cast<char*>(buffer);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    while (length) {
        if (effects->abort(effect)) throw Cancelled{};
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("CorridorKey engine timed out after 10 minutes");
        pollfd wait{fd, short(sending ? POLLOUT : POLLIN), 0};
        const int ready = poll(&wait, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) throw std::runtime_error("Cannot poll CorridorKey engine");
        if (!ready) continue;
        const ssize_t n = sending ? send(fd, p, length, MSG_NOSIGNAL | MSG_DONTWAIT) : recv(fd, p, length, MSG_DONTWAIT);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) continue;
        if (n <= 0) throw std::runtime_error("CorridorKey engine disconnected; check the desktop app log");
        p += n; length -= size_t(n);
    }
}
void put32(char* p, uint32_t v) { std::memcpy(p, &v, 4); }
uint32_t u32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
std::vector<float> infer(Instance& inst, const std::string& path, std::vector<float>& input,
                         int width, int height, int colour, float despill) {
    sockaddr_un address{}; address.sun_family = AF_UNIX;
    if (path.empty() || path.size() >= sizeof(address.sun_path)) throw std::runtime_error("Invalid engine socket path");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    Socket sock;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    for (;;) {
        if (effects->abort(inst.effect)) throw Cancelled{};
        sock.fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (sock.fd < 0) throw std::runtime_error("Cannot create engine connection");
        if (connect(sock.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) break;
        const int error = errno;
        close(sock.fd); sock.fd = -1;
        if (error != EAGAIN && error != EINTR && error != EINPROGRESS)
            throw std::runtime_error("Start Resolve engine in the CorridorKey desktop app, then press Refresh in this effect");
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("CorridorKey engine remained busy for 10 minutes");
        poll(nullptr, 0, 100);
    }
    ucred peer{}; socklen_t peerSize = sizeof(peer);
    if (getsockopt(sock.fd, SOL_SOCKET, SO_PEERCRED, &peer, &peerSize) != 0 || peer.uid != getuid())
        throw std::runtime_error("Engine socket belongs to another user");
    std::array<char, 32> request{};
    std::memcpy(request.data(), "CK2REQ01", 8);
    put32(request.data()+8, uint32_t(width)); put32(request.data()+12, uint32_t(height));
    put32(request.data()+16, 4); put32(request.data()+20, uint32_t(colour));
    std::memcpy(request.data()+24, &despill, 4);
    const uint32_t bytes = uint32_t(input.size() * sizeof(float));
    put32(request.data()+28, bytes);
    transfer(sock.fd, request.data(), request.size(), true, inst.effect);
    transfer(sock.fd, input.data(), bytes, true, inst.effect);
    std::array<char, 16> response{};
    transfer(sock.fd, response.data(), response.size(), false, inst.effect);
    if (std::memcmp(response.data(), "CK2RES01", 8)) throw std::runtime_error("Invalid engine response");
    const uint32_t status = u32(response.data()+8), size = u32(response.data()+12);
    if (status) {
        if (size > 16384) throw std::runtime_error("Oversized engine error");
        std::string error(size, '\0'); transfer(sock.fd, error.data(), size, false, inst.effect);
        throw std::runtime_error(error);
    }
    if (size != bytes) throw std::runtime_error("Engine returned the wrong frame size");
    std::vector<float> output(input.size());
    transfer(sock.fd, output.data(), bytes, false, inst.effect);
    for (float v : output) if (!std::isfinite(v)) throw std::runtime_error("Engine returned non-finite pixels");
    return output;
}

OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle in) {
    auto& inst = *instance(effect);
    std::lock_guard<std::mutex> lock(inst.mutex);
    double time = 0, despill = 0.5; int colour = 0, mode = 0; char* socketName = nullptr;
    check(props->propGetDouble(in, kOfxPropTime, 0, &time));
    char* field = nullptr;
    if (props->propGetString(in, kOfxImageEffectPropFieldToRender, 0, &field) == kOfxStatOK && field &&
        std::strcmp(field, kOfxImageFieldNone) && std::strcmp(field, kOfxImageFieldBoth))
        throw std::runtime_error("CorridorKey requires progressive frames; deinterlace upstream");
    check(params->paramGetValueAtTime(inst.colour, time, &colour));
    check(params->paramGetValueAtTime(inst.mode, time, &mode));
    check(params->paramGetValueAtTime(inst.despill, time, &despill));
    check(params->paramGetValueAtTime(inst.socket, time, &socketName));
    std::string path = socketName ? socketName : "";
    if (colour < 0 || colour > 1 || mode < 0 || mode > 2 || !std::isfinite(despill) || despill < 0 || despill > 1)
        throw std::runtime_error("Invalid keying parameters");
    Image source(inst.source, time), output(inst.output, time);
    const int width = source.bounds[2] - source.bounds[0], height = source.bounds[3] - source.bounds[1];
    // Host coordinates are bottom-up. The model's arrays are top-down.
    std::vector<float> pixels(size_t(width) * height * 4);
    const bool premult = getstr(source.handle, kOfxImageEffectPropPreMultiplication) == kOfxImagePreMultiplied;
    for (int y = 0; y < height; ++y) {
        if (effects->abort(effect)) throw Cancelled{};
        const float* src = source.pixel(source.bounds[0], source.bounds[3] - 1 - y);
        float* dst = pixels.data() + size_t(y) * width * 4;
        for (int x = 0; x < width; ++x, src += 4, dst += 4) {
            for (int c = 0; c < 4; ++c) if (!std::isfinite(src[c])) throw std::runtime_error("Input contains non-finite pixels");
            dst[3] = std::clamp(src[3], 0.f, 1.f);
            for (int c = 0; c < 3; ++c) dst[c] = premult ? (dst[3] > 1e-6f ? src[c] / dst[3] : 0.f) : src[c];
        }
    }
    // Exact source comparison avoids stale results on upstream grade changes.
    // One frame only: bounded memory, independent of timeline duration.
    if (pixels != inst.previous || width != inst.width || height != inst.height ||
        colour != inst.previousColour || float(despill) != inst.previousDespill || path != inst.previousSocket) {
        auto result = infer(inst, path, pixels, width, height, colour, float(despill));
        inst.result = std::move(result); inst.previous = std::move(pixels);
        inst.width = width; inst.height = height; inst.previousColour = colour;
        inst.previousDespill = float(despill); inst.previousSocket = path;
    }
    std::array<int, 4> window{};
    check(props->propGetIntN(in, kOfxImageEffectPropRenderWindow, 4, window.data()));
    for (int y = std::max(window[1], output.bounds[1]); y < std::min(window[3], output.bounds[3]); ++y) {
        if (effects->abort(effect)) throw Cancelled{};
        for (int x = std::max(window[0], output.bounds[0]); x < std::min(window[2], output.bounds[2]); ++x) {
            float* dst = output.pixel(x, y);
            if (!source.pixel(x, y)) { std::fill(dst, dst+4, 0.f); continue; }
            const float* src = inst.result.data() + (size_t(source.bounds[3] - 1 - y) * width + (x - source.bounds[0])) * 4;
            const float alpha = std::clamp(src[3], 0.f, 1.f);
            for (int c = 0; c < 3; ++c) dst[c] = mode == 1 ? alpha : src[c] * (mode == 0 ? alpha : 1.f);
            dst[3] = mode == 0 ? alpha : 1.f;
        }
    }
    if (messages) messages->clearPersistentMessage(effect);
    return kOfxStatOK;
}

OfxStatus describe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle p; check(effects->getPropertySet(effect, &p));
    str(p, kOfxPropLabel, "CorridorKey 2.0 Community");
    str(p, kOfxImageEffectPluginPropGrouping, "Keyer");
    // Optional OFX 1.2 metadata: Resolve's bundled wrapper likewise ignores
    // unsupported-description errors. Essential rendering properties stay strict.
    props->propSetString(p, kOfxPropPluginDescription, 0, "Experimental Linux single-frame keyer. Start Resolve engine in the CorridorKey desktop UI. Use the GUI queue for temporal keying and hints.");
    str(p, kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextFilter);
    str(p, kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextGeneral, 1);
    str(p, kOfxImageEffectPropSupportedPixelDepths, kOfxBitDepthFloat);
    str(p, kOfxImageEffectPluginRenderThreadSafety, kOfxImageEffectRenderInstanceSafe);
    integer(p, kOfxImageEffectPluginPropHostFrameThreading, 0);
    integer(p, kOfxImageEffectPropSupportsTiles, 0);
    integer(p, kOfxImageEffectPropSupportsMultiResolution, 0);
    integer(p, kOfxImageEffectPropTemporalClipAccess, 0);
    return kOfxStatOK;
}
OfxStatus describeContext(OfxImageEffectHandle effect) {
    for (const char* name : {"Source", "Output"}) {
        OfxPropertySetHandle p; check(effects->clipDefine(effect, name, &p));
        str(p, kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGBA);
        integer(p, kOfxImageEffectPropSupportsTiles, 0);
    }
    OfxParamSetHandle set; check(effects->getParamSet(effect, &set));
    auto define = [&](const char* type, const char* name, const char* label) {
        OfxPropertySetHandle p; check(params->paramDefine(set, type, name, &p));
        str(p, kOfxPropLabel, label); return p;
    };
    auto p = define(kOfxParamTypeChoice, "output", "View");
    str(p, kOfxParamPropChoiceOption, "Keyed RGBA (premultiplied)");
    str(p, kOfxParamPropChoiceOption, "Alpha matte", 1);
    str(p, kOfxParamPropChoiceOption, "Foreground RGB", 2);
    integer(p, kOfxParamPropDefault, 0);
    p = define(kOfxParamTypeChoice, "colour", "Input colour");
    str(p, kOfxParamPropChoiceOption, "sRGB transfer / Rec.709 primaries");
    str(p, kOfxParamPropChoiceOption, "Linear / Rec.709 primaries", 1);
    str(p, kOfxParamPropHint, "Convert camera log, ACEScg or wide gamut into sRGB/Rec.709 primaries upstream. This effect does no primary-space transform. Output uses the selected input transfer function.");
    integer(p, kOfxParamPropDefault, 0);
    p = define(kOfxParamTypeDouble, "despill", "Despill");
    check(props->propSetDouble(p, kOfxParamPropDefault, 0, 0.5));
    check(props->propSetDouble(p, kOfxParamPropMin, 0, 0.0));
    check(props->propSetDouble(p, kOfxParamPropMax, 0, 1.0));
    check(props->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0));
    check(props->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0));
    p = define(kOfxParamTypeString, "socket", "Engine socket");
    str(p, kOfxParamPropDefault, defaultSocket().c_str());
    integer(p, kOfxParamPropAnimates, 0);
    p = define(kOfxParamTypePushButton, "refresh", "Refresh / retry engine");
    integer(p, kOfxParamPropEvaluateOnChange, 1);
    p = define(kOfxParamTypePage, "controls", "CorridorKey");
    int index = 0;
    for (const char* name : {"output", "colour", "despill", "socket", "refresh"}) str(p, kOfxParamPropPageChild, name, index++);
    return kOfxStatOK;
}
OfxStatus create(OfxImageEffectHandle effect) {
    auto inst = std::make_unique<Instance>(); inst->effect = effect;
    check(effects->clipGetHandle(effect, "Source", &inst->source, nullptr));
    check(effects->clipGetHandle(effect, "Output", &inst->output, nullptr));
    OfxParamSetHandle set; check(effects->getParamSet(effect, &set));
    check(params->paramGetHandle(set, "output", &inst->mode, nullptr));
    check(params->paramGetHandle(set, "colour", &inst->colour, nullptr));
    check(params->paramGetHandle(set, "despill", &inst->despill, nullptr));
    check(params->paramGetHandle(set, "socket", &inst->socket, nullptr));
    OfxPropertySetHandle p; check(effects->getPropertySet(effect, &p));
    check(props->propSetPointer(p, kOfxPropInstanceData, 0, inst.get())); inst.release();
    return kOfxStatOK;
}
OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    auto effect = (OfxImageEffectHandle)handle;
    try {
        if (!std::strcmp(action, kOfxActionLoad)) {
            props = (OfxPropertySuiteV1*)host->fetchSuite(host->host, kOfxPropertySuite, 1);
            effects = (OfxImageEffectSuiteV1*)host->fetchSuite(host->host, kOfxImageEffectSuite, 1);
            params = (OfxParameterSuiteV1*)host->fetchSuite(host->host, kOfxParameterSuite, 1);
            messages = (OfxMessageSuiteV2*)host->fetchSuite(host->host, kOfxMessageSuite, 2);
            return props && effects && params ? kOfxStatOK : kOfxStatErrMissingHostFeature;
        }
        if (!std::strcmp(action, kOfxActionUnload)) return kOfxStatOK;
        if (!std::strcmp(action, kOfxActionDescribe)) return describe(effect);
        if (!std::strcmp(action, kOfxImageEffectActionDescribeInContext)) return describeContext(effect);
        if (!std::strcmp(action, kOfxActionCreateInstance)) return create(effect);
        if (!std::strcmp(action, kOfxActionDestroyInstance)) {
            delete instance(effect);
            OfxPropertySetHandle p; check(effects->getPropertySet(effect, &p));
            check(props->propSetPointer(p, kOfxPropInstanceData, 0, nullptr));
            return kOfxStatOK;
        }
        if (!std::strcmp(action, kOfxImageEffectActionRender)) return render(effect, in);
        if (!std::strcmp(action, kOfxImageEffectActionGetClipPreferences)) {
            str(out, kOfxImageEffectPropPreMultiplication, kOfxImagePreMultiplied); return kOfxStatOK;
        }
        if (!std::strcmp(action, kOfxActionInstanceChanged)) {
            if (getstr(in, kOfxPropName) == "refresh") {
                auto* inst = instance(effect); std::lock_guard<std::mutex> lock(inst->mutex);
                inst->previous.clear(); inst->result.clear();
            }
            return kOfxStatReplyDefault;
        }
        return kOfxStatReplyDefault;
    } catch (const Cancelled&) { return kOfxStatOK; }
      catch (const std::bad_alloc&) { return kOfxStatErrMemory; }
      catch (const std::exception& error) {
        std::fprintf(stderr, "CorridorKey %s: %s\n", action, error.what());
        if (messages && effect && !std::strcmp(action, kOfxImageEffectActionRender))
            messages->setPersistentMessage(effect, kOfxMessageError, "corridorkey-engine", "%s", error.what());
        return kOfxStatFailed;
    } catch (...) { return kOfxStatFailed; }
}
void setHost(OfxHost* h) { host = h; }
OfxPlugin plugin{kOfxImageEffectPluginApi, 1, "org.corridorkey.community.linux", 0, 1, setHost, entry};
} // namespace

extern "C" __attribute__((visibility("default"))) int OfxGetNumberOfPlugins() { return 1; }
extern "C" __attribute__((visibility("default"))) OfxPlugin* OfxGetPlugin(int n) { return n == 0 ? &plugin : nullptr; }
