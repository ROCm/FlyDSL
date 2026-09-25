#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

struct Dim3 { unsigned x, y, z; };
static thread_local size_t armed_size;
static thread_local std::vector<unsigned char> captured;
static thread_local std::string kernel_name;
static thread_local bool skip_launch;

extern "C" void capture_arm(size_t size) { armed_size = size; captured.clear(); }
extern "C" void capture_arm_skip(size_t size) { capture_arm(size); skip_launch = true; }
extern "C" size_t capture_copy(void* out, size_t capacity) {
    if (capacity >= captured.size()) std::memcpy(out, captured.data(), captured.size());
    return captured.size();
}
extern "C" const char* capture_name() { return kernel_name.c_str(); }
extern "C" int hipLaunchKernel(const void* fn, Dim3 grid, Dim3 block, void** args,
                               size_t shared, void* stream) {
    using Launch = int (*)(const void*, Dim3, Dim3, void**, size_t, void*);
    using Name = const char* (*)(const void*, void*);
    static auto original = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "hipLaunchKernel"));
    static auto get_name = reinterpret_cast<Name>(dlsym(RTLD_NEXT, "hipKernelNameRefByPtr"));
    if (armed_size && grid.x == 256 && block.x == 512) {
        const char* name = get_name ? get_name(fn, stream) : nullptr;
        if (name && std::strstr(name, "pure_mla_moe_layer_kernel")) {
            captured.assign(static_cast<unsigned char*>(args[0]), static_cast<unsigned char*>(args[0]) + armed_size);
            kernel_name = name;
            armed_size = 0;
            if (skip_launch) { skip_launch = false; return 0; }
        }
    }
    return original(fn, grid, block, args, shared, stream);
}
