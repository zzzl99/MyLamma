#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
    CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
    }

    void* CPUDeviceAllocator::allocate(size_t byte_size) const {
        if (!byte_size) {
            return nullptr;
        }
        void* data = malloc(byte_size);
        return data;
    }

    void CPUDeviceAllocator::release(void* ptr) const {
        if (ptr) {
            free(ptr);
        }
    }
    std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base