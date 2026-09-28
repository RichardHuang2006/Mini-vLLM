#include "mini_vllm_ext.h"

#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

void load_library(const char *path) {
    mx::metal::device(mx::Device(mx::Device::gpu)).get_library("mini_vllm_ext", path);
}

}  // namespace mini_vllm_ext
