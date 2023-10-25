#include "cachemem.h"

CacheRef SharedMemCacheCreate (size_t cache_size, std::string name, bool create) {
    auto cache_ptr = std::make_shared<SACacheIndex>(cache_size, name, create);
    return CacheRef(cache_ptr);
}