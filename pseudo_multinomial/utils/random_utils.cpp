#include "random_utils.h"

void seed_mt19937(std::mt19937 &rng, uint32_t seed) {
    rng.seed(static_cast<std::mt19937::result_type>(seed));
}

void seed_mt19937(std::mt19937 &rng) {
    uint32_t t = static_cast<uint32_t>(time(nullptr));
    std::hash<uint32_t> hasher; size_t hashed=hasher(t);
    rng.seed(static_cast<std::mt19937::result_type>(static_cast<uint32_t>(hashed)));
}

unsigned long get_10_rand_bits() {
    std::bitset<10> rand_bits;
    for (size_t i = 0; i < 10; i++) {
        rand_bits.set(i, RAND_BITS_DIST(RNG));
    }
    return rand_bits.to_ulong();
}
