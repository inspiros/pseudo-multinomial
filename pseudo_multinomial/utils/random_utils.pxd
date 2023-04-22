cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() nogil
        mt19937(unsigned int seed) nogil

    cdef cppclass pcg64:
        pcg64() nogil
        pcg64(unsigned int seed) nogil

    cdef cppclass uniform_real_distribution[double]:
        uniform_real_distribution() nogil
        uniform_real_distribution(double a, double b) nogil
        double operator()(pcg64 gen) nogil
        double operator()(mt19937 gen) nogil

cdef extern from *:
    """
    #pragma once
    #include <bitset>
    #include <cstdint>
    #include <functional>
    #include <random>
    #include <time.h>

    void seed_mt19937(std::mt19937 &rng, uint32_t seed) {
        rng.seed(static_cast<std::mt19937::result_type>(seed));
    }

    void seed_mt19937(std::mt19937 &rng) {
        uint32_t t = static_cast<uint32_t>(time(nullptr));
        std::hash<uint32_t> hasher; size_t hashed=hasher(t);
        rng.seed(static_cast<std::mt19937::result_type>(static_cast<uint32_t>(hashed)));
    }

    static std::random_device RD;
    static std::mt19937 RNG(RD());
    static std::bernoulli_distribution RAND_BITS_DIST = std::bernoulli_distribution(0.5);
    unsigned long get_10_rand_bits() {
        std::bitset<10> rand_bits;
        for (size_t i = 0; i < 10; i++) {
            rand_bits.set(i, RAND_BITS_DIST(RNG));
        }
        return rand_bits.to_ulong();
    }
    """
    cdef void seed_mt19937(mt19937 gen) nogil
    cdef void seed_mt19937(mt19937 gen, int seed) nogil
    cdef unsigned long get_10_rand_bits() nogil
