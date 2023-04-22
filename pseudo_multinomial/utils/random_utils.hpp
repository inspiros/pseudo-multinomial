#pragma once

#include <bitset>
#include <cstdint> //`uint32_t`
#include <functional> //`std::hash`
#include <random> //`std::mt19937`
#include <time.h>

static std::random_device RD;
static std::mt19937 RNG(RD());
static std::bernoulli_distribution RAND_BITS_DIST = std::bernoulli_distribution(0.5);

void seed_mt19937(std::mt19937 &rng, uint32_t seed);
void seed_mt19937(std::mt19937 &rng);

unsigned long get_10_rand_bits();
