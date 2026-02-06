#include "blaz.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace {
constexpr std::size_t kBinBits = sizeof(BinType) * 8;
constexpr double kBinRadius = (1ULL << (kBinBits - 1)) - 1.0;
constexpr std::size_t kMaskWordBits = sizeof(MaskWord) * 8;

std::size_t mask_words(std::size_t bit_count) {
    return (bit_count + (kMaskWordBits - 1)) / kMaskWordBits;
}

bool mask_get(const std::vector<MaskWord>& mask, std::size_t bit) {
    const std::size_t word = bit / kMaskWordBits;
    const std::size_t shift = bit % kMaskWordBits;
    return (mask[word] >> shift) & 1ULL;
}

void mask_set(std::vector<MaskWord>& mask, std::size_t bit) {
    const std::size_t word = bit / kMaskWordBits;
    const std::size_t shift = bit % kMaskWordBits;
    mask[word] |= (1ULL << shift);
}

void build_fixed_mask(BlazCompressedComplex& out, std::size_t num_blocks,
    std::size_t block_size, std::size_t keep_count) {
    out.mask.assign(mask_words(num_blocks * block_size), 0);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const std::size_t base = b * block_size;
        for (std::size_t idx = 0; idx < keep_count; ++idx) {
            mask_set(out.mask, base + idx);
        }
    }
}

void build_joint_energy_mask(BlazCompressedComplex& out, std::size_t num_blocks,
    std::size_t block_size, std::size_t keep_count) {
    out.mask.assign(mask_words(num_blocks * block_size), 0);
    if (keep_count >= block_size) {
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const std::size_t base = b * block_size;
            for (std::size_t idx = 0; idx < block_size; ++idx) {
                mask_set(out.mask, base + idx);
            }
        }
        return;
    }

    std::vector<std::size_t> indices(block_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<double> energy(block_size, 0.0);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        std::fill(energy.begin(), energy.end(), 0.0);
        const double scale_r = out.N_r[b] / kBinRadius;
        const double scale_i = out.N_i[b] / kBinRadius;
        for (std::size_t idx = 0; idx < block_size; ++idx) {
            const std::size_t offset = b * block_size + idx;
            const double qr = static_cast<double>(out.F_r[offset]);
            const double qi = static_cast<double>(out.F_i[offset]);
            const double vr = qr * scale_r;
            const double vi = qi * scale_i;
            energy[idx] = (vr * vr) + (vi * vi);
        }

        const auto nth =
            indices.begin() + static_cast<std::ptrdiff_t>(keep_count);
        std::nth_element(indices.begin(), nth, indices.end(),
            [&](std::size_t a, std::size_t b) {
                return energy[a] > energy[b];
            });
        const std::size_t base = b * block_size;
        for (std::size_t k = 0; k < keep_count; ++k) {
            mask_set(out.mask, base + indices[k]);
        }
    }
}

void compact_coeffs(BlazCompressedComplex& out, std::size_t num_blocks,
    std::size_t block_size, std::size_t keep_count) {
    std::vector<BinType> compact_r;
    std::vector<BinType> compact_i;
    compact_r.reserve(num_blocks * keep_count);
    compact_i.reserve(num_blocks * keep_count);

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const std::size_t base = b * block_size;
        for (std::size_t idx = 0; idx < block_size; ++idx) {
            if (!mask_get(out.mask, base + idx)) {
                continue;
            }
            compact_r.push_back(out.F_r[base + idx]);
            compact_i.push_back(out.F_i[base + idx]);
        }
    }

    out.F_r.swap(compact_r);
    out.F_i.swap(compact_i);
}

void dct_1d_from_state(const CTYPE* state, std::size_t block_idx,
    bool real_part, double* out, std::size_t n) {
    const double inv_n = 1.0 / static_cast<double>(n);
    const double scale0 = std::sqrt(inv_n);
    const double scale = std::sqrt(2.0 * inv_n);
    const double pi = std::acos(-1.0);
    const std::size_t base = block_idx * n;
    for (std::size_t k = 0; k < n; ++k) {
        double sum = 0.0;
        for (std::size_t t = 0; t < n; ++t) {
            const CTYPE v = state[base + t];
            const double x = real_part ? std::real(v) : std::imag(v);
            const double angle = pi * (static_cast<double>(t) + 0.5) *
                                 static_cast<double>(k) * inv_n;
            sum += x * std::cos(angle);
        }
        out[k] = (k == 0) ? (scale0 * sum) : (scale * sum);
    }
}

void idct_1d(const double* in, double* out, std::size_t n) {
    const double inv_n = 1.0 / static_cast<double>(n);
    const double scale0 = std::sqrt(inv_n);
    const double scale = std::sqrt(2.0 * inv_n);
    const double pi = std::acos(-1.0);
    for (std::size_t t = 0; t < n; ++t) {
        double sum = scale0 * in[0];
        for (std::size_t k = 1; k < n; ++k) {
            const double angle = pi * (static_cast<double>(t) + 0.5) *
                                 static_cast<double>(k) * inv_n;
            sum += scale * in[k] * std::cos(angle);
        }
        out[t] = sum;
    }
}

double specified_coeff(BinType q, double block_max) {
    if (block_max == 0.0) {
        return 0.0;
    }
    return (static_cast<double>(q) * block_max) / kBinRadius;
}
}  // namespace

BlazConfig& blaz_config() {
    static BlazConfig cfg;
    return cfg;
}

BlazCompressedComplex blaz_compress_1d_complex_array(CTYPE* state, ITYPE dim) {
    (void)state;

    BlazCompressedComplex out;
    const BlazConfig& cfg = blaz_config();
    assert(cfg.block_size > 0);
    assert(cfg.keep_den > 0);
    assert(cfg.keep_num <= cfg.keep_den);
    const std::size_t total = static_cast<std::size_t>(dim);
    const std::size_t block_size =
        (total < cfg.block_size) ? total : cfg.block_size;
    out.s = {total};
    out.i = {block_size};
    assert(block_size % cfg.keep_den == 0);
    const bool keep_all = (total < cfg.block_size);
    const std::size_t keep_count =
        keep_all ? block_size : (block_size * cfg.keep_num / cfg.keep_den);
    assert(total % block_size == 0);
    const std::size_t num_blocks = total / block_size;

    out.N_r.assign(num_blocks, 0.0);
    out.N_i.assign(num_blocks, 0.0);

    std::vector<double> dct_block(block_size, 0.0);

    out.F_r.reserve(num_blocks * block_size);
    out.F_i.reserve(num_blocks * block_size);

    for (std::size_t b = 0; b < num_blocks; ++b) {
        dct_1d_from_state(state, b, true, dct_block.data(), block_size);
        double max_r = 0.0;
        for (double v : dct_block) {
            const double a = std::abs(v);
            if (a > max_r) {
                max_r = a;
            }
        }
        out.N_r[b] = max_r;
        if (max_r == 0.0) {
            out.F_r.insert(out.F_r.end(), block_size, 0);
        } else {
            for (std::size_t idx = 0; idx < block_size; ++idx) {
                const double v = dct_block[idx];
                const double scaled = (kBinRadius * v) / max_r;
                const double rounded = std::round(scaled);
                out.F_r.push_back(static_cast<BinType>(rounded));
            }
        }

        dct_1d_from_state(state, b, false, dct_block.data(), block_size);
        double max_i = 0.0;
        for (double v : dct_block) {
            const double a = std::abs(v);
            if (a > max_i) {
                max_i = a;
            }
        }
        out.N_i[b] = max_i;
        if (max_i == 0.0) {
            out.F_i.insert(out.F_i.end(), block_size, 0);
        } else {
            for (std::size_t idx = 0; idx < block_size; ++idx) {
                const double v = dct_block[idx];
                const double scaled = (kBinRadius * v) / max_i;
                const double rounded = std::round(scaled);
                out.F_i.push_back(static_cast<BinType>(rounded));
            }
        }
    }

    {
        // try different pruning policy
        // build_fixed_mask(out, num_blocks, block_size, keep_count);
        build_joint_energy_mask(out, num_blocks, block_size, keep_count);
        compact_coeffs(out, num_blocks, block_size, keep_count);
    }

    return out;
}

void blaz_decompress_1d_complex_array(
    const BlazCompressedComplex* comp, CTYPE* out_state) {
    assert(comp != nullptr);
    assert(out_state != nullptr);
    assert(comp->s.size() == 1);
    assert(comp->i.size() == 1);

    const std::size_t total = comp->s[0];
    const std::size_t block = comp->i[0];
    assert(block > 0);
    assert(total % block == 0);
    if (total < blaz_config().block_size) {
        assert(block == total);
    }
    const std::size_t num_blocks = total / block;
    assert(comp->mask.size() == mask_words(num_blocks * block));

    const std::size_t kept_count = [&]() {
        std::size_t count = 0;
        for (std::size_t b = 0; b < num_blocks; ++b) {
            for (std::size_t idx = 0; idx < block; ++idx) {
                if (mask_get(comp->mask, b * block + idx)) {
                    ++count;
                }
            }
        }
        return count;
    }();
    assert(comp->N_r.size() == num_blocks);
    assert(comp->N_i.size() == num_blocks);
    assert(comp->F_r.size() == num_blocks * kept_count);
    assert(comp->F_i.size() == num_blocks * kept_count);

    std::vector<double> coeff(block, 0.0);
    std::vector<double> time(block, 0.0);

    std::size_t fr_idx = 0;
    std::size_t fi_idx = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        std::size_t kept_this_block = 0;
        for (std::size_t idx = 0; idx < block; ++idx) {
            if (mask_get(comp->mask, b * block + idx)) {
                ++kept_this_block;
            }
        }

        std::fill(coeff.begin(), coeff.end(), 0.0);
        if (comp->N_r[b] != 0.0) {
            for (std::size_t idx = 0; idx < block; ++idx) {
                if (!mask_get(comp->mask, b * block + idx)) {
                    continue;
                }
                coeff[idx] = specified_coeff(comp->F_r[fr_idx++], comp->N_r[b]);
            }
        } else {
            fr_idx += kept_this_block;
        }
        idct_1d(coeff.data(), time.data(), block);
        for (std::size_t t = 0; t < block; ++t) {
            out_state[b * block + t] = CTYPE(time[t], 0.0);
        }

        std::fill(coeff.begin(), coeff.end(), 0.0);
        if (comp->N_i[b] != 0.0) {
            for (std::size_t idx = 0; idx < block; ++idx) {
                if (!mask_get(comp->mask, b * block + idx)) {
                    continue;
                }
                coeff[idx] = specified_coeff(comp->F_i[fi_idx++], comp->N_i[b]);
            }
        } else {
            fi_idx += kept_this_block;
        }
        idct_1d(coeff.data(), time.data(), block);
        for (std::size_t t = 0; t < block; ++t) {
            const std::size_t pos = b * block + t;
            out_state[pos] = CTYPE(std::real(out_state[pos]), time[t]);
        }
    }
}

CTYPE blaz_dot_product(
    const BlazCompressedComplex* a, const BlazCompressedComplex* b) {
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->s.size() == 1);
    assert(b->s.size() == 1);
    assert(a->i.size() == 1);
    assert(b->i.size() == 1);
    assert(a->s[0] == b->s[0]);
    assert(a->i[0] == b->i[0]);

    const std::size_t total = a->s[0];
    const std::size_t block = a->i[0];
    assert(block > 0);
    assert(total % block == 0);
    const std::size_t num_blocks = total / block;

    assert(a->mask.size() == mask_words(num_blocks * block));
    assert(b->mask.size() == mask_words(num_blocks * block));
    assert(a->N_r.size() == num_blocks);
    assert(a->N_i.size() == num_blocks);
    assert(b->N_r.size() == num_blocks);
    assert(b->N_i.size() == num_blocks);

    const auto kept_count = [&](const BlazCompressedComplex* comp) {
        std::size_t count = 0;
        for (std::size_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
            for (std::size_t idx = 0; idx < block; ++idx) {
                if (mask_get(comp->mask, b_idx * block + idx)) {
                    ++count;
                }
            }
        }
        return count;
    };
    assert(a->F_r.size() == kept_count(a));
    assert(a->F_i.size() == kept_count(a));
    assert(b->F_r.size() == kept_count(b));
    assert(b->F_i.size() == kept_count(b));

    double sum_real = 0.0;
    double sum_imag = 0.0;
    std::size_t fr_a = 0;
    std::size_t fi_a = 0;
    std::size_t fr_b = 0;
    std::size_t fi_b = 0;

    for (std::size_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
        const std::size_t base = b_idx * block;

        for (std::size_t idx = 0; idx < block; ++idx) {
            const bool keep_a = mask_get(a->mask, base + idx);
            const bool keep_b = mask_get(b->mask, base + idx);

            double ar = 0.0;
            double ai = 0.0;
            double br = 0.0;
            double bi = 0.0;

            if (keep_a) {
                ar = specified_coeff(a->F_r[fr_a++], a->N_r[b_idx]);
                ai = specified_coeff(a->F_i[fi_a++], a->N_i[b_idx]);
            }
            if (keep_b) {
                br = specified_coeff(b->F_r[fr_b++], b->N_r[b_idx]);
                bi = specified_coeff(b->F_i[fi_b++], b->N_i[b_idx]);
            }

            if (!keep_a || !keep_b) {
                continue;
            }

            sum_real += (ar * br) - (ai * bi);
            sum_imag += (ar * bi) + (ai * br);
        }
    }

    return CTYPE(sum_real, sum_imag);
}
