//
#ifdef _USE_MPI
#include "MPIutil.hpp"

#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "SZ3c/sz3c.h"
#include "quantize_config.hpp"
#include "utility.hpp"

#ifdef _USE_QUANT
namespace {
using QuantCommInt = int16_t;

inline MPI_Datatype quant_comm_mpi_datatype() {
    if (std::is_same<QuantCommInt, int16_t>::value) return MPI_INT16_T;
    if (std::is_same<QuantCommInt, int32_t>::value) return MPI_INT32_T;
    if (std::is_same<QuantCommInt, int64_t>::value) return MPI_INT64_T;
    throw MPIRuntimeException(
        "quant_comm_mpi_datatype: unsupported QuantCommInt type");
}

inline size_t quantized_int_count_from_complex_count(int complex_count) {
    return static_cast<size_t>(complex_count) * 2;
}

inline size_t quantized_byte_count_from_complex_count(int complex_count) {
    return quantized_int_count_from_complex_count(complex_count) *
           sizeof(QuantCommInt);
}

inline int quantized_mpi_count_from_complex_count(int complex_count) {
    const size_t quant_count =
        quantized_int_count_from_complex_count(complex_count);
    if (quant_count >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw MPIRuntimeException(
            "quantized_mpi_count_from_complex_count: buffer size exceeds "
            "MPI int count range");
    }
    return static_cast<int>(quant_count);
}

std::vector<QuantCommInt> async_packed_send_ring[_MAX_REQUESTS];
std::vector<QuantCommInt> async_packed_recv_ring[_MAX_REQUESTS];
CTYPE* async_recv_dst_ring[_MAX_REQUESTS] = {nullptr};
int async_recv_complex_count_ring[_MAX_REQUESTS] = {0};
bool async_recv_pending_ring[_MAX_REQUESTS] = {false};

inline void validate_quant_step_for_comm_type(double quant_step) {
    const double min_step = 1.0 /
                            static_cast<double>(
                                std::numeric_limits<QuantCommInt>::max());
    if (quant_step < min_step) {
        throw MPIRuntimeException(
            "pack_snapped_complex_to_int_pairs: quantization step is too "
            "small for QuantCommInt range");
    }
}

inline QuantCommInt quant_cast_checked(long long value) {
    const long long hi =
        static_cast<long long>(std::numeric_limits<QuantCommInt>::max());
    const long long lo =
        static_cast<long long>(std::numeric_limits<QuantCommInt>::lowest());
    if (value > hi || value < lo) {
        throw MPIRuntimeException(
            "pack_snapped_complex_to_int_pairs: quantized value overflows "
            "QuantCommInt");
    }
    return static_cast<QuantCommInt>(value);
}

#ifndef NDEBUG
void validate_snapped_complex_buffer(
    const CTYPE* src, int complex_count, double step) {
    const double tolerance = step * 1e-6;
    for (int i = 0; i < complex_count; ++i) {
        const double re = std::real(src[i]);
        const double im = std::imag(src[i]);
        const double qre = std::round(re / step);
        const double qim = std::round(im / step);
        if (std::abs(re - qre * step) > tolerance ||
            std::abs(im - qim * step) > tolerance) {
            throw MPIRuntimeException(
                "pack_snapped_complex_to_int_pairs: input is not snapped to the "
                "current quantization grid");
        }
    }
}
#endif

#if defined(__GNUC__)
__attribute__((unused))
#endif
std::vector<QuantCommInt> pack_snapped_complex_to_int_pairs(
    const CTYPE* src, int complex_count) {
    if (complex_count < 0) {
        throw MPIRuntimeException(
            "pack_snapped_complex_to_int_pairs: complex_count must be "
            "non-negative");
    }
    std::vector<QuantCommInt> out(
        quantized_int_count_from_complex_count(complex_count));
    if (complex_count == 0) return out;

    const double quant_step = get_quantize_step();
    if (quant_step <= 0.0) {
        throw MPIRuntimeException(
            "pack_snapped_complex_to_int_pairs: quantization error bound must "
            "be positive");
    }
    validate_quant_step_for_comm_type(quant_step);
#ifndef NDEBUG
    validate_snapped_complex_buffer(src, complex_count, quant_step);
#endif

    for (int i = 0; i < complex_count; ++i) {
        const double re = std::real(src[i]);
        const double im = std::imag(src[i]);
        const long long qre = llround(re / quant_step);
        const long long qim = llround(im / quant_step);
        out[2 * i] = quant_cast_checked(qre);
        out[2 * i + 1] = quant_cast_checked(qim);
    }
    return out;
}

#if defined(__GNUC__)
__attribute__((unused))
#endif
void unpack_int_pairs_to_snapped_complex(
    const QuantCommInt* quantized, int complex_count, CTYPE* dst) {
    if (complex_count < 0) {
        throw MPIRuntimeException(
            "unpack_int_pairs_to_snapped_complex: complex_count must be "
            "non-negative");
    }
    const double quant_step = get_quantize_step();
    if (quant_step <= 0.0) {
        throw MPIRuntimeException(
            "unpack_int_pairs_to_snapped_complex: quantization error bound "
            "must be positive");
    }

    for (int i = 0; i < complex_count; ++i) {
        const double re = static_cast<double>(quantized[2 * i]) * quant_step;
        const double im =
            static_cast<double>(quantized[2 * i + 1]) * quant_step;
        dst[i] = CTYPE(re, im);
    }
}
}  // namespace
#endif

namespace {
inline uint64_t payload_bytes_from_count(int count, size_t elem_size) {
    if (count < 0) {
        throw MPIRuntimeException(
            "payload_bytes_from_count: count must be non-negative");
    }
    return static_cast<uint64_t>(count) * static_cast<uint64_t>(elem_size);
}
}  // namespace

void MPIutil::MPIFunctionError(
    const std::string &func, UINT ret, const std::string &file, UINT line) {
    std::string msg1 = func;
    std::string msg2 = " error(=";
    std::string msg3 = std::to_string(ret);
    std::string msg4 = "), ";
    std::string msg5 = file;
    std::string msg6 = ", ";
    std::string msg7 = std::to_string(line);
    throw MPIRuntimeException(msg1 + msg2 + msg3 + msg4 + msg5 + msg6 + msg7);
}

MPI_Request *MPIutil::get_request() {
    if (mpireq_cnt >= _MAX_REQUESTS) {
        std::string msg1 = "cannot get a request for communication, ";
        std::string msg2 = __FILE__;
        std::string msg3 = ", ";
        std::string msg4 = std::to_string(__LINE__);
        throw MPIRuntimeException(msg1 + msg2 + msg3 + msg4);
    }

    mpireq_cnt++;
    MPI_Request *ret = &(mpireq[mpireq_idx]);
    mpireq_idx = (mpireq_idx + 1) % _MAX_REQUESTS;
    return ret;
}

void MPIutil::mpi_wait(UINT count) {
    if (mpireq_cnt < count) {
        std::string msg1 = "mpi_wait count(=";
        std::string msg2 = std::to_string(count);
        std::string msg3 = ") is over incompleted requests(=";
        std::string msg4 = std::to_string(mpireq_cnt);
        std::string msg5 = "), ";
        std::string msg6 = __FILE__;
        std::string msg7 = ", ";
        std::string msg8 = std::to_string(__LINE__);
        throw MPIRuntimeException(
            msg1 + msg2 + msg3 + msg4 + msg5 + msg6 + msg7 + msg8);
    }

    for (UINT i = 0; i < count; i++) {
        UINT idx = (_MAX_REQUESTS + mpireq_idx - mpireq_cnt) % _MAX_REQUESTS;
        UINT ret = MPI_Wait(&(mpireq[idx]), &mpistat);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Wait", ret, __FILE__, __LINE__);
#ifdef _USE_QUANT
        if (async_recv_pending_ring[idx]) {
            unpack_int_pairs_to_snapped_complex(
                async_packed_recv_ring[idx].data(),
                async_recv_complex_count_ring[idx], async_recv_dst_ring[idx]);
            async_recv_pending_ring[idx] = false;
            async_recv_dst_ring[idx] = nullptr;
            async_recv_complex_count_ring[idx] = 0;
        }
#endif
        mpireq_cnt--;
    }
}

int MPIutil::get_rank() { return mpirank; }

int MPIutil::get_size() { return mpisize; }

int MPIutil::get_tag() {
    // pthread_mutex_lock(&mutex);
    mpitag ^= 1 << 20;  // max 1M-ranks
    // pthread_mutex_unlock(&mutex);
    return mpitag;
}

void MPIutil::release_workarea() {
    if (workarea != NULL) free(workarea);
    workarea = NULL;
}

CTYPE *MPIutil::get_workarea(ITYPE *dim_work, ITYPE *num_work) {
    ITYPE dim = *dim_work;
    *dim_work = get_min_ll(1 << _NQUBIT_WORK, dim);
    *num_work = get_max_ll(1, dim >> _NQUBIT_WORK);
    if (workarea == NULL) {
#if defined(__ARM_FEATURE_SVE)
        posix_memalign(
            (void **)&workarea, 256, sizeof(CTYPE) * (1 << _NQUBIT_WORK));
#else
        workarea = (CTYPE *)malloc(sizeof(CTYPE) * (1 << _NQUBIT_WORK));
#endif
        if (workarea == NULL) {
            std::string msg1 = "Can't malloc in get_workarea for MPI, ";
            std::string msg2 = __FILE__;
            std::string msg3 = ", ";
            std::string msg4 = std::to_string(__LINE__);
            throw MPIRuntimeException(msg1 + msg2 + msg3 + msg4);
        }
    }
    return workarea;
}

void MPIutil::barrier() {
    UINT ret = MPI_Barrier(mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Barrier", ret, __FILE__, __LINE__);
}

void MPIutil::set_quant_comm_enabled(bool enabled) {
#ifdef _USE_QUANT
    if (mpireq_cnt != 0) {
        throw MPIRuntimeException(
            "set_quant_comm_enabled: cannot switch mode while async MPI "
            "requests are in flight");
    }
    quant_comm_enabled = enabled;
#else
    (void)enabled;
#endif
}

bool MPIutil::is_quant_comm_enabled() const {
#ifdef _USE_QUANT
    return quant_comm_enabled;
#else
    return false;
#endif
}

void MPIutil::m_DC_send(void *sendbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    UINT ret = MPI_Send(
        sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, tag0, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Send", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_recv(void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    UINT ret = MPI_Recv(recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, tag0,
        mpicomm, &mpistat);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Recv", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_sendrecv(
    void *sendbuf, void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
#ifdef _USE_QUANT
    if (quant_comm_enabled) {
        const int quant_count = quantized_mpi_count_from_complex_count(count);
        std::vector<QuantCommInt> send_packed = pack_snapped_complex_to_int_pairs(
            reinterpret_cast<const CTYPE*>(sendbuf), count);
        std::vector<QuantCommInt> recv_packed(
            quantized_int_count_from_complex_count(count));

        UINT ret = MPI_Sendrecv(send_packed.data(), quant_count,
            quant_comm_mpi_datatype(), pair_rank, mpi_tag1, recv_packed.data(),
            quant_count, quant_comm_mpi_datatype(), pair_rank, mpi_tag2,
            mpicomm, &mpistat);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Sendrecv", ret, __FILE__, __LINE__);
        unpack_int_pairs_to_snapped_complex(
            recv_packed.data(), count, reinterpret_cast<CTYPE*>(recvbuf));
        bytes_exchanged_quant +=
            payload_bytes_from_count(quant_count, sizeof(QuantCommInt));
    } else {
        UINT ret = MPI_Sendrecv(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
            mpi_tag1, recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
            mpi_tag2, mpicomm, &mpistat);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Sendrecv", ret, __FILE__, __LINE__);
        bytes_exchanged_no_quant +=
            payload_bytes_from_count(count, sizeof(CTYPE));
    }
#else
    UINT ret = MPI_Sendrecv(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
        mpi_tag1, recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag2,
        mpicomm, &mpistat);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Sendrecv", ret, __FILE__, __LINE__);
#endif
}

void MPIutil::m_DC_sendrecv_compressed(void *sendbuf, void *recvbuf, int count,
    int pair_rank, int errBoundMode, double absErrBound, double relBoundRatio,
    double pwrBoundRatio) {
    static uint64_t seq = 0;
    const uint64_t call_seq = seq++;

    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;

    const size_t elem_count = static_cast<size_t>(count) * 2;
    const size_t raw_bytes = static_cast<size_t>(count) * sizeof(CTYPE);
    size_t out_size = 0;
    const double comp_start = MPI_Wtime();
    unsigned char *compressed =
        SZ_compress_args(SZ_DOUBLE, sendbuf, &out_size, errBoundMode,
            absErrBound, relBoundRatio, pwrBoundRatio, 1, 1, 1, 1, elem_count);
    const double comp_end = MPI_Wtime();
    compress_overhead_time_sum += (comp_end - comp_start);

    if (raw_bytes > 0) {
        compress_ratio_sum +=
            static_cast<double>(out_size) / static_cast<double>(raw_bytes);
        compress_ratio_count++;
    }

    uint64_t send_size = static_cast<uint64_t>(out_size);
    uint64_t recv_size = 0;
    const double size_comm_start = MPI_Wtime();
    UINT ret = MPI_Sendrecv(&send_size, 1, MPI_UINT64_T, pair_rank, mpi_tag1,
        &recv_size, 1, MPI_UINT64_T, pair_rank, mpi_tag2, mpicomm, &mpistat);
    const double size_comm_end = MPI_Wtime();
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Sendrecv(size)", ret, __FILE__, __LINE__);
    compress_overhead_time_sum += (size_comm_end - size_comm_start);

    std::vector<unsigned char> recv_bytes(recv_size);
    ret = MPI_Sendrecv(compressed, static_cast<int>(send_size), MPI_BYTE,
        pair_rank, mpi_tag1, recv_bytes.data(), static_cast<int>(recv_size),
        MPI_BYTE, pair_rank, mpi_tag2, mpicomm, &mpistat);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Sendrecv(bytes)", ret, __FILE__, __LINE__);

    const double decomp_start = MPI_Wtime();
    void *decompressed = SZ_decompress(SZ_DOUBLE, recv_bytes.data(),
        static_cast<size_t>(recv_size), 1, 1, 1, 1, elem_count);
    const double decomp_end = MPI_Wtime();
    compress_overhead_time_sum += (decomp_end - decomp_start);

    std::memcpy(recvbuf, decompressed, elem_count * sizeof(double));

    free_buf(decompressed);
    free_buf(compressed);
}

void MPIutil::m_DC_sendrecv_replace(void *buf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    UINT ret = MPI_Sendrecv_replace(buf, count, MPI_CXX_DOUBLE_COMPLEX,
        pair_rank, mpi_tag1, pair_rank, mpi_tag2, mpicomm, &mpistat);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Sendrecv_replace", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_isendrecv(
    void *sendbuf, void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    MPI_Request *send_request = get_request();
    MPI_Request *recv_request = get_request();

#ifdef _USE_QUANT
    const UINT send_idx = (_MAX_REQUESTS + mpireq_idx - 1) % _MAX_REQUESTS;
    const UINT recv_idx = (_MAX_REQUESTS + mpireq_idx - 1) % _MAX_REQUESTS;
    if (quant_comm_enabled) {
        const int quant_count = quantized_mpi_count_from_complex_count(count);
        async_packed_send_ring[send_idx] = pack_snapped_complex_to_int_pairs(
            reinterpret_cast<const CTYPE*>(sendbuf), count);
        async_packed_recv_ring[recv_idx].resize(
            quantized_int_count_from_complex_count(count));
        async_recv_dst_ring[recv_idx] = reinterpret_cast<CTYPE*>(recvbuf);
        async_recv_complex_count_ring[recv_idx] = count;
        async_recv_pending_ring[recv_idx] = true;

        UINT ret = MPI_Isend(async_packed_send_ring[send_idx].data(),
            quant_count, quant_comm_mpi_datatype(), pair_rank, mpi_tag1,
            mpicomm, send_request);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Isend", ret, __FILE__, __LINE__);
        ret = MPI_Irecv(async_packed_recv_ring[recv_idx].data(), quant_count,
            quant_comm_mpi_datatype(), pair_rank, mpi_tag2, mpicomm,
            recv_request);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Irecv", ret, __FILE__, __LINE__);
        bytes_exchanged_quant +=
            payload_bytes_from_count(quant_count, sizeof(QuantCommInt));
    } else {
        async_recv_pending_ring[recv_idx] = false;
        async_recv_dst_ring[recv_idx] = nullptr;
        async_recv_complex_count_ring[recv_idx] = 0;
        UINT ret = MPI_Isend(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
            mpi_tag1, mpicomm, send_request);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Isend", ret, __FILE__, __LINE__);
        ret = MPI_Irecv(recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
            mpi_tag2, mpicomm, recv_request);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Irecv", ret, __FILE__, __LINE__);
        bytes_exchanged_no_quant +=
            payload_bytes_from_count(count, sizeof(CTYPE));
    }
#else
    UINT ret = MPI_Isend(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
        mpi_tag1, mpicomm, send_request);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Isend", ret, __FILE__, __LINE__);
    ret = MPI_Irecv(recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag2,
        mpicomm, recv_request);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Irecv", ret, __FILE__, __LINE__);
#endif
}

void MPIutil::m_DC_allgather(void *sendbuf, void *recvbuf, int count) {
    UINT ret = MPI_Allgather(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, recvbuf,
        count, MPI_CXX_DOUBLE_COMPLEX, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allgather<CTYPE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_D_allgather(double a, void *recvbuf) {
    UINT ret =
        MPI_Allgather(&a, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allgather<DOUBLE>", ret, __FILE__, __LINE__);
}

void MPIutil::m_I_allreduce(void *buf, UINT count) {
    UINT ret = MPI_Allreduce(
        MPI_IN_PLACE, buf, count, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allreduce<ITYPE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_D_allreduce(void *buf) {
    UINT ret =
        MPI_Allreduce(MPI_IN_PLACE, buf, 1, MPI_DOUBLE, MPI_SUM, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allreduce<DOUBLE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_DC_allreduce(void *buf) {
    UINT ret = MPI_Allreduce(
        MPI_IN_PLACE, buf, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allreduce<CTYPE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_u_bcast(UINT *a) {
    UINT ret = MPI_Bcast(a, 1, MPI_INT, 0, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Bcast<int>", ret, __FILE__, __LINE__);
}

void MPIutil::s_D_bcast(double *a) {
    UINT ret = MPI_Bcast(a, 1, MPI_DOUBLE, 0, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Bcast<DOUBLE>", ret, __FILE__, __LINE__);
}
#endif  // #ifdef _USE_MPI
