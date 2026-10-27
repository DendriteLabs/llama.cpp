#include "common.h"

// F == 1 : norm (no fuse)
// F == 2 : norm + mul
// F == 3 : norm + mul + add
template <typename T, short F>
kernel void kernel_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    T sumft(0.0f);

    float sumf = 0.0f;

    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumft += x[i00];
    }
    sumf = dot(sumft, T(1.0f));
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean = sumf/args.ne00;

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);

    sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        y[i00] = x[i00] - mean;
        sumf += dot(y[i00], y[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float variance = sumf/args.ne00;

    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (y[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (y[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (y[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_norm_fuse_impl<float4, 1>) kernel_norm_fuse_t;

template [[host_name("kernel_norm_f32")]]         kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 1>;
template [[host_name("kernel_norm_mul_f32")]]     kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 2>;
template [[host_name("kernel_norm_mul_add_f32")]] kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 3>;

template [[host_name("kernel_norm_f32_4")]]         kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_norm_mul_f32_4")]]     kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 2>;
template [[host_name("kernel_norm_mul_add_f32_4")]] kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 3>;

// F == 1 : rms_norm (no fuse)
// F == 2 : rms_norm + mul
// F == 3 : rms_norm + mul + add
template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (x[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;

template [[host_name("kernel_rms_norm_f32")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 1>;
template [[host_name("kernel_rms_norm_mul_f32")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;
template [[host_name("kernel_rms_norm_mul_add_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 3>;

template [[host_name("kernel_rms_norm_f32_4")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_rms_norm_mul_f32_4")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 2>;
template [[host_name("kernel_rms_norm_mul_add_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 3>;

template <typename T>
kernel void kernel_add_rms_norm_mul_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        device       char * dst_sum,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x  = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);
    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * b  = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    device T * s = (device T *) (dst_sum + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    device T * y = (device T *) (dst     + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);

    float sumf = 0.0f;

    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        const T v = x[i00] + b[i00];
        s[i00] = v;
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        y[i00] = (s[i00]*scale)*f0[i00];
    }
}

typedef decltype(kernel_add_rms_norm_mul_impl<float4>) kernel_add_rms_norm_mul_t;

template [[host_name("kernel_add_rms_norm_mul_f32")]]   kernel kernel_add_rms_norm_mul_t kernel_add_rms_norm_mul_impl<float>;
template [[host_name("kernel_add_rms_norm_mul_f32_4")]] kernel kernel_add_rms_norm_mul_t kernel_add_rms_norm_mul_impl<float4>;

template <int N, int NB, bool ADD>
kernel void kernel_rms_norm_mul_fwht_impl(
        constant ggml_metal_kargs_norm & args,
        device const char  * src0,
        device const char  * src1_0,
        device const char  * src1_1,
        device const float * signs,
        device       char  * dst,
        device       char  * dst_sum,
        device       float * dst_fwht,
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    constexpr int NW  = N_SIMDWIDTH;
    constexpr int NSG = N / NW;

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    const int tid = tpitg.x;

    device const float * x  = (device const float *) (src0   + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);
    device const float * f0 = (device const float *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const float * b  = (device const float *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    device float * y = (device float *) (dst     + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    device float * s = (device float *) (dst_sum + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    device float * z = dst_fwht + ((int64_t) (i03*args.nef2[0] + i02)*args.nef1[0] + i01)*args.ne00;

    float reg[NB];
    float sumf = 0.0f;

    for (int k = 0; k < NB; ++k) {
        float v = x[k*N + tid];
        if (ADD) {
            v += b[k*N + tid];
            s[k*N + tid] = v;
        }
        reg[k] = v;
        sumf += v*v;
    }
    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = tiisg < NSG ? shmem[tiisg] : 0.0f;
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = 1.0f/sqrt(sumf/args.ne00 + args.eps);
    const float hs    = 1.0f/sqrt((float) N);

    for (int k = 0; k < NB; ++k) {
        const float v = (reg[k]*scale)*f0[k*N + tid];
        y[k*N + tid] = v;
        reg[k] = v*signs[k*N + tid]*hs;
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int k = 0; k < NB; ++k) {
            const float val  = reg[k];
            const float val2 = simd_shuffle_xor(val, i);
            reg[k] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        for (int k = 0; k < NB; ++k) {
            shmem[k*N + tid] = reg[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < NB; ++k) {
            const float val  = reg[k];
            const float val2 = shmem[k*N + (tid ^ i)];
            reg[k] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int k = 0; k < NB; ++k) {
        z[k*N + tid] = reg[k];
    }
}

typedef decltype(kernel_rms_norm_mul_fwht_impl<1024, 1, false>) kernel_rms_norm_mul_fwht_t;

template [[host_name("kernel_rms_norm_mul_fwht_1024_nb1")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 1, false>;
template [[host_name("kernel_rms_norm_mul_fwht_1024_nb2")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 2, false>;
template [[host_name("kernel_rms_norm_mul_fwht_1024_nb3")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 3, false>;
template [[host_name("kernel_rms_norm_mul_fwht_1024_nb4")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 4, false>;
template [[host_name("kernel_rms_norm_mul_fwht_1024_nb5")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 5, false>;
template [[host_name("kernel_rms_norm_mul_fwht_1024_nb6")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 6, false>;
template [[host_name("kernel_rms_norm_mul_fwht_1024_nb7")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 7, false>;
template [[host_name("kernel_rms_norm_mul_fwht_1024_nb8")]]     kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 8, false>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb1")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 1, true>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb2")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 2, true>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb3")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 3, true>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb4")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 4, true>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb5")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 5, true>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb6")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 6, true>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb7")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 7, true>;
template [[host_name("kernel_add_rms_norm_mul_fwht_1024_nb8")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht_impl<1024, 8, true>;

template <int N, int NB>
kernel void kernel_gdn_tail_fwht_impl(
        constant ggml_metal_kargs_gdn_tail & args,
        device const char  * o_raw,
        device const char  * z_raw,
        device const float * w,
        device const float * signs,
        device       float * dst,
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    constexpr int NW = N_SIMDWIDTH;

    const int row = tgpig.x;
    const int tid = tpitg.x;

    device const float * o = (device const float *) (o_raw + row*args.nb_o);
    device const float * z = (device const float *) (z_raw + row*args.nb_z);
    device       float * y = dst + (int64_t) row*NB*N;

    threadgroup float * part = shmem;
    threadgroup float * buf  = shmem + NB*NW;

    const int head = args.head;

    float v[NB];

    for (int k = 0; k < NB; ++k) {
        v[k] = o[k*N + tid];
        const float ps = simd_sum(v[k]*v[k]);
        if (tiisg == 0) {
            part[k*NW + sgitg] = ps;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int sg_per_head = head/NW;
    const int sg0 = (tid/head)*sg_per_head;

    for (int k = 0; k < NB; ++k) {
        float ss = 0.0f;
        for (int j = 0; j < sg_per_head; ++j) {
            ss += part[k*NW + sg0 + j];
        }

        const float scale = 1.0f/sqrt(ss/head + args.eps);

        const int   e   = k*N + tid;
        const float nrm = (v[k]*scale)*w[e % head];
        const float x0  = z[e];
        const float g   = (x0 / (1.0f + exp(-x0)))*nrm;

        const int hd  = e % args.perm_hd;
        const int t   = e / args.perm_hd;
        const int nk  = t % args.perm_nk;
        const int rep = t / args.perm_nk;

        buf[hd + args.perm_hd*(rep + args.perm_rep*nk)] = g;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float hs = 1.0f/sqrt((float) N);

    float reg[NB];
    for (int k = 0; k < NB; ++k) {
        const int gi = k*N + tid;
        reg[k] = buf[gi]*signs[gi]*hs;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 1; i < NW; i *= 2) {
        for (int k = 0; k < NB; ++k) {
            const float val  = reg[k];
            const float val2 = simd_shuffle_xor(val, i);
            reg[k] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        for (int k = 0; k < NB; ++k) {
            buf[k*N + tid] = reg[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < NB; ++k) {
            const float val  = reg[k];
            const float val2 = buf[k*N + (tid ^ i)];
            reg[k] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int k = 0; k < NB; ++k) {
        y[k*N + tid] = reg[k];
    }
}

typedef decltype(kernel_gdn_tail_fwht_impl<1024, 1>) kernel_gdn_tail_fwht_t;

template [[host_name("kernel_gdn_tail_fwht_1024_nb2")]] kernel kernel_gdn_tail_fwht_t kernel_gdn_tail_fwht_impl<1024, 2>;
template [[host_name("kernel_gdn_tail_fwht_1024_nb3")]] kernel kernel_gdn_tail_fwht_t kernel_gdn_tail_fwht_impl<1024, 3>;
template [[host_name("kernel_gdn_tail_fwht_1024_nb4")]] kernel kernel_gdn_tail_fwht_t kernel_gdn_tail_fwht_impl<1024, 4>;
template [[host_name("kernel_gdn_tail_fwht_1024_nb5")]] kernel kernel_gdn_tail_fwht_t kernel_gdn_tail_fwht_impl<1024, 5>;
template [[host_name("kernel_gdn_tail_fwht_1024_nb6")]] kernel kernel_gdn_tail_fwht_t kernel_gdn_tail_fwht_impl<1024, 6>;
template [[host_name("kernel_gdn_tail_fwht_1024_nb8")]] kernel kernel_gdn_tail_fwht_t kernel_gdn_tail_fwht_impl<1024, 8>;

template <typename T0, typename T>
kernel void kernel_l2_norm_impl(
        constant ggml_metal_kargs_l2_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const T0 * x = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T  * y = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f/max(sqrt(sumf), args.eps);

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        y[i00] = x[i00] * scale;
    }
}

typedef decltype(kernel_l2_norm_impl<float, float>) kernel_l2_norm_t;

template [[host_name("kernel_l2_norm_f32_f32")]]   kernel kernel_l2_norm_t kernel_l2_norm_impl<float,  float>;
template [[host_name("kernel_l2_norm_f32_f32_4")]] kernel kernel_l2_norm_t kernel_l2_norm_impl<float4, float4>;

kernel void kernel_group_norm_f32(
        constant ggml_metal_kargs_group_norm & args,
        device const float * src0,
        device       float * dst,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = args.ne00*args.ne01*args.ne02;
    const int64_t gs = args.ne00*args.ne01*((args.ne02 + args.ngrp - 1) / args.ngrp);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}
