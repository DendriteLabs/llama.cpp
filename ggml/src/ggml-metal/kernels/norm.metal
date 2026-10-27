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

// One 1024-thread threadgroup per row, float4 accesses: thread t holds float4 slots t and t + 1024 (the
// same split as kernel_add_rms_norm_mul_impl<float4> with 1024 threads, so the sum of squares is the same).
// Every load is issued before the first store: the device pointers may alias, so a store would otherwise
// hold back the loads after it. Hadamard block N = 1024 = 256 float4: stages 1 and 2 inside the float4,
// 4..64 by lane shuffles, 128..512 through threadgroup memory (NS*1024 float4).
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
    constexpr int NT  = 1024;
    constexpr int NQ  = N/4;
    constexpr int NQT = NB*NQ;
    constexpr int NS  = (NQT + NT - 1)/NT;
    static_assert(N == 1024 && NT % NQ == 0, "kernel_rms_norm_mul_fwht: block of 1024");

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    const int tid = tpitg.x;

    device const float4 * x  = (device const float4 *) (src0   + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);
    device const float4 * f0 = (device const float4 *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const float4 * b  = (device const float4 *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);
    device const float4 * sg = (device const float4 *) signs;

    device float4 * y = (device float4 *) (dst     + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    device float4 * s = (device float4 *) (dst_sum + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    device float4 * z = (device float4 *) (dst_fwht + ((int64_t) (i03*args.nef2[0] + i02)*args.nef1[0] + i01)*args.ne00);

    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }

    float4 reg[NS];
    float4 wv[NS];
    float4 sv[NS];

    for (int j = 0; j < NS; ++j) {
        const int i = tid + j*NT;
        reg[j] = 0.0f;
        wv[j]  = 0.0f;
        sv[j]  = 0.0f;
        if (i < NQT) {
            reg[j] = x[i];
            if (ADD) {
                reg[j] += b[i];
            }
            wv[j] = f0[i];
            sv[j] = sg[i];
        }
    }

    float sumf = 0.0f;
    for (int j = 0; j < NS; ++j) {
        sumf += dot(reg[j], reg[j]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f/sqrt(sumf/args.ne00 + args.eps);
    const float hs    = 1.0f/sqrt((float) N);

    for (int j = 0; j < NS; ++j) {
        const int i = tid + j*NT;
        if (i < NQT) {
            if (ADD) {
                s[i] = reg[j];
            }
            const float4 v = (reg[j]*scale)*wv[j];
            y[i] = v;
            reg[j] = v*sv[j]*hs;
        }
    }

    // element e = 4*i + c; the partner of stage m is e ^ m
    for (int j = 0; j < NS; ++j) {
        const float4 v = reg[j];
        const float4 t = float4(v.y + v.x, v.x - v.y, v.w + v.z, v.z - v.w);
        reg[j] = float4(t.z + t.x, t.w + t.y, t.x - t.z, t.y - t.w);
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NS; ++j) {
            const float4 v  = reg[j];
            const float4 v2 = simd_shuffle_xor(v, i);
            reg[j] = (tiisg & i) == 0 ? v2 + v : v2 - v;
        }
    }

    threadgroup float4 * sh = (threadgroup float4 *) shmem;

    for (int i = NW; i < NQ; i *= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NS; ++j) {
            sh[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NS; ++j) {
            const float4 v  = reg[j];
            const float4 v2 = sh[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? v2 + v : v2 - v;
        }
    }

    for (int j = 0; j < NS; ++j) {
        const int i = tid + j*NT;
        if (i < NQT) {
            z[i] = reg[j];
        }
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

// One 256-thread threadgroup per 1024-wide output block of a row (float4 per thread): each thread finds the
// source element of its output position through the inverse head permutation, so a simdgroup holds one
// whole head (128 values) and its sum of squares is one simd_sum. Then the sign flip and the Hadamard
// transform of the block (stages 1 and 2 inside the float4, 4..64 by lane shuffles, 128..512 through
// threadgroup memory of 256 float4). Grid: x = block, y = row.
kernel void kernel_gdn_tail_fwht(
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
    constexpr int NQ = 256;

    const int blk = tgpig.x;
    const int row = tgpig.y;
    const int tid = tpitg.x;

    device const float4 * o  = (device const float4 *) (o_raw + row*args.nb_o);
    device const float4 * z  = (device const float4 *) (z_raw + row*args.nb_z);
    device const float4 * w4 = (device const float4 *) w;
    device const float4 * sg = (device const float4 *) signs;
    device       float4 * y  = (device float4 *) (dst + (int64_t) row*args.ne) ;

    const int i = blk*NQ + tid;
    const int p = 4*i;

    const int hd  = p % args.perm_hd;
    const int u   = p / args.perm_hd;
    const int rep = u % args.perm_rep;
    const int nk  = u / args.perm_rep;
    const int e   = (rep*args.perm_nk + nk)*args.perm_hd + hd;

    const float4 v  = o[e/4];
    const float  ss = simd_sum(dot(v, v));

    const float  scale = 1.0f/sqrt(ss/args.head + args.eps);
    const float4 nrm   = (v*scale)*w4[hd/4];
    const float4 x0    = z[e/4];
    const float4 g     = (x0 / (1.0f + exp(-x0)))*nrm;

    const float hs = 1.0f/sqrt(1024.0f);

    float4 r = g*sg[i]*hs;

    {
        const float4 t = float4(r.y + r.x, r.x - r.y, r.w + r.z, r.z - r.w);
        r = float4(t.z + t.x, t.w + t.y, t.x - t.z, t.y - t.w);
    }

    for (int m = 1; m < NW; m *= 2) {
        const float4 r2 = simd_shuffle_xor(r, m);
        r = (tiisg & m) == 0 ? r2 + r : r2 - r;
    }

    threadgroup float4 * sh = (threadgroup float4 *) shmem;

    for (int m = NW; m < NQ; m *= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh[tid] = r;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float4 r2 = sh[tid ^ m];
        r = (tid & m) == 0 ? r2 + r : r2 - r;
    }

    y[i] = r;
}

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
