#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
using cfloat = c10::complex<float>;

__device__ __forceinline__ void apply_h_pair(const cfloat* prev, cfloat* out, int i0, int i1){
    cfloat a = prev[i0]; cfloat b = prev[i1];
    const float inv_sqrt2 = 0.7071067811865476f;
    out[i0] = cfloat((a.real()+b.real())*inv_sqrt2, (a.imag()+b.imag())*inv_sqrt2);
    out[i1] = cfloat((a.real()-b.real())*inv_sqrt2, (a.imag()-b.imag())*inv_sqrt2);
}
__device__ __forceinline__ void apply_x_pair(const cfloat* prev, cfloat* out, int i0, int i1){ out[i0]=prev[i1]; out[i1]=prev[i0]; }
__device__ __forceinline__ void apply_z_pair(const cfloat* prev, cfloat* out, int i0, int i1){ out[i0]=prev[i0]; out[i1]=cfloat(-prev[i1].real(), -prev[i1].imag()); }

__device__ __forceinline__ void apply_cx(const cfloat* prev, cfloat* out, int amp, int q1, int q2){
    int bitc = (amp >> q1) & 1; int bitt = (amp >> q2) & 1;
    if(bitc==1 && bitt==0){
        int i0 = amp; int i1 = amp | (1<<q2);
        out[i0] = prev[i1];
        out[i1] = prev[i0];
    }
}
__device__ __forceinline__ void apply_cz_all(const cfloat* prev, cfloat* out, int amp, int q1, int q2){
    int b1 = (amp >> q1) & 1; int b2 = (amp >> q2) & 1;
    cfloat v = prev[amp];
    if(b1==1 && b2==1){ out[amp] = cfloat(-v.real(), -v.imag()); }
    else{ out[amp] = v; }
}

__device__ __forceinline__ void apply_rzrx_pair(const cfloat* prev, cfloat* out, int i0, int i1, float ang_rz, float ang_rx){
    float h = 0.5f * ang_rz; float cph = cosf(h); float sph = sinf(h);
    // phases for i0 (bit0) and i1 (bit1)
    // bit0 -> exp(-i h) = c - i s ; bit1 -> exp(+i h) = c + i s
    float p0r = cph, p0i = -sph; float p1r = cph, p1i = sph;
    float hx = 0.5f * ang_rx; float cx = cosf(hx); float sx = sinf(hx);
    // s_complex = -i*sx -> (0,-sx)
    cfloat a = prev[i0]; cfloat b = prev[i1];
    // a' = (p0)*(a); b' = (p1)*(b)
    float apr = a.real()*p0r - a.imag()*p0i; float api = a.real()*p0i + a.imag()*p0r;
    float bpr = b.real()*p1r - b.imag()*p1i; float bpi = b.real()*p1i + b.imag()*p1r;
    // new0 = cx*a' + (-i sx)*b' => (cx*apr + sx*bpi, cx*api - sx*bpr)
    float n0r = cx*apr + sx*bpi; float n0i = cx*api - sx*bpr;
    // new1 = (-i sx)*a' + cx*b' => (sx*api + cx*bpr, -sx*apr + cx*bpi)
    float n1r = sx*api + cx*bpr; float n1i = -sx*apr + cx*bpi;
    out[i0] = cfloat(n0r, n0i); out[i1] = cfloat(n1r, n1i);
}

__global__ void base_noise_segment_kernel(
    cfloat* __restrict__ states,
    cfloat* __restrict__ scratch,
    const int* __restrict__ gate_kind,  // [B,L]
    const int* __restrict__ q1s,        // [B,L]
    const int* __restrict__ q2s,        // [B,L]
    const float* __restrict__ rz1,      // [B,L]
    const float* __restrict__ rx1,      // [B,L]
    const float* __restrict__ rz2,      // [B,L]
    const float* __restrict__ rx2,      // [B,L]
    int B, int K, int D, int L, int reverse)
{
    int bk = blockIdx.x; if(bk >= B*K) return; int amp = threadIdx.x; if(amp >= D) return;
    size_t offset = (size_t)bk * D;
    cfloat* cur = states + offset;
    cfloat* nxt = scratch + offset;
    if(reverse==0){
        for(int s=0; s<L; ++s){
            int b = bk / K; // sample index in [0,B)
            int g = gate_kind[b*L + s]; if(g < 0) break;
            int q1 = q1s[b*L + s]; int q2 = q2s[b*L + s];
            if(g==0){ if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_h_pair(cur, nxt, i0, i1);} }
            else if(g==1){ if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_x_pair(cur, nxt, i0, i1);} }
            else if(g==2){ if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_z_pair(cur, nxt, i0, i1);} }
            else if(g==3){ if(((amp>>q2)&1)==0){ apply_cx(cur, nxt, amp, q1, q2); } }
            else if(g==4){ apply_cz_all(cur, nxt, amp, q1, q2); }
            else { nxt[amp] = cur[amp]; }
            __syncthreads(); cfloat* tmp = cur; cur = nxt; nxt = tmp;
            // noise q1 then q2
            float a_rz1 = rz1[b*L + s]; float a_rx1 = rx1[b*L + s];
            if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_rzrx_pair(cur, nxt, i0, i1, a_rz1, a_rx1);}        
            __syncthreads(); tmp = cur; cur = nxt; nxt = tmp;
            if(q2 >= 0){
                if(((amp>>q2)&1)==0){ int i0 = amp; int i1 = amp | (1<<q2); apply_rzrx_pair(cur, nxt, i0, i1, rz2[b*L+s], rx2[b*L+s]);}
                __syncthreads(); tmp = cur; cur = nxt; nxt = tmp;
            }
        }
    } else {
        // backward: apply adjoint in reverse order: (RZ2+RX2)^H, (RZ1+RX1)^H, then base gate adjoint
        for(int s=L-1; s>=0; --s){
            int b = bk / K; int g = gate_kind[b*L + s]; if(g < 0) continue; // skip pads
            int q1 = q1s[b*L + s]; int q2 = q2s[b*L + s];
            // inverse noise on q2 then q1 with negated angles
            if(q2 >= 0){
                if(((amp>>q2)&1)==0){ int i0 = amp; int i1 = amp | (1<<q2); apply_rzrx_pair(cur, nxt, i0, i1, -rz2[b*L+s], -rx2[b*L+s]); }
                __syncthreads(); cfloat* tmp = cur; cur = nxt; nxt = tmp;
            }
            if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_rzrx_pair(cur, nxt, i0, i1, -rz1[b*L + s], -rx1[b*L + s]); }
            __syncthreads(); cfloat* tmp2 = cur; cur = nxt; nxt = tmp2;
            // inverse base gate (self-adjoint for H,X,Z,CX,CZ)
            if(g==0){ if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_h_pair(cur, nxt, i0, i1);} }
            else if(g==1){ if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_x_pair(cur, nxt, i0, i1);} }
            else if(g==2){ if(((amp>>q1)&1)==0){ int i0 = amp; int i1 = amp | (1<<q1); apply_z_pair(cur, nxt, i0, i1);} }
            else if(g==3){ if(((amp>>q2)&1)==0){ apply_cx(cur, nxt, amp, q1, q2); } }
            else if(g==4){ apply_cz_all(cur, nxt, amp, q1, q2); }
            else { nxt[amp] = cur[amp]; }
            __syncthreads(); cfloat* tmp3 = cur; cur = nxt; nxt = tmp3;
        }
    }
    // ensure result in states
    if(cur != (states + offset)){
        // write back from cur to states
        states[offset + amp] = cur[amp];
    }
}

torch::Tensor fused_base_noise_segment(torch::Tensor states, torch::Tensor scratch,
    torch::Tensor gate_kind, torch::Tensor q1s, torch::Tensor q2s,
    torch::Tensor rz1, torch::Tensor rx1, torch::Tensor rz2, torch::Tensor rx2,
    int reverse)
{
    TORCH_CHECK(states.is_cuda(), "states must be CUDA complex64 [B,K,D]");
    int B = gate_kind.size(0); int L = gate_kind.size(1);
    int K = states.size(1); int D = states.size(2);
    int threads = D; if(threads>1024) threads = 1024; dim3 grid(B*K), block(threads);
    base_noise_segment_kernel<<<grid, block>>>(
        reinterpret_cast<cfloat*>(states.data_ptr<c10::complex<float>>()),
        reinterpret_cast<cfloat*>(scratch.data_ptr<c10::complex<float>>()),
        gate_kind.data_ptr<int>(), q1s.data_ptr<int>(), q2s.data_ptr<int>(),
        rz1.data_ptr<float>(), rx1.data_ptr<float>(), rz2.data_ptr<float>(), rx2.data_ptr<float>(),
        B, K, D, L, reverse);
    return states;
}
