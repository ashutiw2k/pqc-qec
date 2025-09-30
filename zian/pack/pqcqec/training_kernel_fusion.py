#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal training script (vectorized training kernel; shared-PQC structure).
Implements:
- JSON/JSONL data loading -> Dataset / DataLoader
- Transformer predicting parameter gate angles (outputs sin, cos)
- Minimal statevector simulator: h/x/z/cx/cz + rz/rx
- Multi-initial fidelity loss over K_RANDOM random initial states: loss = 1 - mean(F)
- Optional auxiliary angle supervision (AUX_ANGLE_LOSS)
- Optional cosine scheduler (NOW step-based)
- Vectorized precompute (fast), AND vectorized training replay with shared PQC structure
- Eliminates .item() sync points; angles computed once; optional checkpoint per-step

Removed:
- legacy noise implementation (hash variant), heavy vectorization variants not needed here
"""

from __future__ import annotations
import os, json, math, random, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Sequence, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= Settings =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.complex64
MAX_BASE_LEN=500; MAX_PARAM=75; MAX_QUBITS=5
EMB_DIM=1024; NUM_LAYERS=12; NUM_HEADS=16; FF_DIM=EMB_DIM*4; DROP=0.1  # 更大的模型处理更长序列
K_RANDOM=32  # 减少随机初态数量，10 qubits复杂度高

# ---- 批量与优化相关（调整为10 qubits）----
BASELINE_BS = 32          # 你之前稳定的小批量
BATCH_SIZE  = 4          # 10 qubits需要更小的batch size，内存约需32倍
LR_BASE     = 1e-4        # 增加基础学习率
LR_SCALE    = BATCH_SIZE / BASELINE_BS  # 4/32 = 0.125
LR          = LR_BASE * LR_SCALE * 0.25   # 2e-3 * 0.125 * 0.25 = 6.25e-5，接近之前的有效值
GRAD_CLIP   = 1.0
WARMUP_RATIO = 0.03       # 3% 的总训练步用于 warmup
USE_SCHEDULER = True      # 使用逐步(step)调度

EPOCHS=5
PRECOMPUTE_BASE=True
FAST_BASE_CACHE=True
FAST_NOISE_SCHEDULE=True
PACK_REF_STATES=True
VERBOSE_PRECOMP_TIMINGS=True
PARAM_CHECKPOINT=False  # 暂关 checkpoint；稳定后可重开
DATA_PATH='A:/wings/pqc-qec/zian/pack/pqcqec/5q_500g_circuit_data_processed'; SEED=42
AUX_ANGLE_LOSS=False; AUX_ANGLE_WEIGHT=0.05
PRINT_INTERVAL=50
DIFF_FIDELITY=True  # 训练主损可反传：基座+噪声 no_grad，参数门可微
USE_NOISE = True
NOISE_X_RAD = math.pi/100
NOISE_Z_RAD = math.pi/100
NOISE_DELTA_X = 0.05
NOISE_DELTA_Z = 0.05

# ===== 新增优化与计时开关 =====
FAST_PARAM_SIM = True  # 启用新的快速参数门模拟路径（假设所有 sample 共享 PQC 结构 & n_qubits 相同）
USE_PRECOMPUTED_PARAM_LAYOUT = True  # 预计算每个 base step 对应的参数门分组
MEASURE_BATCH_TIMES = True  # 统计每个 batch 的细分耗时
COMPILE_MODEL = False  # 如使用 PyTorch 2.0+ 可尝试 torch.compile(model)
AGGREGATE_PARAM_ANGLES = True  # 新增: 使用 scatter_add 聚合所有参数角度 (进一步减少 Python 循环)
# ---- 额外性能调试 & 减少 CPU 同步 ----
LOG_INTERVAL = 20            # 每多少个 batch 打印一次详细日志 (减少 item() 同步)
USE_CUDA_EVENTS = True       # 使用 cuda events 计 fwd/ sim / backward (GPU 真实执行时间)
REDUCE_SYNC = True           # 减少 batch 内同步 (不每步调用 .item())
USE_FUSED_BASE_NOISE = True  # 重新启用 fused kernel (CPU 模式下会自动禁用)
DEBUG_VALIDATE_FUSED = False  # 关闭调试验证，正常训练
DEBUG_VALIDATE_FUSED_STEPS = 3  # 验证前 N 个 batch (开启 DEBUG_VALIDATE_FUSED 时才生效)
DEBUG_COMPARE_ONE_SEGMENT = False  # 默认关闭 segment 逐步回放
DEBUG_ZERO_RX_NOISE = False  # 调试：若为 True 则在 fused path 验证阶段强制将 RX 噪声置零

# 新增诊断选项
VERBOSE_FUSED_DIAGNOSTICS = False  # 关闭详细诊断，正常训练
FUSED_TIMING_DEBUG = False  # 关闭时间测量，正常训练

# ===== 实验 fused kernel (lazy load) =====
try:
    from torch.utils.cpp_extension import load_inline as _load_inline_ext
except Exception:
    _load_inline_ext = None
_fused_ext_mod = None
_FUSED_ATTEMPTED = False
_FUSED_FAILED_REASON = None
_FUSED_USED_PRINTED = False  # one-time success message when fused path actually runs
_FUSED_SEGMENT_COUNT = 0  # 统计 fused segment 调用次数  
_FUSED_TOTAL_TIME = 0.0   # 统计 fused 总时间
_TOTAL_SEGMENTS_ATTEMPTED = 0  # 总共尝试的 segment 数
_FALLBACK_SEGMENTS = 0    # 使用 fallback 的 segment 数

def _ensure_msvc_env_windows():
    """On Windows inside VSCode plain PowerShell, cl/nvcc may be missing from PATH.
    Attempt to locate Visual Studio via vswhere and load developer environment vars in-process.
    Safe no-op if already available or non-Windows.
    """
    import os, shutil, subprocess, re
    if os.name != 'nt':
        return
    # Fast path: already have cl
    if shutil.which('cl') and shutil.which('nvcc'):
        return
    pf86 = os.environ.get('ProgramFiles(x86)')
    if not pf86:
        return
    vswhere = os.path.join(pf86, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
    if not os.path.isfile(vswhere):
        return
    try:
        # Require VC tools (component id) for more precise match
        cmd = [vswhere, '-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath']
        install_path = subprocess.check_output(cmd, encoding='utf-8', errors='ignore').strip().splitlines()
        if not install_path:
            return
        vsroot = install_path[0]
        # Prefer VsDevCmd (sets more vars than bare vcvars64)
        candidate = os.path.join(vsroot, 'Common7', 'Tools', 'VsDevCmd.bat')
        if not os.path.isfile(candidate):
            candidate = os.path.join(vsroot, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
            if not os.path.isfile(candidate):
                return
        # Run batch and dump environment. Use cmd /c "... & set"
        proc = subprocess.run(['cmd.exe', '/d', '/c', f'"{candidate}" -arch=x64 >nul 2>nul & set'], capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return
        # Parse KEY=VALUE lines
        for line in proc.stdout.splitlines():
            if '=' not in line: continue
            k,v = line.split('=',1)
            # Avoid overwriting existing PATH extension logic incorrectly; merge for PATH
            if k.upper() == 'PATH':
                # Prepend new paths not already inside PATH
                new_parts = [p for p in v.split(';') if p and p not in os.environ.get('PATH','')]
                if new_parts:
                    os.environ['PATH'] = ';'.join(new_parts + [os.environ.get('PATH','')])
            else:
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass
    # Ensure CUDA bin (nvcc) path appended if CUDA_HOME set but nvcc missing
    if shutil.which('nvcc') is None:
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home:
            bin_path = os.path.join(cuda_home, 'bin')
            if os.path.isdir(bin_path) and bin_path not in os.environ.get('PATH',''):
                os.environ['PATH'] = bin_path + ';' + os.environ.get('PATH','')

def _ensure_fused_extension():
    global _fused_ext_mod
    if _fused_ext_mod is not None:
        return _fused_ext_mod
    if _load_inline_ext is None:
        raise RuntimeError('cpp_extension load_inline not available')
    # Attempt to patch environment (Windows) so cl/nvcc become visible if VSCode terminal lacked dev shell
    _ensure_msvc_env_windows()
    cuda_src = r"""
#include <torch/extension.h>
// Access raw interleaved complex<float> memory: layout = [B,K,D] with element size 2 floats (real,imag)
__global__ void fused_kernel(
    float* __restrict__ data, // interleaved complex
    const int* __restrict__ gate_ids,
    const int* __restrict__ q1,
    const int* __restrict__ q2,
    const float* __restrict__ noise_rz_q1,
    const float* __restrict__ noise_rz_q2,
    const float* __restrict__ noise_rx_q1,
    const float* __restrict__ noise_rx_q2,
    int B,int K,int D,int Lb)
{
    int bk = blockIdx.x; int amp = threadIdx.x; if(amp>=D) return; int b = bk / K; int k = bk % K;
    size_t base_offset = ((size_t)b * K + k) * D * 2; // 2 floats per complex
    for(int t=0; t<Lb; ++t){
        int base = b*Lb + t;
        int gid = gate_ids[base];
        if(gid < 0) continue;
        int a1 = q1[base]; int a2 = q2[base];
        // ---- Base gate ----
        if(gid <= 2){ // 1q h/x/z on a1
            int q=a1;
            if(gid==2){ // pure phase flip only on bit=1 branch
                if(((amp>>q)&1)==1){
                    float* self_ptr = &data[base_offset + 2*amp];
                    self_ptr[0] = -self_ptr[0]; self_ptr[1] = -self_ptr[1];
                }
            } else if(((amp>>q)&1)==0){
                int p = amp | (1<<q);
                float* a_ptr = &data[base_offset + 2*amp];
                float* b_ptr = &data[base_offset + 2*p];
                float ar=a_ptr[0], ai=a_ptr[1];
                float br=b_ptr[0], bi=b_ptr[1];
                if(gid==0){ // h
                    const float inv = 0.70710678118f;
                    a_ptr[0] = (ar+br)*inv; a_ptr[1] = (ai+bi)*inv;
                    b_ptr[0] = (ar-br)*inv; b_ptr[1] = (ai-bi)*inv;
                } else { // x
                    a_ptr[0]=br; a_ptr[1]=bi; b_ptr[0]=ar; b_ptr[1]=ai;
                }
            }
        } else if(gid==3){ // cx (a1 control, a2 target)
            if(a2>=0){
                int c = (amp>>a1)&1; int tbit=(amp>>a2)&1;
                if(c && !tbit){
                    int p = amp | (1<<a2);
                    float* a_ptr = &data[base_offset + 2*amp];
                    float* b_ptr = &data[base_offset + 2*p];
                    float ar=a_ptr[0], ai=a_ptr[1];
                    a_ptr[0]=b_ptr[0]; a_ptr[1]=b_ptr[1];
                    b_ptr[0]=ar; b_ptr[1]=ai;
                }
            }
        } else if(gid==4){ // cz
            if(a2>=0){
                int b1=(amp>>a1)&1; int b2=(amp>>a2)&1; if(b1 && b2){ float* ptr=&data[base_offset + 2*amp]; ptr[0]=-ptr[0]; ptr[1]=-ptr[1]; }
            }
        }
        // ---- Noise (RZ then RX per qubit) ----
        if(noise_rz_q1){
            float ang1 = noise_rz_q1[base];
            if(ang1 != 0.f){
                int bit = (amp>>a1)&1; float h=0.5f*ang1; float c=cosf(h), s=sinf(h); float phase_s = bit ? s : -s; // e^{+/- i h}
                float* ptr=&data[base_offset + 2*amp]; float ar=ptr[0], ai=ptr[1];
                ptr[0]= ar*c - ai*phase_s; ptr[1]= ar*phase_s + ai*c;
            }
            if(a2>=0){
                float ang2 = noise_rz_q2[base];
                if(ang2 != 0.f){
                    int bit2 = (amp>>a2)&1; float h=0.5f*ang2; float c=cosf(h), s=sinf(h); float phase_s = bit2 ? s : -s;
                    float* ptr=&data[base_offset + 2*amp]; float ar=ptr[0], ai=ptr[1];
                    ptr[0]= ar*c - ai*phase_s; ptr[1]= ar*phase_s + ai*c;
                }
            }
        }
        if(noise_rx_q1){
            float ang1x = noise_rx_q1[base];
            if(ang1x != 0.f){
                if(((amp>>a1)&1)==0){
                    int p=amp|(1<<a1); float h=0.5f*ang1x; float c=cosf(h), s=sinf(h);
                    float* a_ptr=&data[base_offset + 2*amp]; float* b_ptr=&data[base_offset + 2*p];
                    float ar=a_ptr[0], ai=a_ptr[1]; float br=b_ptr[0], bi=b_ptr[1];
                    // new_a = c*a + (-i s)*b ; new_b = (-i s)*a + c*b
                    float a_r_new = c*ar + s*bi;
                    float a_i_new = c*ai - s*br;
                    float b_r_new = s*ai + c*br;
                    float b_i_new = -s*ar + c*bi;
                    a_ptr[0]=a_r_new; a_ptr[1]=a_i_new; b_ptr[0]=b_r_new; b_ptr[1]=b_i_new;
                }
            }
            if(a2>=0){
                float ang2x = noise_rx_q2[base];
                if(ang2x != 0.f){
                    if(((amp>>a2)&1)==0){
                        int p=amp|(1<<a2); float h=0.5f*ang2x; float c=cosf(h), s=sinf(h);
                        float* a_ptr=&data[base_offset + 2*amp]; float* b_ptr=&data[base_offset + 2*p];
                        float ar=a_ptr[0], ai=a_ptr[1]; float br=b_ptr[0], bi=b_ptr[1];
                        float a_r_new = c*ar + s*bi;
                        float a_i_new = c*ai - s*br;
                        float b_r_new = s*ai + c*br;
                        float b_i_new = -s*ar + c*bi;
                        a_ptr[0]=a_r_new; a_ptr[1]=a_i_new; b_ptr[0]=b_r_new; b_ptr[1]=b_i_new;
                    }
                }
            }
        }
    }
}

torch::Tensor fused_forward(torch::Tensor states, torch::Tensor gate_ids_flat, torch::Tensor q1_flat, torch::Tensor q2_flat,
                            torch::Tensor rz1, torch::Tensor rz2, torch::Tensor rx1, torch::Tensor rx2, int Lb){
    TORCH_CHECK(states.is_cuda(),"states must be CUDA");
    TORCH_CHECK(states.scalar_type()==at::kComplexFloat, "states dtype must be complex64 (complex float)");
    TORCH_CHECK(states.is_contiguous(), "states must be contiguous (call contiguous() once outside if needed)");
    int B=states.size(0), K=states.size(1), D=states.size(2); int total=B*K; int threads=D; dim3 grid(total), block(threads);
    TORCH_CHECK(gate_ids_flat.numel()==B*Lb, "gate_ids_flat size mismatch");
    TORCH_CHECK(q1_flat.numel()==B*Lb && q2_flat.numel()==B*Lb, "q1/q2 flat size mismatch");
    const float* rz_1 = (rz1.defined() && rz1.numel()>0)? rz1.data_ptr<float>() : nullptr;
    const float* rz_2 = (rz2.defined() && rz2.numel()>0)? rz2.data_ptr<float>() : nullptr;
    const float* rx_1 = (rx1.defined() && rx1.numel()>0)? rx1.data_ptr<float>() : nullptr;
    const float* rx_2 = (rx2.defined() && rx2.numel()>0)? rx2.data_ptr<float>() : nullptr;
    float* data = reinterpret_cast<float*>(states.data_ptr<c10::complex<float>>());
    fused_kernel<<<grid,block>>>(data,
                                 gate_ids_flat.data_ptr<int>(), q1_flat.data_ptr<int>(), q2_flat.data_ptr<int>(),
                                 rz_1, rz_2, rx_1, rx_2,
                                 B,K,D,Lb);
    return states;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){ m.def("fused_forward", &fused_forward, "fused base+noise (float-only, RZ+RX noise, per-sample gates)"); }
"""
    # Allow newer MSVC than nvcc officially supports (risk considered low for our simple kernel)
    # Add support for GH200 (sm_90) architecture
    extra_cuda_flags = [
        "-allow-unsupported-compiler", 
        "-O3", 
        "--use_fast_math",
        "-gencode=arch=compute_90,code=sm_90",  # GH200 support
        "-gencode=arch=compute_86,code=sm_86"   # Also keep older arch for compatibility
    ]
    _fused_ext_mod = _load_inline_ext(name="pqc_fused_bn_v2", cpp_sources="", cuda_sources=cuda_src,
                                      # We provide our own PYBIND11_MODULE above; do not auto-generate wrappers via 'functions'.
                                      extra_cuda_cflags=extra_cuda_flags,
                                      verbose=True)
    return _fused_ext_mod

def try_fused_base_noise(states, gate_ids, q1, q2, rz1_all, rz2_all, rx1_all=None, rx2_all=None):
    global _FUSED_ATTEMPTED, _FUSED_FAILED_REASON, _FUSED_USED_PRINTED
    if not USE_FUSED_BASE_NOISE or states.device.type!='cuda':
        if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS:
            print(f"[FUSED-DEBUG] Skipped: USE_FUSED_BASE_NOISE={USE_FUSED_BASE_NOISE}, device={states.device.type}")
        return None
    if _FUSED_ATTEMPTED and _fused_ext_mod is None:
        if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS:
            print(f"[FUSED-DEBUG] Skipped: Previous attempt failed - {_FUSED_FAILED_REASON}")
        return None  # 已失败且记录
    try:
        mod=_ensure_fused_extension()
        if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS and not _FUSED_USED_PRINTED:
            print(f"[FUSED-DEBUG] Extension loaded successfully, proceeding with fused execution")
    except Exception as e:
        if not _FUSED_ATTEMPTED:
            _FUSED_FAILED_REASON = str(e)
            print(f"[WARN] fused ext load failed (will disable further attempts): {_FUSED_FAILED_REASON}")
        _FUSED_ATTEMPTED = True
        return None
    _FUSED_ATTEMPTED = True
    B=states.size(0); Lb=gate_ids.size(1)
    g_row=gate_ids[0]; q1_row=q1[0]; q2_row=q2[0]
    rz1_t = rz1_all if (rz1_all is not None) else torch.empty(0, device=states.device)
    rz2_t = rz2_all if (rz2_all is not None) else torch.empty(0, device=states.device)
    rx1_t = rx1_all if (rx1_all is not None) else torch.empty(0, device=states.device)
    rx2_t = rx2_all if (rx2_all is not None) else torch.empty(0, device=states.device)
    # 添加时间测量（如果启用调试）
    if 'FUSED_TIMING_DEBUG' in globals() and FUSED_TIMING_DEBUG:
        import time
        start_time = time.perf_counter()
    
    out = mod.fused_forward(states, g_row.to(torch.int32), q1_row.to(torch.int32), q2_row.to(torch.int32),
                            rz1_t.view(B,-1) if rz1_t.numel()>0 else rz1_t,
                            rz2_t.view(B,-1) if rz2_t.numel()>0 else rz2_t,
                            rx1_t.view(B,-1) if rx1_t.numel()>0 else rx1_t,
                            rx2_t.view(B,-1) if rx2_t.numel()>0 else rx2_t)
    
    if 'FUSED_TIMING_DEBUG' in globals() and FUSED_TIMING_DEBUG:
        torch.cuda.synchronize()  # 确保GPU计算完成
        end_time = time.perf_counter()
        fused_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    if not _FUSED_USED_PRINTED:
        print('[INFO] fused base+noise kernel active (float-only)')
        if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS:
            print(f'[FUSED-DEBUG] First execution: B={B}, Lb={Lb}, states_shape={states.shape}')
            if 'FUSED_TIMING_DEBUG' in globals() and FUSED_TIMING_DEBUG:
                print(f'[FUSED-DEBUG] Execution time: {fused_time:.3f}ms')
        _FUSED_USED_PRINTED = True
    return out

def _try_fused_segment(states, g_seg, q1_seg, q2_seg, rz1_seg, rz2_seg, rx1_seg, rx2_seg):
    """Attempt fused execution for a contiguous base-only segment (no param gates inside).
    g_seg, q1_seg, q2_seg: [B, Ls] gate id & qubit tensors (same across K dimension)
    rz1_seg, rz2_seg: [B, Ls] per-step rz noise for q1/q2 (or None / empty)
    Returns True if fused applied.
    Preconditions: states: [B,K,D] complex64, CUDA; segment length Ls>0.
    NOTE: current fused kernel only supports RZ noise (not RX). Caller must ensure
    there is no RX noise in the segment (or it is zero) before invoking.
    """
    if not USE_FUSED_BASE_NOISE:
        return False
    
    if states.device.type != 'cuda':
        return False
        
    try:
        mod=_ensure_fused_extension()
    except Exception as e:
        if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS:
            print(f"[FUSED-SEGMENT] Extension load failed: {e}")
        return False
    # Flatten per-batch segment so kernel can read entry [b*Ls + t]
    B = g_seg.size(0); Ls = g_seg.size(1)
    g_flat  = g_seg.to(torch.int32).contiguous().view(-1)
    q1_flat = q1_seg.to(torch.int32).contiguous().view(-1)
    q2_flat = q2_seg.to(torch.int32).contiguous().view(-1)
    def flat_or_empty(x):
        if x is None:
            return torch.empty(0, device=states.device, dtype=torch.float32)
        return x.contiguous().view(-1)
    rz1_flat = flat_or_empty(rz1_seg)
    rz2_flat = flat_or_empty(rz2_seg)
    rx1_flat = flat_or_empty(rx1_seg)
    rx2_flat = flat_or_empty(rx2_seg)
    if 'DEBUG_ZERO_RX_NOISE' in globals() and DEBUG_ZERO_RX_NOISE and rx1_flat.numel()>0:
        rx1_flat.zero_(); rx2_flat.zero_()
    # Safety: ensure no invalid 2-qubit gates with q2 < 0 (would cause undefined bit shift in CUDA)
    if ((g_seg == BASE_GATES['cx']) | (g_seg == BASE_GATES['cz'])).any():
        bad_mask = ((g_seg == BASE_GATES['cx']) | (g_seg == BASE_GATES['cz'])) & (q2_seg < 0)
        if bad_mask.any():
            idxs = bad_mask.nonzero(as_tuple=False)[:5]
            print('[FUSED][WARN] Found 2-qubit gate(s) with q2 < 0 inside segment, disabling fusion for this run. Examples:', idxs.tolist())
            globals()['USE_FUSED_BASE_NOISE'] = False
            return False
    debug_compare = ('DEBUG_VALIDATE_FUSED' in globals() and DEBUG_VALIDATE_FUSED and B>0 and Ls>0)
    if debug_compare:
        # 保存 segment 起始态
        ref_start = states.detach().clone()
    # 添加时间测量（如果启用调试）
    if 'FUSED_TIMING_DEBUG' in globals() and FUSED_TIMING_DEBUG:
        import time
        start_time = time.perf_counter()
    
    states = mod.fused_forward(states,
                               g_flat, q1_flat, q2_flat,
                               rz1_flat.view(B, Ls) if rz1_flat.numel()>0 else rz1_flat,
                               rz2_flat.view(B, Ls) if rz2_flat.numel()>0 else rz2_flat,
                               rx1_flat.view(B, Ls) if rx1_flat.numel()>0 else rx1_flat,
                               rx2_flat.view(B, Ls) if rx2_flat.numel()>0 else rx2_flat,
                               Ls)
    
    if 'FUSED_TIMING_DEBUG' in globals() and FUSED_TIMING_DEBUG:
        torch.cuda.synchronize()  # 确保GPU计算完成
        end_time = time.perf_counter()
        segment_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    # 更新全局统计
    global _FUSED_SEGMENT_COUNT, _FUSED_TOTAL_TIME
    _FUSED_SEGMENT_COUNT += 1
    if 'FUSED_TIMING_DEBUG' in globals() and FUSED_TIMING_DEBUG:
        _FUSED_TOTAL_TIME += segment_time
    
    # 仅在详细诊断模式下输出执行信息
    if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS:
        print(f'[FUSED-SEGMENT] ✓ KERNEL EXECUTED: B={B}, Ls={Ls}, count={_FUSED_SEGMENT_COUNT}')
        if 'FUSED_TIMING_DEBUG' in globals() and FUSED_TIMING_DEBUG:
            print(f'[FUSED-SEGMENT] Time: {segment_time:.3f}ms, total: {_FUSED_TOTAL_TIME:.3f}ms')
    if debug_compare:
        # 用 Python 逐步回放参考结果
        ref_states = ref_start
        # reconstruct splits once (n comes from D=2^n -> n = log2(D))
        D = states.size(-1); n = int(math.log2(D))
        # We need splits; reuse global make_splits if exists
        try:
            make_splits_fn = globals().get('make_splits', None)
            if make_splits_fn is not None:
                splits_ref = make_splits_fn(n, states.device)
            else:
                # minimal fallback splits construction
                idx_all = torch.arange(D, device=states.device)
                splits_ref = []
                for qb in range(n):
                    sel0 = ((idx_all >> qb) & 1)==0
                    i0 = idx_all[sel0]
                    i1 = i0 | (1<<qb)
                    splits_ref.append((i0,i1))
        except Exception:
            splits_ref = None
        for t in range(Ls):
            # apply base gate step per sample
            gate_ids_step = g_seg[:, t]
            q1_step = q1_seg[:, t]; q2_step = q2_seg[:, t]
            # emulate single gate per sample like reference path
            # build masks
            for bi in range(B):
                gid = int(gate_ids_step[bi].item())
                if gid < 0: continue
                a1 = int(q1_step[bi].item()); a2 = int(q2_step[bi].item())
                if gid == BASE_GATES['h'] or gid == BASE_GATES['x'] or gid == BASE_GATES['z']:
                    i0,i1 = splits_ref[a1]
                    a = ref_states[bi:bi+1, :, i0]; b = ref_states[bi:bi+1, :, i1]
                    if gid == BASE_GATES['h']:
                        new0 = (a + b)/math.sqrt(2); new1 = (a - b)/math.sqrt(2)
                    elif gid == BASE_GATES['x']:
                        new0, new1 = b, a
                    else: # z
                        new0, new1 = a, -b
                    ref_states[bi:bi+1, :, i0] = new0
                    ref_states[bi:bi+1, :, i1] = new1
                elif gid == BASE_GATES['cx']:
                    if a2 < 0: continue
                    # swap when control=1 target=0
                    # brute force iterate amplitudes (small D)
                    cb = 1<<a1; tb = 1<<a2
                    idx = torch.arange(D, device=ref_states.device)
                    sel = ((idx & cb)!=0) & ((idx & tb)==0)
                    i0 = idx[sel]; i1 = i0 | tb
                    tmp = ref_states[bi:bi+1,:,i0].clone()
                    ref_states[bi:bi+1,:,i0] = ref_states[bi:bi+1,:,i1]
                    ref_states[bi:bi+1,:,i1] = tmp
                elif gid == BASE_GATES['cz']:
                    if a2 < 0: continue
                    cb = 1<<a1; tb = 1<<a2
                    idx = torch.arange(D, device=ref_states.device)
                    sel = ((idx & cb)!=0) & ((idx & tb)!=0)
                    ref_states[bi:bi+1,:,idx[sel]] *= -1
            # noise for step t
            if rz1_flat.numel()>0:
                # indexes in flattened noise: b*Ls + t
                for bi in range(B):
                    base = bi*Ls + t
                    ang_rz_q1 = rz1_flat[base].item() if rz1_flat.numel()==B*Ls else 0.0
                    ang_rx_q1 = rx1_flat[base].item() if rx1_flat.numel()==B*Ls else 0.0
                    a1 = int(q1_seg[bi, t].item())
                    if int(g_seg[bi, t].item()) >= 0:
                        # RZ
                        if ang_rz_q1 != 0.0 and a1 >=0:
                            i0,i1 = splits_ref[a1]
                            h=0.5*ang_rz_q1
                            c=math.cos(h); s=math.sin(h)
                            rs = ref_states[bi:bi+1,:,:]
                            # apply to bit=0 -> e^{-i h}; bit=1 -> e^{+i h}
                            rs[:,:,i0] = rs[:,:,i0]*complex(c, -s)
                            rs[:,:,i1] = rs[:,:,i1]*complex(c,  s)
                        # RX
                        if ang_rx_q1 != 0.0 and a1 >=0:
                            i0,i1 = splits_ref[a1]
                            h=0.5*ang_rx_q1; c=math.cos(h); s=math.sin(h)
                            rs = ref_states[bi:bi+1,:,:]
                            a = rs[:,:,i0].clone(); b = rs[:,:,i1].clone()
                            # new_a = c*a + (-i s)*b ; new_b = (-i s)*a + c*b
                            new_a = c*a + (-1j*s)*b
                            new_b = (-1j*s)*a + c*b
                            rs[:,:,i0] = new_a; rs[:,:,i1] = new_b
            # q2 noise
            if rz2_flat.numel()>0:
                for bi in range(B):
                    base = bi*Ls + t
                    a2 = int(q2_seg[bi, t].item())
                    if a2 < 0: continue
                    ang_rz_q2 = rz2_flat[base].item() if rz2_flat.numel()==B*Ls else 0.0
                    ang_rx_q2 = rx2_flat[base].item() if rx2_flat.numel()==B*Ls else 0.0
                    if ang_rz_q2!=0.0:
                        i0,i1 = splits_ref[a2]
                        h=0.5*ang_rz_q2; c=math.cos(h); s=math.sin(h)
                        rs = ref_states[bi:bi+1,:,:]
                        rs[:,:,i0] = rs[:,:,i0]*complex(c, -s)
                        rs[:,:,i1] = rs[:,:,i1]*complex(c,  s)
                    if ang_rx_q2!=0.0:
                        i0,i1 = splits_ref[a2]
                        h=0.5*ang_rx_q2; c=math.cos(h); s=math.sin(h)
                        rs = ref_states[bi:bi+1,:,:]
                        a = rs[:,:,i0].clone(); b = rs[:,:,i1].clone()
                        new_a = c*a + (-1j*s)*b
                        new_b = (-1j*s)*a + c*b
                        rs[:,:,i0] = new_a; rs[:,:,i1] = new_b
        diff = (states - ref_states).abs()
        max_diff = diff.max().item()
        if max_diff > 1e-5:
            # 额外诊断：归一化差异、最大差异位置
            # 取第一个 batch & 第一个随机初态的差异，列出前若干幅度索引
            with torch.no_grad():
                st_f = states[0,0]; st_r = ref_states[0,0]
                norm_f = (st_f.conj()*st_f).real.sum().item()
                norm_r = (st_r.conj()*st_r).real.sum().item()
                # top differing amplitudes
                flat_diff = (st_f - st_r).abs()
                topk = torch.topk(flat_diff, k=min(8, flat_diff.numel()))
                top_indices = topk.indices.tolist()
                top_vals = [float(v) for v in topk.values]
            print(f"[SEG-DEBUG] fused segment diff max={max_diff:.3e} Ls={Ls} norm_f={norm_f:.6f} norm_ref={norm_r:.6f} top_idx={top_indices} top_diff={top_vals} (disable fusion)")
            # 直接回滚到参考态，防止误差扩散
            states.copy_(ref_states)
            globals()['USE_FUSED_BASE_NOISE'] = False
    return True

random.seed(SEED); torch.manual_seed(SEED)

BASE_GATES={'h':0,'x':1,'z':2,'cx':3,'cz':4}
PARAM_GATES={'rz':0,'rx':1}
INV_BASE={v:k for k,v in BASE_GATES.items()}
INV_PARAM={v:k for k,v in PARAM_GATES.items()}
PAD_ID=-1

# ===================== Noise model =====================
def _build_noise_schedule(item:dict):
    Lb=len(item['base_gates'])
    if not USE_NOISE:
        zeros=[0.0]*Lb
        return dict(rx_q1=zeros, rz_q1=zeros, rx_q2=zeros, rz_q2=zeros)
    rx_q1=[]; rz_q1=[]; rx_q2=[]; rz_q2=[]
    for _ in range(Lb):
        rx1=(random.random()*2-1)*NOISE_X_RAD if random.random()<NOISE_DELTA_X else 0.0
        rz1=(random.random()*2-1)*NOISE_Z_RAD if random.random()<NOISE_DELTA_Z else 0.0
        rx2=(random.random()*2-1)*NOISE_X_RAD if random.random()<NOISE_DELTA_X else 0.0
        rz2=(random.random()*2-1)*NOISE_Z_RAD if random.random()<NOISE_DELTA_Z else 0.0
        rx_q1.append(rx1); rz_q1.append(rz1); rx_q2.append(rx2); rz_q2.append(rz2)
    return dict(rx_q1=rx_q1, rz_q1=rz_q1, rx_q2=rx_q2, rz_q2=rz_q2)

# ===== Optimized scalar-angle variants (support 0-dim tensors) =====
def _ensure_scalar_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device, dtype=torch.float32)

def _apply_rz_scalar(batch_states: torch.Tensor, q: int, angle, splits):
    angle = _ensure_scalar_tensor(angle, batch_states.device)
    if torch.all(angle==0): return
    i0, i1 = splits[q]
    em = torch.exp(-0.5j * angle)
    ep = torch.exp(0.5j * angle)
    batch_states[:, i0] *= em
    batch_states[:, i1] *= ep

def _apply_rx_scalar(batch_states: torch.Tensor, q: int, angle, splits):
    angle = _ensure_scalar_tensor(angle, batch_states.device)
    if torch.all(angle==0): return
    i0, i1 = splits[q]
    c = torch.cos(0.5 * angle)
    s = -1j * torch.sin(0.5 * angle)
    s0 = batch_states[:, i0]
    s1 = batch_states[:, i1]
    batch_states[:, i0] = c * s0 + s * s1
    batch_states[:, i1] = s * s0 + c * s1

# ================= Dataset =================
class CircuitDataset(Dataset):
    def __init__(self,path:str):
        self.items: list[dict] = []
        self._next_index = 0
        if not os.path.exists(path):
            print(f"[WARN] Data path does not exist: {path}")
            return
        def process_obj(o:dict):
            # New token format
            if 'base_circuit_tokens' in o and 'pqc_circuit_tokens' in o:
                base_tokens = o['base_circuit_tokens']
                pqc_tokens  = o['pqc_circuit_tokens']
                base_gates=[]; base_q1=[]; base_q2=[]
                for tok in base_tokens:
                    g=tok[0]; qs=tok[1]
                    if g not in BASE_GATES: continue
                    if len(qs)==1:
                        q1=qs[0]; q2=-1
                    elif len(qs)>=2:
                        q1,q2=qs[0],qs[1]
                    else:
                        continue
                    base_gates.append(g); base_q1.append(q1); base_q2.append(q2)
                param_gates=[]; param_qubits=[]; after_list=[]; param_angles=[]
                base_ptr=0; last_base_idx=-1
                def is_same_base(tok, idx):
                    if idx>=len(base_gates): return False
                    g=tok[0]; qs=tok[1]
                    if g!=base_gates[idx]: return False
                    bq1=base_q1[idx]; bq2=base_q2[idx]
                    if len(qs)==1: return qs[0]==bq1 and bq2==-1
                    if len(qs)>=2: return qs[0]==bq1 and qs[1]==bq2
                    return False
                for tok in pqc_tokens:
                    g=tok[0]; qs=tok[1]; params = tok[2] if len(tok)>2 else []
                    if is_same_base(tok, base_ptr):
                        last_base_idx=base_ptr; base_ptr+=1; continue
                    if g in PARAM_GATES:
                        q = qs[0] if qs else 0
                        ang = params[0] if params else 0.0
                        param_gates.append(g); param_qubits.append(q); after_list.append(last_base_idx); param_angles.append(ang)
                n_q=o.get('n_qubits')
                if n_q is None:
                    qs_all=[*base_q1,*[q for q in base_q2 if q>=0],*param_qubits]
                    n_q=(max(qs_all)+1) if qs_all else 1
                if len(base_gates)>MAX_BASE_LEN:
                    base_gates=base_gates[:MAX_BASE_LEN]; base_q1=base_q1[:MAX_BASE_LEN]; base_q2=base_q2[:MAX_BASE_LEN]
                if len(param_gates)>MAX_PARAM:
                    param_gates=param_gates[:MAX_PARAM]; param_qubits=param_qubits[:MAX_PARAM]; after_list=after_list[:MAX_PARAM]; param_angles=param_angles[:MAX_PARAM]
                self.items.append(dict(idx=self._next_index, base_gates=base_gates, base_q1=base_q1, base_q2=base_q2,
                                       param_gates=param_gates, param_qubits=param_qubits,
                                       after=after_list, param_angles_gt=param_angles, n_qubits=n_q))
                self._next_index += 1
                return
            # Old format
            base_g=o['base_gates']; bq=o['base_qubits']
            if len(bq)!=2: raise ValueError('base_qubits must be [q1_list, q2_list]')
            param_g=o.get('param_gates',[]); param_q=o.get('param_qubits',[])
            after=o.get('after',[-1]*len(param_g)); ang=o.get('pqc_angles_gt',[0.0]*len(param_g))
            if not (len(param_g)==len(param_q)==len(after)==len(ang)):
                raise ValueError('parameter list length mismatch')
            if len(base_g)>MAX_BASE_LEN:
                base_g=base_g[:MAX_BASE_LEN]; bq=[bq[0][:MAX_BASE_LEN], bq[1][:MAX_BASE_LEN]]
            if len(param_g)>MAX_PARAM:
                param_g=param_g[:MAX_PARAM]; param_q=param_q[:MAX_PARAM]; after=after[:MAX_PARAM]; ang=ang[:MAX_PARAM]
            n_q=o.get('n_qubits')
            if n_q is None:
                qs=[*bq[0],*bq[1],*param_q]; qs=[q for q in qs if q>=0]; n_q=(max(qs)+1) if qs else 1
            self.items.append(dict(idx=self._next_index, base_gates=base_g, base_q1=bq[0], base_q2=bq[1],
                                   param_gates=param_g, param_qubits=param_q,
                                   after=after, param_angles_gt=ang, n_qubits=n_q))
            self._next_index += 1

        if os.path.isdir(path):
            files=[f for f in os.listdir(path) if f.lower().endswith(('.json','.jsonl'))]
            files.sort()
            iterator = tqdm(files, desc='Reading data files', unit='file') if tqdm else files
            for fname in iterator:
                fp=os.path.join(path,fname)
                try:
                    with open(fp,'r',encoding='utf-8') as fh:
                        for line in fh:
                            if not line.strip(): continue
                            process_obj(json.loads(line))
                            break
                except Exception as e:
                    print(f"[WARN] Failed to read file {fp}: {e}")
            if tqdm:
                print(f"[INFO] Loaded samples: {len(self.items)}")
        else:
            with open(path,'r',encoding='utf-8') as f:
                lines=f.readlines()
            iterator=tqdm(lines, desc='Reading lines', unit='line') if tqdm else lines
            for line in iterator:
                if not line.strip(): continue
                process_obj(json.loads(line))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

@dataclass
class Batch:
    base_g:torch.Tensor; base_q1:torch.Tensor; base_q2:torch.Tensor
    param_g:torch.Tensor; param_q:torch.Tensor; param_after:torch.Tensor
    param_angles_gt:torch.Tensor; base_len:torch.Tensor; param_len:torch.Tensor; n_qubits:torch.Tensor; idx:torch.Tensor
    def to(self,device):
        for k,v in self.__dict__.items():
            if isinstance(v,torch.Tensor): setattr(self,k,v.to(device))
        return self

def _pad(seq,pad,L):
    seq=list(seq); return seq[:L]+[pad]*max(0,L-len(seq))

def collate(samples:List[dict]):
    bg=[]; bq1=[]; bq2=[]; pg=[]; pq=[]; pafter=[]; pang=[]; base_l=[]; param_l=[]; nqs=[]; idxs=[]
    for o in samples:
        g=[BASE_GATES[x] for x in o['base_gates']]; p=[PARAM_GATES[x] for x in o['param_gates']]
        bg.append(_pad(g,PAD_ID,MAX_BASE_LEN)); bq1.append(_pad(o['base_q1'],PAD_ID,MAX_BASE_LEN)); bq2.append(_pad(o['base_q2'],PAD_ID,MAX_BASE_LEN))
        pg.append(_pad(p,PAD_ID,MAX_PARAM)); pq.append(_pad(o['param_qubits'],PAD_ID,MAX_PARAM))
        pafter.append(_pad(o['after'],-999,MAX_PARAM)); pang.append(_pad(o['param_angles_gt'],0.0,MAX_PARAM))
        base_l.append(len(g)); param_l.append(len(p)); nqs.append(o['n_qubits']); idxs.append(o['idx'])
    to_long=lambda x: torch.tensor(x,dtype=torch.long)
    return Batch(to_long(bg),to_long(bq1),to_long(bq2),to_long(pg),to_long(pq),to_long(pafter),
                 torch.tensor(pang,dtype=torch.float32),to_long(base_l),to_long(param_l),to_long(nqs),to_long(idxs))

# ================= Model =================
class AnglePredictor(nn.Module):
    def __init__(self):
        super().__init__(); d=EMB_DIM
        self.base_emb=nn.Embedding(len(BASE_GATES)+1,d,padding_idx=len(BASE_GATES))
        self.qubit_emb=nn.Embedding(MAX_QUBITS+1,d); self.pos_emb=nn.Embedding(MAX_BASE_LEN,d)
        layer=nn.TransformerEncoderLayer(d,NUM_HEADS,FF_DIM,DROP,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,NUM_LAYERS)
        self.param_type_emb=nn.Embedding(len(PARAM_GATES)+1,d,padding_idx=len(PARAM_GATES))
        self.param_pos_emb=nn.Embedding(MAX_PARAM,d); self.query_proj=nn.Linear(d,d)
        self.attn=nn.MultiheadAttention(d,NUM_HEADS,dropout=DROP,batch_first=True); self.out=nn.Linear(d,2)
        # 改善输出层初始化，使初始角度接近0
        with torch.no_grad():
            self.out.weight.data.normal_(0, 0.01)
            self.out.bias.data.zero_()
    def forward(self,b:Batch):
        ids=b.base_g.clone(); mask=(ids==PAD_ID); ids[mask]=len(BASE_GATES)
        x=self.base_emb(ids)+self.qubit_emb(torch.clamp(b.base_q1,0,MAX_QUBITS))+self.qubit_emb(torch.clamp(b.base_q2,0,MAX_QUBITS))
        pos=torch.arange(x.size(1),device=x.device); x=x+self.pos_emb(pos)[None]; x=self.encoder(x,src_key_padding_mask=mask)
        p=b.param_g.clone(); pmask=(p==PAD_ID); p[pmask]=len(PARAM_GATES)
        q=self.param_type_emb(p)+self.param_pos_emb(torch.arange(p.size(1),device=p.device))[None]; q=self.query_proj(q)
        attn,_=self.attn(q,x,x,key_padding_mask=mask)
        return self.out(attn), pmask

# ================= Minimal simulator =================
_SPLIT_CACHE: Dict[Tuple[int, torch.device], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
_CX_SWAP_CACHE: Dict[Tuple[int, torch.device], Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]]] = {}
_CZ_MASK_CACHE: Dict[Tuple[int, torch.device], Dict[Tuple[int,int], torch.Tensor]] = {}

def _get_two_qubit_struct(n:int, device:torch.device):
    key=(n,device)
    if key in _CX_SWAP_CACHE and key in _CZ_MASK_CACHE:
        return _CX_SWAP_CACHE[key], _CZ_MASK_CACHE[key]
    dim = 1 << n
    idx_all = torch.arange(dim, device=device)
    cx_swap = {}; cz_mask = {}
    for c in range(n):
        for t in range(n):
            if c == t: continue
            cb = 1 << c; tb = 1 << t
            sel = ((idx_all & cb) != 0) & ((idx_all & tb) == 0)
            i0 = idx_all[sel]; i1 = i0 | tb
            cx_swap[(c, t)] = (i0, i1)
            sel_cz = ((idx_all & cb) != 0) & ((idx_all & tb) != 0)
            cz_mask[(c, t)] = idx_all[sel_cz]
    _CX_SWAP_CACHE[key] = cx_swap; _CZ_MASK_CACHE[key] = cz_mask
    return cx_swap, cz_mask
def _split_indices(n,device):
    k=(n,device)
    if k in _SPLIT_CACHE: return _SPLIT_CACHE[k]
    dim=1<<n; ar=torch.arange(dim,device=device); out=[]
    for q in range(n):
        bit=(ar>>q)&1; out.append(((bit==0).nonzero(as_tuple=False).squeeze(-1),(bit==1).nonzero(as_tuple=False).squeeze(-1)))
    _SPLIT_CACHE[k]=out; return out

def _apply_const_1q(st,q,kind,splits):
    i0,i1=splits[q]; s0=st[...,i0]; s1=st[...,i1]
    if kind=='h': n0=(s0+s1)/math.sqrt(2); n1=(s0-s1)/math.sqrt(2)
    elif kind=='x': n0,n1=s1,s0
    elif kind=='z': n0,n1=s0,-s1
    else: raise ValueError(kind)
    st[...,i0]=n0; st[...,i1]=n1

def _apply_rz(st,q,a,splits):
    i0,i1=splits[q]; em=torch.exp(-0.5j*a).unsqueeze(-1); ep=torch.exp(0.5j*a).unsqueeze(-1); st[...,i0]*=em; st[...,i1]*=ep

def _apply_rx(st,q,a,splits):
    i0,i1=splits[q]; c=torch.cos(0.5*a).unsqueeze(-1); s=-1j*torch.sin(0.5*a).unsqueeze(-1); s0=st[...,i0]; s1=st[...,i1]; st[...,i0]=c*s0+s*s1; st[...,i1]=s*s0+c*s1

def _apply_cx(st,c,t): dim=st.size(-1); idx=torch.arange(dim,device=st.device); mc=1<<c; mt=1<<t; sel=((idx&mc)!=0)&((idx&mt)==0); i0=idx[sel]; i1=i0|mt; tmp=st[...,i0].clone(); st[...,i0]=st[...,i1]; st[...,i1]=tmp
def _apply_cz(st,q1,q2): dim=st.size(-1); idx=torch.arange(dim,device=st.device); mask=((idx&(1<<q1))!=0)&((idx&(1<<q2))!=0); st[...,idx[mask]]=-st[...,idx[mask]]

def sincos_to_angle(sc):
    # sc: [B, L, 2] or [L,2]
    sc = sc / (sc.norm(dim=-1,keepdim=True)+1e-8)
    return torch.atan2(sc[...,0], sc[...,1])

def angle_supervise_loss(pred,gt,mask):
    if gt is None: return torch.tensor(0.0,device=pred.device)
    valid=~mask
    if valid.sum()==0: return torch.tensor(0.0,device=pred.device)
    sc=pred/(pred.norm(dim=-1,keepdim=True)+1e-9); ang=torch.atan2(sc[...,0],sc[...,1]); diff=torch.angle(torch.exp(1j*(ang-gt))); return (diff[valid]**2).mean()

# ===================== Standalone single-circuit simulator (no noise, supports rz/rx) =====================
def simulate_single_circuit_no_noise(
    circuit_ops: Sequence[Tuple[str, Sequence[int], Optional[float]]],
    num_qubits: int,
    input_state,
    device: torch.device | None = None,
    big_endian_wires: bool = True,
) -> torch.Tensor:
    """Simulate a single circuit using existing gate helpers (h/x/z/cx/cz + rz/rx) without noise.

    This mirrors correctness.py::simulate_with_0924_kernel interface, but reuses the internal
    helper functions (_apply_const_1q, _apply_rz_scalar, _apply_rx_scalar, _apply_cx, _apply_cz).

    Args:
        circuit_ops: iterable of (gate, wires, param_or_None). Wires given externally as either
            big-endian (wire 0 = MSB, default) or little-endian (set big_endian_wires=False).
        num_qubits: total qubits.
        input_state: 1D complex array-like length 2**num_qubits.
        device: target torch.device (defaults to global DEVICE or CUDA if available).
        big_endian_wires: whether provided wires follow PennyLane-style MSB ordering.

    Returns:
        torch.Tensor: final state vector (complex64) shape [2**num_qubits] on 'device'.
    """
    if device is None:
        device = DEVICE
    st = torch.as_tensor(input_state, dtype=DTYPE, device=device).clone()
    if st.ndim != 1:
        raise ValueError("input_state must be 1D")
    dim_expected = 1 << num_qubits
    if st.numel() != dim_expected:
        raise ValueError(f"State length {st.numel()} != 2**{num_qubits}")

    # For parameter gate helpers we expect shape [B, D]; use B=1 wrapper
    st_batch = st.unsqueeze(0)  # [1, 2^n]
    splits = _split_indices(num_qubits, device)

    for gate, wires, param in circuit_ops:
        if not isinstance(wires, (list, tuple)):
            raise ValueError("wires must be a sequence")
        mapped = []
        for w in wires:
            w_int = int(w)
            if w_int < 0 or w_int >= num_qubits:
                raise ValueError(f"wire {w_int} out of range [0,{num_qubits-1}]")
            if big_endian_wires:
                # external MSB -> internal little-endian
                mapped.append(num_qubits - 1 - w_int)
            else:
                mapped.append(w_int)

        if gate in ('h','x','z'):
            if len(mapped) != 1:
                raise ValueError(f"Gate {gate} expects 1 wire")
            # _apply_const_1q expects shape [..., D]; our st_batch works
            _apply_const_1q(st_batch, mapped[0], gate, splits)
        elif gate == 'cx':
            if len(mapped) != 2:
                raise ValueError("cx expects 2 wires [control, target]")
            _apply_cx(st_batch, mapped[0], mapped[1])
        elif gate == 'cz':
            if len(mapped) != 2:
                raise ValueError("cz expects 2 wires [q1, q2]")
            _apply_cz(st_batch, mapped[0], mapped[1])
        elif gate == 'rz':
            if len(mapped) != 1:
                raise ValueError("rz expects 1 wire")
            if param is None:
                raise ValueError("rz gate missing angle")
            _apply_rz_scalar(st_batch, mapped[0], float(param), splits)
        elif gate == 'rx':
            if len(mapped) != 1:
                raise ValueError("rx expects 1 wire")
            if param is None:
                raise ValueError("rx gate missing angle")
            _apply_rx_scalar(st_batch, mapped[0], float(param), splits)
        else:
            raise ValueError(f"Unsupported gate: {gate}")

    return st_batch[0]

# ===================== Precompute (vectorized) =====================
def build_base_cache_vectorized(dataset: CircuitDataset):
    iterator = tqdm(dataset.items, desc='[fast] grouping', unit='sample') if tqdm else dataset.items
    groups: dict[int, list[dict]] = {}
    for it in iterator:
        groups.setdefault(it['n_qubits'], []).append(it)

    init_states_per_n: dict[int, torch.Tensor] = {}
    ref_states_per_idx: dict[int, torch.Tensor] | dict = {}
    noise_schedules: dict[int, dict] | dict = {}
    device = DEVICE
    ref_states_packed = None
    ref_idx2row = {}
    for n, items in groups.items():
        dim = 1 << n
        Bn = len(items)
        if Bn == 0: continue
        L_max = max(len(it['base_gates']) for it in items)
        # Build gate tensors on CPU then H2D once
        gate_ids_cpu = torch.full((Bn, L_max), PAD_ID, dtype=torch.long)
        q1_cpu      = torch.full((Bn, L_max), -1, dtype=torch.long)
        q2_cpu      = torch.full((Bn, L_max), -1, dtype=torch.long)
        sample_idx_list = []
        for bi, it in enumerate(items):
            sample_idx_list.append(it['idx'])
            Lb_i = len(it['base_gates'])
            if Lb_i == 0: continue
            gate_ids_row = [BASE_GATES[g] for g in it['base_gates']]
            gate_ids_cpu[bi, :Lb_i] = torch.tensor(gate_ids_row, dtype=torch.long)
            q1_cpu[bi, :Lb_i] = torch.tensor(it['base_q1'], dtype=torch.long)
            q2_cpu[bi, :Lb_i] = torch.tensor(it['base_q2'], dtype=torch.long)

        gate_ids = gate_ids_cpu.to(device, non_blocking=True)
        q1 = q1_cpu.to(device, non_blocking=True)
        q2 = q2_cpu.to(device, non_blocking=True)

        # shared init states for this n
        if n not in init_states_per_n:
            splits_tmp = _split_indices(n, device)
            states_init = []
            for _ in range(K_RANDOM):
                st = torch.zeros(dim, dtype=DTYPE, device=device); st[0] = 1+0j
                for qb in range(n):
                    r = random.random()
                    if r < 0.33: pass
                    elif r < 0.66: _apply_const_1q(st.unsqueeze(0), qb, 'x', splits_tmp)
                    else: _apply_const_1q(st.unsqueeze(0), qb, 'h', splits_tmp)
                states_init.append(st)
            init_states_per_n[n] = torch.stack(states_init, 0)  # [K, 2^n]

        states = init_states_per_n[n].unsqueeze(0).expand(Bn, -1, -1).clone()
        splits = _split_indices(n, device)
        idx_all = torch.arange(dim, device=device)
        cx_swap = {}
        cz_mask = {}
        for c in range(n):
            for t in range(n):
                if c == t: continue
                cb = 1 << c; tb = 1 << t
                sel = ((idx_all & cb) != 0) & ((idx_all & tb) == 0)
                i0 = idx_all[sel]; i1 = i0 | tb
                cx_swap[(c, t)] = (i0, i1)
                sel_cz = ((idx_all & cb) != 0) & ((idx_all & tb) != 0)
                cz_mask[(c, t)] = idx_all[sel_cz]

        with torch.no_grad():
            for t in range(L_max):
                g_t = gate_ids[:, t]
                if (g_t == PAD_ID).all(): break
                # 1q groups
                for gcode, gname in ((BASE_GATES['h'], 'h'), (BASE_GATES['x'], 'x'), (BASE_GATES['z'], 'z')):
                    mask = (g_t == gcode)
                    if not mask.any(): continue
                    qubits = q1[mask, t]
                    batches = mask.nonzero(as_tuple=False).squeeze(-1)
                    uq = qubits.unique()
                    for qb in uq.tolist():
                        sel = batches[(qubits == qb)]
                        if sel.numel() == 0: continue
                        i0, i1 = splits[qb]
                        states_sel = states.index_select(0, sel)
                        a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
                        if gname == 'h':
                            new0 = (a + b) / math.sqrt(2); new1 = (a - b) / math.sqrt(2)
                        elif gname == 'x':
                            new0, new1 = b, a
                        else:
                            new0, new1 = a, -b
                        states_sel[:, :, i0] = new0
                        states_sel[:, :, i1] = new1
                        states[sel] = states_sel
                # 2q groups
                for gcode, gname in ((BASE_GATES['cx'], 'cx'), (BASE_GATES['cz'], 'cz')):
                    mask = (g_t == gcode)
                    if not mask.any(): continue
                    c_list = q1[mask, t]; t_list = q2[mask, t]
                    batches = mask.nonzero(as_tuple=False).squeeze(-1)
                    pairs = torch.stack([c_list, t_list], dim=1)
                    uniq_pairs, inv_idx = torch.unique(pairs, dim=0, return_inverse=True)
                    for pi, (c_val, t_val) in enumerate(uniq_pairs.tolist()):
                        sel = batches[inv_idx == pi]
                        if sel.numel() == 0: continue
                        if gname == 'cx':
                            i0, i1 = cx_swap[(c_val, t_val)]
                            states_sel = states.index_select(0, sel)
                            tmp = states_sel[:, :, i0].clone()
                            states_sel[:, :, i0] = states_sel[:, :, i1]
                            states_sel[:, :, i1] = tmp
                            states[sel] = states_sel
                        else:
                            m_idx = cz_mask[(c_val, t_val)]
                            states_sel = states.index_select(0, sel)
                            states_sel[:, :, m_idx] = -states_sel[:, :, m_idx]
                            states[sel] = states_sel

        # store reference
        if PACK_REF_STATES:
            if ref_states_packed is None:
                ref_states_packed = torch.empty(len(dataset.items), K_RANDOM, dim, dtype=DTYPE, device=device)
            for bi, sample_idx in enumerate(sample_idx_list):
                row = sample_idx
                ref_states_packed[row].copy_(states[bi])
                ref_idx2row[sample_idx] = row
        else:
            for bi, sample_idx in enumerate(sample_idx_list):
                ref_states_per_idx[sample_idx] = states[bi].clone()

    # tensor-mode noise schedules
    if FAST_NOISE_SCHEDULE:
        items_all = dataset.items
        idx_list = [it['idx'] for it in items_all]
        L_per_sample = [len(it['base_gates']) for it in items_all]
        L_max_global = max(L_per_sample) if L_per_sample else 0
        B_total = len(items_all)
        device = DEVICE
        q2_mat = torch.full((B_total, L_max_global), -1, dtype=torch.long, device=device)
        gate_mask = torch.zeros((B_total, L_max_global), dtype=torch.bool, device=device)
        for row, it in enumerate(items_all):
            Lb = len(it['base_gates'])
            gate_mask[row, :Lb] = True
            if Lb>0:
                q2_vals = torch.tensor(it['base_q2'], dtype=torch.long, device=device)
                q2_mat[row, :Lb] = q2_vals
        if USE_NOISE:
            rx_flag1 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_X) & gate_mask
            rx_amp1  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_X_RAD
            rx_q1 = torch.where(rx_flag1, rx_amp1, torch.zeros(1, device=device))
            rz_flag1 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_Z) & gate_mask
            rz_amp1  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_Z_RAD
            rz_q1 = torch.where(rz_flag1, rz_amp1, torch.zeros(1, device=device))
            valid_q2 = (q2_mat >= 0) & gate_mask
            rx_flag2 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_X) & valid_q2
            rx_amp2  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_X_RAD
            rx_q2 = torch.where(rx_flag2, rx_amp2, torch.zeros(1, device=device))
            rz_flag2 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_Z) & valid_q2
            rz_amp2  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_Z_RAD
            rz_q2 = torch.where(rz_flag2, rz_amp2, torch.zeros(1, device=device))
        else:
            rx_q1 = rz_q1 = rx_q2 = rz_q2 = torch.zeros(B_total, L_max_global, device=device)
        idx2row = {idx: i for i, idx in enumerate(idx_list)}
        noise_schedules = dict(tensor_mode=True, rx_q1=rx_q1, rz_q1=rz_q1, rx_q2=rx_q2, rz_q2=rz_q2,
                               idx2row=idx2row, L_max=L_max_global)

    if PACK_REF_STATES:
        ref_states_per_idx = dict(packed=True, tensor=ref_states_packed, idx2row=ref_idx2row)

    return init_states_per_n, ref_states_per_idx, noise_schedules

# ===================== Vectorized training kernel (shared PQC) =====================

def _apply_base_step_batched(states, gate_ids_step, q1_step, q2_step, splits, cx_swap, cz_mask):
    """Apply base gates at one step for a group: vectorized across samples sharing n."""
    # Ensure tensors are on the same device and have compatible dtypes
    gate_ids_step = gate_ids_step.contiguous()
    
    # 1q: h/x/z
    for gcode, gname in ((BASE_GATES['h'],'h'), (BASE_GATES['x'],'x'), (BASE_GATES['z'],'z')):
        # Force CPU computation for mask to avoid kernel compatibility issues
        try:
            mask = (gate_ids_step == gcode)
        except RuntimeError as e:
            # Fallback: move to CPU for comparison, then back to GPU
            gate_cpu = gate_ids_step.cpu()
            mask = (gate_cpu == gcode).to(gate_ids_step.device)
        if not mask.any(): continue
        qubits = q1_step[mask]
        batches = mask.nonzero(as_tuple=False).squeeze(-1)
        uq = qubits.unique()
        for qb in uq.tolist():
            sel = batches[(qubits == qb)]
            if sel.numel()==0: continue
            i0, i1 = splits[qb]
            states_sel = states.index_select(0, sel)
            a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
            if gname == 'h':
                new0 = (a + b)/math.sqrt(2); new1 = (a - b)/math.sqrt(2)
            elif gname == 'x':
                new0, new1 = b, a
            else:
                new0, new1 = a, -b
            states_sel[:, :, i0] = new0
            states_sel[:, :, i1] = new1
            states[sel] = states_sel
    # 2q: cx/cz
    for gcode, gname in ((BASE_GATES['cx'],'cx'), (BASE_GATES['cz'],'cz')):
        mask = (gate_ids_step == gcode)
        if not mask.any(): continue
        c_list = q1_step[mask]; t_list = q2_step[mask]
        batches = mask.nonzero(as_tuple=False).squeeze(-1)
        pairs = torch.stack([c_list, t_list], dim=1)
        uniq_pairs, inv_idx = torch.unique(pairs, dim=0, return_inverse=True)
        for pi, (c_val, t_val) in enumerate(uniq_pairs.tolist()):
            sel = batches[inv_idx == pi]
            if sel.numel()==0: continue
            if gname == 'cx':
                i0, i1 = cx_swap[(c_val, t_val)]
                states_sel = states.index_select(0, sel)
                tmp = states_sel[:, :, i0].clone()
                states_sel[:, :, i0] = states_sel[:, :, i1]
                states_sel[:, :, i1] = tmp
                states[sel] = states_sel
            else:
                m_idx = cz_mask[(c_val, t_val)]
                states_sel = states.index_select(0, sel)
                states_sel[:, :, m_idx] = -states_sel[:, :, m_idx]
                states[sel] = states_sel

def _apply_noise_step_batched(states, q1_step, q2_step, rx1, rz1, rx2, rz2, splits):
    """Apply sparse Rx/Rz noise after this base step, per-sample qubit; vectorized by grouping same qubit."""
    # qubit 1
    uq = q1_step.unique()
    for qb in uq.tolist():
        mask = (q1_step == qb)
        if not mask.any(): continue
        sel = mask.nonzero(as_tuple=False).squeeze(-1)
        states_sel = states.index_select(0, sel)
        # rz then rx, both per-sample different angles
        ang_rz = rz1[sel]  # [B_sel]
        if ang_rz.abs().sum() != 0:
            i0,i1 = splits[qb]
            em = torch.exp(-0.5j * ang_rz)[:, None, None]
            ep = torch.exp(0.5j  * ang_rz)[:, None, None]
            states_sel[:, :, i0] = states_sel[:, :, i0] * em
            states_sel[:, :, i1] = states_sel[:, :, i1] * ep
        ang_rx = rx1[sel]
        if ang_rx.abs().sum() != 0:
            i0,i1 = splits[qb]
            c = torch.cos(0.5*ang_rx)[:, None, None]
            s = -1j*torch.sin(0.5*ang_rx)[:, None, None]
            a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
            states_sel[:, :, i0] = c*a + s*b
            states_sel[:, :, i1] = s*a + c*b
        states[sel] = states_sel
    # qubit 2 (only if present)
    valid_q2 = (q2_step >= 0)
    if valid_q2.any():
        q2_vals = q2_step[valid_q2]
        uq2 = q2_vals.unique()
        base_idx = valid_q2.nonzero(as_tuple=False).squeeze(-1)
        for qb in uq2.tolist():
            mask_local = (q2_vals == qb)
            sel = base_idx[mask_local]  # indices in batch
            states_sel = states.index_select(0, sel)
            ang_rz = rz2[sel]
            if ang_rz.abs().sum() != 0:
                i0,i1 = splits[qb]
                em = torch.exp(-0.5j * ang_rz)[:, None, None]
                ep = torch.exp(0.5j  * ang_rz)[:, None, None]
                states_sel[:, :, i0] = states_sel[:, :, i0] * em
                states_sel[:, :, i1] = states_sel[:, :, i1] * ep
            ang_rx = rx2[sel]
            if ang_rx.abs().sum() != 0:
                i0,i1 = splits[qb]
                c = torch.cos(0.5*ang_rx)[:, None, None]
                s = -1j*torch.sin(0.5*ang_rx)[:, None, None]
                a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
                states_sel[:, :, i0] = c*a + s*b
                states_sel[:, :, i1] = s*a + c*b
            states[sel] = states_sel

def _apply_params_step_shared_structure(states, angles_all, t, param_pos, param_kind, param_qubit, splits):
    """Apply all param gates at step t (shared structure across samples). Keeps autograd for angles.
       angles_all: [B, Lp] (already atan2 of sin/cos logits)
    """
    I_t = (param_pos == t).nonzero(as_tuple=False).squeeze(-1)
    if I_t.numel() == 0:
        return states
    # RZ group
    I_rz = I_t[(param_kind[I_t] == PARAM_GATES['rz'])]
    if I_rz.numel():
        q_rz = param_qubit[I_rz]                 # [Nr]
        uq, inv = torch.unique(q_rz, return_inverse=True)
        ang_rz = angles_all[:, I_rz]             # [B, Nr]
        for i, q in enumerate(uq.tolist()):
            sel = (inv == i)
            ang_q = ang_rz[:, sel].sum(dim=1)    # [B]
            i0, i1 = splits[q]
            em = torch.exp(-0.5j * ang_q)[:, None, None]
            ep = torch.exp(0.5j  * ang_q)[:, None, None]
            states[:, :, i0] = states[:, :, i0] * em
            states[:, :, i1] = states[:, :, i1] * ep
    # RX group
    I_rx = I_t[(param_kind[I_t] == PARAM_GATES['rx'])]
    if I_rx.numel():
        q_rx = param_qubit[I_rx]
        uq, inv = torch.unique(q_rx, return_inverse=True)
        ang_rx = angles_all[:, I_rx]             # [B, Nx]
        for i, q in enumerate(uq.tolist()):
            sel = (inv == i)
            ang_q = ang_rx[:, sel].sum(dim=1)    # [B]
            i0, i1 = splits[q]
            c = torch.cos(0.5*ang_q)[:, None, None]
            s = -1j*torch.sin(0.5*ang_q)[:, None, None]
            a = states[:, :, i0]; b = states[:, :, i1]
            states[:, :, i0] = c*a + s*b
            states[:, :, i1] = s*a + c*b
    return states

def simulate_loss_cached_vectorized_samepqc(batch: Batch, logits, init_cache, ref_cache, noise_schedules):
    """Vectorized replay assuming all samples share the same PQC structure (positions/types identical).
       - Groups by n_qubits
       - Base+noise executed in no_grad (not in graph)
       - Param gates per step applied batched with shared structure (angles keep grad)
    """
    assert isinstance(noise_schedules, dict) and noise_schedules.get('tensor_mode', False), \
        "Require FAST_NOISE_SCHEDULE tensor-mode noise schedules."

    B = batch.base_g.size(0)
    device = logits.device

    # angles once: [B, MAX_PARAM, 2] -> [B, Lp]
    Lp_list = batch.param_len.tolist()
    Lp = max(Lp_list) if len(set(Lp_list))==1 else min(Lp_list)
    angles_all = sincos_to_angle(logits[:, :Lp, :])  # [B, Lp]

    # shared PQC structure from the first sample
    param_pos  = batch.param_after[0, :Lp].to(device)
    param_kind = batch.param_g[0, :Lp].to(device)     # 0: rz, 1: rx
    param_qubit= batch.param_q[0, :Lp].to(device)

    # group by n_qubits inside this batch
    nvals = batch.n_qubits.tolist()
    groups: Dict[int, torch.Tensor] = {}
    for i, n in enumerate(nvals):
        groups.setdefault(n, []).append(i)
    losses = []

    for n, idx_list in groups.items():
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)  # [Bg]
        Bg = idx_tensor.numel()
        dim = 1 << n
        splits = _split_indices(n, device)
        # shared init clone (no H2D)
        states = init_cache[n].to(device).unsqueeze(0).expand(Bg, -1, -1).clone()  # [Bg, K, 2^n]
        # reference states
        if isinstance(ref_cache, dict) and ref_cache.get('packed', False):
            rows = torch.tensor([ref_cache['idx2row'][int(batch.idx[i])] for i in idx_list], device=device, dtype=torch.long)
            ref = ref_cache['tensor'].index_select(0, rows)  # [Bg, K, 2^n]
        else:
            ref = torch.stack([ref_cache[int(batch.idx[i])].to(device) for i in idx_list], dim=0)

        # per-group base tensors for this step
        Lb_list = [int(batch.base_len[i]) for i in idx_list]
        Lb_max = max(Lb_list)
        gate_ids = batch.base_g.index_select(0, idx_tensor)[:, :Lb_max].to(device)   # [Bg, Lb]
        q1       = batch.base_q1.index_select(0, idx_tensor)[:, :Lb_max].to(device)
        q2       = batch.base_q2.index_select(0, idx_tensor)[:, :Lb_max].to(device)

        idx_all = torch.arange(dim, device=device)
        cx_swap = {}
        cz_mask = {}
        for c in range(n):
            for t in range(n):
                if c == t: continue
                cb = 1 << c; tb = 1 << t
                sel = ((idx_all & cb) != 0) & ((idx_all & tb) == 0)
                i0 = idx_all[sel]; i1 = i0 | tb
                cx_swap[(c, t)] = (i0, i1)
                sel_cz = ((idx_all & cb) != 0) & ((idx_all & tb) != 0)
                cz_mask[(c, t)] = idx_all[sel_cz]

        noise_rows = torch.tensor([noise_schedules['idx2row'][int(batch.idx[i])] for i in idx_list],
                                  device=device, dtype=torch.long)

        angles_grp = angles_all.index_select(0, idx_tensor)  # [Bg, Lp]

        for t in range(Lb_max):
            g_t = gate_ids[:, t]
            all_pad = (g_t == PAD_ID).all()
            if all_pad: break

            # Base gates (no_grad)
            with torch.no_grad():
                q1_t = q1[:, t]; q2_t = q2[:, t]
                _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)

                # Noise (tensor-mode), apply per sample/qubit
                rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, t]
                rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, t]
                rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, t]
                rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, t]
                if USE_NOISE:
                    _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)

            # Param gates (keep graph)
            if DIFF_FIDELITY:
                states = _apply_params_step_shared_structure(states, angles_grp, t, param_pos, param_kind, param_qubit, splits)

        ov = (ref.conj() * states).sum(-1)   # [Bg, K]
        F = (ov.abs()**2).mean()
        losses.append(1 - F)

    return torch.stack(losses).mean()

# ===================== 新的共享结构快速参数模拟 =====================
def build_shared_param_layout(example_batch: Batch, device: torch.device | None = None):
    """基于一个 batch 构造参数门位置 -> RZ/RX 在该 step 的 qubit 分组。"""
    param_pos = example_batch.param_after[0]  # [Lp]
    param_kind = example_batch.param_g[0]
    param_qubit = example_batch.param_q[0]
    Lb = int(example_batch.base_len[0].item())
    Lp = int(example_batch.param_len[0].item())
    layout = [dict(rz={}, rx={}) for _ in range(Lb)]
    for i in range(Lp):
        step = int(param_pos[i].item())
        if step < 0 or step >= Lb:
            continue
        kind = int(param_kind[i].item())  # 0=rz,1=rx
        qb = int(param_qubit[i].item())
        if kind == PARAM_GATES['rz']:
            layout_step = layout[step]['rz']
        else:
            layout_step = layout[step]['rx']
        layout_step.setdefault(qb, []).append(i)
    for step_dict in layout:
        for k in ('rz','rx'):
            for qb, lst in step_dict[k].items():
                t = torch.tensor(lst, dtype=torch.long)
                if device is not None:
                    t = t.to(device)
                step_dict[k][qb] = t
    return layout

def simulate_loss_shared_param_fast(batch: Batch, logits, init_cache, ref_cache, noise_schedules, param_layout, angles_cache=None):
    """假设: 所有 sample n_qubits 相同 & PQC 结构一致。"""
    device = logits.device
    B = batch.base_g.size(0)
    n = int(batch.n_qubits[0].item())
    assert (batch.n_qubits == n).all(), "FAST_PARAM_SIM 要求 batch 内 n_qubits 完全相同"
    dim = 1 << n
    splits = _split_indices(n, device)

    # K 随机初态 -> [B,K,2^n]
    states = init_cache[n].to(device).unsqueeze(0).expand(B, -1, -1).clone()  # [B,K,D]

    # 参考态
    if isinstance(ref_cache, dict) and ref_cache.get('packed', False):
        rows = torch.tensor([ref_cache['idx2row'][int(idx)] for idx in batch.idx.tolist()], device=device)
        ref = ref_cache['tensor'].index_select(0, rows)
    else:
        ref = torch.stack([ref_cache[int(i.item())] for i in batch.idx], dim=0).to(device)

    # angles 计算
    Lp = int(batch.param_len[0].item())
    sc = logits[:, :Lp, :]
    sc = sc / (sc.norm(dim=-1, keepdim=True) + 1e-8)
    angles_all = torch.atan2(sc[...,0], sc[...,1])  # [B, Lp]

    # 基座
    Lb = int(batch.base_len[0].item())
    gate_ids = batch.base_g[:, :Lb].to(device)
    q1 = batch.base_q1[:, :Lb].to(device)
    q2 = batch.base_q2[:, :Lb].to(device)

    # 预构造 2q gate 辅助索引
    cx_swap, cz_mask = _get_two_qubit_struct(n, device)

    noise_rows = torch.tensor([noise_schedules['idx2row'][int(i.item())] for i in batch.idx], device=device)

    # 主循环：每个 base step
    for t in range(Lb):
        g_t = gate_ids[:, t]
        if (g_t == PAD_ID).all():
            break
        with torch.no_grad():
            q1_t = q1[:, t]; q2_t = q2[:, t]
            _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
            # 噪声
            if USE_NOISE:
                rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, t]
                rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, t]
                rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, t]
                rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, t]
                _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)

        # 参数门 (保持梯度)
        if DIFF_FIDELITY:
            layout_step = param_layout[t]
            # 直接原位修改 (states 不需要梯度, 避免 clone)
            for qb, idxs in layout_step['rz'].items():
                if idxs.device != angles_all.device:
                    idxs = idxs.to(angles_all.device)
                ang_q = angles_all.index_select(1, idxs).sum(dim=1)
                if torch.allclose(ang_q, torch.zeros_like(ang_q)):
                    continue
                i0, i1 = splits[qb]
                em = torch.exp(-0.5j * ang_q)[:, None, None]
                ep = torch.exp(0.5j  * ang_q)[:, None, None]
                states[:, :, i0] = states[:, :, i0] * em
                states[:, :, i1] = states[:, :, i1] * ep
            for qb, idxs in layout_step['rx'].items():
                if idxs.device != angles_all.device:
                    idxs = idxs.to(angles_all.device)
                ang_q = angles_all.index_select(1, idxs).sum(dim=1)
                if torch.allclose(ang_q, torch.zeros_like(ang_q)):
                    continue
                i0, i1 = splits[qb]
                c = torch.cos(0.5*ang_q)[:, None, None]
                s = -1j*torch.sin(0.5*ang_q)[:, None, None]
                a = states[:, :, i0]; b = states[:, :, i1]
                states[:, :, i0] = c*a + s*b
                states[:, :, i1] = s*a + c*b

    ov = (ref.conj() * states).sum(-1)  # [B,K]
    F = (ov.abs()**2).mean()
    return 1 - F

# ===================== 更进一步: Scatter 聚合参数角度 (减少 Python 循环) =====================
def simulate_loss_shared_param_fast_agg(batch: Batch, logits, init_cache, ref_cache, noise_schedules, param_layout):
    """与 simulate_loss_shared_param_fast 类似，但将所有 (step, kind, qubit) 的角度在一次张量运算里聚合。
       假设 batch 内:
         - n_qubits 全部相同
         - param_layout 在所有 sample 间共享 (结构一致)
       聚合策略: 为每个 step,qubit 构造 rz_acc[step, qb], rx_acc[step, qb]
                 angles_all: [B, Lp]; layout 给出每个 step 下，每个 qubit 对应的一组 param gate indices (可能多个门叠加)
                 先用 0 张量 scatter_add 到 shape [Lp, step, qb]? 由于 layout 稀疏, 我们先构建映射表: gate_index -> (step, kind, qb)
                 然后一次性将 angles_all 转成 [B, Lp] 根据 gate_index 的映射 scatter_add 到 [B, 2, Lb, n_qubits] (channel 0=rz, 1=rx)
    """
    device = logits.device
    n = int(batch.n_qubits[0].item())
    assert (batch.n_qubits == n).all(), "simulate_loss_shared_param_fast_agg 要求 batch 内 n_qubits 同一值"
    Lb = int(batch.base_len[0].item())
    Lp = int(batch.param_len[0].item())
    B = batch.base_g.size(0)
    dim = 1 << n
    splits = _split_indices(n, device)

    # 初态 & 参考态
    states = init_cache[n].to(device).unsqueeze(0).expand(B, -1, -1).clone()  # [B,K,2^n]
    if isinstance(ref_cache, dict) and ref_cache.get('packed', False):
        rows = torch.tensor([ref_cache['idx2row'][int(i.item())] for i in batch.idx], device=device)
        ref = ref_cache['tensor'].index_select(0, rows)
    else:
        ref = torch.stack([ref_cache[int(i.item())] for i in batch.idx], dim=0).to(device)

    # angles: [B,Lp]
    sc = logits[:, :Lp, :]
    sc = sc / (sc.norm(dim=-1, keepdim=True) + 1e-8)
    angles_all = torch.atan2(sc[...,0], sc[...,1])  # [B,Lp]

    # ------- 构造 gate_index -> (step, kind_id, qubit) -------
    # kind_id: 0=rz,1=rx
    # layout[t]['rz'][qb] = tensor(indices)
    # layout[t]['rx'][qb] = tensor(indices)
    map_entries = []  # (gate_idx, step, kind_id, qb)
    for step in range(Lb):
        lay = param_layout[step]
        for qb, idxs in lay['rz'].items():
            if idxs.numel():
                map_entries.append((idxs.to(device), step, 0, qb))
        for qb, idxs in lay['rx'].items():
            if idxs.numel():
                map_entries.append((idxs.to(device), step, 1, qb))
    if not map_entries:
        # 无参数门，退化为基座+噪声 fidelity
        gate_ids = batch.base_g[:, :Lb].to(device)
        q1 = batch.base_q1[:, :Lb].to(device); q2 = batch.base_q2[:, :Lb].to(device)
        idx_all = torch.arange(dim, device=device)
        cx_swap = {}; cz_mask = {}
        for c in range(n):
            for t in range(n):
                if c==t: continue
                cb = 1<<c; tb = 1<<t
                sel = ((idx_all & cb)!=0) & ((idx_all & tb) == 0)
                i0=idx_all[sel]; i1=i0|tb; cx_swap[(c,t)] = (i0,i1)
                sel_cz = ((idx_all & cb)!=0) & ((idx_all & tb) != 0)
                cz_mask[(c,t)] = idx_all[sel_cz]
        noise_rows = torch.tensor([noise_schedules['idx2row'][int(i.item())] for i in batch.idx], device=device)
        for t in range(Lb):
            g_t = gate_ids[:, t]
            if (g_t == PAD_ID).all(): break
            with torch.no_grad():
                q1_t = q1[:, t]; q2_t = q2[:, t]
                _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
                if USE_NOISE:
                    rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, t]
                    rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, t]
                    rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, t]
                    rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, t]
                    _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)
        ov = (ref.conj()*states).sum(-1); F=(ov.abs()**2).mean(); return 1-F

    # 构造 scatter 索引
    total_param_positions = sum(me[0].numel() for me in map_entries)
    gate_flat = torch.empty(total_param_positions, dtype=torch.long, device=device)
    step_flat = torch.empty(total_param_positions, dtype=torch.long, device=device)
    kind_flat = torch.empty(total_param_positions, dtype=torch.long, device=device)
    qubit_flat= torch.empty(total_param_positions, dtype=torch.long, device=device)
    cursor=0
    for idxs, step, kind_id, qb in map_entries:
        n_local = idxs.numel()
        gate_flat[cursor:cursor+n_local] = idxs
        step_flat[cursor:cursor+n_local] = step
        kind_flat[cursor:cursor+n_local] = kind_id
        qubit_flat[cursor:cursor+n_local]= qb
        cursor += n_local

    # angles_all: [B, Lp] -> select gate_flat -> [B, P]
    selected_angles = angles_all.index_select(1, gate_flat)  # [B,P]
    # 我们需要聚合到 shape [B,2,Lb,n] ; scatter_add 用 view + expand
    acc = torch.zeros(B, 2, Lb, n, dtype=selected_angles.dtype, device=device)
    # 构造目标索引
    # expand selected_angles -> [B, P] -> scatter 到 acc[:, kind_flat, step_flat, qubit_flat]
    # 需把 (kind, step, qubit) 组合成线性下标
    linear_index = kind_flat * (Lb*n) + step_flat * n + qubit_flat  # [P]
    acc_flat = acc.view(B, 2*Lb*n)  # [B, 2*Lb*n]
    # scatter_add_ on dim=1 with linear_index
    acc_flat.scatter_add_(1, linear_index.unsqueeze(0).expand(B,-1), selected_angles)
    acc = acc_flat.view(B,2,Lb,n)

    # 取出基座张量，并按 step 交错执行 base+噪声 + 参数门，以保持拓扑顺序
    gate_ids = batch.base_g[:, :Lb].to(device)
    q1 = batch.base_q1[:, :Lb].to(device); q2 = batch.base_q2[:, :Lb].to(device)
    cx_swap, cz_mask = _get_two_qubit_struct(n, device)
    noise_rows = torch.tensor([noise_schedules['idx2row'][int(i.item())] for i in batch.idx], device=device)
    if USE_NOISE:
        rx1_all = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, :Lb]
        rz1_all = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, :Lb]
        rx2_all = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, :Lb]
        rz2_all = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, :Lb]
    else:
        rx1_all = rz1_all = rx2_all = rz2_all = None

    first_param = True  # 首次出现参数门时开启梯度链路
    # 分段：把没有参数门的连续 base 区段用 fused kernel 执行（已支持 RZ+RX 噪声）
    pending_start = None  # 开始累积的 segment (包括当前 t) 尚未执行
    def flush_segment(end_t_exclusive):
        nonlocal pending_start, states
        if pending_start is None:
            return
        s = pending_start; e = end_t_exclusive
        if e <= s:  # empty
            pending_start=None; return
        seg_len = e - s
        g_seg = gate_ids[:, s:e]
        if (g_seg == PAD_ID).all():
            pending_start=None; return
        
        # 详细的分段信息（仅在详细诊断模式下）
        if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS:
            print(f'[SEGMENT-DEBUG] Flushing segment [{s}:{e}] length={seg_len}')
            print(f'[SEGMENT-DEBUG]   g_seg shape: {g_seg.shape}')
            print(f'[SEGMENT-DEBUG]   unique gates: {g_seg.unique().tolist()}')
        if USE_NOISE:
            rz1_seg = rz1_all[:, s:e]; rz2_seg = rz2_all[:, s:e]
            rx1_seg = rx1_all[:, s:e]; rx2_seg = rx2_all[:, s:e]
        else:
            rz1_seg = rz2_seg = rx1_seg = rx2_seg = None
        global _TOTAL_SEGMENTS_ATTEMPTED, _FALLBACK_SEGMENTS
        _TOTAL_SEGMENTS_ATTEMPTED += 1
        ok = _try_fused_segment(states, g_seg, q1[:, s:e], q2[:, s:e], rz1_seg, rz2_seg, rx1_seg, rx2_seg)
        if not ok:
            _FALLBACK_SEGMENTS += 1
        
        # 仅在详细诊断模式下输出
        if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS:
            if ok:
                print(f'[SEGMENT-DEBUG] ✓ Fused execution SUCCESS for segment [{s}:{e}]')
            else:
                print(f'[SEGMENT-DEBUG] ✗ Fused execution FAILED for segment [{s}:{e}], using fallback')
        if ok and DEBUG_COMPARE_ONE_SEGMENT and 'did_segment_debug' not in globals():
            # 对比：复制一份 states_ref，在里面用逐步方式跑相同 segment，然后比较差异最大值
            import torch as _t
            states_ref = states.detach().clone()
            # 回滚该 segment 之前的状态: 无法直接回溯，改为提前保存。为了简单，只在第一个 segment 做预保存。
            # 所以如果 pending_start==0 才可信。否则跳过。
            if s==0:
                # 重新从初态重跑逐步版本到 e，再与 fused 后 states 对比
                # 由于我们这里只在第一个 segment 时执行，所以初态仍是 init_cache 的复制
                pass
            # 标记避免多次
            globals()['did_segment_debug']=True
        if not ok:
            # fallback 逐步执行
            with torch.no_grad():
                for tt in range(s,e):
                    g_t = gate_ids[:, tt]
                    if (g_t == PAD_ID).all(): break
                    q1_t = q1[:, tt]; q2_t = q2[:, tt]
                    _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
                    if USE_NOISE:
                        rx1_t = rx1_all[:, tt]; rz1_t = rz1_all[:, tt]; rx2_t = rx2_all[:, tt]; rz2_t = rz2_all[:, tt]
                        _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)
        pending_start=None

    segment_analysis = []  # 用于诊断分段情况
    for t in range(Lb):
        g_t = gate_ids[:, t]
        if (g_t == PAD_ID).all():
            flush_segment(t)
            break
        rz_step = acc[:,0,t,:]
        rx_step = acc[:,1,t,:]
        has_param_here = not ((rz_step.abs().sum()==0) and (rx_step.abs().sum()==0))
        segment_analysis.append(('param' if has_param_here else 'base', t))
        if not has_param_here:
            # 继续累积 segment
            if pending_start is None:
                pending_start = t
            continue
        # 有参数门：先 flush 当前 segment (不包含 t)
        flush_segment(t)
        # 执行当前 base step (不能 fuse 因为后面要立刻加参数门)
        with torch.no_grad():
            q1_t = q1[:, t]; q2_t = q2[:, t]
            _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
            if USE_NOISE:
                rx1_t = rx1_all[:, t]; rz1_t = rz1_all[:, t]; rx2_t = rx2_all[:, t]; rz2_t = rz2_all[:, t]
                _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)
        # 现在插入参数门
        if first_param:
            states = states.detach().clone(); states.requires_grad_(True); states = states + 0; first_param=False
        # RZ 参数门
        nz_rz_any = (rz_step.abs() > 0).any(dim=0)
        if nz_rz_any.any():
            for qb in nz_rz_any.nonzero(as_tuple=False).squeeze(-1).tolist():
                ang = rz_step[:, qb]
                if ang.abs().sum()==0: continue
                i0,i1 = splits[qb]
                em = torch.exp(-0.5j * ang)[:, None, None]
                ep = torch.exp(0.5j  * ang)[:, None, None]
                states[:, :, i0] = states[:, :, i0] * em
                states[:, :, i1] = states[:, :, i1] * ep
        # RX 参数门
        nz_rx_any = (rx_step.abs() > 0).any(dim=0)
        if nz_rx_any.any():
            for qb in nz_rx_any.nonzero(as_tuple=False).squeeze(-1).tolist():
                ang = rx_step[:, qb]
                if ang.abs().sum()==0: continue
                i0,i1 = splits[qb]
                c = torch.cos(0.5*ang)[:, None, None]
                s = -1j*torch.sin(0.5*ang)[:, None, None]
                a = states[:, :, i0]; b = states[:, :, i1]
                states[:, :, i0] = c*a + s*b
                states[:, :, i1] = s*a + c*b
    # 循环结束后还有尾部 segment
    flush_segment(Lb)
    
    # 输出分段分析（仅第一次）
    if 'VERBOSE_FUSED_DIAGNOSTICS' in globals() and VERBOSE_FUSED_DIAGNOSTICS and 'segment_analysis_printed' not in globals():
        try:
            base_segments = []
            current_start = None
            for step_type, step_idx in segment_analysis:
                if step_type == 'base':
                    if current_start is None:
                        current_start = step_idx
                else:  # param
                    if current_start is not None:
                        base_segments.append((current_start, step_idx))
                        current_start = None
            if current_start is not None:
                base_segments.append((current_start, Lb))
            
            print(f'[SEGMENT-ANALYSIS] Base segments for fused execution: {base_segments}')
            print(f'[SEGMENT-ANALYSIS] Total segments: {len(base_segments)}, steps: {[(end-start) for start, end in base_segments]}')
        except Exception as e:
            print(f'[SEGMENT-ANALYSIS] Error in analysis: {e}')
        globals()['segment_analysis_printed'] = True

    ov = (ref.conj()*states).sum(-1)
    F = (ov.abs()**2).mean()
    return 1 - F

# ===================== Train =====================
def train():
    global USE_FUSED_BASE_NOISE  # 声明全局变量，以便在调试代码中修改
    
    # GH200 optimized settings for PyTorch 2.5.1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache
        torch.backends.cudnn.benchmark = True  # Enable benchmark mode for performance
        print(f"[INFO] PyTorch 2.5.1 optimized for {torch.cuda.get_device_name(0)}")
    
    if not os.path.exists(DATA_PATH):
        print(f'[WARN] Data file not found: {DATA_PATH}')
        return
    ds=CircuitDataset(DATA_PATH)
    if len(ds)==0:
        print('[WARN] Empty dataset')
        return
    if PRECOMPUTE_BASE:
        print('[INFO] Precomputing random initial states and noiseless base references ...')
        t0=time.perf_counter()
        init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(ds)
        dt = time.perf_counter() - t0
        if isinstance(ref_cache, dict) and ref_cache.get('packed', False):
            print(f'[INFO] ref_cache packed tensor shape = {tuple(ref_cache["tensor"].shape)}')
        print(f'[INFO] Precompute done in {dt:.2f}s (FAST={FAST_BASE_CACHE})')
    else:
        init_cache = ref_cache = None
        noise_schedules = {}

    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
    model=AnglePredictor().to(DEVICE)
    
    # 检查模型参数初始化
    total_params = sum(p.numel() for p in model.parameters())
    total_norm = sum(p.data.norm().item() for p in model.parameters())
    print(f'[INFO] Model has {total_params} parameters, total norm: {total_norm:.6f}')

    # 优化器：大批量更稳的 betas
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.99),weight_decay=0.01)

    # ---- 逐步(step)学习率调度：线性 warmup + 余弦衰减 ----
    steps_per_epoch = math.ceil(len(ds) / BATCH_SIZE)
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMUP_STEPS = max(10, int(TOTAL_STEPS * WARMUP_RATIO))

    def _lr_lambda(step: int):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        rest = max(1, TOTAL_STEPS - WARMUP_STEPS)
        prog = (step - WARMUP_STEPS) / float(rest)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda) if USE_SCHEDULER else None
    global_step = 0

    # 预计算 param layout (只需一次)
    shared_param_layout = None
    if USE_PRECOMPUTED_PARAM_LAYOUT:
        tmp_loader = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)
        try:
            first_batch = next(iter(tmp_loader)).to(DEVICE)
            shared_param_layout = build_shared_param_layout(first_batch, device=DEVICE if DEVICE.type=='cuda' else None)
            print(f"[INFO] Built shared param layout with {len(shared_param_layout)} base steps")
            
            # 简单确认 fused kernel 状态
            if USE_FUSED_BASE_NOISE:
                try:
                    _ensure_fused_extension()
                    print('[INFO] Fused CUDA kernel: ✓ Available')
                except Exception as e:
                    print(f'[INFO] Fused CUDA kernel: ✗ Failed ({str(e)[:50]}...)')
            else:
                print('[INFO] Fused CUDA kernel: Disabled')
        except StopIteration:
            pass

    if COMPILE_MODEL and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print('[INFO] model compiled via torch.compile')
        except Exception as e:
            print(f'[WARN] torch.compile failed: {e}')

    for ep in range(1,EPOCHS+1):
        model.train(); total=0.0
        epoch_iter = enumerate(loader, start=1)
        epoch_iter = tqdm(epoch_iter, total=len(loader), desc=f'Epoch {ep}', unit='batch')

        # 计时累积
        time_acc = dict(fwd=0.0, sim=0.0, aux=0.0, backward=0.0, step=0.0, batch_total=0.0)
        import time as _time

        # CUDA events (可选)
        if USE_CUDA_EVENTS and DEVICE.type == 'cuda':
            # 复用事件对，逐 batch 记录并同步累计
            ev_fwd_start = torch.cuda.Event(enable_timing=True); ev_fwd_end = torch.cuda.Event(enable_timing=True)
            ev_sim_start = torch.cuda.Event(enable_timing=True); ev_sim_end = torch.cuda.Event(enable_timing=True)
            ev_bwd_start = torch.cuda.Event(enable_timing=True); ev_bwd_end = torch.cuda.Event(enable_timing=True)
            gpu_time_acc = dict(fwd=0.0, sim=0.0, backward=0.0)
        else:
            gpu_time_acc = None

        for bi, raw in epoch_iter:
            t_batch0 = _time.perf_counter()
            batch=raw.to(DEVICE)

            # forward (model)
            t0=_time.perf_counter()
            if USE_CUDA_EVENTS and DEVICE.type=='cuda':
                ev_fwd_start.record()
            logits,mask=model(batch)  # logits: [B, MAX_PARAM, 2]
            if USE_CUDA_EVENTS and DEVICE.type=='cuda':
                ev_fwd_end.record()
            t1=_time.perf_counter()

            # simulation
            t2=_time.perf_counter()
            if USE_CUDA_EVENTS and DEVICE.type=='cuda':
                ev_sim_start.record()
            if PRECOMPUTE_BASE:
                if FAST_PARAM_SIM and shared_param_layout is not None:
                    if AGGREGATE_PARAM_ANGLES:
                        main = simulate_loss_shared_param_fast_agg(batch, logits, init_cache, ref_cache, noise_schedules, shared_param_layout)
                    else:
                        main = simulate_loss_shared_param_fast(batch, logits, init_cache, ref_cache, noise_schedules, shared_param_layout)
                else:
                    main = simulate_loss_cached_vectorized_samepqc(batch, logits, init_cache, ref_cache, noise_schedules)
                # ---- Debug: 验证 fused 正确性（前若干 batch）----
                if 'DEBUG_VALIDATE_FUSED' in globals() and DEBUG_VALIDATE_FUSED and ep==1 and bi <= 3:
                    if USE_FUSED_BASE_NOISE:
                        orig_flag = USE_FUSED_BASE_NOISE
                        USE_FUSED_BASE_NOISE = False
                        with torch.no_grad():
                            ref_main = simulate_loss_shared_param_fast_agg(batch, logits, init_cache, ref_cache, noise_schedules, shared_param_layout) if AGGREGATE_PARAM_ANGLES else simulate_loss_shared_param_fast(batch, logits, init_cache, ref_cache, noise_schedules, shared_param_layout)
                        USE_FUSED_BASE_NOISE = orig_flag
                        diff = (main.detach()-ref_main).abs().item()
                        print(f"[DEBUG][ep1 step{bi}] fused_loss={main.item():.6f} ref_loss={ref_main.item():.6f} diff={diff:.3e}")
                        if diff > 5e-5:
                            print('[WARN] fused diff exceeds 5e-5 -> disabling fusion for rest of run')
                            USE_FUSED_BASE_NOISE = False
            else:
                raise RuntimeError("Please enable PRECOMPUTE_BASE for the vectorized training kernel.")
            if USE_CUDA_EVENTS and DEVICE.type=='cuda':
                ev_sim_end.record()
            t3=_time.perf_counter()

            # aux
            t4=_time.perf_counter()
            if AUX_ANGLE_LOSS:
                aux=angle_supervise_loss(logits,batch.param_angles_gt,mask)
            else:
                aux=torch.tensor(0.0,device=logits.device)
            t5=_time.perf_counter()

            loss=main + AUX_ANGLE_WEIGHT*aux

            # backward + step（逐步调度）
            t6=_time.perf_counter()
            opt.zero_grad(set_to_none=True)
            if USE_CUDA_EVENTS and DEVICE.type=='cuda':
                ev_bwd_start.record()
            loss.backward()
            

            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step()
            if sch is not None:
                sch.step()
            global_step += 1
            if USE_CUDA_EVENTS and DEVICE.type=='cuda':
                ev_bwd_end.record()
            t7=_time.perf_counter()

            total+=loss.item()
            t_batch1 = _time.perf_counter()

            # accumulate times
            if MEASURE_BATCH_TIMES:
                time_acc['fwd'] += (t1-t0)
                time_acc['sim'] += (t3-t2)
                time_acc['aux'] += (t5-t4)
                time_acc['backward'] += (t7-t6)
                time_acc['batch_total'] += (t_batch1 - t_batch0)
                time_acc['step'] += (t7 - t6)

            # 记录 GPU event 时间 (需要在本次 iteration 发出所有 kernel 后同步一次)
            if gpu_time_acc is not None:
                # 同步以确保事件完成，然后立刻累计（真实 per-batch GPU 时间）
                torch.cuda.synchronize()
                gpu_time_acc['fwd'] += ev_fwd_start.elapsed_time(ev_fwd_end) / 1000.0  # ms->s
                gpu_time_acc['sim'] += ev_sim_start.elapsed_time(ev_sim_end) / 1000.0
                gpu_time_acc['backward'] += ev_bwd_start.elapsed_time(ev_bwd_end) / 1000.0

            # 调试：检查梯度情况
            if bi <= 10 or bi % 500 == 0:  # 前10个batch和每500个batch
                grad_norm = 0.0
                param_with_grad = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        param_with_grad += 1
                grad_norm = grad_norm ** 0.5
                print(f'[DEBUG] Batch {bi}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}, lr={opt.param_groups[0]["lr"]:.2e}')
            
            # 延迟日志，减少 .item() 同步
            if (bi % LOG_INTERVAL == 0) or (bi == len(loader)) or (not REDUCE_SYNC):
                lr_val = opt.param_groups[0]['lr']
                loss_val = float(loss.detach())
                main_val = float(main.detach())
                aux_val = float(aux.detach())
                epoch_iter.set_postfix(lr=f'{lr_val:.2e}', loss=f'{loss_val:.4f}', main=f'{main_val:.4f}', aux=f'{aux_val:.4f}', sim=f'{(t3-t2):.3f}s')

        avg = total/len(loader)
        print(f'[Epoch {ep}] avg_loss={avg:.6f}')
        if MEASURE_BATCH_TIMES:
            nb = len(loader)
            print('[Timing] per-batch averages: ' + ', '.join(
                f"{k}={v/nb:.4f}s" for k,v in time_acc.items()
            ))
        if gpu_time_acc is not None:
            nb_batches = len(loader)
            print('[GPU Timing] per-batch averages: ' + ', '.join(
                f"{k}={v/nb_batches:.4f}s" for k,v in gpu_time_acc.items()
            ))
    torch.save({'model':model.state_dict()},'minimal_model.pt'); print('Model saved -> minimal_model.pt')
    
    # 输出 fused kernel 统计信息
    print('=' * 60)
    print('[FUSED STATS] Final Summary:')
    print(f'[FUSED STATS]   Total segments attempted: {_TOTAL_SEGMENTS_ATTEMPTED}')
    print(f'[FUSED STATS]   Fused segments executed: {_FUSED_SEGMENT_COUNT}')
    print(f'[FUSED STATS]   Fallback segments: {_FALLBACK_SEGMENTS}')
    if _TOTAL_SEGMENTS_ATTEMPTED > 0:
        fused_rate = (_FUSED_SEGMENT_COUNT / _TOTAL_SEGMENTS_ATTEMPTED) * 100
        print(f'[FUSED STATS]   Fused success rate: {fused_rate:.1f}%')
    
    if _FUSED_SEGMENT_COUNT > 0:
        print(f'[FUSED STATS]   Total fused time: {_FUSED_TOTAL_TIME:.3f}ms')
        print(f'[FUSED STATS]   Avg time/segment: {_FUSED_TOTAL_TIME/_FUSED_SEGMENT_COUNT:.3f}ms')
        print('[FUSED STATS]   🚀 FUSED KERNEL WAS USED!')
    else:
        print('[FUSED STATS]   ⚠️  NO FUSED KERNELS EXECUTED - USING FALLBACK ONLY!')
    print('=' * 60)

if __name__=='__main__':
    train()
