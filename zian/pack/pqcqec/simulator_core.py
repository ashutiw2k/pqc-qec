#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core simulator and data utilities extracted from newest_work_correct.
Provides:
- Dataset/Batch/Loader utilities
- Minimal complex state simulator (H/X/Z/CX/CZ + RZ/RX)
- Vectorized base-cache precompute with K random initial states and tensor-mode noise
- Loss simulation for two training modes:
  * Shared-PQC param layout (Transformer predicts per-parameter angles)
  * Fixed-interval PQC blocks (Direct angle optimization)

This module intentionally avoids loading/compiling any CUDA extensions. It uses
pure PyTorch code and will run on both CPU and CUDA. Callers can pass device.
"""
from __future__ import annotations

import math, os, json, random
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

def _ensure_msvc_env_windows():
    """Attempt to load MSVC developer environment into current process on Windows.
    Safe no-op on non-Windows or if cl/nvcc already available.
    """
    try:
        if os.name != 'nt':
            return
        import shutil
        if shutil.which('cl') and shutil.which('nvcc'):
            return
        pf86 = os.environ.get('ProgramFiles(x86)')
        if not pf86:
            return
        vswhere = os.path.join(pf86, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
        vs_install = None
        if os.path.isfile(vswhere):
            try:
                out = subprocess.check_output([vswhere, '-latest', '-requires', 'Microsoft.Component.MSBuild', '-property', 'installationPath'], encoding='utf-8', stderr=subprocess.STDOUT)
                vs_install = out.strip().splitlines()[0] if out.strip() else None
            except Exception:
                vs_install = None
        candidates = []
        if vs_install:
            candidates.append(os.path.join(vs_install, 'Common7', 'Tools', 'VsDevCmd.bat'))
        candidates.append(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
        vsdev = None
        for c in candidates:
            if os.path.isfile(c):
                vsdev = c; break
        if not vsdev:
            return
        cmd = f'"{vsdev}" -arch=x64 -host_arch=x64 >nul && set'
        out = subprocess.check_output(['cmd.exe', '/V:ON', '/C', cmd], encoding='utf-8', stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            if k.upper() in ('PATH', 'INCLUDE', 'LIB', 'LIBPATH') or k.startswith('VCTools') or k.startswith('WindowsSDK'):
                os.environ[k] = v
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home and shutil.which('nvcc') is None:
            os.environ['PATH'] = os.path.join(cuda_home, 'bin') + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        return

# ---- Basic settings (caller may override via function args) ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
DTYPE = torch.complex64
K_RANDOM_DEFAULT = 32
PAD_ID = -1

BASE_GATES = {"h": 0, "x": 1, "z": 2, "cx": 3, "cz": 4}
PARAM_GATES = {"rz": 0, "rx": 1}

# Optional toggles
# Safer fused toggle (match newest_work semantics):
# - Default OFF on Windows to avoid inline build hangs; ON elsewhere by default
# - Allow override via TKFS_USE_FUSED_BASE_NOISE or PQC_USE_FUSED
_is_win = (os.name == 'nt') or platform.system().lower().startswith('win')
_default_fused = '0' if _is_win else '1'
_tkfs_env = os.environ.get('TKFS_USE_FUSED_BASE_NOISE')
_pqc_env = os.environ.get('PQC_USE_FUSED')
_env_val = (_tkfs_env if _tkfs_env is not None else (_pqc_env if _pqc_env is not None else _default_fused))
try:
    USE_FUSED_BASE_NOISE = bool(int(str(_env_val)))
except Exception:
    USE_FUSED_BASE_NOISE = False

# ---- Noise settings (dense jitter semantics) ----
class NoiseConfig:
    def __init__(self,
                 use_noise: bool = True,
                 noise_x_rad: float = math.pi/10,
                 noise_z_rad: float = math.pi/10,
                 noise_delta_x: float = 0.0,
                 noise_delta_z: float = 0.0):
        self.use_noise = use_noise
        self.noise_x_rad = float(noise_x_rad)
        self.noise_z_rad = float(noise_z_rad)
        self.noise_delta_x = float(noise_delta_x)
        self.noise_delta_z = float(noise_delta_z)


def _delta_fraction(val: float) -> float:
    if val <= 0:
        return 0.0
    if val <= 1:
        return float(val)
    return float(val) / 100.0

# ---- Dataset / Batch ----
class CircuitDataset(Dataset):
    def __init__(self, path: str, max_base_len: int = 1000, max_param: int = 1500, max_qubits: int = 5, num_sample: Optional[int] = None):
        self.items: List[dict] = []
        self._next_index = 0
        self._max_base_len = max_base_len
        self._max_param = max_param
        self._max_qubits = max_qubits
        self._num_limit = int(num_sample) if (num_sample is not None) else None

        class _EarlyStop(Exception):
            pass

        def pad(seq, pad, L):
            seq = list(seq)
            return seq[:L] + [pad] * max(0, L - len(seq))

        def process_obj(o: dict):
            # Prefer new token format
            if 'base_circuit_tokens' in o and 'pqc_circuit_tokens' in o:
                base_tokens = o['base_circuit_tokens']
                pqc_tokens  = o['pqc_circuit_tokens']
                base_gates: List[str] = []
                base_q1: List[int] = []
                base_q2: List[int] = []
                for tok in base_tokens:
                    g = tok[0]; qs = tok[1]
                    if g not in BASE_GATES:
                        continue
                    if len(qs) == 1:
                        q1 = qs[0]; q2 = -1
                    elif len(qs) >= 2:
                        q1, q2 = qs[0], qs[1]
                    else:
                        continue
                    base_gates.append(g); base_q1.append(q1); base_q2.append(q2)
                # Map PQC tokens that are not part of base into param lists
                param_gates: List[str] = []
                param_qubits: List[int] = []
                after_list: List[int] = []
                param_angles: List[float] = []
                base_ptr = 0; last_base_idx = -1

                def is_same_base(tok, idx):
                    if idx >= len(base_gates):
                        return False
                    g = tok[0]; qs = tok[1]
                    if g != base_gates[idx]:
                        return False
                    bq1 = base_q1[idx]; bq2 = base_q2[idx]
                    if len(qs) == 1:
                        return qs[0] == bq1 and bq2 == -1
                    if len(qs) >= 2:
                        return qs[0] == bq1 and qs[1] == bq2
                    return False

                for tok in pqc_tokens:
                    g = tok[0]; qs = tok[1]; params = tok[2] if len(tok) > 2 else []
                    if is_same_base(tok, base_ptr):
                        last_base_idx = base_ptr; base_ptr += 1; continue
                    if g in PARAM_GATES:
                        q = qs[0] if qs else 0
                        ang = params[0] if params else 0.0
                        param_gates.append(g); param_qubits.append(q)
                        after_list.append(last_base_idx); param_angles.append(ang)
                n_q = o.get('n_qubits')
                if n_q is None:
                    qs_all = [*base_q1, *[q for q in base_q2 if q >= 0], *param_qubits]
                    n_q = (max(qs_all) + 1) if qs_all else 1
            else:  # legacy format
                base_g = o['base_gates']
                bq = o['base_qubits']
                if len(bq) != 2:
                    raise ValueError('base_qubits must be [q1_list, q2_list]')
                param_g = o.get('param_gates', [])
                param_q = o.get('param_qubits', [])
                after_list = o.get('after', [-1] * len(param_g))
                param_angles = o.get('pqc_angles_gt', [0.0] * len(param_g))
                base_gates = list(base_g)
                base_q1 = list(bq[0])
                base_q2 = list(bq[1])
                param_gates = list(param_g)
                param_qubits = list(param_q)
                n_q = o.get('n_qubits')
                if n_q is None:
                    qs = [*bq[0], *bq[1], *param_q]
                    qs = [q for q in qs if q >= 0]
                    n_q = (max(qs) + 1) if qs else 1

            # Clip lengths and pack
            if len(base_gates) > self._max_base_len:
                base_gates = base_gates[:self._max_base_len]
                base_q1 = base_q1[:self._max_base_len]
                base_q2 = base_q2[:self._max_base_len]
            if len(param_gates) > self._max_param:
                param_gates = param_gates[:self._max_param]
                param_qubits = param_qubits[:self._max_param]
                after_list = after_list[:self._max_param]
                param_angles = param_angles[:self._max_param]

            self.items.append(dict(
                idx=self._next_index,
                base_gates=base_gates,
                base_q1=base_q1,
                base_q2=base_q2,
                param_gates=param_gates,
                param_qubits=param_qubits,
                after=after_list,
                param_angles_gt=param_angles,
                n_qubits=n_q,
            ))
            self._next_index += 1
            if self._num_limit is not None and self._next_index >= self._num_limit:
                raise _EarlyStop

        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.json', '.jsonl'))]
            files.sort()
            try:
                for fname in files:
                    fp = os.path.join(path, fname)
                    with open(fp, 'r', encoding='utf-8') as fh:
                        for line in fh:
                            if not line.strip():
                                continue
                            process_obj(json.loads(line))
                            break  # read first non-empty JSON line per file
            except _EarlyStop:
                pass
        else:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            try:
                for line in lines:
                    if not line.strip():
                        continue
                    process_obj(json.loads(line))
            except _EarlyStop:
                pass

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class Batch:
    base_g: torch.Tensor
    base_q1: torch.Tensor
    base_q2: torch.Tensor
    param_g: torch.Tensor
    param_q: torch.Tensor
    param_after: torch.Tensor
    param_angles_gt: torch.Tensor
    base_len: torch.Tensor
    param_len: torch.Tensor
    n_qubits: torch.Tensor
    idx: torch.Tensor

    def to(self, device: torch.device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


def _pad(seq, pad, L):
    seq = list(seq)
    return seq[:L] + [pad] * max(0, L - len(seq))


def collate(samples: List[dict], max_base_len: int = 1000, max_param: int = 1500, max_qubits: int = 5) -> Batch:
    bg = []; bq1 = []; bq2 = []; pg = []; pq = []; pafter = []; pang = []
    base_l = []; param_l = []; nqs = []; idxs = []
    for o in samples:
        g = [BASE_GATES[x] for x in o['base_gates']]
        p = [PARAM_GATES[x] for x in o['param_gates']]
        bg.append(_pad(g, PAD_ID, max_base_len))
        bq1.append(_pad(o['base_q1'], PAD_ID, max_base_len))
        bq2.append(_pad(o['base_q2'], PAD_ID, max_base_len))
        pg.append(_pad(p, PAD_ID, max_param))
        pq.append(_pad(o['param_qubits'], PAD_ID, max_param))
        pafter.append(_pad(o['after'], -999, max_param))
        pang.append(_pad(o['param_angles_gt'], 0.0, max_param))
        base_l.append(len(g)); param_l.append(len(p)); nqs.append(o['n_qubits']); idxs.append(o['idx'])
    to_long = lambda x: torch.tensor(x, dtype=torch.long)
    return Batch(
        to_long(bg), to_long(bq1), to_long(bq2), to_long(pg), to_long(pq), to_long(pafter),
        torch.tensor(pang, dtype=torch.float32), to_long(base_l), to_long(param_l), to_long(nqs), to_long(idxs)
    )

# ---- Core indexing caches ----
_SPLIT_CACHE: Dict[Tuple[int, torch.device], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
_CX_SWAP_CACHE: Dict[Tuple[int, torch.device], Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]]] = {}
_CZ_MASK_CACHE: Dict[Tuple[int, torch.device], Dict[Tuple[int,int], torch.Tensor]] = {}


def _split_indices(n: int, device: torch.device):
    k = (n, device)
    if k in _SPLIT_CACHE:
        return _SPLIT_CACHE[k]
    dim = 1 << n
    ar = torch.arange(dim, device=device)
    out = []
    for q in range(n):
        bit = (ar >> q) & 1
        out.append(((bit == 0).nonzero(as_tuple=False).squeeze(-1), (bit == 1).nonzero(as_tuple=False).squeeze(-1)))
    _SPLIT_CACHE[k] = out
    return out


def _get_two_qubit_struct(n: int, device: torch.device):
    key = (n, device)
    if key in _CX_SWAP_CACHE and key in _CZ_MASK_CACHE:
        return _CX_SWAP_CACHE[key], _CZ_MASK_CACHE[key]
    dim = 1 << n
    idx_all = torch.arange(dim, device=device)
    cx_swap = {}; cz_mask = {}
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            cb = 1 << c; tb = 1 << t
            sel = ((idx_all & cb) != 0) & ((idx_all & tb) == 0)
            i0 = idx_all[sel]; i1 = i0 | tb
            cx_swap[(c, t)] = (i0, i1)
            sel_cz = ((idx_all & cb) != 0) & ((idx_all & tb) != 0)
            cz_mask[(c, t)] = idx_all[sel_cz]
    _CX_SWAP_CACHE[key] = cx_swap; _CZ_MASK_CACHE[key] = cz_mask
    return cx_swap, cz_mask

# ---- Gate applications ----

def _apply_const_1q(st, q: int, kind: str, splits):
    i0, i1 = splits[q]
    s0 = st[..., i0]
    s1 = st[..., i1]
    if kind == 'h':
        n0 = (s0 + s1) / math.sqrt(2); n1 = (s0 - s1) / math.sqrt(2)
    elif kind == 'x':
        n0, n1 = s1, s0
    elif kind == 'z':
        n0, n1 = s0, -s1
    else:
        raise ValueError(kind)
    st[..., i0] = n0; st[..., i1] = n1


def _apply_rz(st, q: int, a, splits):
    i0, i1 = splits[q]
    em = torch.exp(-0.5j * a).unsqueeze(-1)
    ep = torch.exp(0.5j * a).unsqueeze(-1)
    st[..., i0] *= em
    st[..., i1] *= ep


def _apply_rx(st, q: int, a, splits):
    i0, i1 = splits[q]
    c = torch.cos(0.5 * a).unsqueeze(-1)
    s = -1j * torch.sin(0.5 * a).unsqueeze(-1)
    s0 = st[..., i0]
    s1 = st[..., i1]
    st[..., i0] = c * s0 + s * s1
    st[..., i1] = s * s0 + c * s1


# ---- Fused single-qubit rotations on (i0,i1) pairs ----
def _apply_rzrx_fused_pairs(states_sub: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor,
                            ang_rz: torch.Tensor, ang_rx: torch.Tensor):
    """Apply R = RX(ang_rx) @ RZ(ang_rz) to amplitudes [i0,i1] for a subset of batches.
    states_sub: [B', K, D], angles: [B'] (broadcasted internally).
    """
    # shapes: [B',1,1]
    e1 = torch.exp(-0.5j * ang_rz)[:, None, None]
    e2 = torch.exp(0.5j * ang_rz)[:, None, None]
    c = torch.cos(0.5 * ang_rx)[:, None, None]
    s = (-1j * torch.sin(0.5 * ang_rx))[:, None, None]
    a = states_sub[:, :, i0]
    b = states_sub[:, :, i1]
    # new0 = (c*e1)*a + (s*e2)*b
    # new1 = (s*e1)*a + (c*e2)*b
    states_sub[:, :, i0] = c * e1 * a + s * e2 * b
    states_sub[:, :, i1] = s * e1 * a + c * e2 * b


def _apply_rzrxrz_fused_pairs(states_sub: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor,
                               ang_rz1: torch.Tensor, ang_rx: torch.Tensor, ang_rz2: torch.Tensor):
    """Apply R = RZ(ang_rz2) @ RX(ang_rx) @ RZ(ang_rz1) to amplitudes [i0,i1] for a subset of batches.
    states_sub: [B', K, D], angles: [B'] (broadcasted internally).
    """
    f1 = torch.exp(-0.5j * ang_rz2)[:, None, None]
    f2 = torch.exp(0.5j * ang_rz2)[:, None, None]
    e1 = torch.exp(-0.5j * ang_rz1)[:, None, None]
    e2 = torch.exp(0.5j * ang_rz1)[:, None, None]
    c = torch.cos(0.5 * ang_rx)[:, None, None]
    s = (-1j * torch.sin(0.5 * ang_rx))[:, None, None]
    a = states_sub[:, :, i0]
    b = states_sub[:, :, i1]
    # new0 = (c*f1*e1)*a + (s*f1*e2)*b
    # new1 = (s*f2*e1)*a + (c*f2*e2)*b
    states_sub[:, :, i0] = c * f1 * e1 * a + s * f1 * e2 * b
    states_sub[:, :, i1] = s * f2 * e1 * a + c * f2 * e2 * b


def _apply_cx(st, cqb: int, tqb: int):
    dim = st.size(-1)
    idx = torch.arange(dim, device=st.device)
    mc = 1 << cqb
    mt = 1 << tqb
    sel = ((idx & mc) != 0) & ((idx & mt) == 0)
    i0 = idx[sel]; i1 = i0 | mt
    tmp = st[..., i0].clone()
    st[..., i0] = st[..., i1]
    st[..., i1] = tmp


def _apply_cz(st, q1: int, q2: int):
    dim = st.size(-1)
    idx = torch.arange(dim, device=st.device)
    mask = ((idx & (1 << q1)) != 0) & ((idx & (1 << q2)) != 0)
    st[..., idx[mask]] = -st[..., idx[mask]]

# ---- Optional fused CUDA extension: base+noise segment ----
_bn_ext = None
_bn_ext_attempted = False
_bn_ext_reason: Optional[str] = None  # human-readable reason when ext is unavailable
_fused_used_count = 0

def get_fused_status() -> Dict[str, object]:
        """Return current fused-kernel status for visibility in training logs.
        Keys:
            - enabled: bool (env toggle PQC_USE_FUSED)
            - attempted: bool (whether we tried to load the inline extension)
            - available: bool (extension module loaded)
            - used_calls: int (times fused path has executed in this process)
            - reason: Optional[str] (why extension is unavailable or guarded)
        """
        return {
                "enabled": USE_FUSED_BASE_NOISE,
                "attempted": bool(_bn_ext_attempted),
                "available": bool(_bn_ext is not None),
                "used_calls": int(_fused_used_count),
                "reason": _bn_ext_reason,
        }

def _ensure_bn_extension():
    global _bn_ext_attempted, _bn_ext
    if _bn_ext_attempted:
        return _bn_ext
    _bn_ext_attempted = True
    # Attempt to prepare Windows dev environment so cl/nvcc are available
    _ensure_msvc_env_windows()
    # Avoid known long hangs when trying to JIT-compile CUDA inline extensions on Windows
    # unless the user explicitly forces it via PQC_FORCE_INLINE=1 and a working toolchain is present.
    try:
        is_win = (os.name == 'nt') or platform.system().lower().startswith('win')
    except Exception:
        is_win = False
    force_inline = str(os.environ.get('PQC_FORCE_INLINE', '0')).strip().lower() in ('1','true','yes','y','on')
    if is_win and not force_inline:
        # Soft guard: if toolchain present and fused is enabled, allow proceeding; else guard
        try:
            import shutil as _sh
            has_nvcc = _sh.which('nvcc') is not None
            has_cl = _sh.which('cl') is not None
        except Exception:
            has_nvcc = has_cl = False
        if not (USE_FUSED_BASE_NOISE and has_nvcc and has_cl):
            _bn_ext = None
            globals()['_bn_ext_reason'] = 'windows_guard (set PQC_FORCE_INLINE=1 to force)'
            return None
    # Quick toolchain preflight: require nvcc; on Windows also require cl
    try:
        import shutil
        if shutil.which('nvcc') is None:
            _bn_ext = None
            globals()['_bn_ext_reason'] = 'nvcc_missing'
            return None
        if is_win and shutil.which('cl') is None:
            _bn_ext = None
            globals()['_bn_ext_reason'] = 'cl_missing'
            return None
    except Exception:
        _bn_ext = None
        globals()['_bn_ext_reason'] = 'toolchain_check_failed'
        return None
    try:
        from torch.utils.cpp_extension import load_inline as _load_inline
    except Exception:
        _bn_ext = None
        globals()['_bn_ext_reason'] = 'no_cpp_extension_api'
        return None
    cuda_src = r"""
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
""";
    try:
        # Reduce parallel build issues on Windows/MSVC
        os.environ.setdefault('MAX_JOBS', '1')
        build_dir = os.path.join(os.path.dirname(__file__), "_extcache")
        os.makedirs(build_dir, exist_ok=True)
        print('[Fused] starting inline build (this may take a minute)...', flush=True)
        # Derive CUDA arch flags from TORCH_CUDA_ARCH_LIST (e.g., "8.6;8.9")
        arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ',').replace(' ', '')
        arch_flags = []
        if arch_list:
            for ent in arch_list.split(','):
                if not ent:
                    continue
                ent = ent.strip()
                if ent.endswith('+PTX'):
                    ent = ent[:-4]
                # handle forms like 8.6 or 86
                if '.' in ent:
                    cc = ent.replace('.', '')
                else:
                    cc = ent
                if cc.isdigit():
                    arch_flags += ["-gencode", f"arch=compute_{cc},code=sm_{cc}"]
        # Add safe flags for Windows + recent MSVC/CUDA combos
        import shutil as _sh
        cl_path = _sh.which('cl')
        extra_cuda = ["-Xcompiler", "/std:c++17", "-lineinfo", "-O2", "-allow-unsupported-compiler"] + arch_flags
        if cl_path:
            ccbin_dir = os.path.dirname(cl_path)
            extra_cuda = ["-ccbin", ccbin_dir] + extra_cuda
        cpp_decl = r"""
#include <torch/extension.h>
torch::Tensor fused_base_noise_segment(
    torch::Tensor states, torch::Tensor scratch,
    torch::Tensor gate_kind, torch::Tensor q1s, torch::Tensor q2s,
    torch::Tensor rz1, torch::Tensor rx1, torch::Tensor rz2, torch::Tensor rx2,
    int reverse);
"""
        _bn_ext = _load_inline(
            name="pqc_bn_seg_v1",
            cpp_sources=cpp_decl,
            cuda_sources=cuda_src,
            functions=["fused_base_noise_segment"],
            verbose=True,
            with_cuda=True,
            build_directory=build_dir,
            extra_cflags=["/std:c++17", "/O2", "/EHsc"],
            extra_cuda_cflags=extra_cuda,
        )
        globals()["_bn_ext_reason"] = None
        print('[Fused] inline build done.', flush=True)
    except Exception as e:
        _bn_ext = None
        globals()["_bn_ext_reason"] = f'inline_build_failed: {e}'
        try:
            print(f'[Fused] inline build failed: {e}', flush=True)
        except Exception:
            pass
    return _bn_ext

def ensure_fused_compiled() -> bool:
    """Public helper to trigger fused inline extension load (if eligible). Returns True if available."""
    ext = _ensure_bn_extension()
    return bool(ext is not None)

class _FusedBaseNoiseSegFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states, gate_ids_seg, q1_seg, q2_seg, rz1_seg, rx1_seg, rz2_seg, rx2_seg):
        ext = _ensure_bn_extension()
        if ext is None or states.device.type != 'cuda':
            raise RuntimeError("Fused extension unavailable")
        # Save for backward
        ctx.save_for_backward(gate_ids_seg, q1_seg, q2_seg, rz1_seg, rx1_seg, rz2_seg, rx2_seg)
        # Do NOT modify input in-place (avoid in-place on leaf tensors). Work on a cloned output.
        out = states.clone()
        scratch = torch.empty_like(out)
        ext.fused_base_noise_segment(out,
                                     scratch,
                                     gate_ids_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                     q1_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                     q2_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                     rz1_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     rx1_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     rz2_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     rx2_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     0)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gate_ids_seg, q1_seg, q2_seg, rz1_seg, rx1_seg, rz2_seg, rx2_seg = ctx.saved_tensors
        ext = _ensure_bn_extension()
        if ext is None or grad_out.device.type != 'cuda':
            raise RuntimeError("Fused extension unavailable in backward")
        # Work on a cloned gradient tensor to avoid in-place on incoming grad view.
        gin = grad_out.clone()
        scratch = torch.empty_like(gin)
        # Apply adjoint segment to gradient (reverse=1)
        ext.fused_base_noise_segment(gin,
                                     scratch,
                                     gate_ids_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                     q1_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                     q2_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                     rz1_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     rx1_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     rz2_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     rx2_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                     1)
        return gin, None, None, None, None, None, None, None

def _try_fused_base_noise_segment(states, gate_ids_seg, q1_seg, q2_seg, rz1_seg, rx1_seg, rz2_seg, rx2_seg):
    if not USE_FUSED_BASE_NOISE:
        return False
    if states.device.type != 'cuda':
        return False
    ext = _ensure_bn_extension()
    if ext is None:
        return False
    # Prepare inputs as contiguous CUDA tensors with expected dtypes
    try:
        if states.requires_grad:
            # Autograd path: compute out-of-place, then copy back into provided states tensor.
            out = _FusedBaseNoiseSegFn.apply(states, gate_ids_seg, q1_seg, q2_seg, rz1_seg, rx1_seg, rz2_seg, rx2_seg)
            states.copy_(out)
            globals()['_fused_used_count'] = globals().get('_fused_used_count', 0) + 1
            return True
        else:
            # Non-grad path
            B, Ls = gate_ids_seg.shape
            scratch = torch.empty_like(states)
            ok = bool(ext.fused_base_noise_segment(states,
                                                     scratch,
                                                     gate_ids_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                                     q1_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                                     q2_seg.to(torch.int32, copy=True, non_blocking=True).contiguous(),
                                                     rz1_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                                     rx1_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                                     rz2_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                                     rx2_seg.to(torch.float32, copy=True, non_blocking=True).contiguous(),
                                                     0) is not None)
            if ok:
                globals()['_fused_used_count'] = globals().get('_fused_used_count', 0) + 1
            return ok
    except Exception:
        return False

# ---- Angle helpers ----

def sincos_to_angle(sc: torch.Tensor) -> torch.Tensor:
    sc = sc / (sc.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.atan2(sc[..., 0], sc[..., 1])


def logits_to_angles(logits: torch.Tensor, Lp: int) -> torch.Tensor:
    if logits.size(-1) == 2:
        sc = logits[:, :Lp, :]
        sc = sc / (sc.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.atan2(sc[..., 0], sc[..., 1])
    elif logits.size(-1) == 1:
        return logits[:, :Lp, 0]
    else:
        raise ValueError(f"Unsupported logits last dim {logits.size(-1)}; expected 2 (sin,cos) or 1 (angle)")

# ---- Shared block-based helpers (PQC blocks pipeline) ----

def compute_block_plan(batch: Batch, gate_blocks: int) -> Tuple[int, int, int]:
    """Return (Lb, n, blocks_needed) for fixed-interval blocks on this batch.
    Assumes all items in batch share same n_qubits and base_len.
    """
    Lb = int(batch.base_len[0].item())
    n = int(batch.n_qubits[0].item())
    blocks_needed = math.ceil(Lb / max(1, gate_blocks)) if Lb > 0 else 1
    return Lb, n, blocks_needed


def angles_from_logits_block_layout(logits: torch.Tensor, blocks_needed: int, n_qubits: int) -> torch.Tensor:
    """Map logits to angles and reshape to [B, blocks_needed, n_qubits, 3].
    If logits are shorter than expected, pad with zeros.
    Supports logits last-dim 2 (sin,cos) or 1 (angle).
    """
    device = logits.device
    B = logits.size(0)
    expected = blocks_needed * n_qubits * 3
    use_L = min(logits.size(1), expected)
    all_angles_flat = logits_to_angles(logits, use_L)
    if all_angles_flat.size(1) < expected:
        pad = torch.zeros(B, expected - all_angles_flat.size(1), device=device, dtype=all_angles_flat.dtype)
        all_angles_flat = torch.cat([all_angles_flat, pad], dim=1)
    return all_angles_flat[:, :expected].view(B, blocks_needed, n_qubits, 3)


def simulate_blocks_with_angles(batch: Batch,
                                angles_blk: torch.Tensor,
                                init_cache: Dict[int, torch.Tensor],
                                ref_cache: dict,
                                noise_schedules: dict,
                                gate_blocks: int,
                                device: Optional[torch.device] = None,
                                detach_base_noise: bool = True) -> torch.Tensor:
    """Shared core: run base+noise and insert RZ-RX-RZ PQC blocks using provided angles_blk.
    angles_blk: [B, blocks_needed, n_qubits, 3]
    """
    if device is None:
        device = angles_blk.device
    B = batch.base_g.size(0)
    n = int(batch.n_qubits[0].item())
    assert (batch.n_qubits == n).all()
    states = init_cache[n].to(device).unsqueeze(0).expand(B, -1, -1).clone()
    rows = torch.tensor([ref_cache['idx2row'][int(i.item())] for i in batch.idx], device=device)
    ref = ref_cache['tensor'].index_select(0, rows)

    Lb = int(batch.base_len[0].item())
    gate_ids = batch.base_g[:, :Lb].to(device)
    q1 = batch.base_q1[:, :Lb].to(device)
    q2 = batch.base_q2[:, :Lb].to(device)
    splits = _split_indices(n, device)
    cx_swap, cz_mask = _get_two_qubit_struct(n, device)
    noise_rows = torch.tensor([noise_schedules['idx2row'][int(i.item())] for i in batch.idx], device=device)

    t = 0; blk_idx = 0; first_block = True
    while t < Lb:
        t_end = min(Lb, (blk_idx + 1) * gate_blocks)
        seg_len = t_end - t
        if seg_len > 0 and noise_schedules.get('use_noise', False):
            g_seg = gate_ids[:, t:t_end].contiguous()
            q1_seg = q1[:, t:t_end].contiguous()
            q2_seg = q2[:, t:t_end].contiguous()
            rz1_seg = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            rx1_seg = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            rz2_seg = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            rx2_seg = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            used = _try_fused_base_noise_segment(states, g_seg, q1_seg, q2_seg, rz1_seg, rx1_seg, rz2_seg, rx2_seg)
            if not used:
                for tt in range(t, t_end):
                    g_t = gate_ids[:, tt]
                    if (g_t == PAD_ID).all():
                        break
                    q1_t = q1[:, tt]; q2_t = q2[:, tt]
                    _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
                    rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, tt]
                    rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, tt]
                    rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, tt]
                    rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, tt]
                    _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)
        else:
            for tt in range(t, t_end):
                g_t = gate_ids[:, tt]
                if (g_t == PAD_ID).all():
                    break
                q1_t = q1[:, tt]; q2_t = q2[:, tt]
                _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
                if noise_schedules.get('use_noise', False):
                    rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, tt]
                    rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, tt]
                    rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, tt]
                    rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, tt]
                    _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)
        t = t_end
        if t < Lb:
            if detach_base_noise and first_block:
                states = states.detach()
                first_block = False
            angs = angles_blk[:, blk_idx]
            for qb in range(n):
                i0, i1 = splits[qb]
                a_rz1 = angs[:, qb, 0]
                a_rx  = angs[:, qb, 1]
                a_rz2 = angs[:, qb, 2]
                _apply_rzrxrz_fused_pairs(states, i0, i1, a_rz1, a_rx, a_rz2)
            blk_idx += 1

    if blk_idx < angles_blk.size(1):
        angs = angles_blk[:, blk_idx]
        for qb in range(n):
            i0, i1 = splits[qb]
            a_rz1 = angs[:, qb, 0]
            a_rx  = angs[:, qb, 1]
            a_rz2 = angs[:, qb, 2]
            _apply_rzrxrz_fused_pairs(states, i0, i1, a_rz1, a_rx, a_rz2)

    # Overlap per-sample, per-random-initial-state (match K with K). Avoid KxK broadcasting.
    ov = (ref.conj() * states).sum(-1)  # [B,K]
    F = (ov.abs() ** 2).mean()
    return 1 - F

# ---- Precompute: base cache and tensor-mode noise ----

def build_base_cache_vectorized(dataset: CircuitDataset, k_random: int = K_RANDOM_DEFAULT, device: Optional[torch.device] = None,
                                 noise: Optional[NoiseConfig] = None):
    if device is None:
        device = DEVICE
    if noise is None:
        noise = NoiseConfig()

    groups: Dict[int, List[dict]] = {}
    for it in dataset.items:
        groups.setdefault(it['n_qubits'], []).append(it)

    init_states_per_n: Dict[int, torch.Tensor] = {}
    ref_states_packed = None
    ref_idx2row: Dict[int, int] = {}

    for n, items in groups.items():
        dim = 1 << n
        Bn = len(items)
        if Bn == 0:
            continue
        L_max = max(len(it['base_gates']) for it in items)
        gate_ids_cpu = torch.full((Bn, L_max), PAD_ID, dtype=torch.long)
        q1_cpu = torch.full((Bn, L_max), -1, dtype=torch.long)
        q2_cpu = torch.full((Bn, L_max), -1, dtype=torch.long)
        sample_idx_list = []
        for bi, it in enumerate(items):
            sample_idx_list.append(it['idx'])
            Lb_i = len(it['base_gates'])
            if Lb_i == 0:
                continue
            gate_ids_row = [BASE_GATES[g] for g in it['base_gates']]
            gate_ids_cpu[bi, :Lb_i] = torch.tensor(gate_ids_row, dtype=torch.long)
            q1_cpu[bi, :Lb_i] = torch.tensor(it['base_q1'], dtype=torch.long)
            q2_cpu[bi, :Lb_i] = torch.tensor(it['base_q2'], dtype=torch.long)

        gate_ids = gate_ids_cpu.to(device)
        q1 = q1_cpu.to(device)
        q2 = q2_cpu.to(device)

        # K random initial states shared for this n
        splits_tmp = _split_indices(n, device)
        if n not in init_states_per_n:
            states_init = []
            for _ in range(k_random):
                st = torch.zeros(dim, dtype=DTYPE, device=device); st[0] = 1 + 0j
                for qb in range(n):
                    r = random.random()
                    if r < 0.33:
                        pass
                    elif r < 0.66:
                        _apply_const_1q(st.unsqueeze(0), qb, 'x', splits_tmp)
                    else:
                        _apply_const_1q(st.unsqueeze(0), qb, 'h', splits_tmp)
                states_init.append(st)
            init_states_per_n[n] = torch.stack(states_init, 0)  # [K, 2^n]

        states = init_states_per_n[n].unsqueeze(0).expand(Bn, -1, -1).clone()  # [B,K,D]
        splits = _split_indices(n, device)
        idx_all = torch.arange(dim, device=device)
        cx_swap = {}; cz_mask = {}
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                cb = 1 << c; tb = 1 << t
                sel = ((idx_all & cb) != 0) & ((idx_all & tb) == 0)
                i0 = idx_all[sel]; i1 = i0 | tb
                cx_swap[(c, t)] = (i0, i1)
                sel_cz = ((idx_all & cb) != 0) & ((idx_all & tb) != 0)
                cz_mask[(c, t)] = idx_all[sel_cz]

        with torch.no_grad():
            for t in range(gate_ids.size(1)):
                g_t = gate_ids[:, t]
                if (g_t == PAD_ID).all():
                    break
                # 1q gates grouped by target qubit
                for gcode, gname in ((BASE_GATES['h'], 'h'), (BASE_GATES['x'], 'x'), (BASE_GATES['z'], 'z')):
                    mask = (g_t == gcode)
                    if not mask.any():
                        continue
                    qubits = q1[mask, t]
                    batches = mask.nonzero(as_tuple=False).squeeze(-1)
                    for qb in qubits.unique().tolist():
                        sel = batches[(qubits == qb)]
                        if sel.numel() == 0:
                            continue
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
                # 2q gates grouped by (c,t)
                for gcode, gname in ((BASE_GATES['cx'], 'cx'), (BASE_GATES['cz'], 'cz')):
                    mask = (g_t == gcode)
                    if not mask.any():
                        continue
                    c_list = q1[mask, t]; t_list = q2[mask, t]
                    batches = mask.nonzero(as_tuple=False).squeeze(-1)
                    pairs = torch.stack([c_list, t_list], dim=1)
                    uniq_pairs, inv_idx = torch.unique(pairs, dim=0, return_inverse=True)
                    for pi, (c_val, t_val) in enumerate(uniq_pairs.tolist()):
                        sel = batches[inv_idx == pi]
                        if sel.numel() == 0:
                            continue
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

        # Pack noiseless reference states
        if ref_states_packed is None:
            ref_states_packed = torch.empty(len(dataset.items), init_states_per_n[n].size(0), dim, dtype=DTYPE, device=device)
        for bi, sample_idx in enumerate(sample_idx_list):
            ref_states_packed[sample_idx].copy_(states[bi])
            ref_idx2row[sample_idx] = sample_idx

    # Build tensor-mode noise schedules across all items
    items_all = dataset.items
    idx_list = [it['idx'] for it in items_all]
    L_per_sample = [len(it['base_gates']) for it in items_all]
    L_max_global = max(L_per_sample) if L_per_sample else 0
    B_total = len(items_all)
    q2_mat = torch.full((B_total, L_max_global), -1, dtype=torch.long, device=device)
    gate_mask = torch.zeros((B_total, L_max_global), dtype=torch.bool, device=device)
    for row, it in enumerate(items_all):
        Lb = len(it['base_gates'])
        gate_mask[row, :Lb] = True
        if Lb > 0:
            q2_vals = torch.tensor(it['base_q2'], dtype=torch.long, device=device)
            q2_mat[row, :Lb] = q2_vals

    if noise.use_noise:
        delta_fx = _delta_fraction(noise.noise_delta_x)
        delta_fz = _delta_fraction(noise.noise_delta_z)
        span_x = noise.noise_x_rad * delta_fx
        span_z = noise.noise_z_rad * delta_fz
        base_x = noise.noise_x_rad
        base_z = noise.noise_z_rad
        rx_full = (torch.rand(B_total, L_max_global, device=device) * 2 - 1) * span_x + base_x
        rz_full = (torch.rand(B_total, L_max_global, device=device) * 2 - 1) * span_z + base_z
        rx_q1 = torch.where(gate_mask, rx_full, torch.zeros(1, device=device))
        rz_q1 = torch.where(gate_mask, rz_full, torch.zeros(1, device=device))
        valid_q2 = (q2_mat >= 0) & gate_mask
        rx_full2 = (torch.rand(B_total, L_max_global, device=device) * 2 - 1) * span_x + base_x
        rz_full2 = (torch.rand(B_total, L_max_global, device=device) * 2 - 1) * span_z + base_z
        rx_q2 = torch.where(valid_q2, rx_full2, torch.zeros(1, device=device))
        rz_q2 = torch.where(valid_q2, rz_full2, torch.zeros(1, device=device))
    else:
        rx_q1 = rz_q1 = rx_q2 = rz_q2 = torch.zeros(B_total, L_max_global, device=device)

    idx2row = {idx: i for i, idx in enumerate(idx_list)}
    noise_schedules = dict(tensor_mode=True, rx_q1=rx_q1, rz_q1=rz_q1, rx_q2=rx_q2, rz_q2=rz_q2,
                           idx2row=idx2row, L_max=L_max_global, use_noise=bool(noise.use_noise))

    ref_cache = dict(packed=True, tensor=ref_states_packed, idx2row=ref_idx2row)
    return init_states_per_n, ref_cache, noise_schedules

# ---- Batched base/noise steps ----

def _apply_base_step_batched(states, gate_ids_step, q1_step, q2_step, splits, cx_swap, cz_mask):
    # 1q: h/x/z
    for gcode, gname in ((BASE_GATES['h'], 'h'), (BASE_GATES['x'], 'x'), (BASE_GATES['z'], 'z')):
        mask = (gate_ids_step == gcode)
        if not mask.any():
            continue
        qubits = q1_step[mask]
        batches = mask.nonzero(as_tuple=False).squeeze(-1)
        uq = qubits.unique()
        for qb in uq.tolist():
            sel = batches[(qubits == qb)]
            if sel.numel() == 0:
                continue
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
    # 2q: cx/cz
    for gcode, gname in ((BASE_GATES['cx'], 'cx'), (BASE_GATES['cz'], 'cz')):
        mask = (gate_ids_step == gcode)
        if not mask.any():
            continue
        c_list = q1_step[mask]; t_list = q2_step[mask]
        batches = mask.nonzero(as_tuple=False).squeeze(-1)
        pairs = torch.stack([c_list, t_list], dim=1)
        uniq_pairs, inv_idx = torch.unique(pairs, dim=0, return_inverse=True)
        for pi, (c_val, t_val) in enumerate(uniq_pairs.tolist()):
            sel = batches[inv_idx == pi]
            if sel.numel() == 0:
                continue
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
    # qubit 1
    uq = q1_step.unique()
    for qb in uq.tolist():
        mask = (q1_step == qb)
        if not mask.any():
            continue
        sel = mask.nonzero(as_tuple=False).squeeze(-1)
        states_sel = states.index_select(0, sel)
        # Fused RZ+RX
        ang_rz = rz1[sel]
        ang_rx = rx1[sel]
        if not (ang_rz.abs().sum() == 0 and ang_rx.abs().sum() == 0):
            i0, i1 = splits[qb]
            _apply_rzrx_fused_pairs(states_sel, i0, i1, ang_rz, ang_rx)
        states[sel] = states_sel
    # qubit 2
    valid_q2 = (q2_step >= 0)
    if valid_q2.any():
        q2_vals = q2_step[valid_q2]
        uq2 = q2_vals.unique()
        base_idx = valid_q2.nonzero(as_tuple=False).squeeze(-1)
        for qb in uq2.tolist():
            mask_local = (q2_vals == qb)
            sel = base_idx[mask_local]
            states_sel = states.index_select(0, sel)
            ang_rz = rz2[sel]
            ang_rx = rx2[sel]
            if not (ang_rz.abs().sum() == 0 and ang_rx.abs().sum() == 0):
                i0, i1 = splits[qb]
                _apply_rzrx_fused_pairs(states_sel, i0, i1, ang_rz, ang_rx)
            states[sel] = states_sel

## Shared-parameter layout path was removed. Transformer/direct now share the same
## fixed-interval blocks pipeline via simulate_loss(..., mode='blocks').

# ---- Fixed-interval RZ-RX-RZ blocks (Direct-angle mode) ----

def simulate_loss_fixed_interval_blocks(batch: Batch, logits: torch.Tensor,
                                        init_cache: Dict[int, torch.Tensor],
                                        ref_cache: dict,
                                        noise_schedules: dict,
                                        gate_blocks: int,
                                        device: Optional[torch.device] = None,
                                        detach_base_noise: bool = True) -> torch.Tensor:
    if device is None:
        device = logits.device
    Lb, n, blocks_needed = compute_block_plan(batch, gate_blocks)
    angles_blk = angles_from_logits_block_layout(logits.to(device), blocks_needed, n)
    return simulate_blocks_with_angles(batch, angles_blk, init_cache, ref_cache, noise_schedules, gate_blocks, device=device, detach_base_noise=detach_base_noise)

# ---- Simple single-circuit simulator (no noise), helpful for unit checks ----

def simulate_single_circuit_no_noise(circuit_ops: Sequence[Tuple[str, Sequence[int], Optional[float]]],
                                     num_qubits: int,
                                     input_state,
                                     device: Optional[torch.device] = None,
                                     big_endian_wires: bool = True) -> torch.Tensor:
    if device is None:
        device = DEVICE
    st = torch.as_tensor(input_state, dtype=DTYPE, device=device).clone()
    if st.ndim != 1:
        raise ValueError("input_state must be 1D")
    dim_expected = 1 << num_qubits
    if st.numel() != dim_expected:
        raise ValueError(f"State length {st.numel()} != 2**{num_qubits}")
    st_batch = st.unsqueeze(0)
    splits = _split_indices(num_qubits, device)
    for gate, wires, param in circuit_ops:
        if not isinstance(wires, (list, tuple)):
            raise ValueError("wires must be a sequence")
        mapped = []
        for w in wires:
            w_int = int(w)
            if w_int < 0 or w_int >= num_qubits:
                raise ValueError(f"wire {w_int} out of range [0,{num_qubits-1}]")
            mapped.append(num_qubits - 1 - w_int if big_endian_wires else w_int)
        if gate in ('h','x','z'):
            if len(mapped) != 1:
                raise ValueError(f"Gate {gate} expects 1 wire")
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
            if len(mapped) != 1 or param is None:
                raise ValueError("rz expects 1 wire and angle")
            _apply_rz(st_batch, mapped[0], torch.as_tensor(float(param), device=device), splits)
        elif gate == 'rx':
            if len(mapped) != 1 or param is None:
                raise ValueError("rx expects 1 wire and angle")
            _apply_rx(st_batch, mapped[0], torch.as_tensor(float(param), device=device), splits)
        else:
            raise ValueError(f"Unsupported gate: {gate}")
    return st_batch[0]

# ---- Unified loss entry (dispatcher) ----
def simulate_loss(batch: Batch,
                  logits: torch.Tensor,
                  init_cache: Dict[int, torch.Tensor],
                  ref_cache: dict,
                  noise_schedules: dict,
                  *,
                  mode: str,
                  gate_blocks: Optional[int] = None,
                  detach_base_noise: bool = True,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """Unified loss API (blocks-only). Transformer 和 direct 共用固定间隔块式 PQC 路径。

    Requires: mode='blocks' and gate_blocks provided.
    """
    if (mode or 'blocks').lower() != 'blocks':
        raise ValueError("Only mode='blocks' is supported now")
    if gate_blocks is None:
        raise ValueError("gate_blocks is required for mode='blocks'")
    return simulate_loss_fixed_interval_blocks(
        batch, logits, init_cache, ref_cache, noise_schedules, gate_blocks,
        device=device, detach_base_noise=detach_base_noise
    )
