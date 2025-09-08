import jax.numpy as jnp
# from jax import random

# ---------------------------
# 1) SU(2) from unit quaternion
# ---------------------------

def normalize_quaternion(q, enforce_w_nonneg=True, eps=1e-8):
    q = jnp.asarray(q, dtype=jnp.float32)
    # Sanitize to avoid NaN/Inf flowing into normalization under JIT/vmap
    q = jnp.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    # If norm is zero, set to default unit quaternion [1,0,0,0]
    is_zero_norm = jnp.all(norm < eps, axis=-1, keepdims=True)
    default_q = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    q = jnp.where(is_zero_norm, default_q, q / (norm + eps))
    if enforce_w_nonneg:
        sign = jnp.where(q[..., 0:1] < 0.0, -1.0, 1.0)
        q = sign * q
    return q # (w,x,y,z), ||q||=1\

def su2_from_quaternion(q):
    """U = w I - i (x σx + y σy + z σz) ∈ SU(2)."""
    q_norm = normalize_quaternion(q)
    w, x, y, z = q_norm[..., 0], q_norm[..., 1], q_norm[..., 2], q_norm[..., 3]
    return jnp.array([[w - 1j*z, -1j*x - y],
                      [-1j*x + y, w + 1j*z]], dtype=jnp.complex64)

# ---------------------------
# 2) Canonical ZXZ decomposition (Rz-Rx-Rz) for compilation
#    Unique off the measure-zero boundaries β∈{0,π}
# ---------------------------
def _wrap_pi(t): 
    return (t + jnp.pi) % (2*jnp.pi) - jnp.pi

def zxz_from_su2(U, eps=1e-12):
    # strip global phase so det≈1
    U = U / (jnp.sqrt(jnp.linalg.det(U)) + eps)
    u11, u12, u21, u22 = U[0,0], U[0,1], U[1,0], U[1,1]

    safe_abs_u11 = jnp.clip(jnp.abs(u11), 0.0, 1.0 - eps)
    beta = 2*jnp.arccos(safe_abs_u11)    

    # --- Calculate the results for all three potential cases ---
    # Case 1: beta is near 0
    # CORRECTED: Flipped the order of subtraction
    alpha_case1 = _wrap_pi(jnp.angle(u22) - jnp.angle(u11))
    beta_case1 = 0.0
    gamma_case1 = 0.0

    # Case 2: beta is near pi
    alpha_case2 = 0.0
    beta_case2 = jnp.pi
    # CORRECTED: Added a negative sign.
    # The negative sign is necessary for the gimbal lock case at β=π because
    # the ZXZ decomposition becomes ambiguous at this boundary. The sign ensures
    # that the resulting Euler angles correctly represent the rotation, matching
    # the convention used for the generic case and avoiding a flipped angle assignment.
    gamma_case2 = _wrap_pi(-(jnp.angle(-u21) - jnp.angle(u12)))

    # Case 3: generic case
    apg = jnp.angle(u22) - jnp.angle(u11)    # α+γ
    amg = jnp.angle(-u21) - jnp.angle(u12)   # α-γ
    alpha_case3 = _wrap_pi(0.5*(apg + amg))
    beta_case3 = beta # Use the calculated beta for the default case
    gamma_case3 = _wrap_pi(0.5*(apg - amg))
    
    # --- Define the conditions ---
    cond1 = beta < eps
    cond2 = jnp.abs(jnp.pi - beta) < eps

    # --- Use nested jnp.where to select the correct value for each variable ---
    # This is equivalent to: if cond1 then case1 else (if cond2 then case2 else case3)
    alpha = jnp.where(cond1, alpha_case1, jnp.where(cond2, alpha_case2, alpha_case3))
    beta_final = jnp.where(cond1, beta_case1, jnp.where(cond2, beta_case2, beta_case3))
    gamma = jnp.where(cond1, gamma_case1, jnp.where(cond2, gamma_case2, gamma_case3))

    return jnp.stack([alpha, beta_final, gamma])

# ---------------------------
# 3) End-to-end helper
# ---------------------------
def quaternion_to_zxz_angles(q):
    U = su2_from_quaternion(q)
    return zxz_from_su2(U)


# ---------------------------
# XZY decomposition (Rx-Rz-Ry)
# ---------------------------
def _rotmat_from_unit_quaternion(q: jnp.ndarray) -> jnp.ndarray:
    """SO(3) rotation matrix from a (normalized) quaternion (w,x,y,z)."""
    w, x, y, z = normalize_quaternion(q)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    two = jnp.array(2.0, dtype=q.dtype)
    one = jnp.array(1.0, dtype=q.dtype)
    return jnp.array([
        [one - two*(yy + zz), two*(xy - wz),     two*(xz + wy)],
        [two*(xy + wz),       one - two*(xx+zz), two*(yz - wx)],
        [two*(xz - wy),       two*(yz + wx),     one - two*(xx+yy)],
    ], dtype=jnp.float32)

def xzy_from_rotmat(R: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Euler angles (a,b,g) for XZY order, i.e., Rx(a) Rz(b) Ry(g).

    Handles the gimbal case cos(b)≈0 by setting g=0 and solving a from first column.
    Returns angles wrapped to (-pi, pi].
    """
    r00, r01, r02 = R[0,0], R[0,1], R[0,2]
    r11, r21 = R[1,1], R[2,1]
    r10, r20 = R[1,0], R[2,0]

    cb = jnp.sqrt(r00*r00 + r02*r02)
    b = jnp.arctan2(-r01, cb)

    # Generic branch
    a_gen = jnp.arctan2(r21, r11)
    g_gen = jnp.arctan2(r02, r00)

    # Gimbal-lock branch (cb≈0): set g=0, solve a from first column
    a_gim = jnp.arctan2(r20, r10)
    g_gim = jnp.array(0.0, dtype=R.dtype)

    use_gim = cb < jnp.array(eps, dtype=R.dtype)
    a = jnp.where(use_gim, a_gim, a_gen)
    g = jnp.where(use_gim, g_gim, g_gen)

    # Wrap to (-pi, pi]
    a = _wrap_pi(a); b = _wrap_pi(b); g = _wrap_pi(g)
    return jnp.stack([a, b, g])

def xzy_from_su2(U: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute XZY Euler angles from an SU(2) matrix U.

    Reconstruct a real unit quaternion (w,x,y,z) from U, map to SO(3), then extract XZY angles.
    """
    u11, u12 = U[0,0], U[0,1]
    u21, u22 = U[1,0], U[1,1]
    # Recover quaternion components (robust to small numeric drift)
    w = 0.5 * (jnp.real(u11) + jnp.real(u22))
    x = -jnp.imag(u12)
    y = -jnp.real(u12)
    z = 0.5 * (jnp.imag(u22) - jnp.imag(u11))
    q = jnp.stack([w, x, y, z]).astype(jnp.float32)
    R = _rotmat_from_unit_quaternion(q)
    return xzy_from_rotmat(R, eps=eps)

def quaternion_to_xzy_angles(q: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Convert quaternion (w,x,y,z) directly to XZY Euler angles for Rx-Rz-Ry."""
    R = _rotmat_from_unit_quaternion(q)
    return xzy_from_rotmat(R, eps=eps)
