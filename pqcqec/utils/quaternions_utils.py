import jax.numpy as jnp
# from jax import random

# ---------------------------
# 1) SU(2) from unit quaternion
# ---------------------------
def normalize_quaternion(q, enforce_w_nonneg=True, eps=1e-8):
    q = jnp.asarray(q, dtype=jnp.float32)
    # Sanitize to avoid NaN/Inf flowing into normalization under JIT/vmap
    q = jnp.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q = q / (jnp.linalg.norm(q) + eps)
    if enforce_w_nonneg:
        q = jnp.where(q[0] < 0.0, -q, q)
    return q  # (w,x,y,z), ||q||=1

def su2_from_quaternion(q):
    """U = w I - i (x σx + y σy + z σz) ∈ SU(2)."""
    w, x, y, z = normalize_quaternion(q)
    # w, x, y, z = q
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
    alpha_case1 = _wrap_pi(jnp.angle(u11) - jnp.angle(u22))
    beta_case1 = 0.0
    gamma_case1 = 0.0

    # Case 2: beta is near pi
    alpha_case2 = 0.0
    beta_case2 = jnp.pi
    gamma_case2 = _wrap_pi(jnp.angle(-u21) - jnp.angle(u12))

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

