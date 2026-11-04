import numpy as np
import matplotlib.pyplot as plt
import math

try:
    from scipy.sparse import lil_matrix  # type: ignore[import-not-found]
    from scipy.sparse.linalg import spsolve  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "SciPy is required for this script. Install it with: pip install scipy"
    ) from exc


# ============================================================
# 1. INPUTS
# ============================================================

nr = 100
ntheta = 180

rpm = 3000
Re = 1200

Tin = 350.0
Twall = 420

swirl = 0.70
wall_htc = 150.0
loss = 0.05

r1 = 0.30
r2 = 0.120
b = 0.010

rho = 997.0
mu = 0.0010
cp = 4180.0
k = 0.60
g = 9.81


omega = rpm * 2 * math.pi / 60

Dh = 2 * b
u_in = Re * mu / (rho * Dh)

r = np.linspace(r1, r2, nr)
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)

dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

R, TH = np.meshgrid(r, theta, indexing="ij")

print(f"Grid = {nr} x {ntheta}")
print(f"dr = {dr:.6e} m")
print(f"dtheta = {dtheta:.6e} rad")


# ============================================================
# 4. VELOCITY
# ============================================================

# Radial velocity decreases as radius increases
u_r = u_in * r1 / r

# Tangential velocity due to rotation
u_theta = swirl * omega * r

# Relative tangential velocity
w_theta = u_theta - omega * r

ur = u_r[:, None] * np.ones((1, ntheta))
ut = u_theta[:, None] * np.ones((1, ntheta))
wr = w_theta[:, None] * np.ones((1, ntheta))


# ============================================================
# 5. PRESSURE FDM
# ============================================================

P = np.zeros(nr)

pressure_term = (
    rho * u_theta**2 / r
    - 12 * mu * u_r / b**2
)

P[1:] = np.cumsum(dr * pressure_term[:-1])


# Exact solution used only to check the FDM result
P_exact = (
    0.5 * rho * swirl**2 * omega**2 * (r**2 - r1**2)
    - 12 * mu * u_in * r1 * np.log(r / r1) / b**2
)


# ============================================================
# 6. THERMAL PROPERTIES
# ============================================================

alpha = k / (rho * cp)

# Heat transfer from the wall
beta = 2 * wall_htc / (rho * cp * b)

# Small heat source due to losses
source = loss * np.abs(u_r) * (omega * r)**2 / (b * cp)


# ============================================================
# 7. CREATE FDM MATRIX
# ============================================================

N = nr * ntheta

A = lil_matrix((N, N))
B = np.zeros(N)


def node(i, j):
    return i * ntheta + j


# ============================================================
# 8. BOUNDARY CONDITIONS
# ============================================================

for j in range(ntheta):

    # Inlet temperature
    row = node(0, j)

    A[row, row] = 1
    B[row] = Tin


    # Outlet: dT/dr = 0
    row = node(nr - 1, j)

    A[row, node(nr - 1, j)] = 3
    A[row, node(nr - 2, j)] = -4
    A[row, node(nr - 3, j)] = 1


# ============================================================
# 9. FDM EQUATION INSIDE THE DOMAIN
# ============================================================

for i in range(1, nr - 1):

    ri = r[i]

    # Diffusion terms
    ar = alpha / dr**2
    at = alpha / (ri**2 * dtheta**2)

    # Local velocities
    vr = u_r[i]
    vt = w_theta[i] / ri


    # Upwind scheme for convection
    aw = ar + max(vr, 0) / dr
    ae = ar + max(-vr, 0) / dr

    ass = at + max(vt, 0) / dtheta
    an = at + max(-vt, 0) / dtheta

    ap = aw + ae + ass + an + beta


    for j in range(ntheta):

        row = node(i, j)

        # Centre node
        A[row, row] = ap

        # Radial neighbours
        A[row, node(i - 1, j)] = -aw
        A[row, node(i + 1, j)] = -ae

        # Angular neighbours
        # % ntheta makes the solution periodic
        A[row, node(i, (j - 1) % ntheta)] = -ass
        A[row, node(i, (j + 1) % ntheta)] = -an

        # Right hand side
        B[row] = beta * Twall + source[i]


# ============================================================
# 10. SOLVE TEMPERATURE FIELD
# ============================================================

A = A.tocsr()

T = spsolve(A, B).reshape(nr, ntheta)

if not np.all(np.isfinite(T)):
    raise ValueError("Temperature solution is not valid")

print("\nFDM solution completed")
print(f"Minimum temperature = {T.min():.4f} K")
print(f"Maximum temperature = {T.max():.4f} K")
print(f"Outlet temperature  = {T[-1, :].mean():.4f} K")


# ============================================================
# 11. VELOCITY IN X-Y DIRECTION
# ============================================================

x = R * np.cos(TH)
y = R * np.sin(TH)

u_abs_x = ur * np.cos(TH) - ut * np.sin(TH)
u_abs_y = ur * np.sin(TH) + ut * np.cos(TH)

u_rel_x = ur * np.cos(TH) - wr * np.sin(TH)
u_rel_y = ur * np.sin(TH) + wr * np.cos(TH)

speed = np.sqrt(u_rel_x**2 + u_rel_y**2)


# ============================================================
# 12. PLOTS
# ============================================================

fig, ax = plt.subplots(1, 3, figsize=(16, 5))


# Temperature
p = ax[0].pcolormesh(
    x, y, T,
    shading="auto",
    cmap="inferno"
)

ax[0].set_title("Temperature [K]")
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
ax[0].set_aspect("equal")
fig.colorbar(p, ax=ax[0])


# Pressure
ax[1].plot(r, P, "o-", label="FDM")
ax[1].plot(r, P_exact, "--", label="Exact")

ax[1].set_xlabel("Radius [m]")
ax[1].set_ylabel("Pressure [Pa]")
ax[1].set_title("Radial Pressure Rise")
ax[1].grid(True)
ax[1].legend()


# Relative velocity
p = ax[2].pcolormesh(
    x, y, speed,
    shading="auto"
)

ax[2].set_title("Relative Velocity")
ax[2].set_xlabel("x [m]")
ax[2].set_ylabel("y [m]")
ax[2].set_aspect("equal")
fig.colorbar(p, ax=ax[2])


plt.tight_layout()
plt.show()


# ============================================================
# 13. VELOCITY VECTORS
# ============================================================

skip_r = max(1, nr // 10)
skip_t = max(1, ntheta // 16)

plt.figure(figsize=(7, 6))

plt.pcolormesh(
    x, y, speed,
    shading="auto"
)

plt.quiver(
    x[::skip_r, ::skip_t],
    y[::skip_r, ::skip_t],
    u_rel_x[::skip_r, ::skip_t],
    u_rel_y[::skip_r, ::skip_t],
    color="white",
    scale=20
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Relative Velocity Vectors")
plt.axis("equal")
plt.colorbar(label="Velocity [m/s]")
plt.show()


# ============================================================
# 14. CHECK THE FDM RESULT
# ============================================================

# Inlet temperature error
inlet_error = np.max(
    np.abs(T[0, :] - Tin)
)


# Outlet dT/dr should be approximately zero
outlet_gradient = np.max(
    np.abs(
        (3*T[-1, :] - 4*T[-2, :] + T[-3, :])
        / (2 * dr)
    )
)


# Temperature should be periodic in theta
periodic_error = np.max(
    np.abs(T[:, 0] - T[:, -1])
)


# Difference between FDM and exact pressure
pressure_error = np.max(
    np.abs(P - P_exact)
)


# ============================================================
# 15. FINAL RESULTS
# ============================================================

mass_flow = rho * u_in * (2 * math.pi * r1 * b)

pump_head = (
    P[-1] - P[0]
) / (rho * g)

heat_gain = (
    mass_flow * cp *
    (T[-1, :].mean() - Tin)
)


print("\n========== FDM CHECK ==========")

print(f"Inlet temperature error       : {inlet_error:.6e} K")
print(f"Outlet Neumann error          : {outlet_gradient:.6e} K/m")
print(f"Angular periodicity error     : {periodic_error:.6e} K")
print(f"Pressure FDM error             : {pressure_error:.6e} Pa")

print("\n========== PUMP RESULTS ==========")

print(f"Pressure rise                 : {P[-1] - P[0]:.4f} Pa")
print(f"Pump head                     : {pump_head:.4f} m")
print(f"Mass flow rate                : {mass_flow:.6f} kg/s")
print(f"Heat gained by fluid          : {heat_gain:.4f} W")

print("\n========== TEMPERATURE ==========")

print(f"Inlet temperature             : {Tin:.2f} K")
print(f"Wall temperature              : {Twall:.2f} K")
print(f"Outlet temperature            : {T[-1, :].mean():.2f} K")
print(f"Temperature range             : {T.min():.2f} - {T.max():.2f} K")
