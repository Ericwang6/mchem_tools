import jax
import jax.numpy as jnp
from ..units import INV_4PI_EPS0


def normalize(vec):
    vec_norm = vec / jnp.linalg.norm(vec)
    return vec_norm


@jax.vmap
@jax.jit
def get_local_to_global_matrix(pos, pos1, pos2, axistype):
    # ZThenX
    zvec = normalize(pos1 - pos)
    xvec = normalize(pos2 - pos)
    # Bisector  
    zvec += xvec * (axistype == 1)
    zvec = normalize(zvec)
    
    xvec = xvec - jnp.sum(zvec * xvec) * zvec
    xvec = normalize(xvec)
    yvec = jnp.cross(zvec, xvec)
    mat = jnp.vstack((xvec, yvec, zvec))
    
    return mat


@jax.vmap
@jax.jit
def rotate_multipoles(mpole, mat):
    dipo = jnp.dot(mpole[1:4], mat)
    quad = jnp.array([
        [mpole[4], mpole[5], mpole[6]],
        [mpole[5], mpole[7], mpole[8]],
        [mpole[6], mpole[8], mpole[9]]
    ])
    quad = jnp.dot(jnp.dot(mat.T, quad), mat)
    mpole = mpole.at[1:4].set(dipo)
    mpole = mpole.at[4:10].set(quad[jnp.triu_indices(3)])
    return mpole


@jax.vmap
@jax.jit
def mpole_elec(mi, mj, drvec, scale):
    dr = jnp.linalg.norm(drvec)
    dr2 = dr * dr
    dr3 = dr2 * dr

    x, y, z = drvec[0], drvec[1], drvec[2]
    xy, xz, yz = x*y, x*z, y*z
    xy_3, xz_3, yz_3 = xy*3, xz*3, yz*3
    x2, y2, z2 = x*x, y*y, z*z
    x2_3, y2_3, z2_3 = x2*3, y2*3, z2*3
    x2_5, y2_5, z2_5 = x2*5, y2*5, z2*5
    x_3, y_3, z_3 = x*3, y*3, z*3
    dr2_x2_5, dr2_y2_5, dr2_z2_5 = dr2-x2_5, dr2-y2_5, dr2-z2_5
    dr2_3 = dr2 * 3

    # T
    t = 1 / dr

    # Ta
    dr3_inv = 1 / dr3
    tx = dr3_inv * (-x)
    ty = dr3_inv * (-y)
    tz = dr3_inv * (-z)

    # Tab
    dr5_inv = dr3_inv / dr2
    x2_3_dr2, y2_3_dr2, z2_3_dr2 = x2_3 - dr2, y2_3 - dr2, z2_3 - dr2
    txx = dr5_inv * x2_3_dr2
    txy = dr5_inv * xy_3
    txz = dr5_inv * xz_3
    tyy = dr5_inv * y2_3_dr2
    tyz = dr5_inv * yz_3
    tzz = dr5_inv * z2_3_dr2 
    
    # Tabc
    dr7_inv = dr5_inv / dr2
    txxx = dr7_inv * x_3 * (dr2_3 - x2_5)
    txxy = dr7_inv * y_3 * dr2_x2_5
    txxz = dr7_inv * z_3 * dr2_x2_5
    tyyy = dr7_inv * y_3 * (dr2_3 - y2_5)
    tyyx = dr7_inv * x_3 * dr2_y2_5
    tyyz = dr7_inv * z_3 * dr2_y2_5
    tzzz = dr7_inv * z_3 * (dr2_3 - z2_5)
    tzzx = dr7_inv * x_3 * dr2_z2_5
    tzzy = dr7_inv * y_3 * dr2_z2_5
    txyz = dr7_inv * (-15 * xy * z)

    # Tabcd
    dr4_9 = dr2_3 * dr2_3

    dr9_inv = dr7_inv / dr2
    txxxx = dr9_inv * (105*x2*x2 - 90*x2*dr2 + dr4_9)
    txxxy = dr9_inv * 15 * xy * (7*x2 - dr2_3)
    txxxz = dr9_inv * 15 * xz * (7*x2 - dr2_3) 
    txxyy = dr9_inv * (105*x2*y2 - dr2_3 * (dr2_3 + dr2_z2_5))
    txxzz = dr9_inv * (105*x2*z2 - dr2_3 * (dr2_3 + dr2_y2_5))
    txxyz = dr9_inv * 15 * yz * (7*x2 - dr2)

    tyyyy = dr9_inv * (105*y2*y2 - 90*y2*dr2 + dr4_9)
    tyyyx = dr9_inv * 15 * xy * (7*y2 - dr2_3)
    tyyyz = dr9_inv * 15 * yz * (7*y2 - dr2_3)
    tyyzz = dr9_inv * (105*y2*z2 - dr2_3 * (dr2_3 + dr2_x2_5))
    tyyxz = dr9_inv * 15 * xz * (7*y2 - dr2)

    tzzzz = dr9_inv * (105*z2*z2 - 90*z2*dr2 + dr4_9)
    tzzzx = dr9_inv * 15 * xz * (7*z2 - dr2_3)
    tzzzy = dr9_inv * 15 * yz * (7*z2 - dr2_3)
    tzzxy = dr9_inv * 15 * xy * (7*z2 - dr2)
    
    ci, cj = mi[0], mj[0]
    di_x, di_y, di_z = mi[1], mi[2], mi[3]
    dj_x, dj_y, dj_z = mj[1], mj[2], mj[3]
    qi_xx, qi_xy, qi_xz, qi_yy, qi_yz, qi_zz = mi[4], mi[5], mi[6], mi[7], mi[8], mi[9]
    qj_xx, qj_xy, qj_xz, qj_yy, qj_yz, qj_zz = mj[4], mj[5], mj[6], mj[7], mj[8], mj[9]
    
    # mono-mono
    cc = t * ci * cj

    # mono-dipo
    cd = tx * (ci * dj_x - cj * di_x) \
        + ty * (ci * dj_y - cj * di_y) \
        + tz * (ci * dj_z - cj * di_z)
    
    # mono-quad
    cq = txx * (ci * qj_xx + cj * qi_xx) \
        + txy * 2 * (ci * qj_xy + cj * qi_xy) \
        + txz * 2 * (ci * qj_xz + cj * qi_xz) \
        + tyy * (ci * qj_yy + cj * qi_yy) \
        + tyz * 2 * (ci * qj_yz + cj * qi_yz) \
        + tzz * (ci * qj_zz + cj * qi_zz)
    cq = cq / 3

    # dipo-dipo
    dd = -txx * di_x * dj_x \
        - txy * (di_x * dj_y + di_y * dj_x) \
        - txz * (di_x * dj_z + di_z * dj_x) \
        - tyy * di_y * dj_y \
        - tyz * (di_y * dj_z + di_z * dj_y) \
        - tzz * di_z * dj_z
    
    # dipo-quad
    dq = txxx * (qi_xx * dj_x - qj_xx * di_x) \
        + txxy * (2 * qi_xy * dj_x + qi_xx * dj_y - 2 * qj_xy * di_x - qj_xx * di_y) \
        + txxz * (2 * qi_xz * dj_x + qi_xx * dj_z - 2 * qj_xz * di_x - qj_xx * di_z) \
        + tyyy * (qi_yy * dj_y - qj_yy * di_y) \
        + tyyx * (2 * qi_xy * dj_y + qi_yy * dj_x - 2 * qj_xy * di_y - qj_yy * di_x) \
        + tyyz * (2 * qi_yz * dj_y + qi_yy * dj_z - 2 * qj_yz * di_y - qj_yy * di_z) \
        + tzzz * (qi_zz * dj_z - qj_zz * di_z) \
        + tzzx * (2 * qi_xz * dj_z + qi_zz * dj_x - 2 * qj_xz * di_z - qj_zz * di_x) \
        + tzzy * (2 * qi_yz * dj_z + qi_zz * dj_y - 2 * qj_yz * di_z - qj_zz * di_y) \
        + txyz * 2 * (qi_xy * dj_z + qi_xz * dj_y + qi_yz * dj_x - qj_xy * di_z - qj_xz * di_y - qj_yz * di_x)
    dq = dq / 3

    # quad-quad
    qq = txxxx * qi_xx * qj_xx \
        + txxxy * 2 * (qi_xx * qj_xy + qi_xy * qj_xx) \
        + txxxz * 2 * (qi_xx * qj_xz + qi_xz * qj_xx) \
        + txxyy * (4 * qi_xy * qj_xy + qi_xx * qj_yy + qi_yy * qj_xx) \
        + txxzz * (4 * qi_xz * qj_xz + qi_xx * qj_zz + qi_zz * qj_xx) \
        + txxyz * 2 * (qi_xx * qj_yz + qi_yz * qj_xx + 2 * qi_xz * qj_xy + 2 * qi_xy * qj_xz) \
        + tyyyy * qi_yy * qj_yy \
        + tyyyx * 2 * (qi_yy * qj_xy + qi_xy * qj_yy) \
        + tyyyz * 2 * (qi_yy * qj_yz + qi_yz * qj_yy) \
        + tyyzz * (4 * qi_yz * qj_yz + qi_yy * qj_zz + qi_zz * qj_yy) \
        + tyyxz * 2 * (qi_yy * qj_xz + qi_xz * qj_yy + 2 * qi_xy * qj_yz + 2 * qi_yz * qj_xy) \
        + tzzzz * qi_zz * qj_zz \
        + tzzzx * 2 * (qi_zz * qj_xz + qi_xz * qj_zz) \
        + tzzzy * 2 * (qi_zz * qj_yz + qi_yz * qj_zz) \
        + tzzxy * 2 * (qi_zz * qj_xy + qi_xy * qj_zz + 2 * qi_xz * qj_yz + 2 * qi_yz * qj_xz)
    qq = qq / 9

    ene = scale * INV_4PI_EPS0 * (cc + cd + cq + dd + dq + qq)
    return ene


def generatePermElectrostaticsJaxFn(terms):
    kz, kx, ky = [], [], []
    axistypes = []
    paramIdxs = []
    for term in terms:
        kz.append(term.kz)
        kx.append(term.kx)
        ky.append(term.ky)
        axistypes.append(term.axistype)
        paramIdxs.append(term.idx)
    kz = jnp.array(kz)
    kx = jnp.array(kx)
    ky = jnp.array(ky)
    axistypes = jnp.array(axistypes)
    
    def perm_elec(coords, pairs, param):
        rot_mats = get_local_to_global_matrix(coords, coords[kz], coords[kx], axistypes)
        multipoles = rotate_multipoles(param['multipoles'][paramIdxs], rot_mats)
        mi = multipoles[pairs[:, 0]]
        mj = multipoles[pairs[:, 1]]
        drvec = coords[pairs[:, 1]] - coords[pairs[:, 0]]
        scales = jnp.ones((pairs.shape[0],))
        ene = mpole_elec(mi, mj, drvec, scales)
        return jnp.sum(ene)
    
    return perm_elec
