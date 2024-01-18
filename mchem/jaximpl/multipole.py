import jax
import jax.numpy as jnp
from ..units import INV_4PI_EPS0



def normalize(vec):
    vec_norm = vec / jnp.linalg.norm(vec)
    return vec_norm


@jax.vmap
@jax.jit
def get_local_to_global_matrix(pos, pos1, pos2, pos3, axistype):
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


@jax.jit
def thole_damp_coefs(alpha_i, alpha_j, dr, thole, fac):
    ai = jnp.mean(jnp.linalg.eigvalsh(alpha_i))
    aj = jnp.mean(jnp.linalg.eigvalsh(alpha_j))
    u = dr / (ai * aj) ** (1 / 6)
    au3 = thole * u**3
    exp_au3 = jnp.exp(-au3)
    thole3 = 1 - exp_au3 * fac
    thole5 = 1 - (1 + au3) * exp_au3 * fac
    thole7 = 1 - (1 + au3 + 0.6*au3*au3) * exp_au3 * fac
    return thole3, thole5, thole7


@jax.vmap
@jax.jit
def get_ind_aux_data(mi, alpha_i, alpha_j, drvec, thole, fac):
    """
    Returns
    -------
    f: The electric field at site j caused by permanent multipoles at site i
    tind: Interaction tensor of induced-induced dipoles
    """
    dr = jnp.linalg.norm(drvec)
    dr2 = dr * dr
    dr3 = dr2 * dr

    thole3, thole5, thole7 = thole_damp_coefs(alpha_i, alpha_j, dr, thole, fac)

    x, y, z = drvec[0], drvec[1], drvec[2]
    xy, xz, yz = x*y, x*z, y*z
    xy_3, xz_3, yz_3 = xy*3, xz*3, yz*3
    x2, y2, z2 = x*x, y*y, z*z
    x2_3, y2_3, z2_3 = x2*3, y2*3, z2*3
    x2_5, y2_5, z2_5 = x2*5, y2*5, z2*5
    x_3, y_3, z_3 = x*3, y*3, z*3
    dr2_x2_5_damp = thole5 * dr2 - thole7 * x2_5
    dr2_y2_5_damp = thole5 * dr2 - thole7 * y2_5
    dr2_z2_5_damp = thole5 * dr2 - thole7 * z2_5
    dr2_3 = dr2 * 3
    
    dr3_inv = 1 / dr3
    tx = dr3_inv * (-x) * thole3
    ty = dr3_inv * (-y) * thole3
    tz = dr3_inv * (-z) * thole3

    dr5_inv = dr3_inv / dr2
    txx = dr5_inv * (x2_3 * thole5 - dr2 * thole3)
    txy = dr5_inv * xy_3 * thole5
    txz = dr5_inv * xz_3 * thole5
    tyy = dr5_inv * (y2_3 * thole5 - dr2 * thole3)
    tyz = dr5_inv * yz_3 * thole5
    tzz = dr5_inv * (z2_3 * thole5 - dr2 * thole3)

    dr7_inv = dr5_inv / dr2
    txxx = dr7_inv * x_3 * (dr2_3 * thole5 - x2_5 * thole7)
    txxy = dr7_inv * y_3 * dr2_x2_5_damp
    txxz = dr7_inv * z_3 * dr2_x2_5_damp
    tyyy = dr7_inv * y_3 * (dr2_3 * thole5 - y2_5 * thole7)
    tyyx = dr7_inv * x_3 * dr2_y2_5_damp
    tyyz = dr7_inv * z_3 * dr2_y2_5_damp
    tzzz = dr7_inv * z_3 * (dr2_3 * thole5 - z2_5 * thole7)
    tzzx = dr7_inv * x_3 * dr2_z2_5_damp
    tzzy = dr7_inv * y_3 * dr2_z2_5_damp
    txyz = dr7_inv * (-15 * xy * z) * thole7

    ci = mi[0]
    di_x, di_y, di_z = mi[1], mi[2], mi[3]
    qi_xx, qi_xy, qi_xz, qi_yy, qi_yz, qi_zz = mi[4], mi[5], mi[6], mi[7], mi[8], mi[9]
    
    fx = -tx * ci \
        + txx * di_x + txy * di_y + txz * di_z \
        - (txxx * qi_xx + 2 * txxy * qi_xy + 2 * txxz * qi_xz + 2 * txyz * qi_yz + tyyx * qi_yy + tzzx * qi_zz) / 3
    fy = -ty * ci \
        + txy * di_x + tyy * di_y + tyz * di_z \
        - (txxy * qi_xx + 2 * tyyx * qi_xy + 2 * txyz * qi_xz + 2 * tyyz * qi_yz + tyyy * qi_yy + tzzy * qi_zz) / 3
    fz = -tz * ci \
        + txz * di_x + tyz * di_y + tzz * di_z \
        - (txxz * qi_xx + 2 * txyz * qi_xy + 2 * tzzx * qi_xz + 2 * tzzy * qi_yz + tyyz * qi_yy + tzzz * qi_zz) / 3
    
    f = jnp.array([fx, fy, fz])

    tind = jnp.array([
        [txx, txy, txz],
        [txy, tyy, tyz],
        [txz, tyz, tzz]
    ])

    return f, tind


@jax.vmap
@jax.jit
def rotate_anisopol_tensor(polar, rmat):
    polar_local = jnp.array([
        [polar[0], polar[1], polar[2]],
        [polar[1], polar[3], polar[4]],
        [polar[2], polar[4], polar[5]]
    ])
    polar_global = jnp.dot(jnp.dot(rmat.T, polar_local), rmat)
    return polar_global


@jax.vmap
@jax.jit
def pairwise_cp_corr(drvec, zi, zj, qi, qj, ai, aj, bi, bj):
    dr = jnp.linalg.norm(drvec)
    exp_ai = (zi - qi) * (1 - jnp.exp(-ai * dr))
    exp_aj = (zj - qj) * (1 - jnp.exp(-aj * dr))
    exp_bi = (zi - qi) * (1 - jnp.exp(-bi * dr))
    exp_bj = (zj - qj) * (1 - jnp.exp(-bj * dr))
    ene_cp = (zi * zj - zi * exp_aj - zj * exp_ai + exp_bi * exp_bj - qi * qj) / dr * INV_4PI_EPS0
    return ene_cp


@jax.jit
def esp_kernel(mi, coord, grid_points):
    drvec = grid_points - coord
    dr = jnp.linalg.norm(drvec, axis=1)
    dr2 = dr * dr
    dr3 = dr2 * dr

    x, y, z = drvec[:, 0], drvec[:, 1], drvec[:, 2]
    # T
    t = 1 / dr

    # Ta
    dr3_inv = 1 / dr3
    tx = dr3_inv * (-x)
    ty = dr3_inv * (-y)
    tz = dr3_inv * (-z)

    # Tab
    dr5_inv = dr3_inv / dr2
    txx = dr5_inv * (x*x*3 - dr2)
    txy = dr5_inv * x * y * 3
    txz = dr5_inv * x * z * 3
    tyy = dr5_inv * (y*y*3 - dr2)
    tyz = dr5_inv * y * z * 3
    tzz = dr5_inv * (z*z*3 - dr2) 

    ci = mi[0]
    di_x, di_y, di_z = mi[1], mi[2], mi[3]
    qi_xx, qi_xy, qi_xz, qi_yy, qi_yz, qi_zz = mi[4], mi[5], mi[6], mi[7], mi[8], mi[9]

    v = ci * t \
        - di_x * tx - di_y * ty - di_z * tz \
        + (qi_xx * txx + 2 * qi_xy * txy + 2 * qi_xz * txz + qi_yy * tyy + 2 * qi_yz * tyz + qi_zz * tzz) / 3
    return v


@jax.jit
def get_electrostatic_potential(mpoles, coords, grid_points):
    esp_kernel_vmap = jax.vmap(esp_kernel, in_axes=(0, 0, None), out_axes=0)
    return jnp.sum(
        esp_kernel_vmap(mpoles, coords, grid_points), axis=0
    ) * INV_4PI_EPS0


def generateMBUCBJaxFn(system):
    mpoleterms = system.data['Multipole']
    cpterms = system.data['MBUCBChargePenetration']
    polterms = system.data['AnisotropicPolarization']
    ctterms = system.data['MBUCBChargeTransfer']

    kz, kx, ky = [], [], []
    axistypes = []
    mpoleParamIdxs = []
    for term in mpoleterms:
        kz.append(term.kz)
        kx.append(term.kx)
        ky.append(term.ky)
        axistypes.append(term.axistype)
        mpoleParamIdxs.append(term.paramIdx)
    kz = jnp.array(kz)
    kx = jnp.array(kx)
    ky = jnp.array(ky)
    axistypes = jnp.array(axistypes)

    mpoleParamIdxs = jnp.array(mpoleParamIdxs, dtype=int)
    cpParamIdxs = jnp.array([term.paramIdx for term in cpterms], dtype=int)
    polParamIdxs = jnp.array([term.paramIdx for term in polterms], dtype=int)
    ctParamIdxs = jnp.array([term.paramIdx for term in ctterms], dtype=int)

    def elec(coords, pairs, param, scales):
        rot_mats = get_local_to_global_matrix(coords, coords[kz], coords[kx], coords[ky], axistypes)
        mpolesLocal = param['MultipoleGenerator']['multipoles']
        mpoles = rotate_multipoles(mpolesLocal[mpoleParamIdxs], rot_mats)
        mi = mpoles[pairs[:, 0]]
        mj = mpoles[pairs[:, 1]]
        drvec = coords[pairs[:, 1]] - coords[pairs[:, 0]]
        ene = mpole_elec(mi, mj, drvec, scales['mpole'])
        e_perm = jnp.sum(ene) / 2

        cp_alphas = param['MBUCBChargePenetrationGenerator']['alpha'][cpParamIdxs]
        cp_betas = param['MBUCBChargePenetrationGenerator']['beta'][cpParamIdxs]
        cp_zs = param['MBUCBChargePenetrationGenerator']['z'][cpParamIdxs]
        e_cp_corr = jnp.sum(pairwise_cp_corr(
            drvec, cp_zs[pairs[:, 0]], cp_zs[pairs[:, 1]], 
            mi[:, 0], mj[:, 0], 
            cp_alphas[pairs[:, 0]], cp_alphas[pairs[:, 1]],
            cp_betas[pairs[:, 0]], cp_betas[pairs[:, 1]]
        ) * scales['mpole']) / 2

        # polarization
        num_atoms = coords.shape[0]
        num_pairs = pairs.shape[0]

        direct_scales = scales['direct']
        mutual_scales = scales['mutual']
        polar_scales = scales['polar']
        polarsLocal = param['AnisotropicPolarizationGenerator']['alpha']
        polars = rotate_anisopol_tensor(polarsLocal[polParamIdxs], rot_mats)
        thole = param['AnisotropicPolarizationGenerator']['thole'][polParamIdxs]

        matrix = jnp.zeros((num_atoms * 3, num_atoms * 3))
        
        for i in range(num_atoms):
            matrix = matrix.at[i*3:(i+1)*3, i*3:(i+1)*3].set(jnp.linalg.inv(polars[i]))
        
        e_field = jnp.zeros((num_atoms * 3,))

        f, tind = get_ind_aux_data(
            mi, polars[pairs[:, 0]], polars[pairs[:, 1]],
            drvec,
            thole[pairs[:, 0]],
            jnp.array([1.0 for _ in range(num_pairs)])
        )

        for i in range(num_pairs):
            index_i_3, index_j_3 = pairs[i, 0] * 3, pairs[i, 1] * 3
            e_field = e_field.at[index_j_3: index_j_3 + 3].set(
                f[i] * direct_scales[i] + e_field[index_j_3: index_j_3 + 3]
            )
            matrix = matrix.at[index_i_3: index_i_3 + 3, index_j_3: index_j_3 + 3].set(-tind[i] * mutual_scales[i])

        ind_dipo = jnp.dot(e_field, jnp.linalg.inv(matrix).T)
        ind_dipo = ind_dipo.reshape(-1, 3)

        e_polar_pairwise = -jnp.sum(f * ind_dipo[pairs[:, 1]], axis=1) * polar_scales
        e_polar = jnp.sum(e_polar_pairwise) / 2 * INV_4PI_EPS0

        # charge transfer
        alpha_ct = param['MBUCBChargeTransferGenerator']['alpha'][ctParamIdxs].reshape(-1, 1, 1)
        alpha_ct = alpha_ct * jnp.array([jnp.eye(3) for _ in range(len(ctParamIdxs))])
        d_ct = param['MBUCBChargeTransferGenerator']['d'][ctParamIdxs]
        b_ct = param['MBUCBChargeTransferGenerator']['b'][ctParamIdxs]

        e_field_ct = jnp.zeros((num_atoms * 3,))
        matrix_ct = jnp.zeros((num_atoms * 3, num_atoms * 3))

        for i in range(num_atoms):
            matrix_ct = matrix_ct.at[i*3:(i+1)*3, i*3:(i+1)*3].set(jnp.linalg.inv(alpha_ct[i]))

        f_ct, tind_ct = get_ind_aux_data(
            mi, alpha_ct[pairs[:, 0]], alpha_ct[pairs[:, 1]],
            drvec,
            b_ct[pairs[:, 0]],
            d_ct[pairs[:, 0]] * d_ct[pairs[:, 1]]
        )

        for i in range(pairs.shape[0]):
            index_i_3, index_j_3 = pairs[i, 0] * 3, pairs[i, 1] * 3
            e_field_ct = e_field_ct.at[index_j_3: index_j_3 + 3].set(
                f_ct[i] * direct_scales[i] + e_field_ct[index_j_3: index_j_3 + 3]
            )
            matrix_ct = matrix_ct.at[index_j_3: index_j_3 + 3, index_i_3: index_i_3 + 3].set(-tind_ct[i] * mutual_scales[i])

        ind_dipo_ct = jnp.dot(e_field_ct, jnp.linalg.inv(matrix_ct).T)
        ind_dipo_ct = ind_dipo_ct.reshape(-1, 3)

        e_ct_pairwise = -jnp.sum(f_ct * ind_dipo_ct[pairs[:, 1]], axis=1) * polar_scales
        e_ct = jnp.sum(e_ct_pairwise) / 2 * INV_4PI_EPS0

        return e_perm, e_cp_corr, e_polar, e_ct

    return elec