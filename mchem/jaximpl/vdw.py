import jax
import jax.numpy as jnp


def generateAmoebaVdwJaxFn(system):
    vdwterms = system.data['AmoebaVdw147']
    parentIdxs, paramIdxs = [], []
    for term in vdwterms:
        paramIdxs.append(term.paramIdx)
        parentIdxs.append(term.parentIdx)
    parentIdxs = jnp.array(parentIdxs, dtype=int)
    paramIdxs = jnp.array(paramIdxs, dtype=int)

    delta = 0.07
    gamma = 0.12

    def vdw(coord, pairs, param, scales):
        eps = param['AmoebaVdwGenerator']['epsilon'][paramIdxs]
        sig = param['AmoebaVdwGenerator']['sigma'][paramIdxs]
        reduction = param['AmoebaVdwGenerator']['reduction'][paramIdxs].reshape(-1, 1)

        eps_i, eps_j = eps[pairs[:, 0]], eps[pairs[:, 1]]
        sig_i, sig_j = sig[pairs[:, 0]], sig[pairs[:, 1]]
        eps_ij = 4 * eps_i * eps_j / ((jnp.sqrt(eps_i) + jnp.sqrt(eps_j)) ** 2)
        sig_ij = (sig_i ** 3 + sig_j ** 3) / (sig_i ** 2 + sig_j ** 2)

        vdwCoord = coord * reduction + coord[parentIdxs] * (1 - reduction)
        dr = jnp.linalg.norm(vdwCoord[pairs[:, 0]] - vdwCoord[pairs[:, 1]], axis=1)
        
        rho = dr / sig_ij
        s1 = ((1 + delta) / (rho + delta)) ** 7
        s2 = (1 + gamma) / (rho ** 7 + gamma) - 2
        e_pairwise = eps_ij * s1 * s2
        ene = jnp.sum(e_pairwise * scales) / 2

        return ene
    
    return vdw