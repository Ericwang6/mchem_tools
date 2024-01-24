import pytest
import itertools
from pathlib import Path
import jax.numpy as jnp

from mchem.fileformats import load_pdb
from mchem.forcefield import ForceField
from mchem.jaximpl.vdw import generateAmoebaVdwJaxFn

def test_vdw():
    top = load_pdb(Path(__file__).parent / "data/water_dimer.pdb")
    # coord = jnp.array(top.coordinates)
    coord = jnp.array([
        [0.583801, 0.000000, 0.759932],
        [0.000000,   0.000000,   0.000000 ],
        [0.000000,   0.000000,   1.530090 ],
        [-0.687305,   0.000000,   2.795684 ],
        [-0.448269,  -0.763921,   3.325671 ],
        [-0.448269,   0.763921,   3.325671]
    ])

    pairs = jnp.array([[a, b] for a, b in itertools.product([0, 1, 2], [3, 4, 5])])
    pairs = jnp.vstack((pairs, pairs[:, [1, 0]]))
    scales = jnp.ones(len(pairs))

    ff = ForceField("amoebabio18.xml")
    system = ff.createSystem(top)
    func = generateAmoebaVdwJaxFn(system)
    param = ff.getParameters()
    print(func(coord / 10, pairs, param, scales) / 4.184) # -0.0490
