from ase.io import read, write
import numpy as np
from ase.io.pov import get_bondpairs, get_hydrogenbonds, set_high_bondorder_pairs
from ase.visualize import view

def supercell(atoms, n0, n1, n2):
    '''
    Create supercell from primitive cell `atoms`.
    This method is better than atoms * ( n0, n1, n2 ) because it keeps the order of atoms.
    '''
    org_atoms = atoms.copy()
    new_atoms = sub_atoms = org_atoms[[0]] * (n0, n1, n2)
    for i in range(1, org_atoms.get_number_of_atoms()):
        new_atoms += org_atoms[[i]] * (n0, n1, n2)
    return new_atoms

def get_atom_settings():
    with open('/shared/apps/VESTA-x86_64/elements.ini', 'r') as f:
        lines = f.readlines()
    elements = {}
    for line in lines:
        tmp = line.split()
        color = (float(tmp[5]), float(tmp[6]), float(tmp[7]))
        rcov = float(tmp[2])
        rvdw = float(tmp[3])
        r = float(tmp[4])
        element = tmp[1]
        elements[element] = {
            'r': r,
            'rvdw': rvdw,
            'rcov': rcov,
            'color': color
        }
    return elements

def get_figure(sys0, fout, rot="-30x", resol=1000, w=15, h=15, radius_factor=1, shift=-1, cutoff=12, hydrogenbond=(), shiftx=0, shifty=0):
    z = 0
    C = 204
    nat = len(sys0)
    system = supercell(sys0, 3, 3, 1)
    elements = get_atom_settings()

    if shift == -1:
        center = sys0[C].position + sys0.cell[0] + sys0.cell[1] - [0, 0, 0]
    else:
        center = sys0[shift].position + sys0.cell[0] + sys0.cell[1]

    for i in range(3):
        system.positions[:, i] -= center[i]

    del_list = [at.index for at in system if at.symbol in ['C', 'N', 'O', 'H', 'P'] and np.linalg.norm(at.position) > cutoff]
    for at in sorted(del_list, reverse=True):
        del system[at]

    radii = []
    colors = []
    for i, at in enumerate(system):
        radius = elements[at.symbol]['rcov']
        radii.append(radius / radius_factor)
        color = elements[at.symbol]['color']

        if at.symbol == 'C':
            color = (0.1, 0.1, 0.1)
        elif at.symbol == 'O':
            color = (1., 0, 0.)
        elif at.symbol == 'S':
            color = (1., 0, 1.)
        elif at.symbol == 'H':
            color = (1.0, 1.0, 1.0)

        colors.append((color[0], color[1], color[2], 0))

    bond_pairs = get_bondpairs(system, radius=0.8)
    hydrogenbond = get_hydrogenbonds(system, atype1='H', atype2=['O'], radius=4, rhbondrange=(1.3, 2.1))
    bond_pairs = set_high_bondorder_pairs(bond_pairs, high_bondorder_pairs=hydrogenbond)

    bbox = (-w / 2. + shiftx, -h / 2. + shifty, w / 2. + shiftx, h / 2. + shifty)
    write(fout + '.pov', system, format='pov', run_povray=True,
          canvas_width=resol,
          radii=radii,
          bbox=bbox,
          colors=colors,
          celllinewidth=0,
          rotation=rot,
          hydrogenbond={'ndots': 9, 'color': [0, 0, 0], 'rdot': 0.1})
