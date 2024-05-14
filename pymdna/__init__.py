from .utils import RigidBody, Shapes
from .spline import SplineFrames, Twister
from .geometry import ReferenceBase, NucleicFrames
from .generators import SequenceGenerator, StructureGenerator
from .modifications import Mutate, Hoogsteen 
from .analysis import GrooveAnalysis, TorsionAnalysis, ContactCount

import numpy as np

def check_input(sequence=None, n_bp=None):

    if sequence is None and n_bp is not None:
        sequence = ''.join(np.random.choice(list('ACGT'), n_bp))
        print('Random sequence:', sequence)

    elif sequence is not None and n_bp is None:
        n_bp = len(sequence)
        print('Sequence:', sequence)
        print('Number of base pairs:', n_bp)

    elif sequence is None and n_bp is None:
        sequence = 'CGCGAATTCGCG'
        n_bp = len(sequence)
        print('Default sequence:', sequence)
        print('Number of base pairs:', n_bp)

    elif sequence is not None and n_bp is not None:
        if n_bp != len(sequence):
            raise ValueError('Sequence length and n_bp do not match')
        print('Sequence:', sequence)
        print('Number of base pairs:', n_bp)
        
    return sequence, n_bp


def sequence_to_pdb(sequence='CGCGAATTCGCG', filename='my_dna', save=True):
    """Sequence to MDtraj object with option to save as pdb file 
        adhering to the AMBER force field format"""

    sequence, _ = check_input(sequence=sequence)

    # Linear strand of control points 
    point = Shapes.line((len(sequence)-1)*0.34)
    # Convert the control points to a spline
    spline = SplineFrames(point)
    # Generate the DNA structure
    generator = StructureGenerator(sequence=sequence,spline=spline)

    # Edit the DNA structure to make it compatible with the AMBER force field
    traj = generator.traj
    phosphor_termini = traj.top.select(f'name P OP1 OP2 and resid 0 {traj.top.chain(0).n_residues}')
    all_atoms = traj.top.select('all')
    traj = traj.atom_slice([at for at in all_atoms if at not in phosphor_termini])

    # Save the DNA structure as pdb file
    if save:
        traj.save(f'./{filename}.pdb')

    return traj