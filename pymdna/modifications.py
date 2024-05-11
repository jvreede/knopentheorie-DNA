import copy
from .utils import get_base_pair_dict, RigidBody, get_data_file_path
import numpy as np
from .geometry import ReferenceBase
import mdtraj as md



class Hoogsteen:

    def __init__(self, traj, fliplist, deg=180):
        self.traj = copy.deepcopy(traj)
        self.fliplist = fliplist # List of resids that need to be flipped
        self.theta = np.deg2rad(deg) # Set the rotation angle to 180 degrees
        self.apply_flips()
        
    def select_atom_by_name(self, traj, name):
        # Select an atom by name returns shape (n_frames, 1, [x,y,z])
        return np.squeeze(traj.xyz[:,[traj.topology.select(f'name {name}')[0]],:],axis=1)
        

    def get_base_indices(self, traj, resid=0):

        # Define the atoms that belong to the nucleotide
        base_atoms = { 'C2','C4','C5','C6','C7','C8','C5M',
                       'N1','N2','N3','N4','N6','N7','N9',
                       'O2','O4','O6',
                       'H1','H2','H3','H5','H6','H8',
                       'H21','H22','H41','H42','H61','H62','H71','H72','H73'}
            
            # 'N9', 'N7', 'C8', 'C5', 'C4', 'N3', 'C2', 'N1', 
            #         'C6', 'C7','O6', 'N2', 'N6', 'O2', 'N4', 'O4', 'C5M',
            #         'H1','H2','H21','H22','H3','H41','H42','H5','H6','H61','H62','H71','H72','H73','H8'}
        # Select atoms that belong to the specified residue
        indices = traj.top.select(f'resid {resid}')
        offset = indices[0]  # Save the initial index of the residue
    
        # Create a subtrajectory containing only the specified residue
        subtraj = traj.atom_slice(indices)

        # Select the atoms that belong to the nucleotide
        sub_indices = subtraj.top.select(f'name {" ".join(base_atoms)}')
        # Return the indices of the atoms that belong to the nucleotide
        return sub_indices + offset
    
    def apply_flips(self):

        # For each residue that needs to be mutated
        for resid in self.fliplist:

            # Get the indices of the atoms that need to be transformed
            nucleobase_selection = self.get_base_indices(self.traj, resid=resid)

            # Get the coordinates of the atoms involved in the rotation
            c1_prime_coords = self.select_atom_by_name(self.traj, f'"C1\'" and resid {resid}')
            n9_coords = self.select_atom_by_name(self.traj, f"N9 and resid {resid}")

            # Calculate the Euler vector for the 180-degree rotation around the specified axis and normalize the axis vector
            rotation_axis = c1_prime_coords - n9_coords
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Update the xyz of the nucleobase in base_A.xyz using the rotation
            relative_positions = self.traj.xyz[:, nucleobase_selection, :] - n9_coords[:, None, :]

            # Apply the rotation to each atom's relative position
            rotated_positions = np.array([RigidBody.rotate_vector(v, rotation_axis[0], self.theta) for v in relative_positions[0]])

            # Translate the rotated positions back to the original coordinate system
            new_xyz = rotated_positions + n9_coords[:, None, :]

            # Update the coordinates in the trajectory
            self.traj.xyz[:, nucleobase_selection, :] = new_xyz


class Mutate:

    def __init__(self, traj, mutations, complementary=True):
    
        self.traj = traj
        self.complementary = complementary
        self.mutations = mutations
        self.mutate()

    def mutate(self):

        # Define the base pair map and the complementary mutant map
        base_pair_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}#,'P':'T','M':'C','H':'T'}

        if self.complementary:
            # Update dict with the complementary mutations
            self.mutations = self.make_complementary_mutations(self.traj, self.mutations, base_pair_map)

        # Apply the mutations
        self.mutant_traj = self.apply_mutations(self.traj, self.mutations, base_pair_map)
            

    def get_base_indices(self, traj, resid=0):

        # Define the atoms that belong to the nucleotide
        base_atoms = { 'C2','C4','C5','C6','C7','C8','C5M',
                       'N1','N2','N3','N4','N6','N7','N9',
                       'O2','O4','O6',
                       'H1','H2','H3','H5','H6','H8',
                       'H21','H22','H41','H42','H61','H62','H71','H72','H73'}
            
            # 'N9', 'N7', 'C8', 'C5', 'C4', 'N3', 'C2', 'N1', 
            #         'C6', 'C7','O6', 'N2', 'N6', 'O2', 'N4', 'O4', 'C5M',
            #         'H1','H2','H21','H22','H3','H41','H42','H5','H6','H61','H62','H71','H72','H73','H8'}
        # Select atoms that belong to the specified residue
        indices = traj.top.select(f'resid {resid}')
        offset = indices[0]  # Save the initial index of the residue
    
        # Create a subtrajectory containing only the specified residue
        subtraj = traj.atom_slice(indices)

        # Select the atoms that belong to the nucleotide
        sub_indices = subtraj.top.select(f'name {" ".join(base_atoms)}')
        # Return the indices of the atoms that belong to the nucleotide
        return sub_indices + offset


    def make_complementary_mutations(self, traj, mutations, base_pair_map):
        
        # Get the basepair dictionary of the trajectory
        basepair_dict = get_base_pair_dict(traj)

        # Iterate over a static list of dictionary items to avoid RuntimeError
        for idx, base in list(mutations.items()):
            
            # Get the complementary base
            comp_base = basepair_dict[traj.top._residues[idx]]
            comp_mutant = base_pair_map[base]
        
            # Update mutations with the complementary base's mutation
            mutations[comp_base.index] = comp_mutant
            
        return mutations

    def update_mutant_topology(self, traj, target_indices, mutant_indices, base, resid, mutation_traj):
        # Store pre-deletion atom names and indices for comparison
        pre_atoms = [(atom.name, atom.index) for atom in traj.top._residues[resid]._atoms]

        # Delete target atoms from the topology
        self._delete_target_atoms(traj, target_indices)

        # Store post-deletion atom names and indices for offset calculation
        post_atoms = [(atom.name, atom.index) for atom in traj.top._residues[resid]._atoms]

        # Determine the insertion offset by comparing pre and post deletion atom names and indices
        offset, insert_id = self._find_insertion_offset(pre_atoms, post_atoms, traj)

        # Insert new atoms into the topology at calculated positions
        self._insert_new_atoms(traj, resid, mutant_indices, mutation_traj, offset, insert_id)

        # Update the residue name to reflect the mutation
        traj.top._residues[resid].name = f'D{base}'

        return traj

    def _delete_target_atoms(self, traj, target_indices):
        """
        Delete target atoms from the topology by sorting indices in reverse order
        to maintain index integrity after each deletion.
        """
        for index in sorted(target_indices, reverse=True):
            traj.top.delete_atom_by_index(index)

    def _find_insertion_offset(self, pre_atoms, post_atoms, traj):
        """
        Determine the correct offset for new atom insertion by comparing
        pre- and post-deletion atom names and indices.
        """
        # Default to the last atom's index in the residue as the insertion point
        offset = post_atoms[-1][1]
        insert_id = len(post_atoms) - 1

        # Check for the actual offset where the first discrepancy in atom names occurs
        # loop over name,index pairs in pre_atoms and post_atoms
        for pre, post in zip(pre_atoms, post_atoms):
            if pre[0] != post[0]:
                offset = pre[1]
                insert_id = post_atoms.index(post)
                break

        return offset, insert_id
    
    def _insert_new_atoms(self, traj, resid, mutant_indices, mutation_traj, offset, insert_id):
        """
        Insert new atoms into the topology, accounting for edge cases when the insertion point
        is at the end of the topology.
        """
        for idx, mutant_index in enumerate(mutant_indices):
            atom = mutation_traj.top.atom(mutant_index)

            # Edge case: If the offset is the last atom in the topology, insert new atoms at the end
            if offset + idx >= traj.top.n_atoms:
                print('Edgecase: inserting at or beyond the last atom in the topology', offset, traj.top.n_atoms)
                traj.top.insert_atom(atom.name, atom.element, traj.top._residues[resid],
                                    index=traj.top.n_atoms + idx, rindex=insert_id + idx)
            else:
                # Regular case: insert new atoms at the calculated offset
                traj.top.insert_atom(atom.name, atom.element, traj.top._residues[resid],
                                    index=offset + idx, rindex=insert_id + idx)

    def get_base_transformation(self, mutant_reference,target_reference):

        # Collect the reference information of the mutation
        mutation_origin = mutant_reference.b_R[0]
        D = mutant_reference.b_D[0]
        L = mutant_reference.b_L[0]
        N = mutant_reference.b_N[0]
        mutation_basis = np.array([D,L,N])

        # Collect the reference information of the target to mutate
        target_ref = ReferenceBase(target_reference)
        target_origin = target_ref.b_R[0]
        target_basis = np.array([target_ref.b_D[0],target_ref.b_L[0],target_ref.b_N[0]])

        # Calculate the transformation 
        rot = np.linalg.solve(target_basis,mutation_basis)
        trans = target_origin - mutation_origin
        return rot, trans


    def apply_mutations(self, traj, mutations, base_pair_map):

        # Make a copy of the original trajectory
        traj = copy.deepcopy(traj)
        
        # This comes now directly from sequence generator.py, either move this to somewhere else such that it is accessible everywhere
        #reference_bases = {base: md.load_pdb(f'/Users/thor/surfdrive/Projects/pymdna/pymdna/atomic/NDB96_{base}.pdb') for base in base_pair_map.keys()}
        reference_bases = {base: md.load_pdb(get_data_file_path(f'./atomic/NDB96_{base}.pdb')) for base in base_pair_map.keys()}
        reference_frames = {letter: ReferenceBase(t) for letter,t in reference_bases.items()}

        # For each residue that needs to be mutated
        for resid,base in mutations.items():

            # Get the mutant trajectory object
            mutation_traj = reference_bases[base] 

            # Get the indices of the atoms that need to be transformed
            mutant_indices = self.get_base_indices(mutation_traj, resid=0)
            target_indices = self.get_base_indices(traj, resid=resid)
            
            # Get the transformation for the local reference frames from the mutant to the target
            mutant_reference = reference_frames[base]
            target_reference = traj.atom_slice(traj.top.select(f'resid {resid}'))
            rot, trans = self.get_base_transformation(mutant_reference, target_reference)
           
            # Transform the mutant atoms to the local reference frame of the target
            mutant_xyz = mutation_traj.xyz[:,mutant_indices,:]
            new_xyz = np.dot(mutant_xyz, rot.T) + trans    

            # Get the original xyz coordinates
            xyz = traj.xyz 

            # Split the xyz in 2 pieces, one before the indices that need to be replaced, and the indices after the indices that need to be replaced
            xyz1 = xyz[:,:target_indices[0],:] 
            xyz2 = xyz[:,target_indices[-1]+1:,]

            # Update the topology
            traj = self.update_mutant_topology(traj, target_indices, mutant_indices, base, resid, mutation_traj)

            # Concatenate the new xyz with the original xyz
            xyz = np.concatenate([xyz1, new_xyz, xyz2], axis=1)
            traj.xyz = xyz

        # Return the mutated trajectory
        return traj

# traj = md.load('/Users/thor/surfdrive/Data/h-ns/BacterialChromatin/FI/0_k/2_ApT/dry_0.pdb')
# traj.remove_solvent(inplace=True)
# traj = traj.atom_slice(traj.topology.select('not protein'))[0]
# # traj = traj.atom_slice(traj.topology.select('resid 0 23'))# and not element symbol H'))

# # Create a DNA object
# dna = mdna.NucleicFrames(traj)

# # Define the base pair map and the complementary mutant map
# base_pair_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}#,'P':'T','M':'C','H':'T'}


# # Get base frames from current DNA sequence
# base_frames = dna.frames

# # # Define the mutation to be performed
# complementary = True
# mutations = {0: 'A', 6: 'T'}

# if complementary:
#     # Update dict with the complementary mutations
#     mutations = make_complementary_mutations(traj, mutations, base_pair_map)

# print(mutations)
# sequence = mdna.utils.get_sequence_letters(traj)
# sequence_pairs = mdna.utils.get_base_pair_letters(traj)
# new_sequences = [mutations.get(idx, seq) for idx, seq in enumerate(sequence)]

# print('WT',sequence)
# # print(new_sequences)

# # Apply the mutations
# mutant_traj = apply_mutations(traj, mutations, base_pair_map)
    
# new_sequence = mdna.utils.get_sequence_letters(mutant_traj)
# print('M ',new_sequence)
# view = nv.show_mdtraj(mutant_traj)
# view.clear_representations()
# view.add_representation('ball+stick', selection='all')
# view