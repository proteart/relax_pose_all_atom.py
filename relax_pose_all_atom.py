
import sys
import os
import math
import numpy as np
from pyrosetta import rosetta
from pyrosetta import init
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.core.pack.task.operation import IncludeCurrent
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.core.scoring import score_type_from_name
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
from pyrosetta.rosetta.core.scoring.constraints import DihedralConstraint
from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
from pyrosetta.rosetta.core.pose import addVirtualResAsRoot
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.moves import DsspMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

print("")
print("Initializing PyRosetta...")

print("")
init("-multithreading:total_threads 1 -ignore_waters false -ignore_unrecognized_res -ex1 -ex2 -ex2aro -ex3 -ex4 -extrachi_cutoff 0 -use_input_sc -detect_disulf -no_optH false -flip_HNQ -mute core.scoring.etable basic.io.database core.chemical.GlobalResidueTypeSet core.import_pose.import_pose core.io.pdb.file_data core.io.pose_from_sfr.PoseFromSFRBuilder core.io.pose_from_sfr.chirality_resolution core.energy_methods.CartesianBondedEnergy protocols.DsspMover core.kinematics.FoldTree")

print("")
print("PyRosetta initialized.")

def windows_to_wsl_path(path):
    path = path.strip('"\'')
    if len(path) >= 3 and path[1] == ':' and path[2] == '\\':
        drive = path[0].lower()
        path = f"/mnt/{drive}" + path[2:]
        path = path.replace('\\', '/')
    return path

def strip_bulk_waters(pose, distance_cutoff=3.5):
    """Remove water molecules that are not within the specified distance of any non-water residue."""
    pose.update_residue_neighbors()
    non_water_indices = []
    water_indices = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        name3 = res.name3().strip()
        if name3 in ["HOH", "WAT", "TP3"]:
            water_indices.append(i)
        elif not res.is_virtual_residue():
            non_water_indices.append(i) 
    waters_to_delete = []
    for wat_idx in water_indices:
        wat_res = pose.residue(wat_idx)
        is_close = False
        if not non_water_indices:
            waters_to_delete.append(wat_idx)
            continue
        for non_wat_idx in non_water_indices:
            non_wat_res = pose.residue(non_wat_idx)
            nbr_dist = wat_res.nbr_atom_xyz().distance(non_wat_res.nbr_atom_xyz())
            if nbr_dist > wat_res.nbr_radius() + non_wat_res.nbr_radius() + distance_cutoff + 2.0:
                continue
            for a_wat in range(1, wat_res.nheavyatoms() + 1):
                xyz_wat = wat_res.xyz(a_wat)
                for a_non in range(1, non_wat_res.nheavyatoms() + 1):
                    xyz_non = non_wat_res.xyz(a_non)
                    if (xyz_wat - xyz_non).norm() <= distance_cutoff:
                        is_close = True
                        break
                if is_close:
                    break
            if is_close:
                break
        if not is_close:
            waters_to_delete.append(wat_idx)
    for wat_idx in sorted(waters_to_delete, reverse=True):
        pose.delete_residue_slow(wat_idx)
    return pose

if len(sys.argv) > 1:
    input_pdb_path = windows_to_wsl_path(sys.argv[1])
else:
    input_pdb_path = windows_to_wsl_path(r"")
    
output_pdb_path = os.path.splitext(input_pdb_path)[0] + "_relaxed.pdb"

print("")
pose = pose_from_pdb(input_pdb_path)
if pose.total_residue() > 0 and pose.residue(pose.total_residue()).type().is_virtual_residue():
    pose.delete_residue_slow(pose.total_residue())
pose = strip_bulk_waters(pose, distance_cutoff=3.5)
print("")
print(f"Input PDB: {input_pdb_path}")

addVirtualResAsRoot(pose)
anchor_res = pose.total_residue()
anchor_atom_id = AtomID(1, anchor_res)

scorefxn_torsion = ScoreFunctionFactory.create_score_function("ref2015")
scorefxn_cartesian = ScoreFunctionFactory.create_score_function("ref2015_cart")

tf = TaskFactory()
tf.push_back(RestrictToRepacking())
tf.push_back(IncludeCurrent())
packer_task = tf.create_task_and_apply_taskoperations(pose)

n_chains = pose.num_chains()
chain_starts = [pose.chain_begin(i+1) for i in range(n_chains)]
chain_ends = [pose.chain_end(i+1) for i in range(n_chains)]
virtual_root = pose.total_residue() 

print("")
print(f"Chains detected: {n_chains}")
print("")
print(f"Virtual root residue: {virtual_root}")
print("")
for i in range(n_chains):
    chain_id = pose.pdb_info().chain(chain_starts[i])
    print(f"Chain {chain_id}: {chain_starts[i]}-{chain_ends[i]}")

ft = rosetta.core.kinematics.FoldTree()
for i in range(n_chains):
    ft.add_edge(virtual_root, chain_starts[i], i+1)
for i in range(n_chains):
    ft.add_edge(chain_starts[i], chain_ends[i], -1)

if ft.check_fold_tree():
    pose.fold_tree(ft)
else:
    print("")
    print("Fold tree invalid!")

print("")

for i in range(n_chains):
    chain_id = pose.pdb_info().chain(chain_starts[i])
    print(f"Jump {i+1}: virtual root -> chain {chain_id} ({chain_starts[i]})")

print("")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# --- SCORE FUNCTION PARAMETERS --- #

fa_atr              = 1.0
fa_rep              = 0.55
fa_sol              = 1.0
fa_intra_rep        = 0.005
fa_intra_sol_xover4 = 1.0
lk_ball_wtd         = 1.0
fa_elec             = 1.0
hbond_sr_bb         = 1.0
hbond_lr_bb         = 1.0
hbond_bb_sc         = 1.0
hbond_sc            = 1.0
dslf_fa13           = 1.25
omega               = 0.4
fa_dun              = 0.7
p_aa_pp             = 0.6
yhh_planarity       = 0.625
ref                 = 1.0
rama_prepro         = 0.45
pro_close           = 1.25
cart_bonded         = 2.0
dna_bb_torsion      = 1.0
dna_sugar_close     = 1.0
rna_torsion         = 1.0
rna_sugar_close     = 1.0
fa_stack            = 1.0
dihedral            = 1.0
atom_pair           = 1.0
coordinate          = 1.0
 
# --- CONSTRAINT PARAMETERS --- #  

interface_distance_cutoff = 5.0                          
interface_distance_stddev = 0.5 

pocket_residue_distance_cutoff = 3.5

coordinate_constraint_stddev = 0.75         

scale_protein = 1.0 
scale_loop  = 1.5 
scale_helix = 0.5   
scale_sheet = 1.0   
scale_nucleic = 1.0 
scale_pocket = 0.0001
scale_water = 2.0
    
watson_crick_coplanarity_stddev = 5.0
watson_crick_distance_cutoff = 3.5  
watson_crick_distance_stddev = 0.5 

# --- RELAXATION PROTOCOL PARAMETERS --- #                 

constrained_relaxation_cycles = 10    
constrained_minimizer_tolerance = 0.0001           
constrained_torsion_minimizer_iterations = 250     
constrained_cartesian_minimizer_iterations = 50                         

polishing_minimizer_tolerance = 0.000001         
polishing_torsion_minimizer_iterations = 500     
polishing_cartesian_minimizer_iterations = 100      

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

def get_interface_residues(pose, chain_indices, cutoff, prefilter_margin=5.0):
    """Fast interface residue detection between two chains using a 5Å cutoff."""
    if cutoff is None:
        return set()
    pdb_info = pose.pdb_info()
    chainA_id = pdb_info.chain(chain_indices[0])
    chainB_id = pdb_info.chain(chain_indices[1])
    def rep_atom(res):
        if res.is_protein():
            return "CA" if res.has("CA") else "C"
        elif res.is_DNA() or res.is_RNA():
            return "C4'" if res.has("C4'") else "P"
        for i in range(1, res.natoms() + 1):
            name = res.atom_name(i).strip()
            if not name.startswith('H'):
                return name
        return None
    total = pose.total_residue()
    groupA = [i for i in range(1, total + 1) if pdb_info.chain(i) == chainA_id]
    groupB = [i for i in range(1, total + 1) if pdb_info.chain(i) == chainB_id]
    A_coords, A_res_indices = [], []
    for i in groupA:
        r = pose.residue(i)
        ra = rep_atom(r)
        if ra and r.has(ra):
            v = r.xyz(ra)
            A_coords.append([v.x, v.y, v.z])
            A_res_indices.append(i)
    B_coords, B_res_indices = [], []
    for j in groupB:
        r = pose.residue(j)
        ra = rep_atom(r)
        if ra and r.has(ra):
            v = r.xyz(ra)
            B_coords.append([v.x, v.y, v.z])
            B_res_indices.append(j)
    if not A_coords or not B_coords:
        return set()
    A_coords = np.asarray(A_coords, dtype=np.float64)
    B_coords = np.asarray(B_coords, dtype=np.float64)
    diff = A_coords[:, None, :] - B_coords[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    cutoff_prefilter_sq = (cutoff + prefilter_margin) ** 2
    candidate_mask = d2 <= cutoff_prefilter_sq
    candidate_pairs = np.argwhere(candidate_mask)
    interface_res = set()
    cutoff_sq = cutoff ** 2
    def heavy_atom_array(res):
        coords = []
        for k in range(1, res.natoms() + 1):
            name = res.atom_name(k).strip()
            if not name.startswith('H'):
                v = res.xyz(k)
                coords.append([v.x, v.y, v.z])
        return np.asarray(coords, dtype=np.float64) if coords else None
    heavy_cache = {}
    def get_heavy(res_idx):
        arr = heavy_cache.get(res_idx)
        if arr is None:
            r = pose.residue(res_idx)
            arr = heavy_atom_array(r)
            heavy_cache[res_idx] = arr
        return arr
    for a_idx, b_idx in candidate_pairs:
        resA = A_res_indices[a_idx]
        resB = B_res_indices[b_idx]
        if resA in interface_res and resB in interface_res:
            continue
        haA = get_heavy(resA)
        haB = get_heavy(resB)
        if haA is None or haB is None:
            continue
        diff_ab = haA[:, None, :] - haB[None, :, :]
        dist2_min = np.min(np.sum(diff_ab * diff_ab, axis=2))
        if dist2_min < cutoff_sq:
            interface_res.add(resA)
            interface_res.add(resB)
    return interface_res

def add_interface_constraints(pose, interface_residues_dict, distance_stddev, distance_cutoff):
    """Add AtomPair constraints for mutual nearest-neighbor interface residue pairs."""
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        print("No PDB info available - skipping interface constraints")
        return 0
    nres_total = pose.total_residue()
    def rep_backbone_atom(res):
        if res.is_protein():
            return "CA" if res.has("CA") else None
        if res.is_DNA() or res.is_RNA():
            return "C4'" if res.has("C4'") else None
        return None
    residue_cache = {}
    def get_atom_data(res_idx):
        if res_idx < 1 or res_idx > nres_total:
            return (None, None)
        cached = residue_cache.get(res_idx)
        if cached is not None:
            return cached
        res = pose.residue(res_idx)
        atom_name = rep_backbone_atom(res)
        if atom_name is None:
            data = (None, None)
        else:
            atom_idx = res.atom_index(atom_name)
            v = res.xyz(atom_name)
            data = (atom_idx, np.array([v.x, v.y, v.z], dtype=np.float64))
        residue_cache[res_idx] = data
        return data
    n_constraints = 0
    added_pairs = set()
    for key, value in interface_residues_dict.items():
        if not key.startswith("interface_"):
            continue
        residues = value.get("residues", [])
        chain_a, chain_b = value.get("chains", (None, None))
        if chain_a is None or chain_b is None:
            continue
        residues = sorted(set(residues))
        group_a = [i for i in residues if 1 <= i <= nres_total and pdb_info.chain(i) == chain_a]
        group_b = [j for j in residues if 1 <= j <= nres_total and pdb_info.chain(j) == chain_b]
        if not group_a or not group_b:
            continue
        a_res, a_atom_idx, a_xyz = [], [], []
        for a in group_a:
            atom_idx, xyz = get_atom_data(a)
            if atom_idx is None:
                continue
            a_res.append(a)
            a_atom_idx.append(atom_idx)
            a_xyz.append(xyz)
        b_res, b_atom_idx, b_xyz = [], [], []
        for b in group_b:
            atom_idx, xyz = get_atom_data(b)
            if atom_idx is None:
                continue
            b_res.append(b)
            b_atom_idx.append(atom_idx)
            b_xyz.append(xyz)
        if not a_res or not b_res:
            continue
        A = np.stack(a_xyz, axis=0)  # (NA, 3)
        B = np.stack(b_xyz, axis=0)  # (NB, 3)
        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        D_masked = np.where(D <= distance_cutoff, D, np.inf)
        best_b_idx_for_a = np.argmin(D_masked, axis=1)
        best_dist_for_a = D_masked[np.arange(D_masked.shape[0]), best_b_idx_for_a]
        best_a_idx_for_b = np.argmin(D_masked, axis=0)
        best_dist_for_b = D_masked[best_a_idx_for_b, np.arange(D_masked.shape[1])]
        for ia, ib in enumerate(best_b_idx_for_a):
            dist = best_dist_for_a[ia]
            if not np.isfinite(dist):
                continue
            if best_a_idx_for_b[ib] != ia:
                continue
            if not np.isfinite(best_dist_for_b[ib]):
                continue
            a = a_res[ia]
            b = b_res[ib]
            pair_key = (a, b) if a < b else (b, a)
            if pair_key in added_pairs:
                continue
            id1 = AtomID(a_atom_idx[ia], a)
            id2 = AtomID(b_atom_idx[ib], b)
            func = HarmonicFunc(float(dist), distance_stddev)
            pose.add_constraint(AtomPairConstraint(id1, id2, func))
            added_pairs.add(pair_key)
            n_constraints += 1
    print(f"Interface constraints added - {n_constraints} mutual nearest-neighbor atom pairs constrained")
    return n_constraints

def identify_watson_crick_pairs_by_criteria(pose, distance_cutoff):
    """Identify Watson-Crick base pairs based on distance and chain criteria."""
    wc_patterns = {
        ('A', 'T'): [('N6', 'O4'), ('N1', 'N3')], ('T', 'A'): [('N3', 'N1'), ('O4', 'N6')],
        ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')], ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
        ('A', 'U'): [('N6', 'O4'), ('N1', 'N3')], ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')]
    }
    def get_base_type(residue):
        name = residue.name3().strip()
        base_map = {
            'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C', 'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'U',
            'rA': 'A', 'rU': 'U', 'rG': 'G', 'rC': 'C', 'ADE': 'A', 'THY': 'T', 'GUA': 'G', 'CYT': 'C', 'URA': 'U'
        }
        return base_map.get(name, name)
    def get_chain_id(pose, res_idx):
        return pose.pdb_info().chain(res_idx) if pose.pdb_info() else 'A'
    def check_base_pair_distance(pose, res1_idx, res2_idx, atom1, atom2):
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        if not (res1.has(atom1) and res2.has(atom2)):
            return False, 0.0
        xyz1 = res1.xyz(atom1)
        xyz2 = res2.xyz(atom2)
        distance = (xyz1 - xyz2).norm()
        return distance <= distance_cutoff, distance
    def meets_chain_criteria(pose, res1_idx, res2_idx, res1, res2):
        chain1 = get_chain_id(pose, res1_idx)
        chain2 = get_chain_id(pose, res2_idx)
        if chain1 != chain2:
            return True, f"({chain1}-{chain2})"
        if res1.is_RNA() and res2.is_RNA() and chain1 == chain2:
            return True, f"({chain1})"
        if res1.is_DNA() and res2.is_DNA() and chain1 == chain2:
            return False, f"({chain1})"
        if ((res1.is_DNA() and res2.is_RNA()) or (res1.is_RNA() and res2.is_DNA())) and chain1 == chain2:
            return True, f"({chain1})"
        return False, "unknown"
    def is_watson_crick_pair(pose, res1_idx, res2_idx):
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        if not ((res1.is_DNA() or res1.is_RNA()) and (res2.is_DNA() or res2.is_RNA())):
            return False, None, 0.0, None
        meets_criteria, criteria_type = meets_chain_criteria(pose, res1_idx, res2_idx, res1, res2)
        if not meets_criteria:
            return False, None, 0.0, criteria_type
        base1 = get_base_type(res1)
        base2 = get_base_type(res2)
        pair_key = (base1, base2)
        if pair_key not in wc_patterns:
            return False, None, 0.0, criteria_type
        required_bonds = wc_patterns[pair_key]
        valid_bonds = 0
        min_distance = float('inf')
        for atom1, atom2 in required_bonds:
            is_valid, distance = check_base_pair_distance(pose, res1_idx, res2_idx, atom1, atom2)
            if is_valid:
                valid_bonds += 1
                min_distance = min(min_distance, distance)
        min_bonds_required = min(2, len(required_bonds))
        if valid_bonds >= min_bonds_required:
            pair_type = f"{base1}-{base2}"
            return True, pair_type, min_distance, criteria_type
        return False, None, 0.0, criteria_type
    protected_pairs = []
    nucleic_residues = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_DNA() or res.is_RNA():
            nucleic_residues.append(i)
    for i, res1_idx in enumerate(nucleic_residues):
        for res2_idx in nucleic_residues[i+1:]:
            is_wc, pair_type, distance, criteria_type = is_watson_crick_pair(pose, res1_idx, res2_idx)
            if is_wc:
                protected_pairs.append((res1_idx, res2_idx, pair_type, distance, criteria_type))
    return protected_pairs

def reset_score_weights(scorefxn, is_cartesian=False):
    """Reset score weights to initial values."""
    scorefxn.set_weight(score_type_from_name("fa_atr"), fa_atr)
    scorefxn.set_weight(score_type_from_name("fa_rep"), fa_rep)
    scorefxn.set_weight(score_type_from_name("fa_sol"), fa_sol)
    scorefxn.set_weight(score_type_from_name("fa_intra_rep"), fa_intra_rep)
    scorefxn.set_weight(score_type_from_name("fa_intra_sol_xover4"), fa_intra_sol_xover4)
    scorefxn.set_weight(score_type_from_name("lk_ball_wtd"), lk_ball_wtd)
    scorefxn.set_weight(score_type_from_name("fa_elec"), fa_elec)
    scorefxn.set_weight(score_type_from_name("hbond_sr_bb"), hbond_sr_bb)
    scorefxn.set_weight(score_type_from_name("hbond_lr_bb"), hbond_lr_bb)
    scorefxn.set_weight(score_type_from_name("hbond_bb_sc"), hbond_bb_sc)
    scorefxn.set_weight(score_type_from_name("hbond_sc"), hbond_sc)
    scorefxn.set_weight(score_type_from_name("dslf_fa13"), dslf_fa13)
    scorefxn.set_weight(score_type_from_name("omega"), omega)
    scorefxn.set_weight(score_type_from_name("fa_dun"), fa_dun)
    scorefxn.set_weight(score_type_from_name("p_aa_pp"), p_aa_pp)
    scorefxn.set_weight(score_type_from_name("yhh_planarity"), yhh_planarity)
    scorefxn.set_weight(score_type_from_name("ref"), ref)
    scorefxn.set_weight(score_type_from_name("rama_prepro"), rama_prepro)
    scorefxn.set_weight(score_type_from_name("dna_bb_torsion"), dna_bb_torsion)
    scorefxn.set_weight(score_type_from_name("dna_sugar_close"), dna_sugar_close)
    scorefxn.set_weight(score_type_from_name("rna_torsion"), rna_torsion)
    scorefxn.set_weight(score_type_from_name("rna_sugar_close"), rna_sugar_close)
    scorefxn.set_weight(score_type_from_name("fa_stack"), fa_stack)
    if is_cartesian:
        scorefxn.set_weight(rosetta.core.scoring.cart_bonded, cart_bonded)
        scorefxn.set_weight(rosetta.core.scoring.pro_close, 0.0)
    else:
        scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 0.0)
        scorefxn.set_weight(rosetta.core.scoring.pro_close, 1.25) 
    scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, dihedral)
    scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, atom_pair)
    scorefxn.set_weight(rosetta.core.scoring.coordinate_constraint, coordinate)
    return scorefxn

def sinusoidal_ramp(start, end, n_steps):
    """Generate a list of weights using a sinusoidal ramp from start to end."""
    return [start + (end - start) * 0.5 * (1 - np.cos(np.pi * i / (n_steps - 1))) for i in range(n_steps)]

def detect_residue_types(pose):
    """Detect what types of residues are present in the pose."""
    has_protein = False
    has_nucleic = False
    has_dna = False
    has_rna = False
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_protein():
            has_protein = True
        elif res.is_DNA():
            has_nucleic = True
            has_dna = True
        elif res.is_RNA():
            has_nucleic = True
            has_rna = True
    return has_protein, has_nucleic, has_dna, has_rna

def add_constraints(pose, interface_residues_dict, add_coordinate=True, add_interface=True, add_basepair=True, initial=None, current=None, alpha=0.0, pocket_res_indices=None, scale_pocket=scale_pocket):
    """Add constraints based on detected residue types."""
    has_protein, has_nucleic, has_dna, has_rna = detect_residue_types(pose)
    print(f"Detected residue types - Protein = {has_protein}, DNA = {has_dna}, RNA = {has_rna}")
    if add_coordinate and initial is not None and current is not None:
        add_coordinate_constraints(pose, anchor_atom_id, coordinate_stddev=coordinate_constraint_stddev, initial=initial, current=current, alpha=alpha, pocket_res_indices=pocket_res_indices, scale_pocket=scale_pocket)
    if add_interface and len(interface_residues_dict) > 0:
        add_interface_constraints(pose, interface_residues_dict, distance_stddev=interface_distance_stddev, distance_cutoff=interface_distance_cutoff)
    else:
        print("No interface residues detected - skipping interface constraints")
    if add_basepair and has_nucleic:
        add_basepair_constraints(pose, hbond_stddev=watson_crick_distance_stddev, coplanarity_stddev_deg=watson_crick_coplanarity_stddev)
    elif add_basepair:
        print("No nucleic acid residues detected - skipping base-pair constraints")

def add_coordinate_constraints(pose, anchor_atom_id, coordinate_stddev, initial, current, alpha, pocket_res_indices, scale_pocket):
    """Add coordinate constraints to backbone atoms, and all atoms for pocket residues."""
    secondary_structure = None
    try:
        secondary_structure = pose.secstruct()
    except Exception:
        secondary_structure = None
    def main_atom(res):
        if res.is_protein() and res.has("CA"):
            return "CA"
        elif (res.is_DNA() or res.is_RNA()):
            if res.has("C4'"):
                return "C4'"
            elif res.has("P"):
                return "P"
        elif res.name3().strip() in ["HOH", "WAT", "TP3"]:
            if res.has("O"):
                return "O"
            elif res.has("OW"):
                return "OW"
        return None
    n_constraints = 0
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        scale = 1.0
        if res.is_protein():
            if secondary_structure and len(secondary_structure) >= i:
                if secondary_structure[i-1] == 'H':
                    scale = scale_helix
                elif secondary_structure[i-1] == 'E':
                    scale = scale_sheet
                else:
                    scale = scale_loop
            else:
                scale = scale_protein
        elif res.is_DNA() or res.is_RNA():
            scale = scale_nucleic
        elif res.name3().strip() in ["HOH", "WAT", "TP3"]:
            scale = scale_water 
        is_pocket = pocket_res_indices and (i in pocket_res_indices)
        if is_pocket:
            scale *= scale_pocket    
        atoms_to_constrain = []
        if is_pocket or res.is_ligand() or res.is_metal():
            for a in range(1, res.natoms() + 1): 
                atoms_to_constrain.append(res.atom_name(a).strip())
        else:
            m_atom = main_atom(res)
            if m_atom and res.has(m_atom):
                atoms_to_constrain.append(m_atom)
        for atom_name in atoms_to_constrain:
            if not res.has(atom_name):
                continue
            atom_id = AtomID(res.atom_index(atom_name), i)
            xyz_init = initial.residue(i).xyz(atom_name)
            xyz_curr = current.residue(i).xyz(atom_name)
            xyz_init_np = np.array([xyz_init.x, xyz_init.y, xyz_init.z])
            xyz_curr_np = np.array([xyz_curr.x, xyz_curr.y, xyz_curr.z])
            xyz_np = xyz_init_np * alpha + xyz_curr_np * (1.0 - alpha)
            xyz = xyzVector_double_t(float(xyz_np[0]), float(xyz_np[1]), float(xyz_np[2]))
            func = HarmonicFunc(0.0, coordinate_stddev * scale)
            pose.add_constraint(CoordinateConstraint(atom_id, anchor_atom_id, xyz, func))
            n_constraints += 1
    print(f"Coordinate constraints added - {n_constraints} atoms constrained")

def get_secondary_structure(pose):
    """Assign DSSP secondary structure and returns a list for each residue."""
    dssp = DsspMover()
    dssp.apply(pose)
    return pose.secstruct()

def get_pocket_residues(pose, distance_cutoff):
    """Identify standard residues within a specified distance of any ligand or metal ion."""
    pose.update_residue_neighbors()
    pocket_residues = set()
    ligand_metal_indices = []
    standard_indices = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        is_water = res.name3().strip() in ["HOH", "WAT", "TP3"]
        if (res.is_ligand() or res.is_metal()) and not is_water:
            ligand_metal_indices.append(i)
        else:
            standard_indices.append(i)
    for lig_idx in ligand_metal_indices:
        lig_res = pose.residue(lig_idx)
        for std_idx in standard_indices:
            if std_idx in pocket_residues:
                continue
            std_res = pose.residue(std_idx)
            nbr_dist = lig_res.nbr_atom_xyz().distance(std_res.nbr_atom_xyz())
            if nbr_dist > lig_res.nbr_radius() + std_res.nbr_radius() + distance_cutoff + 2.0:
                continue
            is_close = False
            for a_lig in range(1, lig_res.nheavyatoms() + 1):
                xyz_lig = lig_res.xyz(a_lig)
                for a_std in range(1, std_res.nheavyatoms() + 1):
                    xyz_std = std_res.xyz(a_std)
                    if (xyz_lig - xyz_std).norm() <= distance_cutoff:
                        is_close = True
                        break
                if is_close:
                    break
            if is_close:
                pocket_residues.add(std_idx)
    return pocket_residues

def add_basepair_constraints(pose, hbond_stddev, coplanarity_stddev_deg):
    """Only add Watson-Crick H-bond distance constraints and optional coplanarity."""
    pose.update_residue_neighbors()
    pose.conformation().detect_bonds()
    coplanarity_stddev_rad = math.radians(coplanarity_stddev_deg)
    protected_pairs = identify_watson_crick_pairs_by_criteria(
        pose, distance_cutoff=watson_crick_distance_cutoff
    )
    wc_patterns = {
        ('A', 'T'): [('N6', 'O4'), ('N1', 'N3')], ('T', 'A'): [('N3', 'N1'), ('O4', 'N6')],
        ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')], ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
        ('A', 'U'): [('N6', 'O4'), ('N1', 'N3')], ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')]
    }
    hbond_count = 0
    coplanarity_count = 0
    for res1_idx, res2_idx, pair_type, distance, criteria_type in protected_pairs:
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        base1 = None
        base2 = None
        if pair_type:
            parts = pair_type.split('-')
            if len(parts) == 2:
                base1, base2 = parts[0], parts[1]
        if base1 is None or base2 is None:
            continue
        pair_key = (base1, base2)
        if pair_key in wc_patterns:
            for atom1_name, atom2_name in wc_patterns[pair_key]:
                if res1.has(atom1_name) and res2.has(atom2_name):
                    atom1_id = AtomID(res1.atom_index(atom1_name), res1_idx)
                    atom2_id = AtomID(res2.atom_index(atom2_name), res2_idx)
                    current_distance = (res1.xyz(atom1_name) - res2.xyz(atom2_name)).norm()
                    func = HarmonicFunc(current_distance, hbond_stddev)
                    pose.add_constraint(AtomPairConstraint(atom1_id, atom2_id, func))
                    hbond_count += 1
        def base_atoms(res):
            return ("N9", "C8") if res.is_purine() else ("N1", "C6")
        a1_i, a2_i = base_atoms(res1)
        a1_j, a2_j = base_atoms(res2)
        if res1.has(a1_i) and res1.has(a2_i) and res2.has(a1_j) and res2.has(a2_j):
            ids = [
                AtomID(res1.atom_index(a1_i), res1_idx), AtomID(res1.atom_index(a2_i), res1_idx),
                AtomID(res2.atom_index(a1_j), res2_idx), AtomID(res2.atom_index(a2_j), res2_idx)
            ]
            current_dihedral = rosetta.numeric.dihedral_degrees(
                res1.xyz(a1_i), res1.xyz(a2_i), res2.xyz(a1_j), res2.xyz(a2_j)
            )
            target_angle = 0.0 if abs(current_dihedral) < 90.0 else 180.0
            func = CircularHarmonicFunc(math.radians(target_angle), coplanarity_stddev_rad)
            pose.add_constraint(DihedralConstraint(*ids, func))
            coplanarity_count += 1
    criteria_counts = {}
    for _, _, _, _, criteria_type in protected_pairs:
        criteria_counts[criteria_type] = criteria_counts.get(criteria_type, 0) + 1
    print(f"Base-pair constraints added - {hbond_count} H-bonds constrained and coplanarity enforced across {coplanarity_count} base pairs")
    print("")
    for criteria, count_pairs in criteria_counts.items():
        print(f"{criteria}: {count_pairs} pairs")
    
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

print("=== INPUT POSE ===")
print("")

initial = pose.clone()

n_chains = pose.num_chains()
chain_ids = [pose.pdb_info().chain(pose.chain_begin(i)) for i in range(1, n_chains + 1)]
print("Detected chain IDs:", chain_ids)
print("") 

pose.remove_constraints()
pose.energies().clear()
scorefxn_clean = ScoreFunctionFactory.create_score_function("ref2015_cart")
pose_for_scoring = pose.clone()
if pose_for_scoring.total_residue() > 0 and pose_for_scoring.residue(pose_for_scoring.total_residue()).type().is_virtual_residue():
    pose_for_scoring.delete_residue_slow(pose_for_scoring.total_residue())
for i in range(pose_for_scoring.total_residue(), 0, -1):
        if pose_for_scoring.residue(i).name3().strip() in ["HOH", "WAT", "TP3"]:
            pose_for_scoring.delete_residue_slow(i)
score_clean = scorefxn_clean(pose_for_scoring)
terms = scorefxn_clean.get_nonzero_weighted_scoretypes()
print("Score decomposition (REU):")
print("")
for term in terms:
    val = pose_for_scoring.energies().total_energies()[term]
    print(f"{str(term).replace('ScoreType.', '• ')} = {val:.3f}")
print("")
print(f"Total score (REU) = {score_clean:.3f}")
print("")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

num_cycles = constrained_relaxation_cycles

initial_coordinate_ramp = sinusoidal_ramp(1.0 * coordinate, 0.0 * coordinate, num_cycles + 1)
current_coordinate_ramp = sinusoidal_ramp(0.0 * coordinate, 1.0 * coordinate, num_cycles + 1)

for cycle in range(num_cycles):

    print(f"=== CONSTRAINED RELAXATION : CYCLE {cycle+1}/{num_cycles} ===")
    print("")

    current = pose
    secondary_structure = get_secondary_structure(pose)
    pocket_res_indices = get_pocket_residues(pose, distance_cutoff=pocket_residue_distance_cutoff)

    interface_residues_dict = {}
    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            residues = get_interface_residues(pose, [pose.chain_begin(i + 1), pose.chain_begin(j + 1)], cutoff=interface_distance_cutoff)
            key = f"interface_{chain_ids[i]}_{chain_ids[j]}"
            interface_residues_dict[key] = {"residues": list(residues), "chains": (chain_ids[i], chain_ids[j])}
    
    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(True)
    for i in range(1, pose.total_residue() + 1):
        if i in pocket_res_indices:
            movemap.set_bb(i, False)
            movemap.set_chi(i, False)
        elif pose.residue(i).is_ligand() or pose.residue(i).is_metal():
            movemap.set_bb(i, False)
            movemap.set_chi(i, False)
    for i in range(n_chains):
        start_res = chain_starts[i]
        jump_id = i + 1
        if pose.residue(start_res).is_ligand() or pose.residue(start_res).is_metal():
            movemap.set_jump(jump_id, False)
    
    pose.remove_constraints()
    pose.update_residue_neighbors()
    scorefxn_torsion = reset_score_weights(scorefxn_torsion, is_cartesian=False)
    scorefxn_cartesian = reset_score_weights(scorefxn_cartesian, is_cartesian=True)

    alpha = initial_coordinate_ramp[cycle+1] / (initial_coordinate_ramp[cycle+1] + current_coordinate_ramp[cycle+1]) if (initial_coordinate_ramp[cycle+1] + current_coordinate_ramp[cycle+1]) > 0 else 0.0

    add_constraints(pose, interface_residues_dict, add_coordinate=True, add_interface=True, add_basepair=True, initial=initial, current=current, alpha=alpha, pocket_res_indices=pocket_res_indices, scale_pocket=scale_pocket)

    print("")

    current_fa_rep = fa_rep * min(1.0, 0.1 * (cycle + 1))
    scorefxn_torsion.set_weight(score_type_from_name("fa_rep"), current_fa_rep)
    scorefxn_cartesian.set_weight(score_type_from_name("fa_rep"), current_fa_rep)

    coordinate_ramp = initial_coordinate_ramp[cycle+1] + current_coordinate_ramp[cycle+1]
    scorefxn_torsion.set_weight(rosetta.core.scoring.coordinate_constraint, coordinate_ramp)
    scorefxn_cartesian.set_weight(rosetta.core.scoring.coordinate_constraint, coordinate_ramp)
    
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if i in pocket_res_indices or res.is_ligand() or res.is_metal():
            packer_task.nonconst_residue_task(i).prevent_repacking()
    pack_mover = PackRotamersMover(scorefxn_torsion, packer_task)
    pack_mover.apply(pose)

    pose.update_residue_neighbors()

    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn_torsion)
    min_mover.min_type('dfpmin_armijo_nonmonotone') 
    min_mover.tolerance(constrained_minimizer_tolerance)
    min_mover.cartesian(False)
    min_mover.max_iter(constrained_torsion_minimizer_iterations)
    min_mover.apply(pose)

    pose.update_residue_neighbors()

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if i in pocket_res_indices or res.is_ligand() or res.is_metal():
            packer_task.nonconst_residue_task(i).prevent_repacking()
    pack_mover = PackRotamersMover(scorefxn_torsion, packer_task)
    pack_mover.apply(pose)

    pose.update_residue_neighbors()

    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn_cartesian)
    min_mover.min_type('lbfgs_armijo_nonmonotone') 
    min_mover.tolerance(constrained_minimizer_tolerance)
    min_mover.cartesian(True)
    min_mover.max_iter(constrained_cartesian_minimizer_iterations)
    min_mover.apply(pose)

    pose.update_residue_neighbors()

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if i in pocket_res_indices or res.is_ligand() or res.is_metal():
            packer_task.nonconst_residue_task(i).prevent_repacking()
    pack_mover = PackRotamersMover(scorefxn_torsion, packer_task)
    pack_mover.apply(pose)

    pose.update_residue_neighbors()

    pose.remove_constraints()
    pose.energies().clear()
    scorefxn_clean = ScoreFunctionFactory.create_score_function("ref2015_cart")
    pose_for_scoring = pose.clone()
    if pose_for_scoring.total_residue() > 0 and pose_for_scoring.residue(pose_for_scoring.total_residue()).type().is_virtual_residue():
        pose_for_scoring.delete_residue_slow(pose_for_scoring.total_residue())
    for i in range(pose_for_scoring.total_residue(), 0, -1):
        if pose_for_scoring.residue(i).name3().strip() in ["HOH", "WAT", "TP3"]:
            pose_for_scoring.delete_residue_slow(i)
    score_clean = scorefxn_clean(pose_for_scoring)
    terms = scorefxn_clean.get_nonzero_weighted_scoretypes()
    print("")
    print("Score decomposition (REU):")
    print("")
    for term in terms:
        val = pose_for_scoring.energies().total_energies()[term]
        print(f"{str(term).replace('ScoreType.', '• ')} = {val:.3f}")
    print("")
    print(f"Total score (REU) = {score_clean:.3f}")
    rmsd = rosetta.core.scoring.CA_rmsd(initial, pose)
    print(f"r.m.s.d. to input = {rmsd:.3f} Å")
    print("")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

print("=== STRUCTURE POLISHING ===") 
print("")

secondary_structure = get_secondary_structure(pose)
pocket_res_indices = get_pocket_residues(pose, distance_cutoff=pocket_residue_distance_cutoff)

interface_residues_dict = {}
for i in range(len(chain_ids)):
    for j in range(i + 1, len(chain_ids)):
        residues = get_interface_residues(pose, [pose.chain_begin(i + 1), pose.chain_begin(j + 1)], cutoff=interface_distance_cutoff)
        key = f"interface_{chain_ids[i]}_{chain_ids[j]}"
        interface_residues_dict[key] = {"residues": list(residues), "chains": (chain_ids[i], chain_ids[j])}

movemap = MoveMap()
movemap.set_bb(True)
movemap.set_chi(True)
movemap.set_jump(True)
for i in range(1, pose.total_residue() + 1):
    if i in pocket_res_indices:
        movemap.set_bb(i, False)
        movemap.set_chi(i, False)
    elif pose.residue(i).is_ligand() or pose.residue(i).is_metal():
        movemap.set_bb(i, False)
        movemap.set_chi(i, False)
for i in range(n_chains):
    start_res = chain_starts[i]
    jump_id = i + 1
    if pose.residue(start_res).is_ligand() or pose.residue(start_res).is_metal():
        movemap.set_jump(jump_id, False)

pose.remove_constraints()
pose.update_residue_neighbors()

scorefxn_torsion = reset_score_weights(scorefxn_torsion, is_cartesian=False)
scorefxn_cartesian = reset_score_weights(scorefxn_cartesian, is_cartesian=True)

alpha = 0.0
current = pose

add_constraints(pose, interface_residues_dict, add_coordinate=True, add_interface=True, add_basepair=True, initial=initial, current=current, alpha=alpha, pocket_res_indices=pocket_res_indices, scale_pocket=scale_pocket)
print("")
    
tf = TaskFactory()
tf.push_back(RestrictToRepacking())
tf.push_back(IncludeCurrent())
packer_task = tf.create_task_and_apply_taskoperations(pose)
for i in range(1, pose.total_residue() + 1):
    res = pose.residue(i)
    if i in pocket_res_indices or res.is_ligand() or res.is_metal():
        packer_task.nonconst_residue_task(i).prevent_repacking()
pack_mover = PackRotamersMover(scorefxn_torsion, packer_task)
pack_mover.apply(pose)

pose.update_residue_neighbors()

min_mover = MinMover()
min_mover.movemap(movemap)
min_mover.score_function(scorefxn_torsion)
min_mover.min_type('dfpmin_armijo_nonmonotone') 
min_mover.tolerance(polishing_minimizer_tolerance)
min_mover.cartesian(False)
min_mover.max_iter(polishing_torsion_minimizer_iterations)
min_mover.apply(pose)

pose.update_residue_neighbors()

tf = TaskFactory()
tf.push_back(RestrictToRepacking())
tf.push_back(IncludeCurrent())
packer_task = tf.create_task_and_apply_taskoperations(pose)
for i in range(1, pose.total_residue() + 1):
    res = pose.residue(i)
    if i in pocket_res_indices or res.is_ligand() or res.is_metal():
        packer_task.nonconst_residue_task(i).prevent_repacking()
pack_mover = PackRotamersMover(scorefxn_torsion, packer_task)
pack_mover.apply(pose)

pose.update_residue_neighbors()

min_mover = MinMover()
min_mover.movemap(movemap)
min_mover.score_function(scorefxn_cartesian)
min_mover.min_type('lbfgs_armijo_nonmonotone') 
min_mover.tolerance(polishing_minimizer_tolerance)
min_mover.cartesian(True)
min_mover.max_iter(polishing_cartesian_minimizer_iterations)
min_mover.apply(pose)

pose.update_residue_neighbors()

tf = TaskFactory()
tf.push_back(RestrictToRepacking())
tf.push_back(IncludeCurrent())
packer_task = tf.create_task_and_apply_taskoperations(pose)
for i in range(1, pose.total_residue() + 1):
    res = pose.residue(i)
    if i in pocket_res_indices or res.is_ligand() or res.is_metal():
        packer_task.nonconst_residue_task(i).prevent_repacking()
pack_mover = PackRotamersMover(scorefxn_torsion, packer_task)
pack_mover.apply(pose)

pose.update_residue_neighbors()

pose.remove_constraints()
pose.energies().clear()
scorefxn_clean = ScoreFunctionFactory.create_score_function("ref2015_cart")
pose_for_scoring = pose.clone()
if pose_for_scoring.total_residue() > 0 and pose_for_scoring.residue(pose_for_scoring.total_residue()).type().is_virtual_residue():
    pose_for_scoring.delete_residue_slow(pose_for_scoring.total_residue())
for i in range(pose_for_scoring.total_residue(), 0, -1):
    if pose_for_scoring.residue(i).name3().strip() in ["HOH", "WAT", "TP3"]:
        pose_for_scoring.delete_residue_slow(i)
score_clean = scorefxn_clean(pose_for_scoring)
terms = scorefxn_clean.get_nonzero_weighted_scoretypes()
print("")
print("Score decomposition (REU):")
print("")
for term in terms:
    val = pose_for_scoring.energies().total_energies()[term]
    print(f"{str(term).replace('ScoreType.', '• ')} = {val:.3f}")
print("")
print(f"Total score (REU) = {score_clean:.3f}")
rmsd = rosetta.core.scoring.CA_rmsd(initial, pose)
print(f"r.m.s.d. to input = {rmsd:.3f} Å")
print("")

if pose.total_residue() > 0 and pose.residue(pose.total_residue()).type().is_virtual_residue():
    pose.delete_residue_slow(pose.total_residue())
pose.dump_pdb(output_pdb_path)
print(f"Dumped final pose to {output_pdb_path}.")
print("") 