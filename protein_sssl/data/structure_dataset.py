import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import pickle
import logging
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import gzip
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress PDB warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

class StructureDataset(Dataset):
    """Dataset for protein structures with robust error handling"""
    
    def __init__(
        self,
        structure_paths: List[str],
        sequences: Optional[List[str]] = None,
        max_length: int = 1024,
        resolution_cutoff: float = 3.0,
        remove_redundancy: bool = True,
        quality_filters: Optional[Dict] = None,
        cache_dir: Optional[str] = None
    ):
        self.structure_paths = structure_paths
        self.sequences = sequences or []
        self.max_length = max_length
        self.resolution_cutoff = resolution_cutoff
        self.remove_redundancy = remove_redundancy
        self.cache_dir = cache_dir
        
        # Quality filters
        if quality_filters is None:
            quality_filters = {
                'min_length': 30,
                'max_missing_residues': 0.1,  # 10% missing residues allowed
                'max_b_factor': 100.0,
                'chain_breaks_allowed': 2
            }
        self.quality_filters = quality_filters
        
        # Initialize PDB parser
        self.pdb_parser = PDBParser(QUIET=True)
        
        # Processed structures cache
        self.processed_structures = []
        self.failed_structures = []
        
        # Distance bins for discretization
        self.distance_bins = np.linspace(2.0, 22.0, 64)
        
        # Process structures
        self._process_structures()
        
    def _process_structures(self):
        """Process and validate all structure files"""
        logger.info(f"Processing {len(self.structure_paths)} structure files...")
        
        # Load from cache if available
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, "structure_cache.pkl")
            if os.path.exists(cache_path):
                logger.info("Loading structures from cache...")
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    self.processed_structures = cache_data['processed_structures']
                    self.failed_structures = cache_data['failed_structures']
                    logger.info(f"Loaded {len(self.processed_structures)} structures from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
        
        # Process each structure
        for i, structure_path in enumerate(self.structure_paths):
            try:
                structure_data = self._process_single_structure(structure_path, i)
                if structure_data is not None:
                    self.processed_structures.append(structure_data)
                else:
                    self.failed_structures.append(structure_path)
                    
            except Exception as e:
                logger.warning(f"Failed to process {structure_path}: {e}")
                self.failed_structures.append(structure_path)
                
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(self.structure_paths)} structures")
        
        # Save cache
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, "structure_cache.pkl")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'processed_structures': self.processed_structures,
                        'failed_structures': self.failed_structures
                    }, f)
                logger.info(f"Saved structure cache to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        logger.info(f"Successfully processed {len(self.processed_structures)} structures")
        logger.info(f"Failed to process {len(self.failed_structures)} structures")
        
    def _process_single_structure(self, structure_path: str, index: int) -> Optional[Dict]:
        """Process a single PDB structure with validation"""
        
        try:
            # Parse structure
            if structure_path.endswith('.gz'):
                with gzip.open(structure_path, 'rt') as f:
                    structure = self.pdb_parser.get_structure(f"structure_{index}", f)
            else:
                structure = self.pdb_parser.get_structure(f"structure_{index}", structure_path)
            
            # Get first model
            model = structure[0] if len(structure) > 0 else None
            if model is None:
                return None
            
            # Extract chain data
            chains = list(model.get_chains())
            if not chains:
                return None
                
            # Process each chain (take first valid chain for now)
            for chain in chains:
                chain_data = self._process_chain(chain, structure_path, index)
                if chain_data is not None:
                    return chain_data
                    
            return None
            
        except Exception as e:
            logger.debug(f"Error processing {structure_path}: {e}")
            return None
            
    def _process_chain(self, chain, structure_path: str, index: int) -> Optional[Dict]:
        """Process a single protein chain"""
        
        # Extract residues
        residues = []
        coordinates = []
        sequence = []
        
        # Amino acid mapping
        aa_mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        missing_residues = 0
        high_bfactor_residues = 0
        chain_breaks = 0
        prev_res_num = None
        
        for residue in chain:
            # Skip non-amino acid residues
            if residue.get_id()[0] != ' ':
                continue
                
            resname = residue.get_resname()
            if resname not in aa_mapping:
                missing_residues += 1
                continue
                
            # Get CA coordinate
            if 'CA' not in residue:
                missing_residues += 1
                continue
                
            ca_atom = residue['CA']
            coord = ca_atom.get_coord()
            bfactor = ca_atom.get_bfactor()
            
            # Check B-factor
            if bfactor > self.quality_filters['max_b_factor']:
                high_bfactor_residues += 1
                
            # Check for chain breaks
            res_num = residue.get_id()[1]
            if prev_res_num is not None and res_num - prev_res_num > 1:
                chain_breaks += 1
            prev_res_num = res_num
            
            # Store data
            residues.append(residue)
            coordinates.append(coord)
            sequence.append(aa_mapping[resname])
            
        # Quality checks
        if len(sequence) < self.quality_filters['min_length']:
            return None
            
        if len(sequence) > self.max_length:
            return None
            
        missing_fraction = missing_residues / (len(sequence) + missing_residues)
        if missing_fraction > self.quality_filters['max_missing_residues']:
            return None
            
        if chain_breaks > self.quality_filters['chain_breaks_allowed']:
            return None
        
        # Convert to arrays
        coordinates = np.array(coordinates)
        sequence_str = ''.join(sequence)
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(coordinates)
        
        # Calculate torsion angles (phi, psi)
        torsion_angles = self._calculate_torsion_angles(residues)
        
        # Calculate secondary structure (simplified)
        secondary_structure = self._assign_secondary_structure(coordinates)
        
        return {
            'structure_path': structure_path,
            'index': index,
            'sequence': sequence_str,
            'coordinates': coordinates,
            'distance_matrix': distance_matrix,
            'torsion_angles': torsion_angles,
            'secondary_structure': secondary_structure,
            'quality_metrics': {
                'resolution': None,  # Would need to parse header
                'missing_residues': missing_residues,
                'high_bfactor_residues': high_bfactor_residues,
                'chain_breaks': chain_breaks,
                'sequence_length': len(sequence)
            }
        }
        
    def _calculate_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate CA-CA distance matrix"""
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        return distance_matrix
        
    def _calculate_torsion_angles(self, residues: List) -> np.ndarray:
        """Calculate phi and psi torsion angles"""
        angles = []
        
        for i, residue in enumerate(residues):
            phi = psi = 0.0
            
            # Calculate phi angle (C-1, N, CA, C)
            if i > 0:
                try:
                    prev_residue = residues[i-1]
                    if 'C' in prev_residue and 'N' in residue and 'CA' in residue and 'C' in residue:
                        phi = self._calculate_dihedral(
                            prev_residue['C'].get_coord(),
                            residue['N'].get_coord(),
                            residue['CA'].get_coord(),
                            residue['C'].get_coord()
                        )
                except:
                    phi = 0.0
                    
            # Calculate psi angle (N, CA, C, N+1)
            if i < len(residues) - 1:
                try:
                    next_residue = residues[i+1]
                    if 'N' in residue and 'CA' in residue and 'C' in residue and 'N' in next_residue:
                        psi = self._calculate_dihedral(
                            residue['N'].get_coord(),
                            residue['CA'].get_coord(),
                            residue['C'].get_coord(),
                            next_residue['N'].get_coord()
                        )
                except:
                    psi = 0.0
                    
            angles.append([phi, psi])
            
        return np.array(angles)
        
    def _calculate_dihedral(self, p1, p2, p3, p4) -> float:
        """Calculate dihedral angle between four points"""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # Normalize
        b1 /= np.linalg.norm(b1)
        b2 /= np.linalg.norm(b2)
        b3 /= np.linalg.norm(b3)
        
        # Calculate normal vectors
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Calculate angle
        m1 = np.cross(n1, b2)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        return np.arctan2(y, x)
        
    def _assign_secondary_structure(self, coordinates: np.ndarray) -> np.ndarray:
        """Simple secondary structure assignment based on CA distances"""
        n = len(coordinates)
        ss = np.zeros(n)  # 0: coil, 1: helix, 2: sheet
        
        # Simple heuristic based on CA distances
        for i in range(2, n-2):
            # Check for helix (i to i+3 distance ~5.4A)
            if i+3 < n:
                dist_i_i3 = np.linalg.norm(coordinates[i] - coordinates[i+3])
                if 4.0 < dist_i_i3 < 7.0:
                    ss[i] = 1  # helix
                    
            # Check for sheet (look for hydrogen bonding patterns)
            # Simplified: look for extended conformations
            if i-1 >= 0 and i+1 < n:
                angle = self._calculate_angle(
                    coordinates[i-1], coordinates[i], coordinates[i+1]
                )
                if angle > 2.8:  # Extended conformation
                    ss[i] = 2  # sheet
                    
        return ss
        
    def _calculate_angle(self, p1, p2, p3) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.arccos(cos_angle)
    
    @classmethod
    def from_pdb_directory(
        cls,
        pdb_dir: str,
        resolution_cutoff: float = 3.0,
        remove_redundancy: bool = True,
        max_structures: Optional[int] = None,
        **kwargs
    ):
        """Load dataset from directory of PDB files"""
        
        pdb_dir = Path(pdb_dir)
        
        # Find PDB files
        pdb_files = []
        for ext in ['*.pdb', '*.pdb.gz', '*.ent', '*.ent.gz']:
            pdb_files.extend(pdb_dir.glob(f"**/{ext}"))
            
        pdb_files = [str(f) for f in pdb_files]
        
        if max_structures:
            pdb_files = pdb_files[:max_structures]
            
        logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
        
        return cls(
            structure_paths=pdb_files,
            resolution_cutoff=resolution_cutoff,
            remove_redundancy=remove_redundancy,
            **kwargs
        )
        
    def __len__(self) -> int:
        return len(self.processed_structures)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get structure data with validation"""
        
        if idx >= len(self.processed_structures):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.processed_structures)}")
            
        try:
            structure_data = self.processed_structures[idx]
            
            # Convert to tensors
            sequence = structure_data['sequence']
            coordinates = torch.FloatTensor(structure_data['coordinates'])
            distance_matrix = torch.FloatTensor(structure_data['distance_matrix'])
            torsion_angles = torch.FloatTensor(structure_data['torsion_angles'])
            secondary_structure = torch.LongTensor(structure_data['secondary_structure'])
            
            # Tokenize sequence
            aa_to_id = {
                'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
                'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
                'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
            }
            
            sequence_ids = [aa_to_id.get(aa, 20) for aa in sequence]
            input_ids = torch.LongTensor(sequence_ids)
            
            # Create attention mask
            attention_mask = torch.ones(len(sequence), dtype=torch.long)
            
            # Discretize distances for loss computation
            distance_bins = torch.FloatTensor(self.distance_bins)
            distance_indices = torch.bucketize(distance_matrix, distance_bins)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'coordinates': coordinates,
                'distance_matrix': distance_matrix,
                'distance_targets': distance_indices,
                'torsion_angles': torsion_angles,
                'secondary_structure': secondary_structure,
                'sequence': sequence,
                'structure_path': structure_data['structure_path']
            }
            
        except Exception as e:
            logger.error(f"Error loading structure at index {idx}: {e}")
            # Return dummy data to prevent training crashes
            return self._get_dummy_data()
            
    def _get_dummy_data(self) -> Dict[str, torch.Tensor]:
        """Return dummy data in case of errors"""
        dummy_length = 50
        
        return {
            'input_ids': torch.zeros(dummy_length, dtype=torch.long),
            'attention_mask': torch.zeros(dummy_length, dtype=torch.long),
            'coordinates': torch.zeros((dummy_length, 3), dtype=torch.float),
            'distance_matrix': torch.zeros((dummy_length, dummy_length), dtype=torch.float),
            'distance_targets': torch.zeros((dummy_length, dummy_length), dtype=torch.long),
            'torsion_angles': torch.zeros((dummy_length, 2), dtype=torch.float),
            'secondary_structure': torch.zeros(dummy_length, dtype=torch.long),
            'sequence': 'X' * dummy_length,
            'structure_path': 'dummy'
        }
        
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.processed_structures:
            return {}
            
        lengths = [len(s['sequence']) for s in self.processed_structures]
        resolutions = [s['quality_metrics'].get('resolution', 0) for s in self.processed_structures if s['quality_metrics'].get('resolution')]
        
        stats = {
            'num_structures': len(self.processed_structures),
            'num_failed': len(self.failed_structures),
            'sequence_length': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths)
            }
        }
        
        if resolutions:
            stats['resolution'] = {
                'mean': np.mean(resolutions),
                'std': np.std(resolutions),
                'min': np.min(resolutions),
                'max': np.max(resolutions)
            }
            
        return stats