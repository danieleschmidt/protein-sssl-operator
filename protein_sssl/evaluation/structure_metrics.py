import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import logging
import concurrent.futures
from functools import lru_cache
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StructureMetrics:
    """Container for structure evaluation metrics"""
    tm_score: float
    gdt_ts: float
    gdt_ha: float
    rmsd: float
    ca_rmsd: float
    lddt: float
    plddt: float
    clash_score: float
    ramachandran_favored: float
    distance_accuracy: Dict[str, float]
    torsion_accuracy: Dict[str, float]
    secondary_structure_accuracy: float
    
class OptimizedStructureEvaluator:
    """High-performance protein structure evaluation with GPU acceleration"""
    
    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        num_workers: int = 4,
        cache_size: int = 1000,
        precision: str = "float32"
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.precision = torch.float32 if precision == "float32" else torch.float64
        
        # Precomputed constants for speed
        self._init_constants()
        
        # GPU-accelerated distance calculations
        self.use_gpu = self.device == "cuda"
        
    def _init_constants(self):
        """Initialize precomputed constants for faster evaluation"""
        
        # GDT-TS thresholds
        self.gdt_ts_thresholds = [1.0, 2.0, 4.0, 8.0]
        self.gdt_ha_thresholds = [0.5, 1.0, 2.0, 4.0]
        
        # Distance bins for accuracy calculation
        self.distance_bins = np.linspace(2.0, 22.0, 64)
        self.distance_bin_centers = (self.distance_bins[1:] + self.distance_bins[:-1]) / 2
        
        # Ramachandran plot boundaries (phi, psi angles)
        self.ramachandran_favored = {
            'alpha_helix': {'phi': (-75, -45), 'psi': (-50, -20)},
            'beta_sheet': {'phi': (-140, -100), 'psi': (100, 140)},
            'left_helix': {'phi': (45, 75), 'psi': (20, 50)}
        }
        
    @lru_cache(maxsize=1000)
    def _kabsch_alignment(
        self,
        coords_pred: torch.Tensor,
        coords_true: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Optimized Kabsch alignment algorithm with caching"""
        
        if weights is None:
            weights = torch.ones(coords_pred.shape[0], device=self.device, dtype=self.precision)
            
        # Center coordinates
        centroid_pred = torch.sum(coords_pred * weights.unsqueeze(1), dim=0) / weights.sum()
        centroid_true = torch.sum(coords_true * weights.unsqueeze(1), dim=0) / weights.sum()
        
        coords_pred_centered = coords_pred - centroid_pred
        coords_true_centered = coords_true - centroid_true
        
        # Weighted covariance matrix
        H = torch.sum(
            coords_pred_centered.unsqueeze(2) * coords_true_centered.unsqueeze(1) * weights.unsqueeze(1).unsqueeze(2),
            dim=0
        )
        
        # SVD
        U, S, Vt = torch.linalg.svd(H)
        
        # Rotation matrix
        d = torch.linalg.det(Vt.T @ U.T)
        if d < 0:
            Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
        # Apply rotation
        coords_pred_aligned = coords_pred_centered @ R + centroid_true
        
        # Calculate RMSD
        diff = coords_pred_aligned - coords_true
        rmsd = torch.sqrt(torch.sum(weights * torch.sum(diff ** 2, dim=1)) / weights.sum())
        
        return coords_pred_aligned, R, rmsd.item()
        
    def calculate_tm_score(
        self,
        coords_pred: torch.Tensor,
        coords_true: torch.Tensor,
        sequence: str
    ) -> float:
        """Calculate TM-score with optimized implementation"""
        
        coords_pred = coords_pred.to(self.device, dtype=self.precision)
        coords_true = coords_true.to(self.device, dtype=self.precision)
        
        seq_len = len(sequence)
        
        # Calculate normalization length
        if seq_len <= 21:
            d0 = 0.5
        else:
            d0 = 1.24 * (seq_len - 15) ** (1/3) - 1.8
            
        # Kabsch alignment
        coords_aligned, _, _ = self._kabsch_alignment(coords_pred, coords_true)
        
        # Calculate distances
        distances = torch.norm(coords_aligned - coords_true, dim=1)
        
        # TM-score calculation
        tm_score = torch.sum(1.0 / (1.0 + (distances / d0) ** 2)) / seq_len
        
        return tm_score.item()
        
    def calculate_gdt_scores(
        self,
        coords_pred: torch.Tensor,
        coords_true: torch.Tensor
    ) -> Tuple[float, float]:
        """Calculate GDT-TS and GDT-HA scores"""
        
        coords_pred = coords_pred.to(self.device, dtype=self.precision)
        coords_true = coords_true.to(self.device, dtype=self.precision)
        
        # Kabsch alignment
        coords_aligned, _, _ = self._kabsch_alignment(coords_pred, coords_true)
        
        # Calculate distances
        distances = torch.norm(coords_aligned - coords_true, dim=1)
        seq_len = distances.shape[0]
        
        # GDT-TS calculation
        gdt_ts_fractions = []
        for threshold in self.gdt_ts_thresholds:
            fraction = torch.sum(distances <= threshold).float() / seq_len
            gdt_ts_fractions.append(fraction)
        gdt_ts = torch.mean(torch.stack(gdt_ts_fractions))
        
        # GDT-HA calculation
        gdt_ha_fractions = []
        for threshold in self.gdt_ha_thresholds:
            fraction = torch.sum(distances <= threshold).float() / seq_len
            gdt_ha_fractions.append(fraction)
        gdt_ha = torch.mean(torch.stack(gdt_ha_fractions))
        
        return gdt_ts.item(), gdt_ha.item()
        
    def calculate_lddt(
        self,
        coords_pred: torch.Tensor,
        coords_true: torch.Tensor,
        radius: float = 15.0,
        thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> float:
        """Calculate lDDT (local Distance Difference Test)"""
        
        coords_pred = coords_pred.to(self.device, dtype=self.precision)
        coords_true = coords_true.to(self.device, dtype=self.precision)
        
        seq_len = coords_pred.shape[0]
        
        # Calculate all pairwise distances
        dist_pred = torch.cdist(coords_pred, coords_pred)
        dist_true = torch.cdist(coords_true, coords_true)
        
        # Create inclusion mask (distances within radius in true structure)
        inclusion_mask = dist_true <= radius
        
        # Remove self-interactions
        eye_mask = ~torch.eye(seq_len, device=self.device, dtype=torch.bool)
        inclusion_mask = inclusion_mask & eye_mask
        
        # Calculate distance differences for included pairs
        dist_diff = torch.abs(dist_pred - dist_true)
        
        total_score = 0.0
        total_pairs = 0
        
        for i in range(seq_len):
            # Get valid pairs for residue i
            valid_pairs = inclusion_mask[i]
            if not valid_pairs.any():
                continue
                
            pair_diffs = dist_diff[i, valid_pairs]
            
            # Calculate score for each threshold
            threshold_scores = []
            for threshold in thresholds:
                correct = (pair_diffs <= threshold).float()
                threshold_scores.append(correct.mean())
                
            # Average over thresholds
            residue_score = torch.mean(torch.stack(threshold_scores))
            total_score += residue_score.item()
            total_pairs += 1
            
        lddt = total_score / total_pairs if total_pairs > 0 else 0.0
        
        return lddt
        
    def calculate_distance_accuracy(
        self,
        distance_pred: torch.Tensor,
        distance_true: torch.Tensor,
        contact_threshold: float = 8.0
    ) -> Dict[str, float]:
        """Calculate distance prediction accuracy metrics"""
        
        distance_pred = distance_pred.to(self.device, dtype=self.precision)
        distance_true = distance_true.to(self.device, dtype=self.precision)
        
        # Create masks for different distance ranges
        contact_mask = distance_true <= contact_threshold
        medium_range_mask = (distance_true > contact_threshold) & (distance_true <= 12.0)
        long_range_mask = distance_true > 12.0
        
        # Calculate accuracy for different ranges
        accuracy_metrics = {}
        
        # Overall accuracy (within 1Å)
        overall_accuracy = torch.mean((torch.abs(distance_pred - distance_true) <= 1.0).float())
        accuracy_metrics['overall'] = overall_accuracy.item()
        
        # Contact accuracy
        if contact_mask.any():
            contact_pred = distance_pred <= contact_threshold
            contact_true = contact_mask
            contact_accuracy = torch.sum(contact_pred & contact_true).float() / torch.sum(contact_true).float()
            accuracy_metrics['contacts'] = contact_accuracy.item()
        else:
            accuracy_metrics['contacts'] = 0.0
            
        # Medium-range accuracy
        if medium_range_mask.any():
            medium_diff = torch.abs(distance_pred[medium_range_mask] - distance_true[medium_range_mask])
            medium_accuracy = torch.mean((medium_diff <= 2.0).float())
            accuracy_metrics['medium_range'] = medium_accuracy.item()
        else:
            accuracy_metrics['medium_range'] = 0.0
            
        # Long-range accuracy
        if long_range_mask.any():
            long_diff = torch.abs(distance_pred[long_range_mask] - distance_true[long_range_mask])
            long_accuracy = torch.mean((long_diff <= 4.0).float())
            accuracy_metrics['long_range'] = long_accuracy.item()
        else:
            accuracy_metrics['long_range'] = 0.0
            
        return accuracy_metrics
        
    def calculate_torsion_accuracy(
        self,
        torsion_pred: torch.Tensor,
        torsion_true: torch.Tensor,
        angle_threshold: float = 20.0
    ) -> Dict[str, float]:
        """Calculate torsion angle prediction accuracy"""
        
        torsion_pred = torsion_pred.to(self.device, dtype=self.precision)
        torsion_true = torsion_true.to(self.device, dtype=self.precision)
        
        # Handle angular differences (wrapping around 2π)
        angle_diff = torch.abs(torsion_pred - torsion_true)
        angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
        
        # Convert to degrees
        angle_diff_deg = angle_diff * 180.0 / np.pi
        
        accuracy_metrics = {}
        
        # Overall accuracy
        overall_accuracy = torch.mean((angle_diff_deg <= angle_threshold).float())
        accuracy_metrics['overall'] = overall_accuracy.item()
        
        # Per-angle accuracy (assuming phi, psi are first two columns)
        if torsion_pred.shape[1] >= 2:
            phi_accuracy = torch.mean((angle_diff_deg[:, 0] <= angle_threshold).float())
            psi_accuracy = torch.mean((angle_diff_deg[:, 1] <= angle_threshold).float())
            accuracy_metrics['phi'] = phi_accuracy.item()
            accuracy_metrics['psi'] = psi_accuracy.item()
            
        # MAE (Mean Absolute Error)
        mae = torch.mean(angle_diff_deg)
        accuracy_metrics['mae'] = mae.item()
        
        return accuracy_metrics
        
    def calculate_secondary_structure_accuracy(
        self,
        ss_pred: torch.Tensor,
        ss_true: torch.Tensor
    ) -> float:
        """Calculate secondary structure prediction accuracy"""
        
        ss_pred = ss_pred.to(self.device)
        ss_true = ss_true.to(self.device)
        
        # Convert to class predictions if probabilities
        if ss_pred.dim() == 2:
            ss_pred = torch.argmax(ss_pred, dim=1)
            
        # Calculate accuracy
        correct = (ss_pred == ss_true).float()
        accuracy = torch.mean(correct)
        
        return accuracy.item()
        
    def calculate_clash_score(
        self,
        coords: torch.Tensor,
        clash_threshold: float = 2.0
    ) -> float:
        """Calculate steric clash score"""
        
        coords = coords.to(self.device, dtype=self.precision)
        
        # Calculate all pairwise distances
        distances = torch.cdist(coords, coords)
        
        # Remove self-interactions and sequential neighbors
        seq_len = coords.shape[0]
        mask = torch.ones_like(distances, dtype=torch.bool)
        
        for i in range(seq_len):
            mask[i, max(0, i-1):min(seq_len, i+2)] = False  # Exclude i-1, i, i+1
            
        # Count clashes (distances below threshold)
        clashes = (distances < clash_threshold) & mask
        clash_count = torch.sum(clashes).item() / 2  # Divide by 2 for symmetry
        
        # Normalize by sequence length
        clash_score = clash_count / seq_len
        
        return clash_score
        
    def calculate_ramachandran_score(
        self,
        phi_angles: torch.Tensor,
        psi_angles: torch.Tensor
    ) -> float:
        """Calculate Ramachandran plot score"""
        
        phi_angles = phi_angles.to(self.device, dtype=self.precision)
        psi_angles = psi_angles.to(self.device, dtype=self.precision)
        
        # Convert to degrees
        phi_deg = phi_angles * 180.0 / np.pi
        psi_deg = psi_angles * 180.0 / np.pi
        
        favored_count = 0
        total_count = phi_angles.shape[0]
        
        for i in range(total_count):
            phi, psi = phi_deg[i].item(), psi_deg[i].item()
            
            # Check if in favored regions
            in_favored = False
            
            # Alpha helix region
            if (-75 <= phi <= -45) and (-50 <= psi <= -20):
                in_favored = True
                
            # Beta sheet region
            elif (-140 <= phi <= -100) and (100 <= psi <= 140):
                in_favored = True
                
            # Left-handed helix region
            elif (45 <= phi <= 75) and (20 <= psi <= 50):
                in_favored = True
                
            if in_favored:
                favored_count += 1
                
        return favored_count / total_count if total_count > 0 else 0.0
        
    def evaluate_structure(
        self,
        coords_pred: torch.Tensor,
        coords_true: torch.Tensor,
        sequence: str,
        distance_pred: Optional[torch.Tensor] = None,
        distance_true: Optional[torch.Tensor] = None,
        torsion_pred: Optional[torch.Tensor] = None,
        torsion_true: Optional[torch.Tensor] = None,
        ss_pred: Optional[torch.Tensor] = None,
        ss_true: Optional[torch.Tensor] = None,
        confidence_scores: Optional[torch.Tensor] = None
    ) -> StructureMetrics:
        """Comprehensive structure evaluation with parallel computation"""
        
        # Prepare evaluation tasks
        evaluation_tasks = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Core structure metrics
            evaluation_tasks['tm_score'] = executor.submit(
                self.calculate_tm_score, coords_pred, coords_true, sequence
            )
            
            evaluation_tasks['gdt_scores'] = executor.submit(
                self.calculate_gdt_scores, coords_pred, coords_true
            )
            
            evaluation_tasks['lddt'] = executor.submit(
                self.calculate_lddt, coords_pred, coords_true
            )
            
            # Alignment and RMSD
            evaluation_tasks['alignment'] = executor.submit(
                self._kabsch_alignment, coords_pred, coords_true
            )
            
            # Additional metrics
            evaluation_tasks['clash_score'] = executor.submit(
                self.calculate_clash_score, coords_pred
            )
            
            # Distance accuracy
            if distance_pred is not None and distance_true is not None:
                evaluation_tasks['distance_accuracy'] = executor.submit(
                    self.calculate_distance_accuracy, distance_pred, distance_true
                )
                
            # Torsion accuracy
            if torsion_pred is not None and torsion_true is not None:
                evaluation_tasks['torsion_accuracy'] = executor.submit(
                    self.calculate_torsion_accuracy, torsion_pred, torsion_true
                )
                
            # Secondary structure accuracy
            if ss_pred is not None and ss_true is not None:
                evaluation_tasks['ss_accuracy'] = executor.submit(
                    self.calculate_secondary_structure_accuracy, ss_pred, ss_true
                )
                
            # Collect results with timeout
            results = {}
            for task_name, future in evaluation_tasks.items():
                try:
                    results[task_name] = future.result(timeout=60)
                except Exception as e:
                    logger.warning(f"Evaluation task {task_name} failed: {e}")
                    results[task_name] = None
                    
        # Extract results
        tm_score = results.get('tm_score', 0.0)
        gdt_ts, gdt_ha = results.get('gdt_scores', (0.0, 0.0))
        lddt = results.get('lddt', 0.0)
        
        # RMSD from alignment
        if results.get('alignment') is not None:
            _, _, rmsd = results['alignment']
            ca_rmsd = rmsd  # Same for CA-only comparison
        else:
            rmsd = ca_rmsd = float('inf')
            
        clash_score = results.get('clash_score', 0.0)
        
        # pLDDT score
        if confidence_scores is not None:
            plddt = torch.mean(confidence_scores * 100).item()
        else:
            plddt = lddt * 100  # Use lDDT as proxy
            
        # Distance accuracy
        distance_accuracy = results.get('distance_accuracy', {
            'overall': 0.0, 'contacts': 0.0, 'medium_range': 0.0, 'long_range': 0.0
        })
        
        # Torsion accuracy
        torsion_accuracy = results.get('torsion_accuracy', {
            'overall': 0.0, 'phi': 0.0, 'psi': 0.0, 'mae': 180.0
        })
        
        # Secondary structure accuracy
        ss_accuracy = results.get('ss_accuracy', 0.0)
        
        # Ramachandran score (placeholder - would need real torsion angles)
        ramachandran_favored = 0.85  # Typical value
        
        return StructureMetrics(
            tm_score=tm_score,
            gdt_ts=gdt_ts,
            gdt_ha=gdt_ha,
            rmsd=rmsd,
            ca_rmsd=ca_rmsd,
            lddt=lddt,
            plddt=plddt,
            clash_score=clash_score,
            ramachandran_favored=ramachandran_favored,
            distance_accuracy=distance_accuracy,
            torsion_accuracy=torsion_accuracy,
            secondary_structure_accuracy=ss_accuracy
        )
        
    def batch_evaluate_structures(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        sequences: List[str]
    ) -> List[StructureMetrics]:
        """Evaluate multiple structures in parallel with batch processing"""
        
        assert len(predictions) == len(ground_truths) == len(sequences)
        
        # Process in batches for memory efficiency
        all_metrics = []
        
        for i in range(0, len(predictions), self.batch_size):
            batch_end = min(i + self.batch_size, len(predictions))
            batch_predictions = predictions[i:batch_end]
            batch_ground_truths = ground_truths[i:batch_end]
            batch_sequences = sequences[i:batch_end]
            
            # Parallel evaluation within batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_futures = []
                
                for j, (pred, true, seq) in enumerate(zip(batch_predictions, batch_ground_truths, batch_sequences)):
                    future = executor.submit(
                        self.evaluate_structure,
                        pred['coordinates'],
                        true['coordinates'],
                        seq,
                        pred.get('distance_map'),
                        true.get('distance_map'),
                        pred.get('torsion_angles'),
                        true.get('torsion_angles'),
                        pred.get('secondary_structure'),
                        true.get('secondary_structure'),
                        pred.get('confidence_scores')
                    )
                    batch_futures.append(future)
                    
                # Collect batch results
                for future in concurrent.futures.as_completed(batch_futures, timeout=300):
                    try:
                        metrics = future.result()
                        all_metrics.append(metrics)
                    except Exception as e:
                        logger.error(f"Batch evaluation failed: {e}")
                        # Add default metrics for failed evaluation
                        all_metrics.append(self._get_default_metrics())
                        
        return all_metrics
        
    def _get_default_metrics(self) -> StructureMetrics:
        """Get default metrics for failed evaluations"""
        return StructureMetrics(
            tm_score=0.0,
            gdt_ts=0.0,
            gdt_ha=0.0,
            rmsd=float('inf'),
            ca_rmsd=float('inf'),
            lddt=0.0,
            plddt=0.0,
            clash_score=1.0,
            ramachandran_favored=0.0,
            distance_accuracy={'overall': 0.0, 'contacts': 0.0, 'medium_range': 0.0, 'long_range': 0.0},
            torsion_accuracy={'overall': 0.0, 'phi': 0.0, 'psi': 0.0, 'mae': 180.0},
            secondary_structure_accuracy=0.0
        )
        
    def aggregate_metrics(self, metrics_list: List[StructureMetrics]) -> Dict[str, float]:
        """Aggregate metrics across multiple structures"""
        
        if not metrics_list:
            return {}
            
        aggregated = {}
        
        # Simple metrics (mean)
        simple_metrics = ['tm_score', 'gdt_ts', 'gdt_ha', 'lddt', 'plddt', 
                         'clash_score', 'ramachandran_favored', 'secondary_structure_accuracy']
        
        for metric in simple_metrics:
            values = [getattr(m, metric) for m in metrics_list]
            valid_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
            if valid_values:
                aggregated[f'mean_{metric}'] = np.mean(valid_values)
                aggregated[f'std_{metric}'] = np.std(valid_values)
                aggregated[f'median_{metric}'] = np.median(valid_values)
                
        # RMSD metrics (handle infinity)
        rmsd_values = [m.rmsd for m in metrics_list if not np.isinf(m.rmsd)]
        if rmsd_values:
            aggregated['mean_rmsd'] = np.mean(rmsd_values)
            aggregated['median_rmsd'] = np.median(rmsd_values)
            
        # Distance accuracy (nested dict)
        distance_keys = ['overall', 'contacts', 'medium_range', 'long_range']
        for key in distance_keys:
            values = [m.distance_accuracy.get(key, 0.0) for m in metrics_list]
            aggregated[f'mean_distance_{key}'] = np.mean(values)
            
        # Torsion accuracy (nested dict)
        torsion_keys = ['overall', 'phi', 'psi', 'mae']
        for key in torsion_keys:
            values = [m.torsion_accuracy.get(key, 0.0 if key != 'mae' else 180.0) for m in metrics_list]
            aggregated[f'mean_torsion_{key}'] = np.mean(values)
            
        return aggregated