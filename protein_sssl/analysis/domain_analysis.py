import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass
import concurrent.futures
from functools import lru_cache
import logging
from pathlib import Path
import pickle
import time
from sklearn.cluster import DBSCAN
from scipy import stats
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DomainPrediction:
    id: str
    start: int
    end: int
    confidence: float
    fold_classification: str
    stability_score: float
    predicted_function: str
    binding_sites: List[int]
    evolutionary_conservation: Optional[np.ndarray] = None

@dataclass
class MultiScaleAnalysis:
    sequence: str
    residue_features: Dict[str, np.ndarray]
    secondary_structure: np.ndarray
    tertiary_contacts: np.ndarray
    quaternary_interfaces: Optional[List[Dict]] = None
    temporal_dynamics: Optional[Dict] = None
    
class OptimizedFeatureCache:
    """Thread-safe LRU cache for expensive feature computations"""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        
    def get(self, key: str):
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
            
    def set(self, key: str, value):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                
            self.cache[key] = value
            self.access_times[key] = time.time()
            
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class DomainSegmenter:
    """High-performance domain segmentation with evolutionary information"""
    
    def __init__(
        self,
        min_domain_length: int = 40,
        max_domains_per_protein: int = 10,
        use_evolutionary_info: bool = True,
        cache_size: int = 500,
        num_workers: int = 4
    ):
        self.min_domain_length = min_domain_length
        self.max_domains_per_protein = max_domains_per_protein
        self.use_evolutionary_info = use_evolutionary_info
        self.num_workers = num_workers
        
        # Feature cache for performance
        self.feature_cache = OptimizedFeatureCache(cache_size)
        
        # Precomputed matrices for common calculations
        self._init_precomputed_matrices()
        
    def _init_precomputed_matrices(self):
        """Initialize precomputed matrices for performance"""
        # Amino acid hydrophobicity scale
        self.hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        # Amino acid charge scale
        self.charge = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
            'G': 0, 'H': 0.1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
            'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
        }
        
        # Secondary structure propensity
        self.ss_propensity = {
            'helix': {'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70},
            'sheet': {'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19},
            'coil': {'A': 0.66, 'R': 0.95, 'N': 1.56, 'D': 1.46, 'C': 1.19}
        }
        
    @lru_cache(maxsize=1000)
    def _compute_sequence_features(self, sequence: str) -> np.ndarray:
        """Compute sequence-based features with caching"""
        
        seq_len = len(sequence)
        features = np.zeros((seq_len, 10))  # 10 features per residue
        
        for i, aa in enumerate(sequence):
            if aa in self.hydrophobicity:
                features[i, 0] = self.hydrophobicity[aa]
                features[i, 1] = self.charge[aa]
                
                # Local hydrophobic moments
                if i >= 2 and i < seq_len - 2:
                    local_window = sequence[i-2:i+3]
                    features[i, 2] = np.mean([self.hydrophobicity.get(a, 0) for a in local_window])
                    
                # Secondary structure propensities
                features[i, 3] = self.ss_propensity['helix'].get(aa, 1.0)
                features[i, 4] = self.ss_propensity['sheet'].get(aa, 1.0)
                features[i, 5] = self.ss_propensity['coil'].get(aa, 1.0)
                
                # Sequence complexity (Shannon entropy in local window)
                if i >= 5 and i < seq_len - 5:
                    window = sequence[i-5:i+6]
                    features[i, 6] = self._calculate_entropy(window)
                    
                # Charge distribution
                if i >= 3 and i < seq_len - 3:
                    charge_window = [self.charge.get(a, 0) for a in sequence[i-3:i+4]]
                    features[i, 7] = np.sum(np.array(charge_window) > 0)  # Positive charges
                    features[i, 8] = np.sum(np.array(charge_window) < 0)  # Negative charges
                    
                # Evolutionary conservation (placeholder - would use real MSA data)
                features[i, 9] = np.random.random()  # Mock conservation score
                
        return features
        
    def _calculate_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of amino acid composition"""
        if not sequence:
            return 0.0
            
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
        total = len(sequence)
        entropy = 0.0
        
        for count in aa_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
                
        return entropy
        
    def segment(
        self,
        sequence: str,
        min_domain_length: Optional[int] = None,
        use_evolutionary_info: Optional[bool] = None,
        return_features: bool = False
    ) -> List[DomainPrediction]:
        """Segment protein into domains with high performance"""
        
        if min_domain_length is None:
            min_domain_length = self.min_domain_length
        if use_evolutionary_info is None:
            use_evolutionary_info = self.use_evolutionary_info
            
        # Check cache first
        cache_key = f"segment_{hash(sequence)}_{min_domain_length}_{use_evolutionary_info}"
        cached_result = self.feature_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        seq_len = len(sequence)
        
        if seq_len < min_domain_length:
            # Single domain
            domain = DomainPrediction(
                id="domain_1",
                start=0,
                end=seq_len,
                confidence=0.8,
                fold_classification="single_domain",
                stability_score=0.75,
                predicted_function="unknown",
                binding_sites=[]
            )
            return [domain]
            
        # Compute features
        features = self._compute_sequence_features(sequence)
        
        # Use multiple segmentation approaches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                'hydrophobic': executor.submit(self._segment_by_hydrophobicity, sequence, features),
                'secondary': executor.submit(self._segment_by_secondary_structure, sequence, features),
                'conservation': executor.submit(self._segment_by_conservation, sequence, features) if use_evolutionary_info else None,
                'clustering': executor.submit(self._segment_by_clustering, sequence, features)
            }
            
            # Collect results
            segmentation_results = {}
            for approach, future in futures.items():
                if future is not None:
                    try:
                        segmentation_results[approach] = future.result(timeout=30)
                    except Exception as e:
                        logger.warning(f"Segmentation approach {approach} failed: {e}")
                        segmentation_results[approach] = []
                        
        # Consensus segmentation
        domains = self._consensus_segmentation(
            sequence, 
            segmentation_results, 
            min_domain_length
        )
        
        # Cache result
        self.feature_cache.set(cache_key, domains)
        
        return domains
        
    def _segment_by_hydrophobicity(
        self, 
        sequence: str, 
        features: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Segment based on hydrophobic regions"""
        
        hydrophobic_profile = features[:, 0]  # Hydrophobicity values
        
        # Smooth the profile
        window_size = min(15, len(sequence) // 10)
        if window_size > 1:
            smoothed = np.convolve(
                hydrophobic_profile, 
                np.ones(window_size) / window_size, 
                mode='same'
            )
        else:
            smoothed = hydrophobic_profile
            
        # Find change points using derivatives
        gradient = np.gradient(smoothed)
        change_points = []
        
        threshold = np.std(gradient) * 1.5
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i]) > threshold:
                # Check if it's a significant change
                if abs(smoothed[i+5] - smoothed[i-5]) > 1.0:  # Significant hydrophobicity change
                    change_points.append(i)
                    
        # Convert change points to segments
        segments = []
        start = 0
        for cp in change_points:
            if cp - start >= self.min_domain_length:
                segments.append((start, cp))
                start = cp
                
        # Add final segment
        if len(sequence) - start >= self.min_domain_length:
            segments.append((start, len(sequence)))
            
        return segments
        
    def _segment_by_secondary_structure(
        self, 
        sequence: str, 
        features: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Segment based on secondary structure propensities"""
        
        # Calculate secondary structure preference at each position
        helix_prop = features[:, 3]
        sheet_prop = features[:, 4]
        coil_prop = features[:, 5]
        
        # Identify regions of consistent secondary structure
        ss_preference = np.argmax(np.column_stack([helix_prop, sheet_prop, coil_prop]), axis=1)
        
        # Find transitions
        transitions = []
        for i in range(1, len(ss_preference)):
            if ss_preference[i] != ss_preference[i-1]:
                transitions.append(i)
                
        # Merge short segments and create domains
        segments = []
        start = 0
        
        for trans in transitions:
            if trans - start >= self.min_domain_length // 2:  # Allow shorter SS-based segments
                segments.append((start, trans))
                start = trans
                
        if len(sequence) - start >= self.min_domain_length // 2:
            segments.append((start, len(sequence)))
            
        return segments
        
    def _segment_by_conservation(
        self, 
        sequence: str, 
        features: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Segment based on evolutionary conservation patterns"""
        
        conservation = features[:, 9]  # Mock conservation scores
        
        # Identify highly conserved regions (potential functional domains)
        high_conservation_threshold = np.percentile(conservation, 75)
        conserved_regions = conservation > high_conservation_threshold
        
        # Find boundaries of conserved regions
        boundaries = []
        in_conserved = False
        
        for i, is_conserved in enumerate(conserved_regions):
            if is_conserved and not in_conserved:
                boundaries.append(i)  # Start of conserved region
                in_conserved = True
            elif not is_conserved and in_conserved:
                boundaries.append(i)  # End of conserved region
                in_conserved = False
                
        # Create segments around conserved regions
        segments = []
        for i in range(0, len(boundaries), 2):
            if i + 1 < len(boundaries):
                start = max(0, boundaries[i] - 10)  # Add some context
                end = min(len(sequence), boundaries[i + 1] + 10)
                
                if end - start >= self.min_domain_length:
                    segments.append((start, end))
                    
        return segments
        
    def _segment_by_clustering(
        self, 
        sequence: str, 
        features: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Segment using feature-based clustering"""
        
        if len(features) < 10:  # Too small for clustering
            return [(0, len(sequence))]
            
        # Use DBSCAN for clustering residue features
        try:
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(features)
            labels = clustering.labels_
            
            # Find boundaries between clusters
            boundaries = []
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    boundaries.append(i)
                    
            # Create segments
            segments = []
            start = 0
            for boundary in boundaries:
                if boundary - start >= self.min_domain_length:
                    segments.append((start, boundary))
                    start = boundary
                    
            if len(sequence) - start >= self.min_domain_length:
                segments.append((start, len(sequence)))
                
            return segments
            
        except Exception as e:
            logger.warning(f"Clustering segmentation failed: {e}")
            return [(0, len(sequence))]  # Fallback to single domain
            
    def _consensus_segmentation(
        self,
        sequence: str,
        segmentation_results: Dict[str, List[Tuple[int, int]]],
        min_domain_length: int
    ) -> List[DomainPrediction]:
        """Create consensus segmentation from multiple approaches"""
        
        all_boundaries = set()
        
        # Collect all boundary points
        for approach, segments in segmentation_results.items():
            for start, end in segments:
                all_boundaries.add(start)
                all_boundaries.add(end)
                
        # Sort boundaries
        sorted_boundaries = sorted(list(all_boundaries))
        
        # Create consensus domains
        domains = []
        domain_id = 1
        
        for i in range(len(sorted_boundaries) - 1):
            start = sorted_boundaries[i]
            end = sorted_boundaries[i + 1]
            
            if end - start >= min_domain_length:
                # Calculate confidence based on agreement between methods
                confidence = self._calculate_segment_confidence(
                    start, end, segmentation_results
                )
                
                # Predict domain properties
                domain_seq = sequence[start:end]
                fold_class = self._predict_fold_classification(domain_seq)
                stability = self._predict_stability(domain_seq)
                function = self._predict_function(domain_seq)
                binding_sites = self._predict_binding_sites(domain_seq, start)
                
                domain = DomainPrediction(
                    id=f"domain_{domain_id}",
                    start=start,
                    end=end,
                    confidence=confidence,
                    fold_classification=fold_class,
                    stability_score=stability,
                    predicted_function=function,
                    binding_sites=binding_sites
                )
                
                domains.append(domain)
                domain_id += 1
                
        # If no domains found, create single domain
        if not domains:
            domains = [DomainPrediction(
                id="domain_1",
                start=0,
                end=len(sequence),
                confidence=0.5,
                fold_classification="unknown",
                stability_score=0.5,
                predicted_function="unknown",
                binding_sites=[]
            )]
            
        return domains[:self.max_domains_per_protein]  # Limit number of domains
        
    def _calculate_segment_confidence(
        self,
        start: int,
        end: int,
        segmentation_results: Dict[str, List[Tuple[int, int]]]
    ) -> float:
        """Calculate confidence based on agreement between segmentation methods"""
        
        votes = 0
        total_methods = len(segmentation_results)
        
        for segments in segmentation_results.values():
            for seg_start, seg_end in segments:
                # Check overlap
                overlap_start = max(start, seg_start)
                overlap_end = min(end, seg_end)
                
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / (end - start)
                    if overlap_ratio > 0.5:  # Significant overlap
                        votes += 1
                        break
                        
        return votes / total_methods if total_methods > 0 else 0.5
        
    def _predict_fold_classification(self, sequence: str) -> str:
        """Predict fold class based on sequence composition"""
        
        # Simple heuristics for fold classification
        composition = {}
        for aa in sequence:
            composition[aa] = composition.get(aa, 0) + 1
            
        total = len(sequence)
        
        # Calculate fractions
        helix_promoting = composition.get('A', 0) + composition.get('E', 0) + composition.get('L', 0)
        sheet_promoting = composition.get('V', 0) + composition.get('I', 0) + composition.get('F', 0)
        coil_promoting = composition.get('G', 0) + composition.get('P', 0) + composition.get('N', 0)
        
        helix_frac = helix_promoting / total
        sheet_frac = sheet_promoting / total
        coil_frac = coil_promoting / total
        
        if helix_frac > 0.4:
            return "all_alpha"
        elif sheet_frac > 0.3:
            return "all_beta"
        elif helix_frac > 0.2 and sheet_frac > 0.2:
            return "alpha_beta"
        else:
            return "small_protein"
            
    def _predict_stability(self, sequence: str) -> float:
        """Predict domain stability score"""
        
        # Simple stability prediction based on amino acid composition
        stabilizing_aa = {'C': 0.1, 'W': 0.05, 'F': 0.04, 'Y': 0.03}
        destabilizing_aa = {'G': -0.02, 'P': -0.03}
        
        stability_score = 0.5  # Baseline
        
        for aa in sequence:
            stability_score += stabilizing_aa.get(aa, 0)
            stability_score += destabilizing_aa.get(aa, 0)
            
        # Normalize to 0-1 range
        stability_score = max(0.0, min(1.0, stability_score))
        
        return stability_score
        
    def _predict_function(self, sequence: str) -> str:
        """Predict domain function based on sequence features"""
        
        # Simple function prediction based on motifs and composition
        if 'ATP' in sequence or 'GTP' in sequence:
            return "nucleotide_binding"
        elif sequence.count('C') >= 4:  # Potential zinc finger or disulfide bonds
            return "metal_binding"
        elif sequence.count('D') + sequence.count('E') > len(sequence) * 0.2:
            return "catalytic"
        elif sequence.count('R') + sequence.count('K') > len(sequence) * 0.25:
            return "DNA_binding"
        else:
            return "structural"
            
    def _predict_binding_sites(self, sequence: str, offset: int) -> List[int]:
        """Predict potential binding sites in the domain"""
        
        binding_sites = []
        
        # Look for potential catalytic residues
        catalytic_residues = ['D', 'E', 'H', 'C', 'S', 'T', 'Y']
        
        for i, aa in enumerate(sequence):
            if aa in catalytic_residues:
                # Check local environment for binding potential
                if i >= 2 and i < len(sequence) - 2:
                    local_env = sequence[i-2:i+3]
                    # Simple heuristic: hydrophobic environment around catalytic residue
                    hydrophobic_count = sum(1 for a in local_env if a in 'AILMFWV')
                    if hydrophobic_count >= 2:
                        binding_sites.append(offset + i)
                        
        return binding_sites[:5]  # Limit to top 5 predictions

class MultiScaleAnalyzer:
    """High-performance multi-scale protein structure analysis"""
    
    def __init__(
        self, 
        model: Optional[nn.Module] = None,
        cache_size: int = 200,
        num_workers: int = 4,
        gpu_acceleration: bool = True
    ):
        self.model = model
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.device = "cuda" if gpu_acceleration and torch.cuda.is_available() else "cpu"
        
        # Feature cache
        self.analysis_cache = OptimizedFeatureCache(cache_size)
        
        # Precompute analysis parameters
        self.scales = {
            'residue': {'window_size': 1, 'features': ['local_environment', 'flexibility']},
            'secondary': {'window_size': 8, 'features': ['ss_structure', 'turns', 'loops']},
            'tertiary': {'window_size': 20, 'features': ['contacts', 'packing', 'accessibility']},
            'quaternary': {'window_size': 50, 'features': ['interfaces', 'allosteric_networks']}
        }
        
    def analyze_domain(
        self,
        sequence: str,
        context_sequence: Optional[str] = None,
        scales: List[str] = None,
        coordinates: Optional[torch.Tensor] = None
    ) -> MultiScaleAnalysis:
        """Analyze protein domain at multiple scales with optimized performance"""
        
        if scales is None:
            scales = ['residue', 'secondary', 'tertiary']
            
        # Check cache
        cache_key = f"analyze_{hash(sequence)}_{hash(str(scales))}_{hash(str(context_sequence))}"
        cached_result = self.analysis_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Parallel analysis at different scales
        analysis_tasks = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for scale in scales:
                if scale in self.scales:
                    analysis_tasks[scale] = executor.submit(
                        self._analyze_at_scale,
                        sequence,
                        scale,
                        context_sequence,
                        coordinates
                    )
                    
            # Collect results
            scale_results = {}
            for scale, future in analysis_tasks.items():
                try:
                    scale_results[scale] = future.result(timeout=60)
                except Exception as e:
                    logger.warning(f"Analysis at scale {scale} failed: {e}")
                    scale_results[scale] = self._get_default_scale_result(scale, len(sequence))
                    
        # Combine results into multi-scale analysis
        analysis = self._combine_scale_results(sequence, scale_results, coordinates)
        
        # Cache result
        self.analysis_cache.set(cache_key, analysis)
        
        return analysis
        
    def _analyze_at_scale(
        self,
        sequence: str,
        scale: str,
        context_sequence: Optional[str],
        coordinates: Optional[torch.Tensor]
    ) -> Dict:
        """Analyze protein at a specific scale"""
        
        seq_len = len(sequence)
        scale_params = self.scales[scale]
        window_size = scale_params['window_size']
        features = scale_params['features']
        
        results = {}
        
        if scale == 'residue':
            results = self._analyze_residue_scale(sequence, context_sequence)
        elif scale == 'secondary':
            results = self._analyze_secondary_scale(sequence, window_size)
        elif scale == 'tertiary':
            results = self._analyze_tertiary_scale(sequence, coordinates, window_size)
        elif scale == 'quaternary':
            results = self._analyze_quaternary_scale(sequence, coordinates)
            
        return results
        
    def _analyze_residue_scale(
        self,
        sequence: str,
        context_sequence: Optional[str]
    ) -> Dict:
        """Analyze at residue level with optimized computations"""
        
        seq_len = len(sequence)
        
        # Local environment analysis
        local_env = np.zeros((seq_len, 5))  # 5 environmental features
        flexibility = np.zeros(seq_len)
        
        # Vectorized amino acid property lookups
        hydrophobicity_map = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        flexibility_map = {
            'G': 0.9, 'P': 0.2, 'S': 0.8, 'T': 0.7, 'A': 0.6,
            'C': 0.3, 'D': 0.8, 'E': 0.8, 'F': 0.4, 'H': 0.7,
            'I': 0.3, 'K': 0.8, 'L': 0.4, 'M': 0.6, 'N': 0.8,
            'Q': 0.8, 'R': 0.8, 'V': 0.3, 'W': 0.4, 'Y': 0.5
        }
        
        # Efficient computation using numpy
        hydrophobic_values = np.array([hydrophobicity_map.get(aa, 0) for aa in sequence])
        flexibility_values = np.array([flexibility_map.get(aa, 0.5) for aa in sequence])
        
        # Window-based analysis
        for i in range(seq_len):
            window_start = max(0, i - 2)
            window_end = min(seq_len, i + 3)
            
            # Local hydrophobic moment
            local_env[i, 0] = np.mean(hydrophobic_values[window_start:window_end])
            
            # Local charge density
            charge_window = sequence[window_start:window_end]
            positive_charge = sum(1 for aa in charge_window if aa in 'RKH')
            negative_charge = sum(1 for aa in charge_window if aa in 'DE')
            local_env[i, 1] = positive_charge - negative_charge
            
            # Local complexity
            local_env[i, 2] = len(set(sequence[window_start:window_end]))
            
            # Boundary preferences
            if i == 0 or i == seq_len - 1:
                local_env[i, 3] = 1.0  # Terminal residue
            else:
                local_env[i, 3] = 0.0
                
            # Conservation (mock - would use real evolutionary data)
            local_env[i, 4] = np.random.random()
            
            # Flexibility
            flexibility[i] = flexibility_values[i]
            
        return {
            'local_environment': local_env,
            'flexibility': flexibility
        }
        
    def _analyze_secondary_scale(self, sequence: str, window_size: int) -> Dict:
        """Analyze secondary structure patterns"""
        
        seq_len = len(sequence)
        
        # Secondary structure prediction (simplified)
        ss_prediction = np.zeros((seq_len, 3))  # Helix, sheet, coil
        turns = np.zeros(seq_len)
        loops = np.zeros(seq_len)
        
        # Chou-Fasman like prediction (simplified)
        helix_propensity = {
            'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'Q': 1.11,
            'R': 0.98, 'K': 1.16, 'H': 1.00, 'F': 1.13, 'Y': 0.69
        }
        
        sheet_propensity = {
            'V': 1.60, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.37,
            'L': 1.30, 'T': 1.19, 'C': 1.19, 'A': 0.83, 'R': 0.93
        }
        
        # Sliding window analysis
        for i in range(seq_len):
            window_start = max(0, i - window_size // 2)
            window_end = min(seq_len, i + window_size // 2)
            
            window_seq = sequence[window_start:window_end]
            
            # Calculate propensities
            helix_score = np.mean([helix_propensity.get(aa, 1.0) for aa in window_seq])
            sheet_score = np.mean([sheet_propensity.get(aa, 1.0) for aa in window_seq])
            coil_score = 2.0 - helix_score - sheet_score
            
            # Normalize
            total = helix_score + sheet_score + coil_score
            ss_prediction[i, 0] = helix_score / total
            ss_prediction[i, 1] = sheet_score / total
            ss_prediction[i, 2] = coil_score / total
            
            # Turn prediction (simplified)
            if i >= 2 and i < seq_len - 2:
                turn_residues = 'GNPST'
                turn_count = sum(1 for aa in sequence[i-2:i+3] if aa in turn_residues)
                turns[i] = turn_count / 5.0
                
            # Loop regions (between secondary structures)
            if ss_prediction[i, 2] > 0.6:  # High coil probability
                loops[i] = 1.0
                
        return {
            'ss_structure': ss_prediction,
            'turns': turns,
            'loops': loops
        }
        
    def _analyze_tertiary_scale(
        self,
        sequence: str,
        coordinates: Optional[torch.Tensor],
        window_size: int
    ) -> Dict:
        """Analyze tertiary structure features"""
        
        seq_len = len(sequence)
        
        if coordinates is not None and coordinates.shape[0] == seq_len:
            # Real structure analysis
            contacts = self._calculate_contact_map(coordinates)
            packing = self._calculate_packing_density(coordinates)
            accessibility = self._calculate_accessibility(coordinates)
        else:
            # Predicted structure features
            contacts = self._predict_contact_map(sequence)
            packing = self._predict_packing_density(sequence)
            accessibility = self._predict_accessibility(sequence)
            
        return {
            'contacts': contacts,
            'packing': packing,
            'accessibility': accessibility
        }
        
    def _analyze_quaternary_scale(
        self,
        sequence: str,
        coordinates: Optional[torch.Tensor]
    ) -> Dict:
        """Analyze quaternary structure and interfaces"""
        
        # Mock quaternary analysis
        interfaces = []
        allosteric_networks = {}
        
        if coordinates is not None:
            # Would analyze protein-protein interfaces
            # For now, return empty results
            pass
            
        return {
            'interfaces': interfaces,
            'allosteric_networks': allosteric_networks
        }
        
    def _calculate_contact_map(self, coordinates: torch.Tensor) -> np.ndarray:
        """Calculate contact map from coordinates"""
        
        seq_len = coordinates.shape[0]
        distances = torch.cdist(coordinates, coordinates)
        
        # Contact threshold (8 Angstroms)
        contacts = (distances < 8.0).float().cpu().numpy()
        
        return contacts
        
    def _calculate_packing_density(self, coordinates: torch.Tensor) -> np.ndarray:
        """Calculate packing density around each residue"""
        
        seq_len = coordinates.shape[0]
        packing = np.zeros(seq_len)
        
        for i in range(seq_len):
            # Count neighbors within 12A
            distances = torch.norm(coordinates - coordinates[i], dim=1)
            neighbors = (distances < 12.0).sum().item() - 1  # Exclude self
            packing[i] = neighbors / 20.0  # Normalize
            
        return packing
        
    def _calculate_accessibility(self, coordinates: torch.Tensor) -> np.ndarray:
        """Calculate solvent accessibility"""
        
        seq_len = coordinates.shape[0]
        accessibility = np.zeros(seq_len)
        
        # Simplified accessibility calculation
        for i in range(seq_len):
            distances = torch.norm(coordinates - coordinates[i], dim=1)
            close_residues = (distances < 6.0).sum().item() - 1
            accessibility[i] = max(0, 1.0 - close_residues / 10.0)
            
        return accessibility
        
    def _predict_contact_map(self, sequence: str) -> np.ndarray:
        """Predict contact map from sequence"""
        
        seq_len = len(sequence)
        contacts = np.zeros((seq_len, seq_len))
        
        # Simple contact prediction based on sequence separation and composition
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                sep = j - i
                
                # Local contacts (sequence neighbors)
                if sep <= 4:
                    contacts[i, j] = contacts[j, i] = 0.8
                elif sep <= 12:
                    contacts[i, j] = contacts[j, i] = 0.3
                else:
                    # Long-range contacts based on amino acid pairing
                    aa_i, aa_j = sequence[i], sequence[j]
                    contact_prob = self._predict_aa_contact_probability(aa_i, aa_j, sep)
                    contacts[i, j] = contacts[j, i] = contact_prob
                    
        return contacts
        
    def _predict_aa_contact_probability(self, aa_i: str, aa_j: str, separation: int) -> float:
        """Predict contact probability between two amino acids"""
        
        # Simplified contact prediction
        hydrophobic_aa = set('AILMFWV')
        charged_aa = {'R': 1, 'K': 1, 'D': -1, 'E': -1}
        polar_aa = set('STYNQH')
        
        base_prob = max(0.01, 0.2 * np.exp(-separation / 30))
        
        # Hydrophobic contacts
        if aa_i in hydrophobic_aa and aa_j in hydrophobic_aa:
            return min(0.8, base_prob * 3)
            
        # Electrostatic contacts
        if aa_i in charged_aa and aa_j in charged_aa:
            if charged_aa[aa_i] * charged_aa[aa_j] < 0:  # Opposite charges
                return min(0.7, base_prob * 2.5)
            else:  # Same charges
                return base_prob * 0.3
                
        # Polar contacts
        if aa_i in polar_aa and aa_j in polar_aa:
            return min(0.5, base_prob * 1.5)
            
        return base_prob
        
    def _predict_packing_density(self, sequence: str) -> np.ndarray:
        """Predict packing density from sequence"""
        
        seq_len = len(sequence)
        packing = np.zeros(seq_len)
        
        # Simple packing prediction based on amino acid properties
        for i, aa in enumerate(sequence):
            if aa in 'AILMFWV':  # Hydrophobic, well-packed
                packing[i] = 0.8
            elif aa in 'GP':  # Glycine and proline, less packed
                packing[i] = 0.3
            else:
                packing[i] = 0.5
                
        return packing
        
    def _predict_accessibility(self, sequence: str) -> np.ndarray:
        """Predict solvent accessibility from sequence"""
        
        seq_len = len(sequence)
        accessibility = np.zeros(seq_len)
        
        # Simple accessibility prediction
        for i, aa in enumerate(sequence):
            if aa in 'RKDEQNHST':  # Polar/charged, likely accessible
                accessibility[i] = 0.7
            elif aa in 'AILMFWV':  # Hydrophobic, likely buried
                accessibility[i] = 0.2
            else:
                accessibility[i] = 0.4
                
        # Terminal residues more accessible
        if seq_len > 0:
            accessibility[0] = min(1.0, accessibility[0] + 0.3)
            accessibility[-1] = min(1.0, accessibility[-1] + 0.3)
            
        return accessibility
        
    def _combine_scale_results(
        self,
        sequence: str,
        scale_results: Dict[str, Dict],
        coordinates: Optional[torch.Tensor]
    ) -> MultiScaleAnalysis:
        """Combine results from different scales into unified analysis"""
        
        # Combine residue features
        residue_features = {}
        if 'residue' in scale_results:
            residue_features.update(scale_results['residue'])
            
        # Get secondary structure
        secondary_structure = np.zeros(len(sequence))
        if 'secondary' in scale_results:
            ss_probs = scale_results['secondary'].get('ss_structure', np.zeros((len(sequence), 3)))
            secondary_structure = np.argmax(ss_probs, axis=1)
            
        # Get tertiary contacts
        tertiary_contacts = np.zeros((len(sequence), len(sequence)))
        if 'tertiary' in scale_results:
            tertiary_contacts = scale_results['tertiary'].get('contacts', tertiary_contacts)
            
        # Quaternary interfaces
        quaternary_interfaces = []
        if 'quaternary' in scale_results:
            quaternary_interfaces = scale_results['quaternary'].get('interfaces', [])
            
        return MultiScaleAnalysis(
            sequence=sequence,
            residue_features=residue_features,
            secondary_structure=secondary_structure,
            tertiary_contacts=tertiary_contacts,
            quaternary_interfaces=quaternary_interfaces
        )
        
    def _get_default_scale_result(self, scale: str, seq_len: int) -> Dict:
        """Get default result for failed scale analysis"""
        
        if scale == 'residue':
            return {
                'local_environment': np.zeros((seq_len, 5)),
                'flexibility': np.ones(seq_len) * 0.5
            }
        elif scale == 'secondary':
            return {
                'ss_structure': np.ones((seq_len, 3)) / 3,
                'turns': np.zeros(seq_len),
                'loops': np.ones(seq_len) * 0.3
            }
        elif scale == 'tertiary':
            return {
                'contacts': np.zeros((seq_len, seq_len)),
                'packing': np.ones(seq_len) * 0.5,
                'accessibility': np.ones(seq_len) * 0.5
            }
        elif scale == 'quaternary':
            return {
                'interfaces': [],
                'allosteric_networks': {}
            }
        else:
            return {}
            
    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()