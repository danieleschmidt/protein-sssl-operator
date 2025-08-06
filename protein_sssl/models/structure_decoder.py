import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, NamedTuple
from dataclasses import dataclass
import math

@dataclass
class StructurePrediction:
    coordinates: torch.Tensor  # [seq_len, 3] CA coordinates
    confidence: float
    plddt_score: float
    predicted_tm: float
    distance_map: torch.Tensor
    torsion_angles: torch.Tensor
    secondary_structure: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    sequence: str = ""
    
    def save_pdb(self, filename: str):
        """Save predicted structure as PDB file"""
        coords = self.coordinates.cpu().numpy()
        
        with open(filename, 'w') as f:
            f.write("HEADER    PREDICTED PROTEIN STRUCTURE\n")
            f.write("TITLE     PROTEIN-SSSL-OPERATOR PREDICTION\n")
            f.write("MODEL        1\n")
            
            for i, (x, y, z) in enumerate(coords):
                f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{self.confidence*100:6.2f}           C\n")
                       
            f.write("ENDMDL\n")
            f.write("END\n")
            
    def save_confidence_plot(self, filename: str):
        """Save confidence plot"""
        try:
            import matplotlib.pyplot as plt
            
            if self.uncertainty is not None:
                uncertainty = self.uncertainty.cpu().numpy()
                confidence = 100 * (1 - uncertainty)
            else:
                confidence = np.full(len(self.coordinates), self.confidence * 100)
                
            plt.figure(figsize=(12, 4))
            plt.plot(confidence)
            plt.xlabel('Residue Position')
            plt.ylabel('Confidence (%)')
            plt.title('Per-Residue Confidence')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("Matplotlib not available for plotting")

class DistanceGeometry(nn.Module):
    """Convert distance maps to 3D coordinates"""
    
    def __init__(self):
        super().__init__()
        self.distance_bins = torch.linspace(2.0, 22.0, 64)
        
    def distances_to_coords(
        self, 
        distance_probs: torch.Tensor,
        torsion_angles: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert distance probability matrix to 3D coordinates
        Args:
            distance_probs: [seq_len, seq_len, n_bins] distance probabilities
            torsion_angles: [seq_len, n_angles] torsion angles
        Returns:
            coords: [seq_len, 3] 3D coordinates
        """
        seq_len = distance_probs.shape[0]
        device = distance_probs.device
        
        # Convert probabilities to expected distances
        distances = torch.sum(
            distance_probs * self.distance_bins.to(device).unsqueeze(0).unsqueeze(0), 
            dim=-1
        )
        
        # Simple multidimensional scaling approximation
        # In practice, would use more sophisticated methods
        coords = self._mds_embedding(distances)
        
        # Refine with torsion angles if available
        if torsion_angles is not None:
            coords = self._refine_with_torsions(coords, torsion_angles)
            
        return coords
        
    def _mds_embedding(self, distances: torch.Tensor) -> torch.Tensor:
        """Classical multidimensional scaling"""
        n = distances.shape[0]
        device = distances.device
        
        # Double centering
        H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
        B = -0.5 * H @ (distances ** 2) @ H
        
        # Eigendecomposition  
        eigenvals, eigenvecs = torch.linalg.eigh(B)
        
        # Take top 3 eigenvalues/vectors
        idx = torch.argsort(eigenvals, descending=True)[:3]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Ensure positive eigenvalues
        eigenvals = torch.clamp(eigenvals, min=0.0)
        
        # Coordinates
        coords = eigenvecs @ torch.diag(torch.sqrt(eigenvals))
        
        return coords
        
    def _refine_with_torsions(
        self, 
        coords: torch.Tensor, 
        torsion_angles: torch.Tensor
    ) -> torch.Tensor:
        """Refine coordinates using torsion angle constraints"""
        # Simplified torsion refinement
        # In practice, would use more sophisticated molecular geometry
        return coords

class StructurePredictor(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
        num_ensemble_models: int = 1,
        confidence_threshold: float = 0.8,
    ):
        super().__init__()
        
        self.device = device
        self.num_ensemble_models = num_ensemble_models  
        self.confidence_threshold = confidence_threshold
        
        # Load model
        if model_path is not None:
            self.model = torch.load(model_path, map_location=device)
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")
            
        self.model.eval()
        self.model.to(device)
        
        # Distance geometry module
        self.distance_geometry = DistanceGeometry()
        
        # Tokenizer
        from .ssl_encoder import ProteinTokenizer
        self.tokenizer = ProteinTokenizer()
        
    def predict(
        self,
        sequence: str,
        return_confidence: bool = True,
        num_recycles: int = 3,
        temperature: float = 0.1
    ) -> StructurePrediction:
        """Predict protein structure from sequence"""
        
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer.encode(sequence, max_length=len(sequence))
        input_ids = inputs["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = inputs["attention_mask"].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                return_uncertainty=return_confidence
            )
            
            # Extract predictions
            distance_logits = outputs["distance_logits"].squeeze(0)  # [seq_len, seq_len, 64]
            torsion_angles = outputs["torsion_angles"].squeeze(0)    # [seq_len, 8]  
            secondary_structure = outputs["secondary_structure"].squeeze(0)  # [seq_len, 8]
            
            # Convert to probabilities
            distance_probs = F.softmax(distance_logits, dim=-1)
            
            # Generate 3D coordinates
            coordinates = self.distance_geometry.distances_to_coords(
                distance_probs, 
                torsion_angles
            )
            
            # Calculate confidence scores
            confidence, plddt_score = self._calculate_confidence(outputs, return_confidence)
            
            # Predicted TM-score (simplified heuristic)
            predicted_tm = self._predict_tm_score(distance_probs, confidence)
            
            # Extract uncertainty
            uncertainty = outputs.get("uncertainty", None)
            if uncertainty is not None:
                uncertainty = uncertainty.squeeze(0)
                
        return StructurePrediction(
            coordinates=coordinates,
            confidence=confidence,
            plddt_score=plddt_score,
            predicted_tm=predicted_tm,
            distance_map=distance_probs,
            torsion_angles=torsion_angles,
            secondary_structure=F.softmax(secondary_structure, dim=-1),
            uncertainty=uncertainty,
            sequence=sequence
        )
        
    def _calculate_confidence(
        self, 
        outputs: Dict[str, torch.Tensor], 
        return_confidence: bool
    ) -> tuple:
        """Calculate confidence metrics"""
        
        if not return_confidence or "uncertainty" not in outputs:
            # Default confidence
            return 0.75, 75.0
            
        uncertainty = outputs["uncertainty"].squeeze(0)  # [seq_len]
        
        # Convert uncertainty to confidence
        confidence_per_residue = 1.0 - uncertainty
        mean_confidence = confidence_per_residue.mean().item()
        
        # pLDDT-like score (0-100)
        plddt = (confidence_per_residue * 100).mean().item()
        
        return mean_confidence, plddt
        
    def _predict_tm_score(
        self, 
        distance_probs: torch.Tensor, 
        confidence: float
    ) -> float:
        """Predict TM-score from distance probabilities and confidence"""
        
        # Simple heuristic: TM-score correlates with confidence and contact accuracy
        seq_len = distance_probs.shape[0]
        
        # Count probable contacts (distance < 8Å)
        contact_probs = distance_probs[:, :, :20].sum(dim=-1)  # First 20 bins < 8Å
        contact_mask = contact_probs > 0.5
        
        # Calculate contact accuracy proxy
        contact_accuracy = contact_mask.float().mean().item()
        
        # Combine with confidence
        predicted_tm = 0.5 * confidence + 0.5 * contact_accuracy
        
        return min(predicted_tm, 1.0)
        
    def analyze_uncertainty(self, prediction: StructurePrediction) -> dict:
        """Analyze uncertainty in the prediction"""
        
        if prediction.uncertainty is None:
            return {
                "uncertain_regions": [],
                "stabilizing_mutations": [],
                "confidence_summary": "No uncertainty information available"
            }
            
        uncertainty = prediction.uncertainty.cpu().numpy()
        
        # Find high uncertainty regions
        high_uncertainty_threshold = np.percentile(uncertainty, 75)
        uncertain_residues = np.where(uncertainty > high_uncertainty_threshold)[0]
        
        # Group consecutive uncertain residues
        uncertain_regions = []
        if len(uncertain_residues) > 0:
            current_region = [uncertain_residues[0]]
            for i in range(1, len(uncertain_residues)):
                if uncertain_residues[i] - uncertain_residues[i-1] == 1:
                    current_region.append(uncertain_residues[i])
                else:
                    uncertain_regions.append(current_region)
                    current_region = [uncertain_residues[i]]
            uncertain_regions.append(current_region)
            
        # Convert to ranges
        uncertain_ranges = []
        for region in uncertain_regions:
            if len(region) >= 3:  # Only consider regions of 3+ residues
                uncertain_ranges.append(f"{region[0]+1}-{region[-1]+1}")
                
        # Suggest stabilizing mutations (simplified heuristic)
        stabilizing_mutations = []
        sequence = prediction.sequence
        for region in uncertain_regions:
            for pos in region:
                if pos < len(sequence):
                    current_aa = sequence[pos]
                    # Simple heuristic: suggest more stable amino acids
                    if current_aa in ['G', 'P']:  # Glycine, Proline can be destabilizing
                        stabilizing_mutations.append(f"{current_aa}{pos+1}A")
                        
        return {
            "uncertain_regions": uncertain_ranges,
            "stabilizing_mutations": stabilizing_mutations[:5],  # Top 5 suggestions
            "confidence_summary": f"Mean confidence: {(1-uncertainty.mean())*100:.1f}%"
        }