# User Guides by Persona

## Overview

This comprehensive guide provides specialized documentation for different user types, including detailed tutorials, workflows, and best practices tailored to specific roles and use cases.

## Table of Contents

1. [Researchers & Scientists](#researchers--scientists)
2. [Bioinformatics Developers](#bioinformatics-developers)
3. [Data Scientists & ML Engineers](#data-scientists--ml-engineers)
4. [Pharmaceutical Industry](#pharmaceutical-industry)
5. [Academic Instructors](#academic-instructors)
6. [Operations & DevOps Teams](#operations--devops-teams)
7. [Integration Partners](#integration-partners)

## Researchers & Scientists

### Quick Start for Research

#### Setting Up Your Research Environment

**Step 1: Account Registration**
```bash
# Register for academic access (free tier)
curl -X POST https://api.protein-sssl.terragonlabs.ai/v1/auth/academic-register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "researcher@university.edu",
    "institution": "University of Research",
    "department": "Biochemistry",
    "research_focus": "protein folding dynamics",
    "orcid": "0000-0000-0000-0000",
    "expected_usage": "structure_prediction"
  }'
```

**Step 2: Install Research Tools**
```bash
# Install Python dependencies
pip install protein-sssl-sdk biopython matplotlib seaborn pandas numpy

# Install visualization tools
pip install py3Dmol nglview jupyter

# Install structural analysis tools
pip install mdanalysis pymol-open-source
```

**Step 3: Jupyter Notebook Setup**
```python
# research_setup.ipynb
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from protein_sssl_sdk import ProteinSSLClient
import py3Dmol
from Bio import SeqIO, PDB
import warnings
warnings.filterwarnings('ignore')

# Initialize client
client = ProteinSSLClient(api_key='YOUR_ACADEMIC_API_KEY')

# Test connection
status = client.get_status()
print(f"✅ Connected to Protein-SSL API v{status['version']}")
print(f"📊 Current queue depth: {status['system_load']['queue_depth']}")
print(f"🔄 Rate limit: {status['rate_limits']['remaining']}/{status['rate_limits']['requests_per_hour']} remaining")
```

### Research Workflows

#### Workflow 1: Single Protein Analysis

**Objective**: Predict and analyze the structure of a single protein of interest

```python
class SingleProteinAnalyzer:
    def __init__(self, api_key):
        self.client = ProteinSSLClient(api_key=api_key)
        
    def analyze_protein(self, sequence, protein_name="Unknown", save_results=True):
        """Complete analysis pipeline for a single protein"""
        
        print(f"🧬 Analyzing protein: {protein_name}")
        print(f"📏 Sequence length: {len(sequence)} residues")
        
        # Step 1: Basic validation
        if not self._validate_sequence(sequence):
            return None
            
        # Step 2: Structure prediction
        print("🔄 Predicting structure...")
        prediction = self.client.predict(
            sequence=sequence,
            return_confidence=True,
            return_uncertainty=True,
            metadata={'protein_name': protein_name}
        )
        
        # Step 3: Quality assessment
        quality_metrics = self._assess_quality(prediction)
        
        # Step 4: Domain analysis
        print("🔍 Analyzing domains...")
        domains = self.client.analyze_domains(
            prediction_id=prediction.id,
            include_functional_annotation=True
        )
        
        # Step 5: Generate report
        report = self._generate_report(prediction, domains, quality_metrics)
        
        # Step 6: Save results
        if save_results:
            self._save_results(prediction, domains, report, protein_name)
        
        return {
            'prediction': prediction,
            'domains': domains,
            'quality_metrics': quality_metrics,
            'report': report
        }
    
    def _validate_sequence(self, sequence):
        """Validate protein sequence"""
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(sequence.upper()) - valid_chars
        
        if invalid_chars:
            print(f"❌ Invalid characters found: {invalid_chars}")
            return False
            
        if len(sequence) < 20:
            print("❌ Sequence too short (minimum 20 residues)")
            return False
            
        return True
    
    def _assess_quality(self, prediction):
        """Assess prediction quality"""
        confidence = prediction.metrics['confidence']
        plddt = prediction.metrics['plddt_score']
        
        # Quality classification
        if confidence >= 0.9 and plddt >= 90:
            quality = "Excellent"
        elif confidence >= 0.8 and plddt >= 80:
            quality = "Good"
        elif confidence >= 0.7 and plddt >= 70:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        return {
            'overall_quality': quality,
            'confidence_score': confidence,
            'plddt_score': plddt,
            'low_confidence_regions': self._find_low_confidence_regions(prediction)
        }
    
    def _find_low_confidence_regions(self, prediction, threshold=0.7):
        """Identify regions with low confidence"""
        confidence_scores = prediction.structure['confidence_scores']
        low_conf_regions = []
        
        in_region = False
        start_pos = None
        
        for i, score in enumerate(confidence_scores):
            if score < threshold and not in_region:
                start_pos = i + 1  # 1-indexed
                in_region = True
            elif score >= threshold and in_region:
                low_conf_regions.append((start_pos, i))
                in_region = False
        
        # Handle region at end of sequence
        if in_region:
            low_conf_regions.append((start_pos, len(confidence_scores)))
        
        return low_conf_regions
    
    def _generate_report(self, prediction, domains, quality_metrics):
        """Generate comprehensive analysis report"""
        
        report = f"""
# Protein Structure Analysis Report

## Basic Information
- **Prediction ID**: {prediction.id}
- **Sequence Length**: {len(prediction.sequence)} residues
- **Processing Time**: {prediction.metrics['processing_time']:.2f} seconds
- **Prediction Date**: {prediction.created_at}

## Quality Assessment
- **Overall Quality**: {quality_metrics['overall_quality']}
- **Confidence Score**: {quality_metrics['confidence_score']:.3f}
- **pLDDT Score**: {quality_metrics['plddt_score']:.1f}

## Low Confidence Regions
"""
        
        if quality_metrics['low_confidence_regions']:
            for start, end in quality_metrics['low_confidence_regions']:
                report += f"- Residues {start}-{end}\n"
        else:
            report += "- None detected\n"
        
        report += f"""
## Domain Analysis
- **Number of Domains**: {len(domains.domains)}

"""
        
        for i, domain in enumerate(domains.domains, 1):
            report += f"""
### Domain {i}
- **Position**: {domain['start']}-{domain['end']}
- **Type**: {domain['type']}
- **Confidence**: {domain['confidence']:.3f}
"""
            
            if 'functional_annotation' in domain:
                func = domain['functional_annotation']
                report += f"- **Predicted Function**: {func.get('predicted_function', 'Unknown')}\n"
                if 'go_terms' in func:
                    report += f"- **GO Terms**: {', '.join(func['go_terms'])}\n"
        
        return report
    
    def _save_results(self, prediction, domains, report, protein_name):
        """Save analysis results"""
        import os
        from datetime import datetime
        
        # Create output directory
        safe_name = "".join(c for c in protein_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_dir = f"results_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PDB structure
        prediction.save_pdb(f"{output_dir}/structure.pdb")
        
        # Save confidence plot
        prediction.save_confidence_plot(f"{output_dir}/confidence_plot.png")
        
        # Save report
        with open(f"{output_dir}/analysis_report.md", 'w') as f:
            f.write(report)
        
        # Save domain data
        import json
        with open(f"{output_dir}/domains.json", 'w') as f:
            json.dump(domains.to_dict(), f, indent=2)
        
        print(f"📁 Results saved to: {output_dir}/")

# Usage example
analyzer = SingleProteinAnalyzer('YOUR_API_KEY')

# Example: Analyze a protein of interest
sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
results = analyzer.analyze_protein(
    sequence=sequence,
    protein_name="Example Protein",
    save_results=True
)

print("✅ Analysis complete!")
print(f"🎯 Quality: {results['quality_metrics']['overall_quality']}")
print(f"🧩 Domains found: {len(results['domains'].domains)}")
```

#### Workflow 2: Comparative Structure Analysis

**Objective**: Compare structures of related proteins or protein variants

```python
class ComparativeAnalyzer:
    def __init__(self, api_key):
        self.client = ProteinSSLClient(api_key=api_key)
        
    def compare_protein_variants(self, sequences_dict, reference_name=None):
        """Compare multiple protein variants"""
        
        print(f"🔬 Comparing {len(sequences_dict)} protein variants...")
        
        # Step 1: Predict all structures
        predictions = {}
        for name, sequence in sequences_dict.items():
            print(f"  🔄 Predicting {name}...")
            pred = self.client.predict(
                sequence=sequence,
                return_confidence=True,
                metadata={'variant_name': name}
            )
            predictions[name] = pred
        
        # Step 2: Structural comparisons
        print("📐 Performing structural alignments...")
        comparisons = self._perform_pairwise_comparisons(predictions)
        
        # Step 3: Analyze differences
        analysis = self._analyze_structural_differences(predictions, comparisons)
        
        # Step 4: Create visualizations
        self._create_comparison_plots(predictions, comparisons, analysis)
        
        return {
            'predictions': predictions,
            'comparisons': comparisons,
            'analysis': analysis
        }
    
    def _perform_pairwise_comparisons(self, predictions):
        """Perform all pairwise structural comparisons"""
        comparisons = {}
        names = list(predictions.keys())
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                comparison_key = f"{name1}_vs_{name2}"
                
                print(f"  📊 Comparing {name1} vs {name2}...")
                
                comparison = self.client.compare_structures(
                    structure1_id=predictions[name1].id,
                    structure2_id=predictions[name2].id,
                    alignment_method='tm_align',
                    include_detailed_analysis=True
                )
                
                comparisons[comparison_key] = comparison
        
        return comparisons
    
    def _analyze_structural_differences(self, predictions, comparisons):
        """Analyze structural differences between variants"""
        
        # Create similarity matrix
        names = list(predictions.keys())
        n = len(names)
        similarity_matrix = pd.DataFrame(
            index=names, 
            columns=names, 
            dtype=float
        )
        
        # Fill diagonal
        for name in names:
            similarity_matrix.loc[name, name] = 1.0
        
        # Fill from comparisons
        for comp_key, comp_data in comparisons.items():
            name1, name2 = comp_key.split('_vs_')
            tm_score = comp_data.alignment_results['tm_score']
            similarity_matrix.loc[name1, name2] = tm_score
            similarity_matrix.loc[name2, name1] = tm_score
        
        # Identify most/least similar pairs
        upper_triangle = similarity_matrix.where(
            np.triu(np.ones(similarity_matrix.shape), k=1).astype(bool)
        )
        
        most_similar = upper_triangle.stack().idxmax()
        least_similar = upper_triangle.stack().idxmin()
        
        return {
            'similarity_matrix': similarity_matrix,
            'most_similar_pair': {
                'variants': most_similar,
                'tm_score': upper_triangle.loc[most_similar]
            },
            'least_similar_pair': {
                'variants': least_similar,
                'tm_score': upper_triangle.loc[least_similar]
            },
            'average_similarity': upper_triangle.stack().mean()
        }
    
    def _create_comparison_plots(self, predictions, comparisons, analysis):
        """Create visualization plots"""
        
        # 1. Similarity heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            analysis['similarity_matrix'], 
            annot=True, 
            cmap='viridis',
            vmin=0, 
            vmax=1,
            square=True
        )
        plt.title('Structural Similarity Matrix (TM-scores)')
        plt.tight_layout()
        plt.savefig('similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Confidence comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confidence scores
        names = list(predictions.keys())
        confidences = [pred.metrics['confidence'] for pred in predictions.values()]
        plddt_scores = [pred.metrics['plddt_score'] for pred in predictions.values()]
        
        axes[0,0].bar(names, confidences)
        axes[0,0].set_title('Confidence Scores')
        axes[0,0].set_ylabel('Confidence')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        axes[0,1].bar(names, plddt_scores)
        axes[0,1].set_title('pLDDT Scores')
        axes[0,1].set_ylabel('pLDDT')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Per-residue confidence for first two variants
        if len(names) >= 2:
            pred1, pred2 = predictions[names[0]], predictions[names[1]]
            conf1 = pred1.structure['confidence_scores']
            conf2 = pred2.structure['confidence_scores']
            
            axes[1,0].plot(conf1, label=names[0], alpha=0.7)
            axes[1,0].plot(conf2, label=names[1], alpha=0.7)
            axes[1,0].set_title('Per-Residue Confidence Comparison')
            axes[1,0].set_xlabel('Residue Position')
            axes[1,0].set_ylabel('Confidence')
            axes[1,0].legend()
        
        # TM-score distribution
        tm_scores = []
        for comp_data in comparisons.values():
            tm_scores.append(comp_data.alignment_results['tm_score'])
        
        axes[1,1].hist(tm_scores, bins=10, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('TM-Score Distribution')
        axes[1,1].set_xlabel('TM-Score')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
comparator = ComparativeAnalyzer('YOUR_API_KEY')

# Example: Compare wild-type and mutant variants
variants = {
    'Wild_Type': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
    'Mutant_L10A': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYAPPSQAIQDLLKRMKV',  # L->A
    'Mutant_V47F': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKF',  # V->F
    'Double_Mutant': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYAPPSQAIQDLLKRMKF'  # Both
}

results = comparator.compare_protein_variants(variants)

print("🔍 Comparative Analysis Complete!")
print(f"Most similar: {results['analysis']['most_similar_pair']['variants']} "
      f"(TM-score: {results['analysis']['most_similar_pair']['tm_score']:.3f})")
print(f"Least similar: {results['analysis']['least_similar_pair']['variants']} "
      f"(TM-score: {results['analysis']['least_similar_pair']['tm_score']:.3f})")
```

#### Workflow 3: High-Throughput Screening

**Objective**: Screen large numbers of protein sequences for specific properties

```python
class HighThroughputScreener:
    def __init__(self, api_key, max_concurrent=5):
        self.client = ProteinSSLClient(api_key=api_key)
        self.max_concurrent = max_concurrent
        
    def screen_protein_library(self, sequences_file, screening_criteria):
        """Screen protein library based on specified criteria"""
        
        # Load sequences
        sequences = self._load_sequences(sequences_file)
        print(f"🧬 Loaded {len(sequences)} sequences for screening")
        
        # Batch prediction
        print("🔄 Running batch predictions...")
        batch_results = self._run_batch_predictions(sequences)
        
        # Apply screening criteria
        print("🔍 Applying screening criteria...")
        hits = self._apply_screening_criteria(batch_results, screening_criteria)
        
        # Analyze hits
        analysis = self._analyze_hits(hits, sequences)
        
        # Generate report
        report = self._generate_screening_report(hits, analysis, screening_criteria)
        
        return {
            'total_screened': len(sequences),
            'hits': hits,
            'hit_rate': len(hits) / len(sequences),
            'analysis': analysis,
            'report': report
        }
    
    def _load_sequences(self, sequences_file):
        """Load sequences from FASTA file"""
        sequences = []
        for record in SeqIO.parse(sequences_file, 'fasta'):
            sequences.append({
                'id': record.id,
                'description': record.description,
                'sequence': str(record.seq),
                'length': len(record.seq)
            })
        return sequences
    
    def _run_batch_predictions(self, sequences):
        """Run predictions in batches"""
        batch_size = 50  # Adjust based on your quota
        results = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            print(f"  📦 Processing batch {i//batch_size + 1} ({len(batch)} sequences)")
            
            # Prepare batch request
            batch_sequences = [
                {
                    'id': seq['id'],
                    'sequence': seq['sequence'],
                    'metadata': {'description': seq['description']}
                }
                for seq in batch
            ]
            
            # Submit batch
            batch_job = self.client.predict_batch(
                sequences=batch_sequences,
                options={
                    'return_confidence': True,
                    'priority': 'normal'
                }
            )
            
            # Wait for completion
            while not batch_job.is_complete():
                time.sleep(30)
                batch_job.refresh()
                print(f"    ⏳ Progress: {batch_job.progress.percentage:.1f}%")
            
            # Collect results
            for result in batch_job.results:
                if result.status == 'completed':
                    results.append({
                        'id': result.id,
                        'prediction': result.prediction,
                        'sequence': next(s for s in batch if s['id'] == result.id)
                    })
                else:
                    print(f"    ❌ Failed: {result.id} - {result.error}")
        
        return results
    
    def _apply_screening_criteria(self, batch_results, criteria):
        """Apply screening criteria to filter hits"""
        hits = []
        
        for result in batch_results:
            pred = result['prediction']
            seq = result['sequence']
            
            # Check each criterion
            meets_criteria = True
            
            if 'min_confidence' in criteria:
                if pred.metrics['confidence'] < criteria['min_confidence']:
                    meets_criteria = False
            
            if 'min_plddt' in criteria:
                if pred.metrics['plddt_score'] < criteria['min_plddt']:
                    meets_criteria = False
            
            if 'max_length' in criteria:
                if seq['length'] > criteria['max_length']:
                    meets_criteria = False
            
            if 'min_length' in criteria:
                if seq['length'] < criteria['min_length']:
                    meets_criteria = False
            
            if 'required_domains' in criteria:
                # Analyze domains
                domains = self.client.analyze_domains(pred.id)
                domain_types = [d['type'] for d in domains.domains]
                
                for required_domain in criteria['required_domains']:
                    if required_domain not in domain_types:
                        meets_criteria = False
                        break
            
            if 'max_disorder' in criteria:
                # Check disorder content
                analysis = self.client.analyze_structure(pred.id)
                if analysis.disorder_fraction > criteria['max_disorder']:
                    meets_criteria = False
            
            if meets_criteria:
                hits.append({
                    'id': result['id'],
                    'sequence': seq,
                    'prediction': pred,
                    'score': self._calculate_hit_score(pred, seq, criteria)
                })
        
        # Sort hits by score
        hits.sort(key=lambda x: x['score'], reverse=True)
        
        return hits
    
    def _calculate_hit_score(self, prediction, sequence, criteria):
        """Calculate composite score for ranking hits"""
        score = 0
        
        # Confidence contribution
        score += prediction.metrics['confidence'] * 0.4
        
        # pLDDT contribution
        score += (prediction.metrics['plddt_score'] / 100) * 0.4
        
        # Length penalty (prefer medium-length proteins)
        optimal_length = criteria.get('optimal_length', 300)
        length_penalty = 1 - abs(sequence['length'] - optimal_length) / optimal_length
        score += max(0, length_penalty) * 0.2
        
        return score
    
    def _analyze_hits(self, hits, all_sequences):
        """Analyze screening hits"""
        
        if not hits:
            return {'message': 'No hits found'}
        
        # Length distribution
        hit_lengths = [h['sequence']['length'] for h in hits]
        all_lengths = [s['length'] for s in all_sequences]
        
        # Confidence distribution
        hit_confidences = [h['prediction'].metrics['confidence'] for h in hits]
        
        # pLDDT distribution
        hit_plddt = [h['prediction'].metrics['plddt_score'] for h in hits]
        
        return {
            'hit_count': len(hits),
            'length_stats': {
                'hits': {'mean': np.mean(hit_lengths), 'std': np.std(hit_lengths)},
                'all': {'mean': np.mean(all_lengths), 'std': np.std(all_lengths)}
            },
            'confidence_stats': {
                'mean': np.mean(hit_confidences),
                'std': np.std(hit_confidences),
                'min': np.min(hit_confidences),
                'max': np.max(hit_confidences)
            },
            'plddt_stats': {
                'mean': np.mean(hit_plddt),
                'std': np.std(hit_plddt),
                'min': np.min(hit_plddt),
                'max': np.max(hit_plddt)
            },
            'top_hits': hits[:10]  # Top 10 hits
        }
    
    def _generate_screening_report(self, hits, analysis, criteria):
        """Generate screening report"""
        
        report = f"""
# High-Throughput Screening Report

## Screening Criteria
"""
        for key, value in criteria.items():
            report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        report += f"""
## Results Summary
- **Total Hits**: {len(hits)}
- **Hit Rate**: {len(hits)/analysis.get('total_screened', len(hits))*100:.1f}%

## Hit Statistics
- **Confidence**: {analysis['confidence_stats']['mean']:.3f} ± {analysis['confidence_stats']['std']:.3f}
- **pLDDT Score**: {analysis['plddt_stats']['mean']:.1f} ± {analysis['plddt_stats']['std']:.1f}
- **Length**: {analysis['length_stats']['hits']['mean']:.0f} ± {analysis['length_stats']['hits']['std']:.0f} residues

## Top 10 Hits
"""
        
        for i, hit in enumerate(analysis['top_hits'], 1):
            report += f"""
### {i}. {hit['id']}
- **Score**: {hit['score']:.3f}
- **Confidence**: {hit['prediction'].metrics['confidence']:.3f}
- **pLDDT**: {hit['prediction'].metrics['plddt_score']:.1f}
- **Length**: {hit['sequence']['length']} residues
- **Description**: {hit['sequence']['description']}
"""
        
        return report

# Usage example
screener = HighThroughputScreener('YOUR_API_KEY')

# Define screening criteria
criteria = {
    'min_confidence': 0.8,
    'min_plddt': 70,
    'min_length': 100,
    'max_length': 500,
    'required_domains': ['globular'],
    'max_disorder': 0.3,
    'optimal_length': 250
}

# Run screening
results = screener.screen_protein_library(
    sequences_file='protein_library.fasta',
    screening_criteria=criteria
)

print(f"🎯 Screening Complete!")
print(f"📊 {results['total_screened']} sequences screened")
print(f"✅ {len(results['hits'])} hits found ({results['hit_rate']*100:.1f}% hit rate)")
```

### Research Tips and Best Practices

#### Data Management for Research

```python
class ResearchDataManager:
    def __init__(self, project_name):
        self.project_name = project_name
        self.base_dir = f"research_data_{project_name}"
        os.makedirs(self.base_dir, exist_ok=True)
        
    def create_experiment(self, experiment_name, description=""):
        """Create new experiment directory with metadata"""
        exp_dir = os.path.join(self.base_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create metadata file
        metadata = {
            'experiment_name': experiment_name,
            'description': description,
            'created_date': datetime.now().isoformat(),
            'sequences_analyzed': 0,
            'predictions_made': 0,
            'analyses_performed': []
        }
        
        with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return exp_dir
    
    def log_analysis(self, experiment_name, analysis_type, details):
        """Log analysis performed"""
        metadata_file = os.path.join(self.base_dir, experiment_name, 'metadata.json')
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['analyses_performed'].append({
            'type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def backup_experiment(self, experiment_name):
        """Create backup of experiment data"""
        import shutil
        
        exp_dir = os.path.join(self.base_dir, experiment_name)
        backup_name = f"{experiment_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = os.path.join(self.base_dir, 'backups', backup_name)
        
        os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
        shutil.copytree(exp_dir, backup_dir)
        
        print(f"📦 Experiment backed up to: {backup_dir}")
        return backup_dir

# Usage
data_manager = ResearchDataManager("protein_folding_study")
exp_dir = data_manager.create_experiment(
    "mutation_effects",
    "Study of mutation effects on protein stability"
)
```

#### Publication-Ready Figures

```python
def create_publication_figure(predictions_data, figure_type="comparison"):
    """Create publication-quality figures"""
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    
    if figure_type == "comparison":
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel A: Confidence comparison
        names = list(predictions_data.keys())
        confidences = [pred['confidence'] for pred in predictions_data.values()]
        
        ax1.bar(names, confidences, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Confidence Score')
        ax1.set_title('A. Prediction Confidence', fontweight='bold', loc='left')
        ax1.tick_params(axis='x', rotation=45)
        
        # Panel B: pLDDT comparison
        plddt_scores = [pred['plddt_score'] for pred in predictions_data.values()]
        
        ax2.bar(names, plddt_scores, color='darkgreen', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('pLDDT Score')
        ax2.set_title('B. Local Confidence (pLDDT)', fontweight='bold', loc='left')
        ax2.tick_params(axis='x', rotation=45)
        
        # Panel C: Per-residue confidence
        colors = plt.cm.Set1(np.linspace(0, 1, len(names)))
        
        for i, (name, pred) in enumerate(predictions_data.items()):
            residue_conf = pred['per_residue_confidence']
            ax3.plot(residue_conf, label=name, color=colors[i], linewidth=2)
        
        ax3.set_xlabel('Residue Position')
        ax3.set_ylabel('Confidence Score')
        ax3.set_title('C. Per-Residue Confidence', fontweight='bold', loc='left')
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        
        # Panel D: Structural similarity heatmap
        # (Requires similarity matrix from comparative analysis)
        if 'similarity_matrix' in predictions_data:
            im = ax4.imshow(predictions_data['similarity_matrix'], cmap='viridis', vmin=0, vmax=1)
            ax4.set_title('D. Structural Similarity', fontweight='bold', loc='left')
            ax4.set_xticks(range(len(names)))
            ax4.set_yticks(range(len(names)))
            ax4.set_xticklabels(names, rotation=45)
            ax4.set_yticklabels(names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            cbar.set_label('TM-Score')
        
        plt.tight_layout()
        plt.savefig('publication_figure.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
        
    return fig

# Usage
figure = create_publication_figure(analysis_results, "comparison")
plt.show()
```

## Bioinformatics Developers

### Development Environment Setup

#### Complete Development Stack

```bash
# development_setup.sh
#!/bin/bash

echo "🛠️ Setting up Protein-SSL development environment..."

# Create virtual environment
python3 -m venv protein_ssl_dev
source protein_ssl_dev/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install protein-sssl-sdk

# Development tools
pip install jupyter jupyterlab
pip install pytest pytest-cov black isort flake8 mypy
pip install pre-commit

# Bioinformatics libraries
pip install biopython biotite mdanalysis pymol-open-source
pip install numpy pandas matplotlib seaborn plotly

# Machine learning libraries
pip install scikit-learn torch torchvision
pip install transformers datasets

# Database connections
pip install psycopg2-binary sqlalchemy

# API development
pip install fastapi uvicorn requests-cache

# Setup pre-commit hooks
pre-commit install

echo "✅ Development environment ready!"
```

#### Project Structure Template

```python
# create_project_structure.py
import os

def create_bioinformatics_project(project_name):
    """Create standardized project structure"""
    
    structure = {
        project_name: {
            'src': {
                'data': ['__init__.py', 'loaders.py', 'preprocessors.py'],
                'models': ['__init__.py', 'predictors.py', 'analyzers.py'],
                'utils': ['__init__.py', 'api_client.py', 'visualization.py'],
                'pipelines': ['__init__.py', 'workflows.py']
            },
            'tests': {
                'unit': ['__init__.py', 'test_api_client.py', 'test_models.py'],
                'integration': ['__init__.py', 'test_pipelines.py'],
                'data': ['sample_sequences.fasta', 'test_structures.pdb']
            },
            'notebooks': ['exploration.ipynb', 'analysis.ipynb'],
            'scripts': ['run_pipeline.py', 'batch_analysis.py'],
            'configs': ['development.yaml', 'production.yaml'],
            'docs': ['README.md', 'API.md', 'USAGE.md'],
            '.': ['requirements.txt', 'setup.py', '.gitignore', 'pytest.ini']
        }
    }
    
    def create_structure(base_path, structure_dict):
        for name, content in structure_dict.items():
            if name == '.':
                current_path = base_path
            else:
                current_path = os.path.join(base_path, name)
                os.makedirs(current_path, exist_ok=True)
            
            if isinstance(content, dict):
                create_structure(current_path, content)
            elif isinstance(content, list):
                for filename in content:
                    filepath = os.path.join(current_path, filename)
                    if not os.path.exists(filepath):
                        with open(filepath, 'w') as f:
                            if filename.endswith('.py'):
                                f.write('"""Module docstring."""\n')
                            elif filename.endswith('.md'):
                                f.write(f'# {filename.replace(".md", "").title()}\n\n')
    
    create_structure('.', structure)
    print(f"✅ Project structure created for {project_name}")

# Usage
create_bioinformatics_project("my_protein_analysis")
```

### Advanced API Integration

#### Custom API Client with Advanced Features

```python
# src/utils/api_client.py
import asyncio
import aiohttp
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class AdvancedProteinSSLClient:
    """Advanced API client with caching, rate limiting, and error recovery"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.protein-sssl.terragonlabs.ai/v1",
        cache_dir: str = "./cache",
        max_retries: int = 3,
        rate_limit_per_minute: int = 60,
        enable_caching: bool = True
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_retries = max_retries
        self.rate_limit_per_minute = rate_limit_per_minute
        self.enable_caching = enable_caching
        
        # Rate limiting
        self.request_times = []
        self.min_interval = 60.0 / rate_limit_per_minute
        
        # Session management
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, endpoint: str, data: Dict) -> str:
        """Generate cache key for request"""
        content = f"{endpoint}:{str(sorted(data.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get response from cache"""
        if not self.enable_caching:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is still valid (24 hours)
                if time.time() - cached_data['timestamp'] < 86400:
                    logger.info(f"Cache hit for key: {cache_key}")
                    return cached_data['response']
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict) -> None:
        """Save response to cache"""
        if not self.enable_caching:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'timestamp': time.time(),
                    'response': response
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request with retries and error handling"""
        
        await self._rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/{endpoint.lstrip('/')}"
                
                if method.upper() == "GET":
                    async with self.session.get(url, params=params) as response:
                        return await self._handle_response(response)
                elif method.upper() == "POST":
                    async with self.session.post(url, json=data, params=params) as response:
                        return await self._handle_response(response)
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except aiohttp.ClientError as e:
                logger.error(f"Client error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        raise RuntimeError(f"Failed to complete request after {self.max_retries} attempts")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Handle API response"""
        content = await response.text()
        
        if response.status == 200:
            return await response.json()
        elif response.status == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limited, retrying after {retry_after}s")
            await asyncio.sleep(retry_after)
            raise aiohttp.ClientError("Rate limited")
        else:
            try:
                error_data = await response.json()
                error_msg = error_data.get('error', {}).get('message', content)
            except:
                error_msg = content
            
            raise aiohttp.ClientError(f"API error {response.status}: {error_msg}")
    
    async def predict_structure(
        self,
        sequence: str,
        options: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """Predict protein structure"""
        
        data = {
            'sequence': sequence,
            'options': options or {}
        }
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key('predict', data)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Make request
        response = await self._make_request('POST', 'predict', data)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, response)
        
        return response
    
    async def batch_predict(
        self,
        sequences: List[Dict],
        batch_size: int = 50,
        monitor_progress: bool = True
    ) -> List[Dict]:
        """Batch prediction with progress monitoring"""
        
        # Submit batch
        batch_data = {
            'sequences': sequences,
            'options': {'batch_size': batch_size}
        }
        
        batch_response = await self._make_request('POST', 'predict/batch', batch_data)
        batch_id = batch_response['batch_id']
        
        logger.info(f"Batch {batch_id} submitted with {len(sequences)} sequences")
        
        # Monitor progress
        if monitor_progress:
            while True:
                status_response = await self._make_request('GET', f'predict/batch/{batch_id}')
                
                status = status_response['status']
                progress = status_response['progress']
                
                logger.info(f"Batch {batch_id}: {progress['percentage']:.1f}% complete")
                
                if status == 'completed':
                    logger.info(f"Batch {batch_id} completed successfully")
                    break
                elif status == 'failed':
                    logger.error(f"Batch {batch_id} failed")
                    raise RuntimeError("Batch processing failed")
                
                await asyncio.sleep(30)  # Check every 30 seconds
        
        # Collect results
        results = []
        for seq_result in status_response['sequences']:
            if seq_result['status'] == 'completed':
                # Get individual result
                result_response = await self._make_request(
                    'GET', 
                    f"results/{seq_result['prediction_id']}"
                )
                results.append({
                    'id': seq_result['id'],
                    'prediction': result_response,
                    'status': 'success'
                })
            else:
                results.append({
                    'id': seq_result['id'],
                    'error': seq_result.get('error', 'Unknown error'),
                    'status': 'failed'
                })
        
        return results
    
    async def analyze_domains(
        self,
        sequence: str,
        structure_id: Optional[str] = None,
        options: Optional[Dict] = None
    ) -> Dict:
        """Analyze protein domains"""
        
        data = {
            'sequence': sequence,
            'options': options or {}
        }
        
        if structure_id:
            data['structure_id'] = structure_id
        
        return await self._make_request('POST', 'analyze/domains', data)
    
    async def compare_structures(
        self,
        structure1_id: str,
        structure2_id: str,
        alignment_method: str = 'tm_align'
    ) -> Dict:
        """Compare two protein structures"""
        
        data = {
            'structures': [
                {'id': 'struct_1', 'source': 'prediction_id', 'value': structure1_id},
                {'id': 'struct_2', 'source': 'prediction_id', 'value': structure2_id}
            ],
            'analysis_type': 'structural_alignment',
            'options': {'alignment_method': alignment_method}
        }
        
        return await self._make_request('POST', 'analyze/compare', data)

# Usage example
async def main():
    async with AdvancedProteinSSLClient('YOUR_API_KEY') as client:
        # Single prediction
        result = await client.predict_structure(
            'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV'
        )
        print(f"Prediction confidence: {result['metrics']['confidence']:.3f}")
        
        # Batch prediction
        sequences = [
            {'id': 'seq1', 'sequence': 'MKFL...'},
            {'id': 'seq2', 'sequence': 'ACDE...'}
        ]
        
        batch_results = await client.batch_predict(sequences)
        print(f"Batch completed: {len(batch_results)} results")

# Run async main
if __name__ == "__main__":
    asyncio.run(main())
```

#### Pipeline Development Framework

```python
# src/pipelines/workflows.py
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

class PipelineTask(ABC):
    """Abstract base class for pipeline tasks"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.dependencies = []
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the task"""
        pass
    
    def add_dependency(self, task: 'PipelineTask'):
        """Add task dependency"""
        self.dependencies.append(task)

class StructurePredictionTask(PipelineTask):
    """Task for protein structure prediction"""
    
    def __init__(self, task_id: str, sequence: str, options: Optional[Dict] = None):
        super().__init__(task_id)
        self.sequence = sequence
        self.options = options or {}
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute structure prediction"""
        start_time = time.time()
        
        try:
            client = context['api_client']
            result = await client.predict_structure(
                sequence=self.sequence,
                options=self.options
            )
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {self.task_id} failed: {e}")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

class DomainAnalysisTask(PipelineTask):
    """Task for domain analysis"""
    
    def __init__(self, task_id: str, prediction_task_id: str, options: Optional[Dict] = None):
        super().__init__(task_id)
        self.prediction_task_id = prediction_task_id
        self.options = options or {}
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute domain analysis"""
        start_time = time.time()
        
        try:
            # Get prediction result from context
            prediction_result = context['task_results'][self.prediction_task_id]
            if prediction_result.status != TaskStatus.COMPLETED:
                raise ValueError(f"Dependency task {self.prediction_task_id} not completed")
            
            structure_id = prediction_result.result['prediction_id']
            sequence = prediction_result.result['sequence']
            
            client = context['api_client']
            result = await client.analyze_domains(
                sequence=sequence,
                structure_id=structure_id,
                options=self.options
            )
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {self.task_id} failed: {e}")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

class ComparisonTask(PipelineTask):
    """Task for structural comparison"""
    
    def __init__(self, task_id: str, prediction_task_ids: List[str], method: str = 'tm_align'):
        super().__init__(task_id)
        self.prediction_task_ids = prediction_task_ids
        self.method = method
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute structural comparison"""
        start_time = time.time()
        
        try:
            # Get prediction results
            structure_ids = []
            for pred_task_id in self.prediction_task_ids:
                pred_result = context['task_results'][pred_task_id]
                if pred_result.status != TaskStatus.COMPLETED:
                    raise ValueError(f"Dependency task {pred_task_id} not completed")
                structure_ids.append(pred_result.result['prediction_id'])
            
            if len(structure_ids) != 2:
                raise ValueError("Comparison requires exactly 2 structures")
            
            client = context['api_client']
            result = await client.compare_structures(
                structure1_id=structure_ids[0],
                structure2_id=structure_ids[1],
                alignment_method=self.method
            )
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {self.task_id} failed: {e}")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

class Pipeline:
    """Pipeline execution engine"""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks = {}
        self.execution_order = []
        
    def add_task(self, task: PipelineTask):
        """Add task to pipeline"""
        self.tasks[task.task_id] = task
        self._update_execution_order()
    
    def _update_execution_order(self):
        """Update task execution order based on dependencies"""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(task_id):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving task {task_id}")
            if task_id in visited:
                return
                
            temp_visited.add(task_id)
            
            task = self.tasks[task_id]
            for dep_task in task.dependencies:
                visit(dep_task.task_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            order.append(task_id)
        
        for task_id in self.tasks:
            if task_id not in visited:
                visit(task_id)
        
        self.execution_order = order
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, TaskResult]:
        """Execute pipeline"""
        logger.info(f"Starting pipeline execution: {self.name}")
        
        # Initialize context
        context['task_results'] = {}
        results = {}
        
        # Execute tasks in order
        for task_id in self.execution_order:
            task = self.tasks[task_id]
            logger.info(f"Executing task: {task_id}")
            
            # Check dependencies
            for dep_task in task.dependencies:
                dep_result = context['task_results'].get(dep_task.task_id)
                if not dep_result or dep_result.status != TaskStatus.COMPLETED:
                    error_msg = f"Dependency {dep_task.task_id} not completed"
                    logger.error(error_msg)
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=error_msg
                    )
                    continue
            
            # Execute task
            result = await task.execute(context)
            results[task_id] = result
            context['task_results'][task_id] = result
            
            logger.info(f"Task {task_id} completed with status: {result.status}")
            
            if result.status == TaskStatus.FAILED:
                logger.error(f"Task {task_id} failed: {result.error}")
                # Optionally stop pipeline on failure
                # break
        
        logger.info(f"Pipeline execution completed: {self.name}")
        return results

# Usage example
async def create_comparative_analysis_pipeline():
    """Create pipeline for comparative protein analysis"""
    
    pipeline = Pipeline("comparative_analysis")
    
    # Define sequences
    sequences = {
        'wild_type': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
        'mutant': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKA'  # V->A mutation
    }
    
    # Create prediction tasks
    for name, sequence in sequences.items():
        pred_task = StructurePredictionTask(
            task_id=f"predict_{name}",
            sequence=sequence,
            options={'return_confidence': True}
        )
        pipeline.add_task(pred_task)
        
        # Add domain analysis task
        domain_task = DomainAnalysisTask(
            task_id=f"domains_{name}",
            prediction_task_id=f"predict_{name}",
            options={'include_functional_annotation': True}
        )
        domain_task.add_dependency(pred_task)
        pipeline.add_task(domain_task)
    
    # Add comparison task
    comp_task = ComparisonTask(
        task_id="compare_structures",
        prediction_task_ids=["predict_wild_type", "predict_mutant"]
    )
    comp_task.add_dependency(pipeline.tasks["predict_wild_type"])
    comp_task.add_dependency(pipeline.tasks["predict_mutant"])
    pipeline.add_task(comp_task)
    
    return pipeline

# Execute pipeline
async def run_pipeline():
    pipeline = await create_comparative_analysis_pipeline()
    
    # Setup context
    async with AdvancedProteinSSLClient('YOUR_API_KEY') as client:
        context = {'api_client': client}
        
        # Execute pipeline
        results = await pipeline.execute(context)
        
        # Process results
        for task_id, result in results.items():
            if result.status == TaskStatus.COMPLETED:
                print(f"✅ {task_id}: Success ({result.execution_time:.2f}s)")
            else:
                print(f"❌ {task_id}: Failed - {result.error}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
```

### Testing and Quality Assurance

#### Comprehensive Test Suite

```python
# tests/unit/test_api_client.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.utils.api_client import AdvancedProteinSSLClient

@pytest.fixture
def mock_client():
    """Mock API client for testing"""
    return AdvancedProteinSSLClient(
        api_key="test_key",
        cache_dir="./test_cache",
        enable_caching=False
    )

@pytest.mark.asyncio
async def test_structure_prediction_success(mock_client):
    """Test successful structure prediction"""
    
    mock_response = {
        'prediction_id': 'test_pred_123',
        'metrics': {'confidence': 0.95, 'plddt_score': 85.2},
        'sequence': 'MKFLKFSLLTAV'
    }
    
    with patch.object(mock_client, '_make_request', return_value=mock_response):
        async with mock_client:
            result = await mock_client.predict_structure('MKFLKFSLLTAV')
        
        assert result['prediction_id'] == 'test_pred_123'
        assert result['metrics']['confidence'] == 0.95

@pytest.mark.asyncio
async def test_rate_limiting(mock_client):
    """Test rate limiting functionality"""
    
    # Set very low rate limit for testing
    mock_client.rate_limit_per_minute = 2
    mock_client.min_interval = 30.0  # 30 seconds between requests
    
    with patch.object(mock_client, '_make_request', return_value={}):
        async with mock_client:
            # Make first request
            start_time = time.time()
            await mock_client.predict_structure('TEST')
            
            # Make second request - should be rate limited
            await mock_client.predict_structure('TEST')
            elapsed = time.time() - start_time
            
            # Should have waited for rate limit
            assert elapsed >= 30.0

@pytest.mark.asyncio
async def test_caching_functionality():
    """Test caching mechanism"""
    
    client = AdvancedProteinSSLClient(
        api_key="test_key",
        cache_dir="./test_cache",
        enable_caching=True
    )
    
    mock_response = {'prediction_id': 'cached_pred_123'}
    
    with patch.object(client, '_make_request', return_value=mock_response) as mock_request:
        async with client:
            # First call should hit API
            result1 = await client.predict_structure('TESTSEQ')
            
            # Second call should use cache
            result2 = await client.predict_structure('TESTSEQ')
        
        # API should only be called once
        assert mock_request.call_count == 1
        assert result1 == result2

# tests/integration/test_pipelines.py
import pytest
from src.pipelines.workflows import Pipeline, StructurePredictionTask, DomainAnalysisTask

@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_pipeline():
    """Test simple prediction pipeline"""
    
    pipeline = Pipeline("test_pipeline")
    
    # Create prediction task
    pred_task = StructurePredictionTask(
        task_id="predict_test",
        sequence="MKFLKFSLLTAV",
        options={'return_confidence': True}
    )
    pipeline.add_task(pred_task)
    
    # Create domain analysis task
    domain_task = DomainAnalysisTask(
        task_id="domains_test",
        prediction_task_id="predict_test"
    )
    domain_task.add_dependency(pred_task)
    pipeline.add_task(domain_task)
    
    # Mock API client
    mock_client = AsyncMock()
    mock_client.predict_structure.return_value = {
        'prediction_id': 'test_123',
        'sequence': 'MKFLKFSLLTAV',
        'metrics': {'confidence': 0.9}
    }
    mock_client.analyze_domains.return_value = {
        'domains': [{'start': 1, 'end': 12, 'type': 'globular'}]
    }
    
    # Execute pipeline
    context = {'api_client': mock_client}
    results = await pipeline.execute(context)
    
    # Verify results
    assert len(results) == 2
    assert results['predict_test'].status.value == 'completed'
    assert results['domains_test'].status.value == 'completed'

# Performance tests
@pytest.mark.performance
def test_batch_processing_performance():
    """Test batch processing performance"""
    import time
    
    sequences = ['MKFL' * 50] * 100  # 100 sequences of 200 residues each
    
    start_time = time.time()
    # Run batch processing
    # ... (implement actual test)
    elapsed = time.time() - start_time
    
    # Should process 100 sequences in reasonable time
    assert elapsed < 300  # 5 minutes max
```

#### Performance Monitoring

```python
# src/utils/monitoring.py
import time
import psutil
import logging
from functools import wraps
from typing import Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor API performance and resource usage"""
    
    def __init__(self):
        self.metrics = {
            'api_calls': 0,
            'total_time': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def track_api_call(self, func):
        """Decorator to track API call performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = await func(*args, **kwargs)
                self.metrics['api_calls'] += 1
                return result
                
            except Exception as e:
                self.metrics['errors'] += 1
                logger.error(f"API call failed: {e}")
                raise
                
            finally:
                elapsed = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                self.metrics['total_time'] += elapsed
                
                logger.info(f"API call completed in {elapsed:.2f}s, "
                           f"memory delta: {end_memory - start_memory:.1f}MB")
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        avg_time = (
            self.metrics['total_time'] / self.metrics['api_calls'] 
            if self.metrics['api_calls'] > 0 else 0
        )
        
        error_rate = (
            self.metrics['errors'] / (self.metrics['api_calls'] + self.metrics['errors'])
            if (self.metrics['api_calls'] + self.metrics['errors']) > 0 else 0
        )
        
        cache_hit_rate = (
            self.metrics['cache_hits'] / 
            (self.metrics['cache_hits'] + self.metrics['cache_misses'])
            if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        )
        
        return {
            'total_api_calls': self.metrics['api_calls'],
            'total_errors': self.metrics['errors'],
            'average_response_time': avg_time,
            'error_rate': error_rate,
            'cache_hit_rate': cache_hit_rate,
            'total_time': self.metrics['total_time'],
            'timestamp': datetime.now().isoformat()
        }
    
    def save_report(self, filename: str = None):
        """Save performance report to file"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.get_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {filename}")

# Usage
monitor = PerformanceMonitor()

@monitor.track_api_call
async def monitored_predict(client, sequence):
    return await client.predict_structure(sequence)
```

---

This comprehensive user guide provides detailed documentation for different personas, enabling each user type to effectively utilize the protein-sssl-operator according to their specific needs and expertise levels.