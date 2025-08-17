# Complete API Reference & User Guide

## Overview

This comprehensive guide provides complete API documentation and user guides for the protein-sssl-operator, covering different user personas including researchers, developers, and operations teams. The guide includes detailed API references, tutorials, integration examples, and best practices.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Core API Endpoints](#core-api-endpoints)
4. [User Guides by Persona](#user-guides-by-persona)
5. [Integration Examples](#integration-examples)
6. [SDK Documentation](#sdk-documentation)
7. [Tutorial Sequences](#tutorial-sequences)
8. [Best Practices](#best-practices)
9. [Error Handling](#error-handling)
10. [Rate Limiting & Quotas](#rate-limiting--quotas)

## API Overview

### Base URL and Versioning

```
Production: https://api.protein-sssl.terragonlabs.ai/v1
Staging: https://staging-api.protein-sssl.terragonlabs.ai/v1
```

### Supported Formats
- **Request**: JSON, multipart/form-data (for file uploads)
- **Response**: JSON, PDB files, CSV
- **Authentication**: Bearer tokens, API keys

### API Design Principles
- RESTful architecture
- Consistent naming conventions
- Comprehensive error responses
- Pagination for large datasets
- Rate limiting and throttling
- Comprehensive logging and monitoring

## Authentication & Authorization

### API Key Authentication

#### Obtaining an API Key
```bash
# Register for an API key
curl -X POST https://api.protein-sssl.terragonlabs.ai/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "researcher@university.edu",
    "organization": "University Research Lab",
    "use_case": "protein structure prediction research",
    "expected_usage": "academic_research"
  }'
```

#### Using API Keys
```bash
# Include API key in headers
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.protein-sssl.terragonlabs.ai/v1/predict
```

### OAuth 2.0 Integration

#### Authorization Flow
```bash
# Step 1: Redirect user to authorization URL
https://api.protein-sssl.terragonlabs.ai/v1/oauth/authorize?
  client_id=YOUR_CLIENT_ID&
  response_type=code&
  scope=predict,analyze&
  redirect_uri=YOUR_REDIRECT_URI

# Step 2: Exchange code for access token
curl -X POST https://api.protein-sssl.terragonlabs.ai/v1/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code&code=AUTH_CODE&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET"
```

### Permission Scopes

| Scope | Description | Access Level |
|-------|-------------|--------------|
| `predict` | Basic structure prediction | Standard |
| `analyze` | Advanced analysis features | Standard |
| `batch` | Batch processing capabilities | Premium |
| `admin` | Administrative functions | Admin |
| `training` | Model training capabilities | Enterprise |

## Core API Endpoints

### Health and Status

#### Health Check
```http
GET /v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "api": "healthy",
    "database": "healthy",
    "model_service": "healthy",
    "cache": "healthy"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### System Status
```http
GET /v1/status
```

**Response:**
```json
{
  "system_load": {
    "cpu_usage": 65.2,
    "memory_usage": 78.5,
    "gpu_usage": 82.1,
    "queue_depth": 15
  },
  "rate_limits": {
    "requests_per_hour": 1000,
    "remaining": 847,
    "reset_time": "2024-01-15T11:00:00Z"
  },
  "model_info": {
    "version": "2.1.0",
    "accuracy": 0.924,
    "last_updated": "2024-01-10T09:00:00Z"
  }
}
```

### Structure Prediction

#### Single Sequence Prediction
```http
POST /v1/predict
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request:**
```json
{
  "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
  "options": {
    "return_confidence": true,
    "return_uncertainty": true,
    "num_recycles": 3,
    "model_version": "latest",
    "output_format": "pdb"
  },
  "metadata": {
    "protein_name": "Example Protein",
    "organism": "Homo sapiens",
    "reference": "DOI:10.1000/example"
  }
}
```

**Response:**
```json
{
  "prediction_id": "pred_1234567890",
  "status": "completed",
  "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
  "structure": {
    "coordinates": "base64_encoded_coordinates",
    "pdb_content": "ATOM      1  N   MET A   1      20.154  16.967   4.339...",
    "confidence_scores": [0.95, 0.92, 0.88, ...],
    "uncertainty_estimates": [0.05, 0.08, 0.12, ...]
  },
  "metrics": {
    "plddt_score": 0.924,
    "predicted_tm_score": 0.887,
    "confidence": 0.916,
    "processing_time": 2.45
  },
  "analysis": {
    "secondary_structure": "HHHHHHHHLLLLLLHHHHHHHHHHLLLLLL...",
    "domain_boundaries": [{"start": 1, "end": 120, "type": "globular"}],
    "binding_sites": [{"residues": [45, 46, 47], "type": "active_site"}]
  },
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-22T10:30:00Z"
}
```

#### Batch Prediction
```http
POST /v1/predict/batch
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request:**
```json
{
  "sequences": [
    {
      "id": "protein_1",
      "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
      "metadata": {"name": "Protein 1"}
    },
    {
      "id": "protein_2", 
      "sequence": "ACDEFGHIKLMNPQRSTVWY",
      "metadata": {"name": "Protein 2"}
    }
  ],
  "options": {
    "batch_size": 10,
    "priority": "normal",
    "notification_webhook": "https://your-server.com/webhook",
    "output_format": "json"
  }
}
```

**Response:**
```json
{
  "batch_id": "batch_1234567890",
  "status": "queued",
  "total_sequences": 2,
  "estimated_completion": "2024-01-15T10:45:00Z",
  "sequences": [
    {
      "id": "protein_1",
      "status": "queued",
      "position": 1
    },
    {
      "id": "protein_2",
      "status": "queued", 
      "position": 2
    }
  ],
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Batch Status Check
```http
GET /v1/predict/batch/{batch_id}
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "batch_id": "batch_1234567890",
  "status": "processing",
  "progress": {
    "completed": 1,
    "failed": 0,
    "pending": 1,
    "total": 2,
    "percentage": 50.0
  },
  "sequences": [
    {
      "id": "protein_1",
      "status": "completed",
      "prediction_id": "pred_1234567891",
      "download_url": "/v1/results/pred_1234567891"
    },
    {
      "id": "protein_2",
      "status": "processing",
      "estimated_completion": "2024-01-15T10:45:00Z"
    }
  ],
  "updated_at": "2024-01-15T10:35:00Z"
}
```

### Advanced Analysis

#### Structural Comparison
```http
POST /v1/analyze/compare
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request:**
```json
{
  "structures": [
    {
      "id": "structure_1",
      "source": "prediction_id",
      "value": "pred_1234567890"
    },
    {
      "id": "structure_2", 
      "source": "pdb_file",
      "value": "base64_encoded_pdb_content"
    }
  ],
  "analysis_type": "structural_alignment",
  "options": {
    "alignment_method": "tm_align",
    "include_rmsd": true,
    "include_sequence_alignment": true
  }
}
```

**Response:**
```json
{
  "comparison_id": "comp_1234567890",
  "alignment_results": {
    "tm_score": 0.845,
    "rmsd": 2.34,
    "aligned_residues": 234,
    "total_residues": 267,
    "sequence_identity": 0.78
  },
  "structural_alignment": {
    "transformation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "aligned_coordinates": "base64_encoded_coordinates"
  },
  "visualization_url": "/v1/visualize/comp_1234567890"
}
```

#### Domain Analysis
```http
POST /v1/analyze/domains
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request:**
```json
{
  "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
  "structure_id": "pred_1234567890",
  "options": {
    "min_domain_length": 40,
    "use_evolutionary_info": true,
    "include_functional_annotation": true
  }
}
```

**Response:**
```json
{
  "analysis_id": "domain_1234567890",
  "domains": [
    {
      "id": "domain_1",
      "start": 1,
      "end": 120,
      "type": "globular",
      "confidence": 0.92,
      "functional_annotation": {
        "go_terms": ["GO:0003824", "GO:0008152"],
        "pfam_domains": ["PF00001"],
        "predicted_function": "enzyme activity"
      },
      "structural_features": {
        "secondary_structure_content": {
          "helix": 0.45,
          "sheet": 0.30,
          "loop": 0.25
        },
        "binding_sites": [
          {
            "residues": [45, 46, 47],
            "type": "active_site",
            "confidence": 0.89
          }
        ]
      }
    }
  ],
  "overall_metrics": {
    "domain_count": 1,
    "structured_fraction": 0.89,
    "disorder_fraction": 0.11
  }
}
```

### File Management

#### Upload Structure File
```http
POST /v1/files/upload
Content-Type: multipart/form-data
Authorization: Bearer YOUR_API_KEY
```

**Request:**
```bash
curl -X POST https://api.protein-sssl.terragonlabs.ai/v1/files/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@structure.pdb" \
  -F "file_type=pdb" \
  -F "metadata={\"protein_name\":\"Example Protein\"}"
```

**Response:**
```json
{
  "file_id": "file_1234567890",
  "filename": "structure.pdb",
  "file_type": "pdb",
  "size": 15234,
  "checksum": "sha256:abcdef123456...",
  "download_url": "/v1/files/file_1234567890/download",
  "metadata": {
    "protein_name": "Example Protein",
    "residue_count": 267,
    "chain_count": 1
  },
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-22T10:30:00Z"
}
```

#### Download Results
```http
GET /v1/results/{prediction_id}
Authorization: Bearer YOUR_API_KEY
Accept: application/json
```

**Query Parameters:**
- `format`: json, pdb, mmcif, csv
- `include_confidence`: true/false
- `compression`: none, gzip

**Response Headers:**
```
Content-Type: application/json
Content-Disposition: attachment; filename="prediction_1234567890.json"
X-Prediction-ID: pred_1234567890
X-Model-Version: 2.1.0
```

## User Guides by Persona

### For Researchers

#### Getting Started as a Researcher

**1. Account Setup**
```bash
# Request academic access
curl -X POST https://api.protein-sssl.terragonlabs.ai/v1/auth/academic-access \
  -H "Content-Type: application/json" \
  -d '{
    "email": "researcher@university.edu",
    "institution": "University of Research",
    "department": "Biochemistry",
    "research_area": "protein folding",
    "orcid": "0000-0000-0000-0000"
  }'
```

**2. Simple Structure Prediction**
```python
import requests

# Basic prediction
response = requests.post(
    'https://api.protein-sssl.terragonlabs.ai/v1/predict',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={
        'sequence': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
        'options': {
            'return_confidence': True,
            'return_uncertainty': True
        }
    }
)

prediction = response.json()
print(f"Confidence: {prediction['metrics']['confidence']:.3f}")
```

**3. Analyzing Results**
```python
# Download and analyze PDB structure
import requests
from Bio.PDB import PDBParser

# Get PDB content
pdb_response = requests.get(
    f"https://api.protein-sssl.terragonlabs.ai/v1/results/{prediction_id}",
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    params={'format': 'pdb'}
)

# Parse structure
with open('predicted_structure.pdb', 'w') as f:
    f.write(pdb_response.text)

parser = PDBParser()
structure = parser.get_structure('protein', 'predicted_structure.pdb')

# Calculate basic metrics
residue_count = len(list(structure.get_residues()))
print(f"Predicted structure has {residue_count} residues")
```

#### Research Workflow Examples

**Comparative Study Workflow**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Compare multiple related sequences
sequences = {
    'wild_type': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
    'mutant_1': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKA',  # V->A mutation
    'mutant_2': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKF'   # V->F mutation
}

results = {}
for name, seq in sequences.items():
    response = requests.post(
        'https://api.protein-sssl.terragonlabs.ai/v1/predict',
        headers={'Authorization': 'Bearer YOUR_API_KEY'},
        json={'sequence': seq, 'options': {'return_confidence': True}}
    )
    results[name] = response.json()

# Compare confidence scores
confidence_data = {
    name: result['metrics']['confidence'] 
    for name, result in results.items()
}

df = pd.DataFrame.from_dict(confidence_data, orient='index', columns=['Confidence'])
df.plot(kind='bar', title='Prediction Confidence Comparison')
plt.ylabel('Confidence Score')
plt.show()
```

### For Developers

#### SDK Integration

**Python SDK Installation**
```bash
pip install protein-sssl-sdk
```

**Basic SDK Usage**
```python
from protein_sssl_sdk import ProteinSSLClient
import asyncio

# Initialize client
client = ProteinSSLClient(
    api_key='YOUR_API_KEY',
    base_url='https://api.protein-sssl.terragonlabs.ai/v1'
)

async def predict_structure():
    # Single prediction
    prediction = await client.predict(
        sequence='MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
        return_confidence=True
    )
    
    print(f"Prediction ID: {prediction.id}")
    print(f"Confidence: {prediction.confidence:.3f}")
    
    # Save structure
    await prediction.save_pdb('structure.pdb')
    
    # Get detailed analysis
    analysis = await prediction.analyze_domains()
    print(f"Domains found: {len(analysis.domains)}")

# Run async function
asyncio.run(predict_structure())
```

**Batch Processing with SDK**
```python
async def batch_prediction():
    sequences = [
        {'id': 'protein_1', 'sequence': 'MKFL...'},
        {'id': 'protein_2', 'sequence': 'ACDE...'},
        # ... more sequences
    ]
    
    # Submit batch
    batch = await client.predict_batch(sequences)
    print(f"Batch ID: {batch.id}")
    
    # Monitor progress
    while not batch.is_complete():
        await asyncio.sleep(30)
        await batch.refresh()
        print(f"Progress: {batch.progress.percentage:.1f}%")
    
    # Process results
    for result in batch.results:
        if result.status == 'completed':
            print(f"{result.id}: Confidence {result.confidence:.3f}")
        else:
            print(f"{result.id}: Failed - {result.error}")

asyncio.run(batch_prediction())
```

#### Error Handling Best Practices

```python
from protein_sssl_sdk import ProteinSSLClient, ProteinSSLError
import logging

logger = logging.getLogger(__name__)

async def robust_prediction(sequence):
    client = ProteinSSLClient(api_key='YOUR_API_KEY')
    
    try:
        # Validate sequence before submission
        if not client.validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
        
        # Make prediction with retry logic
        prediction = await client.predict(
            sequence=sequence,
            return_confidence=True,
            timeout=300,  # 5 minute timeout
            retry_count=3
        )
        
        # Check prediction quality
        if prediction.confidence < 0.7:
            logger.warning(f"Low confidence prediction: {prediction.confidence:.3f}")
        
        return prediction
        
    except ProteinSSLError.RateLimitError as e:
        logger.error(f"Rate limit exceeded. Retry after: {e.retry_after}")
        await asyncio.sleep(e.retry_after)
        return await robust_prediction(sequence)  # Retry
        
    except ProteinSSLError.ValidationError as e:
        logger.error(f"Validation error: {e.message}")
        raise
        
    except ProteinSSLError.ServerError as e:
        logger.error(f"Server error: {e.message}")
        # Implement exponential backoff
        await asyncio.sleep(min(2 ** attempt, 60))
        if attempt < 3:
            return await robust_prediction(sequence)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### For Operations Teams

#### Monitoring Integration

**Prometheus Metrics Collection**
```python
from prometheus_client import Counter, Histogram, Gauge
import requests
import time

# Define metrics
api_requests = Counter('protein_sssl_api_requests_total', 'Total API requests', ['endpoint', 'status'])
api_duration = Histogram('protein_sssl_api_request_duration_seconds', 'API request duration')
prediction_queue = Gauge('protein_sssl_prediction_queue_depth', 'Current prediction queue depth')

def monitor_api_health():
    """Monitor API health and collect metrics"""
    while True:
        try:
            # Health check
            start_time = time.time()
            response = requests.get(
                'https://api.protein-sssl.terragonlabs.ai/v1/health',
                timeout=10
            )
            duration = time.time() - start_time
            
            # Record metrics
            api_requests.labels(endpoint='health', status=response.status_code).inc()
            api_duration.observe(duration)
            
            if response.status_code == 200:
                health_data = response.json()
                if 'system_load' in health_data:
                    prediction_queue.set(health_data['system_load'].get('queue_depth', 0))
            
        except requests.RequestException as e:
            api_requests.labels(endpoint='health', status='error').inc()
            logger.error(f"Health check failed: {e}")
        
        time.sleep(30)  # Check every 30 seconds
```

**Log Analysis for Operations**
```python
import json
import re
from datetime import datetime, timedelta

def analyze_api_logs(log_file_path):
    """Analyze API logs for operational insights"""
    errors = []
    slow_requests = []
    popular_endpoints = {}
    
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                
                # Track errors
                if log_entry.get('level') == 'ERROR':
                    errors.append(log_entry)
                
                # Track slow requests
                if 'duration' in log_entry and log_entry['duration'] > 5.0:
                    slow_requests.append(log_entry)
                
                # Track endpoint popularity
                endpoint = log_entry.get('endpoint')
                if endpoint:
                    popular_endpoints[endpoint] = popular_endpoints.get(endpoint, 0) + 1
                    
            except json.JSONDecodeError:
                continue
    
    # Generate operational report
    report = {
        'error_count': len(errors),
        'slow_request_count': len(slow_requests),
        'most_popular_endpoints': sorted(
            popular_endpoints.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5],
        'analysis_time': datetime.now().isoformat()
    }
    
    return report
```

## Integration Examples

### Jupyter Notebook Integration

```python
# protein_analysis.ipynb
import requests
import pandas as pd
import matplotlib.pyplot as plt
import py3Dmol
from IPython.display import HTML

class ProteinAnalysisNotebook:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.protein-sssl.terragonlabs.ai/v1'
        
    def predict_and_visualize(self, sequence, protein_name="Unknown"):
        """Predict structure and create visualization"""
        # Make prediction
        response = requests.post(
            f'{self.base_url}/predict',
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={
                'sequence': sequence,
                'options': {'return_confidence': True}
            }
        )
        
        prediction = response.json()
        
        # Create visualization
        view = py3Dmol.view(width=800, height=600)
        view.addModel(prediction['structure']['pdb_content'], 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.zoomTo()
        
        # Display results
        print(f"Protein: {protein_name}")
        print(f"Sequence Length: {len(sequence)}")
        print(f"Confidence: {prediction['metrics']['confidence']:.3f}")
        print(f"pLDDT Score: {prediction['metrics']['plddt_score']:.3f}")
        
        return view, prediction

# Usage in notebook
analyzer = ProteinAnalysisNotebook('YOUR_API_KEY')
view, prediction = analyzer.predict_and_visualize(
    'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
    'Example Protein'
)
view.show()
```

### Bioinformatics Pipeline Integration

```python
# pipeline_integration.py
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor
import requests
import json

class ProteinPipelineIntegrator:
    def __init__(self, api_key, max_workers=5):
        self.api_key = api_key
        self.base_url = 'https://api.protein-sssl.terragonlabs.ai/v1'
        self.max_workers = max_workers
        
    def process_fasta_file(self, fasta_path, output_dir):
        """Process all sequences in a FASTA file"""
        sequences = []
        
        # Read sequences
        for record in SeqIO.parse(fasta_path, 'fasta'):
            sequences.append({
                'id': record.id,
                'description': record.description,
                'sequence': str(record.seq)
            })
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_single_sequence, seq)
                for seq in sequences
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing sequence: {e}")
        
        # Save results
        self._save_results(results, output_dir)
        return results
    
    def _process_single_sequence(self, seq_data):
        """Process a single sequence"""
        response = requests.post(
            f'{self.base_url}/predict',
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={
                'sequence': seq_data['sequence'],
                'options': {'return_confidence': True},
                'metadata': {
                    'protein_id': seq_data['id'],
                    'description': seq_data['description']
                }
            }
        )
        
        if response.status_code == 200:
            prediction = response.json()
            return {
                'protein_id': seq_data['id'],
                'prediction_id': prediction['prediction_id'],
                'confidence': prediction['metrics']['confidence'],
                'plddt_score': prediction['metrics']['plddt_score']
            }
        else:
            return {
                'protein_id': seq_data['id'],
                'error': response.text
            }
    
    def _save_results(self, results, output_dir):
        """Save processing results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        with open(f'{output_dir}/summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV for analysis
        df = pd.DataFrame(results)
        df.to_csv(f'{output_dir}/results.csv', index=False)

# Usage
integrator = ProteinPipelineIntegrator('YOUR_API_KEY')
results = integrator.process_fasta_file('proteins.fasta', 'output/')
```

### Web Application Integration

```javascript
// frontend_integration.js
class ProteinSSLClient {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = 'https://api.protein-sssl.terragonlabs.ai/v1';
    }
    
    async predictStructure(sequence, options = {}) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sequence: sequence,
                options: {
                    return_confidence: true,
                    ...options
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async pollPredictionStatus(predictionId) {
        const response = await fetch(`${this.baseUrl}/results/${predictionId}`, {
            headers: {
                'Authorization': `Bearer ${this.apiKey}`
            }
        });
        
        return await response.json();
    }
}

// React component example
function ProteinPredictionComponent() {
    const [sequence, setSequence] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const client = new ProteinSSLClient(process.env.REACT_APP_API_KEY);
    
    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        
        try {
            const result = await client.predictStructure(sequence);
            setPrediction(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="protein-prediction">
            <textarea
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="Enter protein sequence..."
                rows={5}
                cols={80}
            />
            <br />
            <button onClick={handlePredict} disabled={loading || !sequence}>
                {loading ? 'Predicting...' : 'Predict Structure'}
            </button>
            
            {error && <div className="error">Error: {error}</div>}
            
            {prediction && (
                <div className="prediction-results">
                    <h3>Prediction Results</h3>
                    <p>Confidence: {prediction.metrics.confidence.toFixed(3)}</p>
                    <p>pLDDT Score: {prediction.metrics.plddt_score.toFixed(3)}</p>
                    {/* Add 3D visualization component here */}
                </div>
            )}
        </div>
    );
}
```

## SDK Documentation

### Python SDK

#### Installation
```bash
pip install protein-sssl-sdk
```

#### Quick Start
```python
from protein_sssl_sdk import ProteinSSLClient

# Initialize client
client = ProteinSSLClient(api_key='YOUR_API_KEY')

# Simple prediction
prediction = client.predict('MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV')
print(f"Confidence: {prediction.confidence}")

# Save results
prediction.save_pdb('structure.pdb')
```

#### Advanced Usage
```python
import asyncio
from protein_sssl_sdk import AsyncProteinSSLClient

async def advanced_analysis():
    client = AsyncProteinSSLClient(api_key='YOUR_API_KEY')
    
    # Batch prediction
    sequences = ['MKFL...', 'ACDE...', 'EFGH...']
    batch = await client.predict_batch(sequences)
    
    # Monitor progress
    while not batch.is_complete():
        await asyncio.sleep(10)
        await batch.refresh()
        print(f"Progress: {batch.progress}%")
    
    # Analyze results
    for result in batch.results:
        if result.success:
            domains = await result.analyze_domains()
            print(f"Protein {result.id}: {len(domains)} domains")

asyncio.run(advanced_analysis())
```

### JavaScript SDK

#### Installation
```bash
npm install protein-sssl-sdk
```

#### Usage
```javascript
import { ProteinSSLClient } from 'protein-sssl-sdk';

const client = new ProteinSSLClient({
    apiKey: 'YOUR_API_KEY',
    baseUrl: 'https://api.protein-sssl.terragonlabs.ai/v1'
});

// Promise-based API
client.predict('MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV')
    .then(prediction => {
        console.log(`Confidence: ${prediction.confidence}`);
        return prediction.downloadPDB();
    })
    .then(pdbContent => {
        console.log('PDB downloaded successfully');
    })
    .catch(error => {
        console.error('Prediction failed:', error);
    });

// Async/await API
async function predictStructure() {
    try {
        const prediction = await client.predict('MKFL...');
        const analysis = await prediction.analyzeDomains();
        return { prediction, analysis };
    } catch (error) {
        console.error('Error:', error);
    }
}
```

### R SDK

#### Installation
```r
install.packages("devtools")
devtools::install_github("terragonlabs/protein-sssl-r-sdk")
```

#### Usage
```r
library(proteinSSL)

# Initialize client
client <- ProteinSSLClient$new(api_key = "YOUR_API_KEY")

# Make prediction
sequence <- "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
prediction <- client$predict(sequence)

# View results
cat("Confidence:", prediction$confidence, "\n")
cat("pLDDT Score:", prediction$plddt_score, "\n")

# Save structure
prediction$save_pdb("structure.pdb")

# Batch processing
sequences <- c("MKFL...", "ACDE...", "EFGH...")
batch_results <- client$predict_batch(sequences)

# Create data frame for analysis
results_df <- data.frame(
  sequence_id = batch_results$ids,
  confidence = batch_results$confidences,
  plddt_score = batch_results$plddt_scores
)

# Visualize results
library(ggplot2)
ggplot(results_df, aes(x = confidence, y = plddt_score)) +
  geom_point() +
  labs(title = "Prediction Quality Analysis",
       x = "Confidence Score",
       y = "pLDDT Score")
```

## Tutorial Sequences

### Tutorial 1: Getting Started with Basic Predictions

**Objective**: Learn to make your first protein structure prediction

**Prerequisites**: API key, basic understanding of protein sequences

**Steps**:

1. **Setup and Authentication**
```bash
# Test your API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.protein-sssl.terragonlabs.ai/v1/health
```

2. **Your First Prediction**
```python
import requests

sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"

response = requests.post(
    'https://api.protein-sssl.terragonlabs.ai/v1/predict',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={'sequence': sequence}
)

prediction = response.json()
print(f"Prediction ID: {prediction['prediction_id']}")
print(f"Confidence: {prediction['metrics']['confidence']:.3f}")
```

3. **Understanding Results**
```python
# Analyze the prediction
print("\n=== Prediction Analysis ===")
print(f"Sequence length: {len(sequence)} residues")
print(f"Confidence score: {prediction['metrics']['confidence']:.3f}")
print(f"pLDDT score: {prediction['metrics']['plddt_score']:.3f}")
print(f"Processing time: {prediction['metrics']['processing_time']:.2f} seconds")

# Check confidence distribution
confidence_scores = prediction['structure']['confidence_scores']
avg_confidence = sum(confidence_scores) / len(confidence_scores)
print(f"Average per-residue confidence: {avg_confidence:.3f}")
```

4. **Saving Results**
```python
# Download PDB file
pdb_response = requests.get(
    f"https://api.protein-sssl.terragonlabs.ai/v1/results/{prediction['prediction_id']}",
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    params={'format': 'pdb'}
)

with open('my_first_prediction.pdb', 'w') as f:
    f.write(pdb_response.text)

print("Structure saved as 'my_first_prediction.pdb'")
```

### Tutorial 2: Batch Processing for High-Throughput Analysis

**Objective**: Process multiple protein sequences efficiently

**Prerequisites**: Completed Tutorial 1, list of protein sequences

**Steps**:

1. **Prepare Sequence Data**
```python
# Example sequences (truncated for brevity)
sequences = [
    {'id': 'protein_1', 'sequence': 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV'},
    {'id': 'protein_2', 'sequence': 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGH'},
    {'id': 'protein_3', 'sequence': 'QWERTYUIOPASDFGHJKLZXCVBNMQWERTYUIOPASDFGHJKLZX'},
    # ... more sequences
]

print(f"Preparing to process {len(sequences)} sequences")
```

2. **Submit Batch Job**
```python
batch_response = requests.post(
    'https://api.protein-sssl.terragonlabs.ai/v1/predict/batch',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={
        'sequences': sequences,
        'options': {
            'batch_size': 10,
            'priority': 'normal',
            'output_format': 'json'
        }
    }
)

batch_info = batch_response.json()
batch_id = batch_info['batch_id']
print(f"Batch submitted: {batch_id}")
print(f"Estimated completion: {batch_info['estimated_completion']}")
```

3. **Monitor Progress**
```python
import time

def check_batch_status(batch_id):
    response = requests.get(
        f'https://api.protein-sssl.terragonlabs.ai/v1/predict/batch/{batch_id}',
        headers={'Authorization': 'Bearer YOUR_API_KEY'}
    )
    return response.json()

# Poll for completion
while True:
    status = check_batch_status(batch_id)
    progress = status['progress']
    
    print(f"Progress: {progress['completed']}/{progress['total']} "
          f"({progress['percentage']:.1f}%)")
    
    if status['status'] == 'completed':
        print("Batch processing completed!")
        break
    elif status['status'] == 'failed':
        print("Batch processing failed!")
        break
    
    time.sleep(30)  # Check every 30 seconds
```

4. **Process Results**
```python
import pandas as pd

# Collect all results
results = []
final_status = check_batch_status(batch_id)

for seq_result in final_status['sequences']:
    if seq_result['status'] == 'completed':
        # Download individual result
        pred_response = requests.get(
            f"https://api.protein-sssl.terragonlabs.ai/v1/results/{seq_result['prediction_id']}",
            headers={'Authorization': 'Bearer YOUR_API_KEY'}
        )
        
        pred_data = pred_response.json()
        results.append({
            'protein_id': seq_result['id'],
            'confidence': pred_data['metrics']['confidence'],
            'plddt_score': pred_data['metrics']['plddt_score'],
            'processing_time': pred_data['metrics']['processing_time']
        })

# Create summary DataFrame
df = pd.DataFrame(results)
print("\n=== Batch Results Summary ===")
print(df.describe())

# Save results
df.to_csv('batch_results.csv', index=False)
print("Results saved to 'batch_results.csv'")
```

### Tutorial 3: Advanced Analysis and Comparison

**Objective**: Perform advanced structural analysis and comparisons

**Prerequisites**: Completed Tutorials 1-2, understanding of structural biology

**Steps**:

1. **Domain Analysis**
```python
# Analyze domains in your prediction
prediction_id = "pred_1234567890"  # From previous tutorial

domain_response = requests.post(
    'https://api.protein-sssl.terragonlabs.ai/v1/analyze/domains',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={
        'structure_id': prediction_id,
        'options': {
            'min_domain_length': 40,
            'include_functional_annotation': True
        }
    }
)

domain_analysis = domain_response.json()

print("=== Domain Analysis ===")
for i, domain in enumerate(domain_analysis['domains'], 1):
    print(f"Domain {i}: residues {domain['start']}-{domain['end']}")
    print(f"  Type: {domain['type']}")
    print(f"  Confidence: {domain['confidence']:.3f}")
    
    if 'functional_annotation' in domain:
        print(f"  Function: {domain['functional_annotation']['predicted_function']}")
```

2. **Structural Comparison**
```python
# Compare two structures
structure_1_id = "pred_1234567890"
structure_2_id = "pred_0987654321"

comparison_response = requests.post(
    'https://api.protein-sssl.terragonlabs.ai/v1/analyze/compare',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={
        'structures': [
            {'id': 'struct_1', 'source': 'prediction_id', 'value': structure_1_id},
            {'id': 'struct_2', 'source': 'prediction_id', 'value': structure_2_id}
        ],
        'analysis_type': 'structural_alignment',
        'options': {
            'alignment_method': 'tm_align',
            'include_rmsd': True
        }
    }
)

comparison = comparison_response.json()

print("=== Structural Comparison ===")
print(f"TM-score: {comparison['alignment_results']['tm_score']:.3f}")
print(f"RMSD: {comparison['alignment_results']['rmsd']:.2f} Å")
print(f"Aligned residues: {comparison['alignment_results']['aligned_residues']}")
print(f"Sequence identity: {comparison['alignment_results']['sequence_identity']:.3f}")
```

3. **Visualization Setup**
```python
# Create interactive visualization
import py3Dmol

def visualize_comparison(structure_1_pdb, structure_2_pdb):
    """Create side-by-side structure visualization"""
    
    # Create viewer
    view = py3Dmol.view(width=1000, height=500, viewergrid=(1,2))
    
    # Add first structure
    view.addModel(structure_1_pdb, 'pdb', viewer=(0,0))
    view.setStyle({'cartoon': {'color': 'blue'}}, viewer=(0,0))
    view.setBackgroundColor('white', viewer=(0,0))
    view.zoomTo(viewer=(0,0))
    
    # Add second structure
    view.addModel(structure_2_pdb, 'pdb', viewer=(0,1))
    view.setStyle({'cartoon': {'color': 'red'}}, viewer=(0,1))
    view.setBackgroundColor('white', viewer=(0,1))
    view.zoomTo(viewer=(0,1))
    
    return view

# Get PDB content for both structures
pdb_1 = requests.get(
    f"https://api.protein-sssl.terragonlabs.ai/v1/results/{structure_1_id}",
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    params={'format': 'pdb'}
).text

pdb_2 = requests.get(
    f"https://api.protein-sssl.terragonlabs.ai/v1/results/{structure_2_id}",
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    params={'format': 'pdb'}
).text

# Create visualization
view = visualize_comparison(pdb_1, pdb_2)
view.show()
```

## Best Practices

### API Usage Best Practices

**1. Rate Limiting and Throttling**
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    """Decorator to enforce rate limiting"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(calls_per_minute=30)  # Stay under rate limit
def make_prediction(sequence):
    return requests.post(...)
```

**2. Error Handling and Retries**
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
    """Create session with automatic retries"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Usage
session = create_session_with_retries()
response = session.post(
    'https://api.protein-sssl.terragonlabs.ai/v1/predict',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={'sequence': sequence},
    timeout=30
)
```

**3. Efficient Batch Processing**
```python
def efficient_batch_processing(sequences, batch_size=50):
    """Process sequences in optimal batches"""
    
    # Group sequences by similar length for efficiency
    length_groups = {}
    for seq in sequences:
        length_bucket = (len(seq['sequence']) // 100) * 100  # Group by 100s
        if length_bucket not in length_groups:
            length_groups[length_bucket] = []
        length_groups[length_bucket].append(seq)
    
    results = []
    
    # Process each length group
    for length_bucket, group_sequences in length_groups.items():
        print(f"Processing {len(group_sequences)} sequences of ~{length_bucket} residues")
        
        # Split into batches
        for i in range(0, len(group_sequences), batch_size):
            batch = group_sequences[i:i + batch_size]
            
            # Submit batch
            batch_response = requests.post(
                'https://api.protein-sssl.terragonlabs.ai/v1/predict/batch',
                headers={'Authorization': 'Bearer YOUR_API_KEY'},
                json={'sequences': batch}
            )
            
            if batch_response.status_code == 200:
                results.append(batch_response.json()['batch_id'])
            else:
                print(f"Batch submission failed: {batch_response.text}")
    
    return results
```

### Performance Optimization

**1. Caching Results**
```python
import hashlib
import pickle
import os

class PredictionCache:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, sequence, options=None):
        """Generate cache key for sequence and options"""
        content = sequence + str(sorted((options or {}).items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, sequence, options=None):
        """Get cached prediction if available"""
        cache_key = self._get_cache_key(sequence, options)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, sequence, options, prediction):
        """Cache prediction result"""
        cache_key = self._get_cache_key(sequence, options)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(prediction, f)

# Usage
cache = PredictionCache()

def cached_predict(sequence, options=None):
    # Check cache first
    cached_result = cache.get(sequence, options)
    if cached_result:
        print("Using cached result")
        return cached_result
    
    # Make API call
    response = requests.post(
        'https://api.protein-sssl.terragonlabs.ai/v1/predict',
        headers={'Authorization': 'Bearer YOUR_API_KEY'},
        json={'sequence': sequence, 'options': options or {}}
    )
    
    prediction = response.json()
    
    # Cache result
    cache.set(sequence, options, prediction)
    
    return prediction
```

**2. Parallel Processing**
```python
import concurrent.futures
import threading

class ParallelPredictor:
    def __init__(self, api_key, max_workers=5):
        self.api_key = api_key
        self.max_workers = max_workers
        self.session = create_session_with_retries()
        
    def predict_many(self, sequences):
        """Predict multiple sequences in parallel"""
        
        def predict_single(sequence_data):
            try:
                response = self.session.post(
                    'https://api.protein-sssl.terragonlabs.ai/v1/predict',
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    json={
                        'sequence': sequence_data['sequence'],
                        'metadata': {'id': sequence_data['id']}
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    return {
                        'id': sequence_data['id'],
                        'success': True,
                        'prediction': response.json()
                    }
                else:
                    return {
                        'id': sequence_data['id'],
                        'success': False,
                        'error': response.text
                    }
                    
            except Exception as e:
                return {
                    'id': sequence_data['id'],
                    'success': False,
                    'error': str(e)
                }
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(predict_single, seq) for seq in sequences]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results

# Usage
predictor = ParallelPredictor('YOUR_API_KEY', max_workers=3)
results = predictor.predict_many([
    {'id': 'seq1', 'sequence': 'MKFL...'},
    {'id': 'seq2', 'sequence': 'ACDE...'},
    # ... more sequences
])
```

### Security Best Practices

**1. API Key Management**
```python
import os
from cryptography.fernet import Fernet

class SecureAPIKeyManager:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self):
        key_file = os.path.expanduser('~/.protein_ssl_key')
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Read-only for owner
            return key
    
    def store_api_key(self, api_key):
        encrypted_key = self.cipher.encrypt(api_key.encode())
        key_file = os.path.expanduser('~/.protein_ssl_api_key')
        with open(key_file, 'wb') as f:
            f.write(encrypted_key)
        os.chmod(key_file, 0o600)
    
    def get_api_key(self):
        key_file = os.path.expanduser('~/.protein_ssl_api_key')
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                encrypted_key = f.read()
            return self.cipher.decrypt(encrypted_key).decode()
        return None

# Usage
key_manager = SecureAPIKeyManager()
# key_manager.store_api_key('your-actual-api-key')
api_key = key_manager.get_api_key()
```

**2. Input Validation**
```python
import re

def validate_protein_sequence(sequence):
    """Validate protein sequence format and content"""
    
    # Remove whitespace
    sequence = sequence.strip().upper()
    
    # Check for valid amino acid characters
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(c in valid_chars for c in sequence):
        invalid_chars = set(sequence) - valid_chars
        raise ValueError(f"Invalid characters in sequence: {invalid_chars}")
    
    # Check length
    if len(sequence) < 10:
        raise ValueError("Sequence too short (minimum 10 residues)")
    if len(sequence) > 5000:
        raise ValueError("Sequence too long (maximum 5000 residues)")
    
    # Check for suspicious patterns
    if len(set(sequence)) < 3:
        raise ValueError("Sequence has too little diversity")
    
    return sequence

def safe_predict(sequence, **kwargs):
    """Make prediction with input validation"""
    try:
        validated_sequence = validate_protein_sequence(sequence)
        
        response = requests.post(
            'https://api.protein-sssl.terragonlabs.ai/v1/predict',
            headers={'Authorization': f'Bearer {api_key}'},
            json={'sequence': validated_sequence, **kwargs}
        )
        
        return response.json()
        
    except ValueError as e:
        return {'error': f'Validation error: {e}'}
    except Exception as e:
        return {'error': f'Prediction error: {e}'}
```

## Error Handling

### Common Error Codes and Solutions

| Status Code | Error Type | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | Bad Request | Invalid sequence or parameters | Validate input data |
| 401 | Unauthorized | Invalid or missing API key | Check API key |
| 403 | Forbidden | Insufficient permissions | Upgrade account or check scopes |
| 429 | Rate Limited | Too many requests | Implement rate limiting |
| 500 | Server Error | Internal server error | Retry with exponential backoff |
| 503 | Service Unavailable | System maintenance | Check status page |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_SEQUENCE",
    "message": "Protein sequence contains invalid characters",
    "details": {
      "invalid_characters": ["X", "B"],
      "position": 45,
      "suggestion": "Replace invalid characters with standard amino acids"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_1234567890"
  }
}
```

### Comprehensive Error Handling

```python
class ProteinSSLError(Exception):
    """Base exception for Protein SSL API errors"""
    pass

class ValidationError(ProteinSSLError):
    """Input validation error"""
    pass

class AuthenticationError(ProteinSSLError):
    """Authentication error"""
    pass

class RateLimitError(ProteinSSLError):
    """Rate limit exceeded"""
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

class ServerError(ProteinSSLError):
    """Server-side error"""
    pass

def handle_api_response(response):
    """Handle API response and raise appropriate exceptions"""
    
    if response.status_code == 200:
        return response.json()
    
    try:
        error_data = response.json().get('error', {})
    except:
        error_data = {}
    
    error_message = error_data.get('message', 'Unknown error')
    
    if response.status_code == 400:
        raise ValidationError(f"Validation error: {error_message}")
    elif response.status_code == 401:
        raise AuthenticationError(f"Authentication error: {error_message}")
    elif response.status_code == 403:
        raise AuthenticationError(f"Permission denied: {error_message}")
    elif response.status_code == 429:
        retry_after = response.headers.get('Retry-After', 60)
        raise RateLimitError(f"Rate limit exceeded: {error_message}", int(retry_after))
    elif response.status_code >= 500:
        raise ServerError(f"Server error: {error_message}")
    else:
        raise ProteinSSLError(f"API error ({response.status_code}): {error_message}")

# Usage with proper error handling
def robust_prediction(sequence):
    """Make prediction with comprehensive error handling"""
    try:
        response = requests.post(
            'https://api.protein-sssl.terragonlabs.ai/v1/predict',
            headers={'Authorization': f'Bearer {api_key}'},
            json={'sequence': sequence},
            timeout=60
        )
        
        return handle_api_response(response)
        
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        return None
        
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        # Possibly refresh API key or prompt user
        return None
        
    except RateLimitError as e:
        logger.warning(f"Rate limited: {e}")
        if e.retry_after:
            logger.info(f"Retrying after {e.retry_after} seconds")
            time.sleep(e.retry_after)
            return robust_prediction(sequence)  # Retry
        return None
        
    except ServerError as e:
        logger.error(f"Server error: {e}")
        # Implement exponential backoff
        return None
        
    except requests.RequestException as e:
        logger.error(f"Network error: {e}")
        return None
```

## Rate Limiting & Quotas

### Rate Limits by Plan

| Plan | Requests/Hour | Batch Size | Concurrent Requests |
|------|---------------|------------|-------------------|
| Free | 100 | 10 | 2 |
| Academic | 1,000 | 50 | 5 |
| Professional | 10,000 | 200 | 20 |
| Enterprise | Unlimited | Unlimited | 100 |

### Quota Management

```python
class QuotaManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.protein-sssl.terragonlabs.ai/v1'
    
    def get_quota_status(self):
        """Get current quota usage and limits"""
        response = requests.get(
            f'{self.base_url}/account/quota',
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        
        return response.json()
    
    def check_quota_before_request(self, estimated_cost=1):
        """Check if quota allows for additional requests"""
        quota = self.get_quota_status()
        
        remaining = quota['limits']['requests_per_hour'] - quota['usage']['requests_this_hour']
        
        if remaining >= estimated_cost:
            return True
        else:
            reset_time = quota['reset_time']
            print(f"Quota exceeded. Resets at {reset_time}")
            return False

# Usage
quota_manager = QuotaManager('YOUR_API_KEY')

if quota_manager.check_quota_before_request():
    prediction = make_prediction(sequence)
else:
    print("Quota exceeded, waiting for reset...")
```

---

This comprehensive API reference and user guide provides everything needed to successfully integrate and use the protein-sssl-operator API across different use cases and user personas. Regular updates ensure the documentation stays current with API evolution.