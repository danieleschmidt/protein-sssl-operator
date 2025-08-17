"""
Multi-Region Deployment System for protein-sssl-operator
Provides geographic data residency, regional data processing rules,
cross-border data transfer compliance, and local data sovereignty.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class Region(Enum):
    """Supported deployment regions"""
    # North America
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    CANADA_CENTRAL = "canada-central"
    
    # Europe
    EU_WEST_1 = "eu-west-1"  # Ireland
    EU_CENTRAL_1 = "eu-central-1"  # Frankfurt
    EU_NORTH_1 = "eu-north-1"  # Stockholm
    UK_SOUTH = "uk-south"  # London
    
    # Asia Pacific
    ASIA_EAST_1 = "asia-east-1"  # Hong Kong
    ASIA_SOUTHEAST_1 = "asia-southeast-1"  # Singapore
    ASIA_NORTHEAST_1 = "asia-northeast-1"  # Tokyo
    ASIA_SOUTH_1 = "asia-south-1"  # Mumbai
    AUSTRALIA_EAST = "australia-east"  # Sydney
    
    # South America
    SOUTH_AMERICA_EAST = "south-america-east"  # São Paulo
    
    # Africa
    AFRICA_SOUTH = "africa-south"  # Cape Town
    
    # Middle East
    MIDDLE_EAST_1 = "middle-east-1"  # Bahrain

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class DataSovereigntyLevel(Enum):
    """Data sovereignty requirements"""
    NONE = "none"  # No sovereignty requirements
    PREFER_LOCAL = "prefer_local"  # Prefer local but allow cross-border
    STRICT_LOCAL = "strict_local"  # Must stay within jurisdiction
    GOVERNMENT_CLOUD = "government_cloud"  # Government cloud only

class ComplianceRegion(Enum):
    """Compliance regions"""
    GDPR = "gdpr"  # European Union + EEA
    CCPA = "ccpa"  # California, USA
    PDPA_SG = "pdpa_sg"  # Singapore
    PDPA_TH = "pdpa_th"  # Thailand
    LGPD = "lgpd"  # Brazil
    PIPEDA = "pipeda"  # Canada
    APPI = "appi"  # Japan
    KPIPA = "kpipa"  # South Korea

@dataclass
class RegionConfig:
    """Configuration for a deployment region"""
    region: Region
    country_codes: List[str]
    jurisdiction: str
    compliance_frameworks: List[ComplianceRegion]
    data_sovereignty_level: DataSovereigntyLevel
    allowed_data_types: List[DataClassification]
    cross_border_allowed: bool
    encryption_required: bool = True
    audit_logging_required: bool = True
    backup_regions: List[Region] = field(default_factory=list)
    latency_requirements_ms: int = 100
    availability_zones: List[str] = field(default_factory=list)
    edge_locations: List[str] = field(default_factory=list)
    storage_encryption_keys: Dict[str, str] = field(default_factory=dict)
    network_restrictions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataResidencyRule:
    """Data residency rule"""
    rule_id: str
    name: str
    data_types: List[DataClassification]
    required_regions: List[Region]
    prohibited_regions: List[Region] = field(default_factory=list)
    max_cross_border_transfers: int = 0
    retention_period_days: Optional[int] = None
    pseudonymization_required: bool = False
    encryption_at_rest_required: bool = True
    encryption_in_transit_required: bool = True
    audit_trail_required: bool = True
    legal_basis: Optional[str] = None
    description: str = ""

@dataclass
class DataTransferRequest:
    """Cross-border data transfer request"""
    request_id: str
    source_region: Region
    destination_region: Region
    data_classification: DataClassification
    data_size_bytes: int
    purpose: str
    legal_basis: str
    requestor: str
    timestamp: float
    approval_status: str = "pending"  # pending, approved, denied
    approver: Optional[str] = None
    expiry_timestamp: Optional[float] = None
    safeguards: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegionalDeployment:
    """Regional deployment instance"""
    deployment_id: str
    region: Region
    namespace: str
    cluster_name: str
    data_residency_rules: List[str]  # Rule IDs
    storage_configuration: Dict[str, Any]
    network_configuration: Dict[str, Any]
    security_configuration: Dict[str, Any]
    monitoring_configuration: Dict[str, Any]
    backup_configuration: Dict[str, Any]
    status: str = "active"  # active, inactive, maintenance
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

class RegionMapper:
    """Maps jurisdictions to regions and compliance requirements"""
    
    REGION_MAPPINGS = {
        # North America
        Region.US_EAST_1: RegionConfig(
            region=Region.US_EAST_1,
            country_codes=["US"],
            jurisdiction="US",
            compliance_frameworks=[ComplianceRegion.CCPA],
            data_sovereignty_level=DataSovereigntyLevel.PREFER_LOCAL,
            allowed_data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            cross_border_allowed=True,
            availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
            edge_locations=["IAD", "DCA", "BWI"]
        ),
        Region.CANADA_CENTRAL: RegionConfig(
            region=Region.CANADA_CENTRAL,
            country_codes=["CA"],
            jurisdiction="CA",
            compliance_frameworks=[ComplianceRegion.PIPEDA],
            data_sovereignty_level=DataSovereigntyLevel.STRICT_LOCAL,
            allowed_data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            cross_border_allowed=True,
            availability_zones=["ca-central-1a", "ca-central-1b"],
            edge_locations=["YTO", "YUL"]
        ),
        
        # Europe
        Region.EU_WEST_1: RegionConfig(
            region=Region.EU_WEST_1,
            country_codes=["IE", "EU"],
            jurisdiction="EU",
            compliance_frameworks=[ComplianceRegion.GDPR],
            data_sovereignty_level=DataSovereigntyLevel.STRICT_LOCAL,
            allowed_data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            cross_border_allowed=True,
            availability_zones=["eu-west-1a", "eu-west-1b", "eu-west-1c"],
            edge_locations=["DUB", "MAN", "LHR"]
        ),
        Region.EU_CENTRAL_1: RegionConfig(
            region=Region.EU_CENTRAL_1,
            country_codes=["DE", "EU"],
            jurisdiction="EU",
            compliance_frameworks=[ComplianceRegion.GDPR],
            data_sovereignty_level=DataSovereigntyLevel.STRICT_LOCAL,
            allowed_data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            cross_border_allowed=True,
            availability_zones=["eu-central-1a", "eu-central-1b", "eu-central-1c"],
            edge_locations=["FRA", "MUC", "BER"]
        ),
        
        # Asia Pacific
        Region.ASIA_SOUTHEAST_1: RegionConfig(
            region=Region.ASIA_SOUTHEAST_1,
            country_codes=["SG"],
            jurisdiction="SG",
            compliance_frameworks=[ComplianceRegion.PDPA_SG],
            data_sovereignty_level=DataSovereigntyLevel.PREFER_LOCAL,
            allowed_data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            cross_border_allowed=True,
            availability_zones=["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"],
            edge_locations=["SIN", "KUL"]
        ),
        Region.ASIA_NORTHEAST_1: RegionConfig(
            region=Region.ASIA_NORTHEAST_1,
            country_codes=["JP"],
            jurisdiction="JP",
            compliance_frameworks=[ComplianceRegion.APPI],
            data_sovereignty_level=DataSovereigntyLevel.PREFER_LOCAL,
            allowed_data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            cross_border_allowed=True,
            availability_zones=["ap-northeast-1a", "ap-northeast-1b", "ap-northeast-1c"],
            edge_locations=["NRT", "KIX", "ITM"]
        ),
        
        # South America
        Region.SOUTH_AMERICA_EAST: RegionConfig(
            region=Region.SOUTH_AMERICA_EAST,
            country_codes=["BR"],
            jurisdiction="BR",
            compliance_frameworks=[ComplianceRegion.LGPD],
            data_sovereignty_level=DataSovereigntyLevel.STRICT_LOCAL,
            allowed_data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            cross_border_allowed=True,
            availability_zones=["sa-east-1a", "sa-east-1b", "sa-east-1c"],
            edge_locations=["GRU", "GIG"]
        ),
    }
    
    @classmethod
    def get_region_config(cls, region: Region) -> Optional[RegionConfig]:
        """Get configuration for a region"""
        return cls.REGION_MAPPINGS.get(region)
    
    @classmethod
    def get_regions_by_jurisdiction(cls, jurisdiction: str) -> List[Region]:
        """Get all regions in a jurisdiction"""
        regions = []
        for region, config in cls.REGION_MAPPINGS.items():
            if config.jurisdiction == jurisdiction or jurisdiction in config.country_codes:
                regions.append(region)
        return regions
    
    @classmethod
    def get_regions_by_compliance(cls, compliance: ComplianceRegion) -> List[Region]:
        """Get all regions supporting a compliance framework"""
        regions = []
        for region, config in cls.REGION_MAPPINGS.items():
            if compliance in config.compliance_frameworks:
                regions.append(region)
        return regions
    
    @classmethod
    def is_cross_border_allowed(cls, source_region: Region, dest_region: Region) -> bool:
        """Check if cross-border transfer is allowed between regions"""
        source_config = cls.get_region_config(source_region)
        dest_config = cls.get_region_config(dest_region)
        
        if not source_config or not dest_config:
            return False
        
        # Check if source allows cross-border transfers
        if not source_config.cross_border_allowed:
            return False
        
        # Check data sovereignty levels
        if source_config.data_sovereignty_level == DataSovereigntyLevel.STRICT_LOCAL:
            # Must stay within same jurisdiction
            return source_config.jurisdiction == dest_config.jurisdiction
        
        if source_config.data_sovereignty_level == DataSovereigntyLevel.GOVERNMENT_CLOUD:
            # Government cloud only
            return False
        
        return True

class DataResidencyManager:
    """Manages data residency rules and compliance"""
    
    def __init__(self):
        self.rules: Dict[str, DataResidencyRule] = {}
        self.transfer_requests: List[DataTransferRequest] = []
        self._lock = threading.RLock()
        self.region_mapper = RegionMapper()
        
        # Load default rules
        self._create_default_rules()
    
    def _create_default_rules(self):
        """Create default data residency rules"""
        
        # GDPR rule
        gdpr_rule = DataResidencyRule(
            rule_id="gdpr_personal_data",
            name="GDPR Personal Data Residency",
            data_types=[DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED],
            required_regions=[Region.EU_WEST_1, Region.EU_CENTRAL_1, Region.EU_NORTH_1],
            prohibited_regions=[Region.US_EAST_1, Region.US_WEST_2],
            max_cross_border_transfers=0,
            pseudonymization_required=True,
            legal_basis="GDPR Article 6",
            description="Personal data must remain within EU/EEA"
        )
        
        # US Government data rule
        us_gov_rule = DataResidencyRule(
            rule_id="us_government_data",
            name="US Government Data Residency",
            data_types=[DataClassification.RESTRICTED, DataClassification.TOP_SECRET],
            required_regions=[Region.US_EAST_1, Region.US_WEST_2],
            prohibited_regions=[],  # Prohibit all non-US regions
            max_cross_border_transfers=0,
            legal_basis="FedRAMP/FISMA",
            description="Government data must remain in US"
        )
        
        # Health data rule
        health_data_rule = DataResidencyRule(
            rule_id="health_data_residency",
            name="Health Data Residency",
            data_types=[DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED],
            required_regions=[],  # Depends on patient location
            max_cross_border_transfers=0,
            pseudonymization_required=True,
            legal_basis="HIPAA/GDPR",
            description="Health data must remain in patient's jurisdiction"
        )
        
        # Research data rule
        research_rule = DataResidencyRule(
            rule_id="research_data_collaboration",
            name="Research Data Collaboration",
            data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL],
            required_regions=[],  # Allow all regions
            max_cross_border_transfers=10,
            pseudonymization_required=True,
            legal_basis="Research collaboration agreements",
            description="Research data can be shared internationally with safeguards"
        )
        
        with self._lock:
            self.rules[gdpr_rule.rule_id] = gdpr_rule
            self.rules[us_gov_rule.rule_id] = us_gov_rule
            self.rules[health_data_rule.rule_id] = health_data_rule
            self.rules[research_rule.rule_id] = research_rule
    
    def add_residency_rule(self, rule: DataResidencyRule):
        """Add a data residency rule"""
        with self._lock:
            self.rules[rule.rule_id] = rule
        
        logger.info(f"Added data residency rule: {rule.rule_id}")
    
    def get_applicable_rules(self, 
                           data_classification: DataClassification,
                           source_region: Region) -> List[DataResidencyRule]:
        """Get applicable residency rules for data"""
        applicable_rules = []
        
        with self._lock:
            for rule in self.rules.values():
                if data_classification in rule.data_types:
                    # Check if source region is prohibited
                    if source_region not in rule.prohibited_regions:
                        applicable_rules.append(rule)
        
        return applicable_rules
    
    def validate_data_placement(self,
                              data_classification: DataClassification,
                              proposed_region: Region,
                              source_region: Optional[Region] = None) -> Tuple[bool, List[str]]:
        """Validate if data can be placed in proposed region"""
        violations = []
        
        # Get applicable rules
        rules = self.get_applicable_rules(data_classification, proposed_region)
        
        for rule in rules:
            # Check required regions
            if rule.required_regions and proposed_region not in rule.required_regions:
                violations.append(f"Rule {rule.rule_id}: Region {proposed_region.value} not in required regions")
            
            # Check prohibited regions
            if proposed_region in rule.prohibited_regions:
                violations.append(f"Rule {rule.rule_id}: Region {proposed_region.value} is prohibited")
            
            # Check cross-border transfer limits
            if source_region and source_region != proposed_region:
                region_config = self.region_mapper.get_region_config(proposed_region)
                if not region_config or not region_config.cross_border_allowed:
                    violations.append(f"Rule {rule.rule_id}: Cross-border transfer not allowed to {proposed_region.value}")
        
        return len(violations) == 0, violations
    
    def request_data_transfer(self,
                            source_region: Region,
                            destination_region: Region,
                            data_classification: DataClassification,
                            data_size_bytes: int,
                            purpose: str,
                            legal_basis: str,
                            requestor: str) -> DataTransferRequest:
        """Request cross-border data transfer"""
        
        request_id = str(uuid.uuid4())
        
        # Validate transfer
        is_valid, violations = self.validate_data_placement(
            data_classification, destination_region, source_region
        )
        
        # Check if cross-border transfer is allowed by region policies
        if not self.region_mapper.is_cross_border_allowed(source_region, destination_region):
            violations.append("Cross-border transfer not allowed by region policy")
            is_valid = False
        
        request = DataTransferRequest(
            request_id=request_id,
            source_region=source_region,
            destination_region=destination_region,
            data_classification=data_classification,
            data_size_bytes=data_size_bytes,
            purpose=purpose,
            legal_basis=legal_basis,
            requestor=requestor,
            timestamp=time.time(),
            approval_status="denied" if not is_valid else "pending",
            metadata={"violations": violations if not is_valid else []}
        )
        
        with self._lock:
            self.transfer_requests.append(request)
        
        logger.info(f"Data transfer request created: {request_id} ({request.approval_status})")
        return request
    
    def approve_transfer_request(self, request_id: str, approver: str, 
                               safeguards: Optional[List[str]] = None) -> bool:
        """Approve a data transfer request"""
        
        with self._lock:
            for request in self.transfer_requests:
                if request.request_id == request_id:
                    if request.approval_status == "pending":
                        request.approval_status = "approved"
                        request.approver = approver
                        request.safeguards = safeguards or []
                        
                        logger.info(f"Data transfer request approved: {request_id}")
                        return True
        
        return False
    
    def get_residency_report(self) -> Dict[str, Any]:
        """Generate data residency compliance report"""
        
        with self._lock:
            total_requests = len(self.transfer_requests)
            approved_requests = len([r for r in self.transfer_requests if r.approval_status == "approved"])
            denied_requests = len([r for r in self.transfer_requests if r.approval_status == "denied"])
            pending_requests = len([r for r in self.transfer_requests if r.approval_status == "pending"])
            
            # Group by regions
            region_stats = defaultdict(lambda: {"source": 0, "destination": 0})
            for request in self.transfer_requests:
                region_stats[request.source_region.value]["source"] += 1
                region_stats[request.destination_region.value]["destination"] += 1
            
            # Group by data classification
            classification_stats = defaultdict(int)
            for request in self.transfer_requests:
                classification_stats[request.data_classification.value] += 1
        
        return {
            "timestamp": time.time(),
            "total_rules": len(self.rules),
            "transfer_requests": {
                "total": total_requests,
                "approved": approved_requests,
                "denied": denied_requests,
                "pending": pending_requests
            },
            "region_statistics": dict(region_stats),
            "classification_statistics": dict(classification_stats),
            "active_rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "data_types": [dt.value for dt in rule.data_types],
                    "required_regions": [r.value for r in rule.required_regions],
                    "prohibited_regions": [r.value for r in rule.prohibited_regions]
                }
                for rule in self.rules.values()
            ]
        }

class MultiRegionDeploymentManager:
    """Manages multi-region deployments with compliance"""
    
    def __init__(self):
        self.deployments: Dict[str, RegionalDeployment] = {}
        self.residency_manager = DataResidencyManager()
        self.region_mapper = RegionMapper()
        self._lock = threading.RLock()
        
        logger.info("Multi-region deployment manager initialized")
    
    def create_regional_deployment(self,
                                 region: Region,
                                 namespace: str,
                                 cluster_name: str,
                                 data_residency_rules: List[str],
                                 storage_config: Optional[Dict[str, Any]] = None,
                                 network_config: Optional[Dict[str, Any]] = None) -> RegionalDeployment:
        """Create a new regional deployment"""
        
        deployment_id = f"{region.value}-{namespace}-{int(time.time())}"
        
        # Get region configuration
        region_config = self.region_mapper.get_region_config(region)
        if not region_config:
            raise ValueError(f"Unsupported region: {region.value}")
        
        # Default configurations
        default_storage = {
            "encryption_at_rest": True,
            "encryption_key_rotation": True,
            "backup_enabled": True,
            "replication_factor": 3,
            "storage_class": "regional-ssd"
        }
        
        default_network = {
            "vpc_isolation": True,
            "private_subnets": True,
            "network_acls": True,
            "security_groups": ["default-sg", "protein-sssl-sg"],
            "load_balancer_type": "internal"
        }
        
        deployment = RegionalDeployment(
            deployment_id=deployment_id,
            region=region,
            namespace=namespace,
            cluster_name=cluster_name,
            data_residency_rules=data_residency_rules,
            storage_configuration=storage_config or default_storage,
            network_configuration=network_config or default_network,
            security_configuration={
                "encryption_required": region_config.encryption_required,
                "audit_logging": region_config.audit_logging_required,
                "compliance_frameworks": [cf.value for cf in region_config.compliance_frameworks],
                "data_sovereignty_level": region_config.data_sovereignty_level.value
            },
            monitoring_configuration={
                "metrics_retention_days": 90,
                "log_retention_days": 365,
                "alerting_enabled": True,
                "compliance_monitoring": True
            },
            backup_configuration={
                "backup_regions": [r.value for r in region_config.backup_regions],
                "backup_frequency": "daily",
                "retention_period_days": 30,
                "cross_region_replication": len(region_config.backup_regions) > 0
            }
        )
        
        with self._lock:
            self.deployments[deployment_id] = deployment
        
        logger.info(f"Created regional deployment: {deployment_id} in {region.value}")
        return deployment
    
    def get_optimal_region(self,
                          user_location: str,
                          data_classification: DataClassification,
                          compliance_requirements: List[ComplianceRegion]) -> Optional[Region]:
        """Get optimal region for deployment based on requirements"""
        
        candidate_regions = []
        
        # Filter regions by compliance requirements
        for compliance in compliance_requirements:
            regions = self.region_mapper.get_regions_by_compliance(compliance)
            candidate_regions.extend(regions)
        
        if not candidate_regions:
            # No specific compliance requirements, use all regions
            candidate_regions = list(Region)
        
        # Filter by data classification
        suitable_regions = []
        for region in candidate_regions:
            region_config = self.region_mapper.get_region_config(region)
            if region_config and data_classification in region_config.allowed_data_types:
                suitable_regions.append(region)
        
        if not suitable_regions:
            return None
        
        # Simple selection based on user location (in practice, use latency-based selection)
        location_preferences = {
            "US": [Region.US_EAST_1, Region.US_WEST_2],
            "CA": [Region.CANADA_CENTRAL],
            "EU": [Region.EU_WEST_1, Region.EU_CENTRAL_1],
            "SG": [Region.ASIA_SOUTHEAST_1],
            "JP": [Region.ASIA_NORTHEAST_1],
            "BR": [Region.SOUTH_AMERICA_EAST]
        }
        
        preferred_regions = location_preferences.get(user_location, [])
        for region in preferred_regions:
            if region in suitable_regions:
                return region
        
        # Return first suitable region if no preference match
        return suitable_regions[0] if suitable_regions else None
    
    def validate_deployment_compliance(self, deployment_id: str) -> Tuple[bool, List[str]]:
        """Validate deployment compliance"""
        
        with self._lock:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                return False, [f"Deployment not found: {deployment_id}"]
        
        violations = []
        
        # Check region configuration
        region_config = self.region_mapper.get_region_config(deployment.region)
        if not region_config:
            violations.append(f"Invalid region configuration: {deployment.region.value}")
            return False, violations
        
        # Check data residency rules
        for rule_id in deployment.data_residency_rules:
            rule = self.residency_manager.rules.get(rule_id)
            if not rule:
                violations.append(f"Data residency rule not found: {rule_id}")
                continue
            
            # Validate region against rule
            if rule.required_regions and deployment.region not in rule.required_regions:
                violations.append(f"Rule {rule_id}: Region {deployment.region.value} not in required regions")
            
            if deployment.region in rule.prohibited_regions:
                violations.append(f"Rule {rule_id}: Region {deployment.region.value} is prohibited")
        
        # Check security configuration
        if region_config.encryption_required and not deployment.security_configuration.get("encryption_required"):
            violations.append("Encryption required by region policy but not enabled")
        
        if region_config.audit_logging_required and not deployment.security_configuration.get("audit_logging"):
            violations.append("Audit logging required by region policy but not enabled")
        
        return len(violations) == 0, violations
    
    def get_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        
        with self._lock:
            deployments_by_region = defaultdict(list)
            deployments_by_status = defaultdict(int)
            
            for deployment in self.deployments.values():
                deployments_by_region[deployment.region.value].append(deployment.deployment_id)
                deployments_by_status[deployment.status] += 1
            
            compliance_coverage = defaultdict(int)
            for deployment in self.deployments.values():
                frameworks = deployment.security_configuration.get("compliance_frameworks", [])
                for framework in frameworks:
                    compliance_coverage[framework] += 1
        
        residency_report = self.residency_manager.get_residency_report()
        
        return {
            "timestamp": time.time(),
            "total_deployments": len(self.deployments),
            "deployments_by_region": dict(deployments_by_region),
            "deployments_by_status": dict(deployments_by_status),
            "compliance_coverage": dict(compliance_coverage),
            "data_residency": residency_report,
            "supported_regions": [region.value for region in Region],
            "compliance_frameworks": [cf.value for cf in ComplianceRegion]
        }

# Global managers
_global_residency_manager: Optional[DataResidencyManager] = None
_global_deployment_manager: Optional[MultiRegionDeploymentManager] = None

def get_residency_manager() -> Optional[DataResidencyManager]:
    """Get global data residency manager"""
    return _global_residency_manager

def get_deployment_manager() -> Optional[MultiRegionDeploymentManager]:
    """Get global multi-region deployment manager"""
    return _global_deployment_manager

def initialize_multi_region() -> Tuple[DataResidencyManager, MultiRegionDeploymentManager]:
    """Initialize global multi-region managers"""
    global _global_residency_manager, _global_deployment_manager
    
    _global_residency_manager = DataResidencyManager()
    _global_deployment_manager = MultiRegionDeploymentManager()
    
    return _global_residency_manager, _global_deployment_manager

# Convenience functions
def get_optimal_deployment_region(user_location: str,
                                data_classification: DataClassification,
                                compliance_requirements: List[ComplianceRegion]) -> Optional[Region]:
    """Get optimal region for deployment"""
    if _global_deployment_manager:
        return _global_deployment_manager.get_optimal_region(
            user_location, data_classification, compliance_requirements
        )
    return None

def validate_data_residency(data_classification: DataClassification,
                          proposed_region: Region,
                          source_region: Optional[Region] = None) -> Tuple[bool, List[str]]:
    """Validate data residency compliance"""
    if _global_residency_manager:
        return _global_residency_manager.validate_data_placement(
            data_classification, proposed_region, source_region
        )
    return True, []