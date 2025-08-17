"""
Global Deployment Manager for protein-sssl-operator
Orchestrates all global deployment features including internationalization,
compliance, accessibility, cultural adaptation, and legal requirements.
"""

import json
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Import our global deployment frameworks
from .i18n_framework import (
    I18nManager, Language, LocaleConfig, initialize_i18n, get_locale_config
)
from .compliance_framework import (
    ComplianceManager, ComplianceRegulation, initialize_compliance,
    create_multi_region_policy
)
from .multi_region_deployment import (
    MultiRegionDeploymentManager, DataResidencyManager, Region,
    DataClassification, ComplianceRegion, initialize_multi_region
)
from .accessibility_framework import (
    AccessibilityManager, AccessibilityProfile, WCAGLevel,
    initialize_accessibility
)
from .cultural_adaptation import (
    CulturalAdaptationManager, CulturalRegion, ScientificNotation,
    initialize_cultural_adaptation
)
from .legal_compliance import (
    LegalComplianceManager, LegalJurisdiction, initialize_legal_compliance
)

logger = logging.getLogger(__name__)

class DeploymentMode(Enum):
    """Deployment modes"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"

class DeploymentStatus(Enum):
    """Deployment status"""
    INITIALIZING = "initializing"
    CONFIGURING = "configuring"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    TERMINATED = "terminated"

@dataclass
class GlobalDeploymentConfig:
    """Global deployment configuration"""
    deployment_id: str
    deployment_name: str
    mode: DeploymentMode
    target_regions: List[Region]
    primary_region: Region
    
    # Compliance settings
    compliance_frameworks: List[ComplianceRegulation]
    data_classification: DataClassification
    cross_border_transfers_allowed: bool = True
    
    # Internationalization settings
    default_language: Language = Language.ENGLISH
    supported_languages: List[Language] = field(default_factory=lambda: [Language.ENGLISH])
    default_locale: str = "en-US"
    
    # Accessibility settings
    wcag_compliance_level: WCAGLevel = WCAGLevel.AA
    accessibility_features_enabled: bool = True
    
    # Cultural adaptation settings
    cultural_regions: List[CulturalRegion] = field(default_factory=list)
    scientific_notation: ScientificNotation = ScientificNotation.DECIMAL_POINT
    
    # Legal compliance settings
    legal_jurisdictions: List[LegalJurisdiction] = field(default_factory=list)
    export_control_enabled: bool = True
    data_retention_enabled: bool = True
    
    # Technical settings
    encryption_required: bool = True
    audit_logging_enabled: bool = True
    monitoring_enabled: bool = True
    backup_enabled: bool = True
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    last_updated: float = field(default_factory=time.time)
    version: str = "1.0.0"

@dataclass
class DeploymentValidationResult:
    """Deployment validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_timestamp: float = field(default_factory=time.time)

@dataclass
class GlobalDeployment:
    """Global deployment instance"""
    deployment_id: str
    config: GlobalDeploymentConfig
    status: DeploymentStatus
    regional_deployments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compliance_status: Dict[str, Any] = field(default_factory=dict)
    accessibility_status: Dict[str, Any] = field(default_factory=dict)
    cultural_adaptations: Dict[str, Any] = field(default_factory=dict)
    legal_compliance_status: Dict[str, Any] = field(default_factory=dict)
    health_status: Dict[str, Any] = field(default_factory=dict)
    last_health_check: float = field(default_factory=time.time)
    deployment_logs: List[Dict[str, Any]] = field(default_factory=list)

class GlobalDeploymentValidator:
    """Validates global deployment configurations"""
    
    @staticmethod
    def validate_config(config: GlobalDeploymentConfig) -> DeploymentValidationResult:
        """Validate deployment configuration"""
        errors = []
        warnings = []
        recommendations = []
        
        # Validate regions
        if not config.target_regions:
            errors.append("At least one target region must be specified")
        
        if config.primary_region not in config.target_regions:
            errors.append("Primary region must be included in target regions")
        
        # Validate compliance frameworks
        if not config.compliance_frameworks:
            warnings.append("No compliance frameworks specified")
        
        # Check region-compliance alignment
        for region in config.target_regions:
            if region.value.startswith("eu-") and ComplianceRegulation.GDPR not in config.compliance_frameworks:
                warnings.append(f"GDPR compliance recommended for EU region {region.value}")
            
            if region.value.startswith("us-") and ComplianceRegulation.CCPA not in config.compliance_frameworks:
                warnings.append(f"CCPA compliance recommended for US region {region.value}")
        
        # Validate language support
        if not config.supported_languages:
            errors.append("At least one supported language must be specified")
        
        if config.default_language not in config.supported_languages:
            errors.append("Default language must be included in supported languages")
        
        # Validate cultural regions
        if config.cultural_regions:
            for cultural_region in config.cultural_regions:
                # Check if cultural region aligns with target regions
                region_alignment = GlobalDeploymentValidator._check_cultural_region_alignment(
                    cultural_region, config.target_regions
                )
                if not region_alignment:
                    warnings.append(f"Cultural region {cultural_region.value} may not align with target regions")
        
        # Security validations
        if not config.encryption_required:
            warnings.append("Encryption should be required for production deployments")
        
        if not config.audit_logging_enabled:
            warnings.append("Audit logging should be enabled for compliance")
        
        # Generate recommendations
        if config.mode == DeploymentMode.PRODUCTION:
            recommendations.extend([
                "Enable all security features for production",
                "Set up comprehensive monitoring and alerting",
                "Establish disaster recovery procedures",
                "Conduct regular compliance audits"
            ])
        
        if len(config.target_regions) > 1:
            recommendations.extend([
                "Implement cross-region data synchronization",
                "Set up region failover mechanisms",
                "Test cross-border data transfer compliance"
            ])
        
        if config.accessibility_features_enabled:
            recommendations.append("Conduct accessibility testing with actual users")
        
        return DeploymentValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    @staticmethod
    def _check_cultural_region_alignment(cultural_region: CulturalRegion, 
                                       target_regions: List[Region]) -> bool:
        """Check if cultural region aligns with target regions"""
        # Simplified alignment check
        cultural_to_regions = {
            CulturalRegion.WESTERN_EUROPE: [Region.EU_WEST_1, Region.EU_CENTRAL_1, Region.EU_NORTH_1],
            CulturalRegion.NORTH_AMERICA: [Region.US_EAST_1, Region.US_WEST_2, Region.CANADA_CENTRAL],
            CulturalRegion.EAST_ASIA: [Region.ASIA_NORTHEAST_1, Region.ASIA_EAST_1],
            CulturalRegion.SOUTHEAST_ASIA: [Region.ASIA_SOUTHEAST_1],
            CulturalRegion.LATIN_AMERICA: [Region.SOUTH_AMERICA_EAST],
        }
        
        expected_regions = cultural_to_regions.get(cultural_region, [])
        return any(region in expected_regions for region in target_regions)

class GlobalDeploymentManager:
    """Main global deployment manager"""
    
    def __init__(self):
        self.deployments: Dict[str, GlobalDeployment] = {}
        self.validator = GlobalDeploymentValidator()
        self._lock = threading.RLock()
        
        # Initialize all sub-managers
        self.i18n_manager: Optional[I18nManager] = None
        self.compliance_manager: Optional[ComplianceManager] = None
        self.multi_region_manager: Optional[MultiRegionDeploymentManager] = None
        self.accessibility_manager: Optional[AccessibilityManager] = None
        self.cultural_manager: Optional[CulturalAdaptationManager] = None
        self.legal_manager: Optional[LegalComplianceManager] = None
        
        # Health monitoring
        self._health_check_interval = 300  # 5 minutes
        self._start_health_monitoring()
        
        logger.info("Global deployment manager initialized")
    
    def initialize_all_frameworks(self):
        """Initialize all global deployment frameworks"""
        try:
            # Initialize i18n
            self.i18n_manager = initialize_i18n()
            logger.info("I18n framework initialized")
            
            # Initialize compliance
            multi_region_policy = create_multi_region_policy()
            self.compliance_manager = initialize_compliance(multi_region_policy)
            logger.info("Compliance framework initialized")
            
            # Initialize multi-region deployment
            _, self.multi_region_manager = initialize_multi_region()
            logger.info("Multi-region deployment framework initialized")
            
            # Initialize accessibility
            self.accessibility_manager = initialize_accessibility()
            logger.info("Accessibility framework initialized")
            
            # Initialize cultural adaptation
            self.cultural_manager = initialize_cultural_adaptation()
            logger.info("Cultural adaptation framework initialized")
            
            # Initialize legal compliance
            self.legal_manager = initialize_legal_compliance()
            logger.info("Legal compliance framework initialized")
            
            logger.info("All global deployment frameworks initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing frameworks: {e}")
            raise
    
    def create_deployment(self, config: GlobalDeploymentConfig) -> GlobalDeployment:
        """Create a new global deployment"""
        
        # Validate configuration
        validation_result = self.validator.validate_config(config)
        if not validation_result.is_valid:
            error_msg = f"Invalid deployment configuration: {'; '.join(validation_result.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log warnings
        for warning in validation_result.warnings:
            logger.warning(f"Deployment config warning: {warning}")
        
        # Create deployment
        deployment = GlobalDeployment(
            deployment_id=config.deployment_id,
            config=config,
            status=DeploymentStatus.INITIALIZING
        )
        
        # Add deployment log
        deployment.deployment_logs.append({
            "timestamp": time.time(),
            "event": "deployment_created",
            "details": {"validation_warnings": validation_result.warnings}
        })
        
        with self._lock:
            self.deployments[deployment.deployment_id] = deployment
        
        logger.info(f"Created global deployment: {deployment.deployment_id}")
        
        # Start asynchronous deployment process
        self._deploy_async(deployment)
        
        return deployment
    
    def _deploy_async(self, deployment: GlobalDeployment):
        """Deploy asynchronously"""
        def deploy_worker():
            try:
                self._execute_deployment(deployment)
            except Exception as e:
                logger.error(f"Deployment error for {deployment.deployment_id}: {e}")
                deployment.status = DeploymentStatus.ERROR
                deployment.deployment_logs.append({
                    "timestamp": time.time(),
                    "event": "deployment_error",
                    "details": {"error": str(e)}
                })
        
        thread = threading.Thread(target=deploy_worker, daemon=True)
        thread.start()
    
    def _execute_deployment(self, deployment: GlobalDeployment):
        """Execute deployment process"""
        config = deployment.config
        deployment.status = DeploymentStatus.CONFIGURING
        
        try:
            # Configure i18n
            if self.i18n_manager:
                locale_config = get_locale_config(config.default_locale)
                if locale_config:
                    self.i18n_manager.set_locale(locale_config)
                deployment.deployment_logs.append({
                    "timestamp": time.time(),
                    "event": "i18n_configured",
                    "details": {"locale": config.default_locale, "languages": [l.value for l in config.supported_languages]}
                })
            
            # Configure compliance
            if self.compliance_manager:
                compliance_status = self.compliance_manager.validate_compliance()
                deployment.compliance_status = {
                    "is_compliant": compliance_status[0],
                    "issues": compliance_status[1],
                    "frameworks": [cf.value for cf in config.compliance_frameworks]
                }
                deployment.deployment_logs.append({
                    "timestamp": time.time(),
                    "event": "compliance_configured",
                    "details": deployment.compliance_status
                })
            
            # Configure regional deployments
            if self.multi_region_manager:
                for region in config.target_regions:
                    regional_deployment = self.multi_region_manager.create_regional_deployment(
                        region=region,
                        namespace=f"protein-sssl-{config.deployment_name}",
                        cluster_name=f"cluster-{region.value}",
                        data_residency_rules=["gdpr_personal_data", "us_research_data"]
                    )
                    
                    deployment.regional_deployments[region.value] = {
                        "deployment_id": regional_deployment.deployment_id,
                        "status": regional_deployment.status,
                        "created_at": regional_deployment.created_at
                    }
                
                deployment.deployment_logs.append({
                    "timestamp": time.time(),
                    "event": "regional_deployments_created",
                    "details": {"regions": [r.value for r in config.target_regions]}
                })
            
            # Configure accessibility
            if self.accessibility_manager and config.accessibility_features_enabled:
                accessibility_report = self.accessibility_manager.get_accessibility_report()
                deployment.accessibility_status = {
                    "wcag_level": config.wcag_compliance_level.value,
                    "features_enabled": True,
                    "report": accessibility_report
                }
                deployment.deployment_logs.append({
                    "timestamp": time.time(),
                    "event": "accessibility_configured",
                    "details": {"wcag_level": config.wcag_compliance_level.value}
                })
            
            # Configure cultural adaptation
            if self.cultural_manager and config.cultural_regions:
                for cultural_region in config.cultural_regions:
                    preferences = self.cultural_manager.get_cultural_preferences(cultural_region)
                    if preferences:
                        deployment.cultural_adaptations[cultural_region.value] = {
                            "scientific_notation": preferences.scientific_notation.value,
                            "communication_style": preferences.communication_style.value,
                            "color_culture": preferences.color_culture.value
                        }
                
                deployment.deployment_logs.append({
                    "timestamp": time.time(),
                    "event": "cultural_adaptation_configured",
                    "details": {"regions": [cr.value for cr in config.cultural_regions]}
                })
            
            # Configure legal compliance
            if self.legal_manager:
                legal_status = self.legal_manager.get_compliance_status()
                deployment.legal_compliance_status = {
                    "overall_status": legal_status["overall_status"],
                    "export_control_enabled": config.export_control_enabled,
                    "data_retention_enabled": config.data_retention_enabled,
                    "jurisdictions": [lj.value for lj in config.legal_jurisdictions]
                }
                deployment.deployment_logs.append({
                    "timestamp": time.time(),
                    "event": "legal_compliance_configured",
                    "details": deployment.legal_compliance_status
                })
            
            # Finalize deployment
            deployment.status = DeploymentStatus.ACTIVE
            deployment.last_health_check = time.time()
            deployment.deployment_logs.append({
                "timestamp": time.time(),
                "event": "deployment_completed",
                "details": {"status": "active"}
            })
            
            logger.info(f"Global deployment completed successfully: {deployment.deployment_id}")
            
        except Exception as e:
            deployment.status = DeploymentStatus.ERROR
            deployment.deployment_logs.append({
                "timestamp": time.time(),
                "event": "deployment_failed",
                "details": {"error": str(e)}
            })
            raise
    
    def get_deployment(self, deployment_id: str) -> Optional[GlobalDeployment]:
        """Get deployment by ID"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> List[GlobalDeployment]:
        """List all deployments"""
        return list(self.deployments.values())
    
    def update_deployment_config(self, deployment_id: str, 
                               config_updates: Dict[str, Any]) -> bool:
        """Update deployment configuration"""
        with self._lock:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                return False
            
            # Update config
            for key, value in config_updates.items():
                if hasattr(deployment.config, key):
                    setattr(deployment.config, key, value)
            
            deployment.config.last_updated = time.time()
            deployment.deployment_logs.append({
                "timestamp": time.time(),
                "event": "config_updated",
                "details": config_updates
            })
            
            logger.info(f"Updated deployment config: {deployment_id}")
            return True
    
    def terminate_deployment(self, deployment_id: str) -> bool:
        """Terminate deployment"""
        with self._lock:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                return False
            
            deployment.status = DeploymentStatus.TERMINATED
            deployment.deployment_logs.append({
                "timestamp": time.time(),
                "event": "deployment_terminated",
                "details": {"terminated_by": "admin"}
            })
            
            logger.info(f"Terminated deployment: {deployment_id}")
            return True
    
    def _start_health_monitoring(self):
        """Start health monitoring for all deployments"""
        def health_monitor():
            while True:
                try:
                    self._perform_health_checks()
                    time.sleep(self._health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
        
        thread = threading.Thread(target=health_monitor, daemon=True)
        thread.start()
    
    def _perform_health_checks(self):
        """Perform health checks on all deployments"""
        current_time = time.time()
        
        with self._lock:
            for deployment in self.deployments.values():
                if deployment.status == DeploymentStatus.ACTIVE:
                    health_status = self._check_deployment_health(deployment)
                    deployment.health_status = health_status
                    deployment.last_health_check = current_time
                    
                    if not health_status.get("healthy", True):
                        logger.warning(f"Health check failed for deployment: {deployment.deployment_id}")
    
    def _check_deployment_health(self, deployment: GlobalDeployment) -> Dict[str, Any]:
        """Check health of a specific deployment"""
        health_status = {
            "healthy": True,
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check compliance status
        if self.compliance_manager:
            compliance_check = self.compliance_manager.validate_compliance()
            health_status["checks"]["compliance"] = {
                "status": "healthy" if compliance_check[0] else "unhealthy",
                "issues": compliance_check[1] if not compliance_check[0] else []
            }
            if not compliance_check[0]:
                health_status["healthy"] = False
        
        # Check regional deployments
        unhealthy_regions = []
        for region, regional_info in deployment.regional_deployments.items():
            # Simplified health check - in practice would check actual infrastructure
            if regional_info.get("status") != "active":
                unhealthy_regions.append(region)
        
        health_status["checks"]["regional_deployments"] = {
            "status": "healthy" if not unhealthy_regions else "unhealthy",
            "unhealthy_regions": unhealthy_regions
        }
        
        if unhealthy_regions:
            health_status["healthy"] = False
        
        # Check legal compliance
        if self.legal_manager:
            legal_status = self.legal_manager.get_compliance_status()
            health_status["checks"]["legal_compliance"] = {
                "status": "healthy" if legal_status["overall_status"] == "compliant" else "unhealthy",
                "issues": legal_status.get("issues", [])
            }
            if legal_status["overall_status"] != "compliant":
                health_status["healthy"] = False
        
        return health_status
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global deployment status across all frameworks"""
        current_time = time.time()
        
        with self._lock:
            total_deployments = len(self.deployments)
            active_deployments = len([d for d in self.deployments.values() if d.status == DeploymentStatus.ACTIVE])
            error_deployments = len([d for d in self.deployments.values() if d.status == DeploymentStatus.ERROR])
        
        # Aggregate framework status
        framework_status = {}
        
        if self.i18n_manager:
            framework_status["internationalization"] = {
                "status": "active",
                "supported_languages": len(self.i18n_manager.get_supported_languages()),
                "locale_info": self.i18n_manager.get_locale_info()
            }
        
        if self.compliance_manager:
            compliance_report = self.compliance_manager.get_compliance_report()
            framework_status["compliance"] = {
                "status": "active",
                "total_subjects": compliance_report["data_subjects"]["total"],
                "total_requests": compliance_report["subject_requests"]["total"]
            }
        
        if self.multi_region_manager:
            region_report = self.multi_region_manager.get_deployment_report()
            framework_status["multi_region"] = {
                "status": "active",
                "total_deployments": region_report["total_deployments"],
                "supported_regions": len(region_report["supported_regions"])
            }
        
        if self.accessibility_manager:
            accessibility_report = self.accessibility_manager.get_accessibility_report()
            framework_status["accessibility"] = {
                "status": "active",
                "total_users": accessibility_report["total_users"],
                "wcag_guidelines": accessibility_report["wcag_guidelines_supported"]
            }
        
        if self.cultural_manager:
            cultural_report = self.cultural_manager.get_adaptation_report()
            framework_status["cultural_adaptation"] = {
                "status": "active",
                "supported_regions": cultural_report["supported_regions"],
                "localized_terms": cultural_report["localized_terms"]
            }
        
        if self.legal_manager:
            legal_report = self.legal_manager.generate_compliance_report()
            framework_status["legal_compliance"] = {
                "status": "active",
                "overall_compliance": legal_report["overall_status"],
                "audit_requirements": legal_report["audit_compliance"]["total_requirements"]
            }
        
        return {
            "timestamp": current_time,
            "global_deployment_status": {
                "total_deployments": total_deployments,
                "active_deployments": active_deployments,
                "error_deployments": error_deployments,
                "health_check_interval_seconds": self._health_check_interval
            },
            "framework_status": framework_status,
            "system_readiness": {
                "all_frameworks_initialized": all([
                    self.i18n_manager is not None,
                    self.compliance_manager is not None,
                    self.multi_region_manager is not None,
                    self.accessibility_manager is not None,
                    self.cultural_manager is not None,
                    self.legal_manager is not None
                ]),
                "ready_for_deployment": active_deployments > 0 or total_deployments == 0
            }
        }
    
    def generate_deployment_report(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive deployment report"""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return None
        
        return {
            "deployment_info": {
                "deployment_id": deployment.deployment_id,
                "name": deployment.config.deployment_name,
                "mode": deployment.config.mode.value,
                "status": deployment.status.value,
                "created_at": deployment.config.created_at,
                "last_updated": deployment.config.last_updated
            },
            "configuration": {
                "target_regions": [r.value for r in deployment.config.target_regions],
                "primary_region": deployment.config.primary_region.value,
                "compliance_frameworks": [cf.value for cf in deployment.config.compliance_frameworks],
                "supported_languages": [l.value for l in deployment.config.supported_languages],
                "cultural_regions": [cr.value for cr in deployment.config.cultural_regions],
                "legal_jurisdictions": [lj.value for lj in deployment.config.legal_jurisdictions]
            },
            "status": {
                "compliance": deployment.compliance_status,
                "accessibility": deployment.accessibility_status,
                "cultural_adaptations": deployment.cultural_adaptations,
                "legal_compliance": deployment.legal_compliance_status,
                "health": deployment.health_status
            },
            "regional_deployments": deployment.regional_deployments,
            "deployment_logs": deployment.deployment_logs[-10:],  # Last 10 log entries
            "recommendations": self._generate_deployment_recommendations(deployment)
        }
    
    def _generate_deployment_recommendations(self, deployment: GlobalDeployment) -> List[str]:
        """Generate recommendations for deployment optimization"""
        recommendations = []
        
        # Health-based recommendations
        if not deployment.health_status.get("healthy", True):
            recommendations.append("Address health check failures immediately")
        
        # Compliance recommendations
        if not deployment.compliance_status.get("is_compliant", True):
            recommendations.append("Resolve compliance issues before processing personal data")
        
        # Regional recommendations
        if len(deployment.config.target_regions) == 1:
            recommendations.append("Consider adding backup regions for disaster recovery")
        
        # Legal recommendations
        if deployment.legal_compliance_status.get("overall_status") != "compliant":
            recommendations.append("Address legal compliance violations")
        
        # General recommendations
        recommendations.extend([
            "Regularly review and update deployment configuration",
            "Monitor performance metrics across all regions",
            "Conduct periodic compliance audits",
            "Test disaster recovery procedures"
        ])
        
        return recommendations

# Global deployment manager instance
_global_deployment_manager: Optional[GlobalDeploymentManager] = None

def get_global_deployment_manager() -> Optional[GlobalDeploymentManager]:
    """Get global deployment manager instance"""
    return _global_deployment_manager

def initialize_global_deployment() -> GlobalDeploymentManager:
    """Initialize global deployment manager"""
    global _global_deployment_manager
    _global_deployment_manager = GlobalDeploymentManager()
    _global_deployment_manager.initialize_all_frameworks()
    return _global_deployment_manager

# Convenience functions
def create_global_deployment(config: GlobalDeploymentConfig) -> Optional[GlobalDeployment]:
    """Create a global deployment"""
    if _global_deployment_manager:
        return _global_deployment_manager.create_deployment(config)
    return None

def get_deployment_status() -> Dict[str, Any]:
    """Get global deployment status"""
    if _global_deployment_manager:
        return _global_deployment_manager.get_global_status()
    return {"error": "Global deployment manager not initialized"}