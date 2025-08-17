"""
Legal and Regulatory Compliance Framework for protein-sssl-operator
Provides data retention policies, export control compliance, local content regulations,
regional audit requirements, cross-border legal frameworks, and dispute resolution.
"""

import json
import logging
import threading
import time
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)

class LegalJurisdiction(Enum):
    """Legal jurisdictions"""
    UNITED_STATES = "US"
    EUROPEAN_UNION = "EU"
    UNITED_KINGDOM = "GB"
    CANADA = "CA"
    AUSTRALIA = "AU"
    JAPAN = "JP"
    SINGAPORE = "SG"
    BRAZIL = "BR"
    INDIA = "IN"
    CHINA = "CN"
    SOUTH_KOREA = "KR"
    SWITZERLAND = "CH"
    NORWAY = "NO"
    ISRAEL = "IL"
    SOUTH_AFRICA = "ZA"

class ExportControlRegime(Enum):
    """Export control regimes"""
    EAR = "ear"          # Export Administration Regulations (US)
    ITAR = "itar"        # International Traffic in Arms Regulations (US)
    EU_DUAL_USE = "eu_dual_use"  # EU Dual-Use Regulation
    WASSENAAR = "wassenaar"      # Wassenaar Arrangement
    AUSTRALIA_GROUP = "australia_group"  # Australia Group
    MTCR = "mtcr"        # Missile Technology Control Regime
    NSG = "nsg"          # Nuclear Suppliers Group

class DataRetentionCategory(Enum):
    """Data retention categories"""
    RESEARCH_DATA = "research_data"
    PERSONAL_DATA = "personal_data"
    FINANCIAL_DATA = "financial_data"
    AUDIT_LOGS = "audit_logs"
    SECURITY_LOGS = "security_logs"
    COMMUNICATION_RECORDS = "communication_records"
    BACKUP_DATA = "backup_data"
    METADATA = "metadata"
    SYSTEM_LOGS = "system_logs"

class LegalBasis(Enum):
    """Legal basis for data processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    RESEARCH_EXEMPTION = "research_exemption"
    NATIONAL_SECURITY = "national_security"

class DisputeResolutionMethod(Enum):
    """Dispute resolution methods"""
    NEGOTIATION = "negotiation"
    MEDIATION = "mediation"
    ARBITRATION = "arbitration"
    LITIGATION = "litigation"
    REGULATORY_COMPLAINT = "regulatory_complaint"
    OMBUDSMAN = "ombudsman"

@dataclass
class DataRetentionPolicy:
    """Data retention policy"""
    policy_id: str
    name: str
    jurisdiction: LegalJurisdiction
    data_category: DataRetentionCategory
    retention_period_days: int
    legal_basis: str
    destruction_method: str
    review_period_days: int = 365
    exemptions: List[str] = field(default_factory=list)
    automated_deletion: bool = True
    archive_before_deletion: bool = True
    notification_required: bool = False
    approval_required: bool = False
    audit_trail_required: bool = True
    created_at: float = field(default_factory=time.time)
    last_reviewed: float = field(default_factory=time.time)
    
    def is_expired(self, data_timestamp: float) -> bool:
        """Check if data has exceeded retention period"""
        return (time.time() - data_timestamp) > (self.retention_period_days * 24 * 3600)
    
    def days_until_expiry(self, data_timestamp: float) -> int:
        """Calculate days until data expires"""
        expiry_time = data_timestamp + (self.retention_period_days * 24 * 3600)
        days_remaining = (expiry_time - time.time()) / (24 * 3600)
        return max(0, int(days_remaining))

@dataclass
class ExportControlRule:
    """Export control rule"""
    rule_id: str
    name: str
    regime: ExportControlRegime
    controlled_items: List[str]
    restricted_countries: List[str]
    restricted_entities: List[str]
    license_required: bool
    license_type: str
    screening_required: bool
    documentation_required: List[str]
    approval_authority: str
    penalty_description: str
    last_updated: float = field(default_factory=time.time)
    
    def requires_license(self, destination_country: str, 
                        end_user: str, item: str) -> bool:
        """Check if export license is required"""
        if not self.license_required:
            return False
        
        if destination_country in self.restricted_countries:
            return True
        
        if end_user in self.restricted_entities:
            return True
        
        if item in self.controlled_items:
            return True
        
        return False

@dataclass
class LegalDocument:
    """Legal document (contract, agreement, etc.)"""
    document_id: str
    document_type: str
    title: str
    parties: List[str]
    jurisdiction: LegalJurisdiction
    governing_law: str
    effective_date: float
    expiry_date: Optional[float]
    auto_renewal: bool
    content_hash: str
    storage_location: str
    access_permissions: List[str]
    review_required: bool = False
    next_review_date: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditRequirement:
    """Audit requirement"""
    requirement_id: str
    name: str
    jurisdiction: LegalJurisdiction
    regulation: str
    frequency: str  # "annual", "biannual", "quarterly", "monthly"
    scope: List[str]
    auditor_requirements: Dict[str, Any]
    documentation_required: List[str]
    retention_period_years: int
    notification_period_days: int
    remediation_period_days: int
    penalties: Dict[str, str]
    last_audit_date: Optional[float] = None
    next_audit_due: Optional[float] = None
    compliance_status: str = "pending"

@dataclass
class CrossBorderTransferAgreement:
    """Cross-border data transfer agreement"""
    agreement_id: str
    agreement_type: str  # "SCC", "BCR", "Adequacy", "DPA"
    sender_jurisdiction: LegalJurisdiction
    recipient_jurisdiction: LegalJurisdiction
    data_categories: List[str]
    transfer_purpose: str
    legal_safeguards: List[str]
    monitoring_mechanisms: List[str]
    breach_notification_procedure: str
    dispute_resolution_method: DisputeResolutionMethod
    effective_date: float
    expiry_date: Optional[float]
    renewal_terms: str
    termination_conditions: List[str]
    compliance_certifications: List[str] = field(default_factory=list)

@dataclass
class LegalIncident:
    """Legal incident or violation"""
    incident_id: str
    incident_type: str
    jurisdiction: LegalJurisdiction
    regulation_violated: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_data_subjects: int
    potential_penalties: Dict[str, Any]
    mitigation_actions: List[str]
    notification_requirements: Dict[str, Any]
    response_deadline: float
    status: str = "open"  # "open", "investigating", "resolved", "closed"
    assigned_to: Optional[str] = None
    reported_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    lessons_learned: Optional[str] = None

class ExportControlScreening:
    """Export control screening system"""
    
    def __init__(self):
        self.restricted_parties = self._load_restricted_parties()
        self.controlled_items = self._load_controlled_items()
        self.country_restrictions = self._load_country_restrictions()
    
    def _load_restricted_parties(self) -> Dict[str, Dict[str, Any]]:
        """Load restricted parties lists"""
        # In practice, this would load from official government sources
        return {
            "denied_persons": {
                "source": "US BIS Denied Persons List",
                "entities": [],
                "last_updated": time.time()
            },
            "sdn_list": {
                "source": "US OFAC SDN List", 
                "entities": [],
                "last_updated": time.time()
            },
            "entity_list": {
                "source": "US BIS Entity List",
                "entities": [],
                "last_updated": time.time()
            }
        }
    
    def _load_controlled_items(self) -> Dict[str, List[str]]:
        """Load controlled items classifications"""
        return {
            "EAR99": ["General software", "Research publications"],
            "5D002": ["Cryptographic software"],
            "5A002": ["Information security equipment"],
            "4A003": ["Computing equipment"],
            "9E003": ["Software for modeling"]
        }
    
    def _load_country_restrictions(self) -> Dict[str, Dict[str, Any]]:
        """Load country-specific restrictions"""
        return {
            "embargoed": {
                "countries": ["IR", "KP", "SY", "CU"],
                "description": "Comprehensive embargo"
            },
            "restricted": {
                "countries": ["CN", "RU"],
                "description": "Specific technology restrictions"
            },
            "sanctioned": {
                "countries": ["AF", "BY", "MM"],
                "description": "Targeted sanctions"
            }
        }
    
    def screen_party(self, party_name: str, country: str) -> Tuple[bool, Dict[str, Any]]:
        """Screen party against restricted lists"""
        screening_result = {
            "party_name": party_name,
            "country": country,
            "matches": [],
            "risk_level": "low",
            "requires_license": False,
            "recommendations": []
        }
        
        # Check against restricted parties (simplified)
        party_lower = party_name.lower()
        
        # Check country restrictions
        for restriction_type, info in self.country_restrictions.items():
            if country in info["countries"]:
                screening_result["matches"].append({
                    "list": f"Country {restriction_type}",
                    "reason": info["description"]
                })
                screening_result["risk_level"] = "high"
                screening_result["requires_license"] = True
        
        # Additional screening logic would go here
        
        # Generate recommendations
        if screening_result["requires_license"]:
            screening_result["recommendations"].append("Obtain export license before transfer")
        
        if screening_result["risk_level"] == "high":
            screening_result["recommendations"].append("Conduct enhanced due diligence")
        
        return len(screening_result["matches"]) == 0, screening_result
    
    def classify_technology(self, technology_description: str) -> Dict[str, Any]:
        """Classify technology for export control purposes"""
        classification = {
            "description": technology_description,
            "eccn": "EAR99",  # Default classification
            "jurisdiction": "EAR",
            "license_required": False,
            "reasoning": "General purpose software"
        }
        
        # Simplified classification logic
        description_lower = technology_description.lower()
        
        if "cryptography" in description_lower or "encryption" in description_lower:
            classification.update({
                "eccn": "5D002",
                "license_required": True,
                "reasoning": "Contains cryptographic functionality"
            })
        elif "artificial intelligence" in description_lower or "machine learning" in description_lower:
            classification.update({
                "eccn": "3E001",
                "license_required": True,
                "reasoning": "AI/ML technology with potential dual-use"
            })
        elif "protein" in description_lower and "modeling" in description_lower:
            classification.update({
                "eccn": "1E001",
                "license_required": False,
                "reasoning": "Research software for protein modeling"
            })
        
        return classification

class DataRetentionManager:
    """Manage data retention policies and enforcement"""
    
    def __init__(self):
        self.policies: Dict[str, DataRetentionPolicy] = {}
        self.retention_schedule: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Create default policies
        self._create_default_policies()
        
        # Start automated retention enforcement
        self._start_retention_enforcement()
    
    def _create_default_policies(self):
        """Create default retention policies for common jurisdictions"""
        
        # GDPR policy
        gdpr_policy = DataRetentionPolicy(
            policy_id="gdpr_personal_data",
            name="GDPR Personal Data Retention",
            jurisdiction=LegalJurisdiction.EUROPEAN_UNION,
            data_category=DataRetentionCategory.PERSONAL_DATA,
            retention_period_days=365,
            legal_basis="Data protection law compliance",
            destruction_method="Secure deletion with verification",
            review_period_days=180,
            automated_deletion=True,
            notification_required=True
        )
        
        # US research data policy
        us_research_policy = DataRetentionPolicy(
            policy_id="us_research_data",
            name="US Research Data Retention",
            jurisdiction=LegalJurisdiction.UNITED_STATES,
            data_category=DataRetentionCategory.RESEARCH_DATA,
            retention_period_days=2555,  # 7 years
            legal_basis="Federal research regulations",
            destruction_method="Secure deletion",
            review_period_days=365,
            automated_deletion=False,  # Requires review
            archive_before_deletion=True
        )
        
        # Audit logs policy
        audit_policy = DataRetentionPolicy(
            policy_id="global_audit_logs",
            name="Global Audit Logs Retention",
            jurisdiction=LegalJurisdiction.UNITED_STATES,  # Default
            data_category=DataRetentionCategory.AUDIT_LOGS,
            retention_period_days=2555,  # 7 years
            legal_basis="Audit and compliance requirements",
            destruction_method="Secure deletion with audit trail",
            automated_deletion=True,
            audit_trail_required=True
        )
        
        with self._lock:
            self.policies[gdpr_policy.policy_id] = gdpr_policy
            self.policies[us_research_policy.policy_id] = us_research_policy
            self.policies[audit_policy.policy_id] = audit_policy
    
    def _start_retention_enforcement(self):
        """Start automated retention enforcement"""
        def enforcement_loop():
            while True:
                try:
                    self._enforce_retention_policies()
                    time.sleep(24 * 3600)  # Check daily
                except Exception as e:
                    logger.error(f"Error in retention enforcement: {e}")
        
        thread = threading.Thread(target=enforcement_loop, daemon=True)
        thread.start()
    
    def add_policy(self, policy: DataRetentionPolicy):
        """Add retention policy"""
        with self._lock:
            self.policies[policy.policy_id] = policy
        
        logger.info(f"Added retention policy: {policy.policy_id}")
    
    def schedule_retention(self, policy_id: str, data_id: str, 
                         data_timestamp: float, metadata: Optional[Dict[str, Any]] = None):
        """Schedule data for retention management"""
        
        policy = self.policies.get(policy_id)
        if not policy:
            logger.error(f"Unknown retention policy: {policy_id}")
            return
        
        expiry_timestamp = data_timestamp + (policy.retention_period_days * 24 * 3600)
        
        retention_entry = {
            "data_id": data_id,
            "policy_id": policy_id,
            "data_timestamp": data_timestamp,
            "expiry_timestamp": expiry_timestamp,
            "metadata": metadata or {},
            "scheduled_at": time.time(),
            "status": "scheduled"
        }
        
        with self._lock:
            self.retention_schedule[policy_id].append(retention_entry)
        
        logger.info(f"Scheduled retention for data {data_id} under policy {policy_id}")
    
    def _enforce_retention_policies(self):
        """Enforce retention policies"""
        current_time = time.time()
        enforced_count = 0
        
        with self._lock:
            for policy_id, entries in self.retention_schedule.items():
                policy = self.policies.get(policy_id)
                if not policy:
                    continue
                
                expired_entries = []
                for entry in entries:
                    if entry["expiry_timestamp"] <= current_time and entry["status"] == "scheduled":
                        if policy.automated_deletion:
                            # Perform deletion
                            self._delete_data(entry, policy)
                            entry["status"] = "deleted"
                            entry["deleted_at"] = current_time
                            expired_entries.append(entry)
                            enforced_count += 1
                        else:
                            # Mark for manual review
                            entry["status"] = "review_required"
                            entry["review_due"] = current_time
        
        if enforced_count > 0:
            logger.info(f"Enforced retention policies for {enforced_count} data items")
    
    def _delete_data(self, entry: Dict[str, Any], policy: DataRetentionPolicy):
        """Delete data according to policy"""
        # This would integrate with actual data storage systems
        logger.info(f"Deleting data {entry['data_id']} per policy {policy.policy_id}")
        
        # Create audit trail
        audit_entry = {
            "action": "data_deletion",
            "data_id": entry["data_id"],
            "policy_id": policy.policy_id,
            "deletion_method": policy.destruction_method,
            "timestamp": time.time(),
            "verification_hash": hashlib.sha256(f"{entry['data_id']}{time.time()}".encode()).hexdigest()
        }
        
        # Log audit entry (would be stored in audit system)
        logger.info(f"Audit: {audit_entry}")
    
    def get_retention_report(self) -> Dict[str, Any]:
        """Generate retention compliance report"""
        
        with self._lock:
            total_policies = len(self.policies)
            total_scheduled = sum(len(entries) for entries in self.retention_schedule.values())
            
            status_counts = defaultdict(int)
            policy_stats = {}
            
            for policy_id, entries in self.retention_schedule.items():
                policy_stats[policy_id] = {
                    "total_entries": len(entries),
                    "by_status": defaultdict(int)
                }
                
                for entry in entries:
                    status_counts[entry["status"]] += 1
                    policy_stats[policy_id]["by_status"][entry["status"]] += 1
        
        return {
            "timestamp": time.time(),
            "total_policies": total_policies,
            "total_scheduled_items": total_scheduled,
            "status_distribution": dict(status_counts),
            "policy_statistics": {
                pid: {
                    "total_entries": stats["total_entries"],
                    "by_status": dict(stats["by_status"])
                }
                for pid, stats in policy_stats.items()
            },
            "policies": [
                {
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "jurisdiction": policy.jurisdiction.value,
                    "retention_days": policy.retention_period_days,
                    "automated": policy.automated_deletion
                }
                for policy in self.policies.values()
            ]
        }

class LegalComplianceManager:
    """Main legal compliance management system"""
    
    def __init__(self):
        self.export_control = ExportControlScreening()
        self.retention_manager = DataRetentionManager()
        
        # Legal documents and agreements
        self.legal_documents: Dict[str, LegalDocument] = {}
        self.audit_requirements: Dict[str, AuditRequirement] = {}
        self.transfer_agreements: Dict[str, CrossBorderTransferAgreement] = {}
        self.legal_incidents: List[LegalIncident] = []
        
        self._lock = threading.RLock()
        
        # Load default compliance frameworks
        self._load_default_frameworks()
        
        logger.info("Legal compliance manager initialized")
    
    def _load_default_frameworks(self):
        """Load default legal compliance frameworks"""
        
        # GDPR audit requirement
        gdpr_audit = AuditRequirement(
            requirement_id="gdpr_audit",
            name="GDPR Compliance Audit",
            jurisdiction=LegalJurisdiction.EUROPEAN_UNION,
            regulation="GDPR Article 35",
            frequency="annual",
            scope=["data_processing", "privacy_controls", "breach_procedures"],
            auditor_requirements={"certification": "CISA", "independence": True},
            documentation_required=["DPO_reports", "breach_logs", "consent_records"],
            retention_period_years=7,
            notification_period_days=30,
            remediation_period_days=90,
            penalties={"fine": "up to 4% of annual revenue", "sanctions": "processing restrictions"}
        )
        
        # SOX audit requirement
        sox_audit = AuditRequirement(
            requirement_id="sox_audit",
            name="Sarbanes-Oxley Compliance Audit",
            jurisdiction=LegalJurisdiction.UNITED_STATES,
            regulation="SOX Section 404",
            frequency="annual",
            scope=["financial_controls", "IT_controls", "data_integrity"],
            auditor_requirements={"certification": "CPA", "independence": True},
            documentation_required=["internal_controls", "risk_assessments", "financial_reports"],
            retention_period_years=7,
            notification_period_days=15,
            remediation_period_days=60,
            penalties={"fine": "up to $5M", "criminal": "up to 20 years imprisonment"}
        )
        
        with self._lock:
            self.audit_requirements[gdpr_audit.requirement_id] = gdpr_audit
            self.audit_requirements[sox_audit.requirement_id] = sox_audit
    
    def screen_export(self, technology: str, destination_country: str, 
                     end_user: str) -> Tuple[bool, Dict[str, Any]]:
        """Screen export for compliance"""
        
        # Classify technology
        classification = self.export_control.classify_technology(technology)
        
        # Screen party
        party_approved, screening_result = self.export_control.screen_party(end_user, destination_country)
        
        # Combine results
        export_approved = party_approved and not classification["license_required"]
        
        result = {
            "export_approved": export_approved,
            "technology_classification": classification,
            "party_screening": screening_result,
            "recommendations": [],
            "required_actions": []
        }
        
        if not party_approved:
            result["required_actions"].append("Cannot export to restricted party")
        
        if classification["license_required"]:
            result["required_actions"].append(f"Obtain export license ({classification['eccn']})")
        
        if not export_approved:
            result["recommendations"].extend([
                "Consult with export control compliance team",
                "Review applicable regulations",
                "Consider alternative destinations or technologies"
            ])
        
        return export_approved, result
    
    def add_legal_document(self, document: LegalDocument):
        """Add legal document"""
        with self._lock:
            self.legal_documents[document.document_id] = document
        
        logger.info(f"Added legal document: {document.document_id}")
    
    def add_transfer_agreement(self, agreement: CrossBorderTransferAgreement):
        """Add cross-border transfer agreement"""
        with self._lock:
            self.transfer_agreements[agreement.agreement_id] = agreement
        
        logger.info(f"Added transfer agreement: {agreement.agreement_id}")
    
    def report_legal_incident(self, incident: LegalIncident):
        """Report legal incident or violation"""
        with self._lock:
            self.legal_incidents.append(incident)
        
        # Trigger notifications if required
        if incident.severity in ["high", "critical"]:
            self._handle_critical_incident(incident)
        
        logger.warning(f"Legal incident reported: {incident.incident_id} ({incident.severity})")
    
    def _handle_critical_incident(self, incident: LegalIncident):
        """Handle critical legal incident"""
        # This would trigger immediate notifications and response procedures
        logger.critical(f"Critical legal incident requires immediate attention: {incident.incident_id}")
        
        # Auto-assign to legal team (in practice, would integrate with ticketing system)
        incident.assigned_to = "legal_team"
        incident.status = "investigating"
    
    def validate_cross_border_transfer(self, from_country: str, to_country: str,
                                     data_type: str) -> Tuple[bool, List[str]]:
        """Validate cross-border data transfer"""
        
        # Check for applicable transfer agreements
        applicable_agreements = []
        
        with self._lock:
            for agreement in self.transfer_agreements.values():
                if (agreement.sender_jurisdiction.value == from_country and
                    agreement.recipient_jurisdiction.value == to_country and
                    data_type in agreement.data_categories):
                    applicable_agreements.append(agreement)
        
        if not applicable_agreements:
            return False, [f"No valid transfer agreement for {data_type} from {from_country} to {to_country}"]
        
        # Check agreement validity
        current_time = time.time()
        valid_agreements = []
        
        for agreement in applicable_agreements:
            if agreement.expiry_date and current_time > agreement.expiry_date:
                continue
            valid_agreements.append(agreement)
        
        if not valid_agreements:
            return False, ["Transfer agreements have expired"]
        
        return True, [f"Transfer authorized under agreement: {valid_agreements[0].agreement_id}"]
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status"""
        
        current_time = time.time()
        
        with self._lock:
            # Check audit compliance
            overdue_audits = []
            upcoming_audits = []
            
            for audit in self.audit_requirements.values():
                if audit.next_audit_due and audit.next_audit_due < current_time:
                    overdue_audits.append(audit.requirement_id)
                elif audit.next_audit_due and audit.next_audit_due < current_time + (30 * 24 * 3600):
                    upcoming_audits.append(audit.requirement_id)
            
            # Check document status
            expiring_documents = []
            for doc in self.legal_documents.values():
                if doc.expiry_date and doc.expiry_date < current_time + (90 * 24 * 3600):
                    expiring_documents.append(doc.document_id)
            
            # Check incident status
            open_incidents = [i for i in self.legal_incidents if i.status == "open"]
            critical_incidents = [i for i in open_incidents if i.severity == "critical"]
        
        return {
            "timestamp": current_time,
            "overall_status": "compliant" if not overdue_audits and not critical_incidents else "non_compliant",
            "audit_compliance": {
                "overdue_audits": overdue_audits,
                "upcoming_audits": upcoming_audits,
                "total_requirements": len(self.audit_requirements)
            },
            "document_management": {
                "expiring_documents": expiring_documents,
                "total_documents": len(self.legal_documents)
            },
            "incident_management": {
                "open_incidents": len(open_incidents),
                "critical_incidents": len(critical_incidents),
                "total_incidents": len(self.legal_incidents)
            },
            "export_control": {
                "screening_enabled": True,
                "technology_classification_enabled": True
            },
            "data_retention": self.retention_manager.get_retention_report()
        }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        status = self.get_compliance_status()
        
        with self._lock:
            # Add detailed statistics
            jurisdiction_coverage = defaultdict(int)
            for req in self.audit_requirements.values():
                jurisdiction_coverage[req.jurisdiction.value] += 1
            
            agreement_stats = {
                "total_agreements": len(self.transfer_agreements),
                "by_type": defaultdict(int),
                "by_jurisdiction_pair": defaultdict(int)
            }
            
            for agreement in self.transfer_agreements.values():
                agreement_stats["by_type"][agreement.agreement_type] += 1
                pair = f"{agreement.sender_jurisdiction.value}-{agreement.recipient_jurisdiction.value}"
                agreement_stats["by_jurisdiction_pair"][pair] += 1
        
        report = {
            **status,
            "detailed_analysis": {
                "jurisdiction_coverage": dict(jurisdiction_coverage),
                "transfer_agreements": {
                    **agreement_stats,
                    "by_type": dict(agreement_stats["by_type"]),
                    "by_jurisdiction_pair": dict(agreement_stats["by_jurisdiction_pair"])
                },
                "export_control_classifications": len(self.export_control.controlled_items),
                "restricted_parties_lists": len(self.export_control.restricted_parties)
            },
            "recommendations": self._generate_compliance_recommendations()
        }
        
        return report
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        status = self.get_compliance_status()
        
        if status["audit_compliance"]["overdue_audits"]:
            recommendations.append("Schedule overdue audits immediately")
        
        if status["document_management"]["expiring_documents"]:
            recommendations.append("Review and renew expiring legal documents")
        
        if status["incident_management"]["critical_incidents"] > 0:
            recommendations.append("Address critical legal incidents with priority")
        
        if len(self.transfer_agreements) == 0:
            recommendations.append("Establish cross-border data transfer agreements")
        
        recommendations.extend([
            "Conduct regular export control training",
            "Review data retention policies annually",
            "Implement automated compliance monitoring",
            "Establish legal incident response procedures"
        ])
        
        return recommendations

# Global legal compliance manager
_global_legal_manager: Optional[LegalComplianceManager] = None

def get_legal_manager() -> Optional[LegalComplianceManager]:
    """Get global legal compliance manager"""
    return _global_legal_manager

def initialize_legal_compliance() -> LegalComplianceManager:
    """Initialize global legal compliance manager"""
    global _global_legal_manager
    _global_legal_manager = LegalComplianceManager()
    return _global_legal_manager

# Convenience functions
def screen_export_request(technology: str, destination: str, end_user: str) -> Tuple[bool, Dict[str, Any]]:
    """Screen export request"""
    if _global_legal_manager:
        return _global_legal_manager.screen_export(technology, destination, end_user)
    return True, {"message": "Export control not initialized"}

def validate_data_transfer(from_country: str, to_country: str, data_type: str) -> Tuple[bool, List[str]]:
    """Validate cross-border data transfer"""
    if _global_legal_manager:
        return _global_legal_manager.validate_cross_border_transfer(from_country, to_country, data_type)
    return True, ["Legal compliance not initialized"]

def schedule_data_retention(policy_id: str, data_id: str, data_timestamp: float, metadata: Optional[Dict[str, Any]] = None):
    """Schedule data for retention management"""
    if _global_legal_manager:
        _global_legal_manager.retention_manager.schedule_retention(policy_id, data_id, data_timestamp, metadata)