"""
Comprehensive Compliance Framework for protein-sssl-operator
Provides GDPR, CCPA, HIPAA, and other regulatory compliance features
including data protection, privacy controls, and audit capabilities.
"""

import time
import logging
import json
import hashlib
import threading
import uuid
from typing import (
    Dict, List, Optional, Any, Union, Callable, 
    Set, Tuple, NamedTuple, Protocol
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
import functools

logger = logging.getLogger(__name__)

class ComplianceRegulation(Enum):
    """Supported compliance regulations"""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    SOC2 = "soc2"  # SOC 2 Compliance Framework
    ISO_27001 = "iso_27001"  # Information Security Management
    SOX = "sox"  # Sarbanes-Oxley Act (US)
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    NIST = "nist"  # NIST Cybersecurity Framework
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    CCPA_CPRA = "ccpa_cpra"  # California Privacy Rights Act (CPRA)

class DataCategory(Enum):
    """Categories of data for compliance"""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"
    HEALTH_DATA = "health_data"
    BIOMETRIC_DATA = "biometric_data"
    GENETIC_DATA = "genetic_data"
    FINANCIAL_DATA = "financial_data"
    RESEARCH_DATA = "research_data"
    PROTEIN_SEQUENCE_DATA = "protein_sequence_data"
    STRUCTURAL_DATA = "structural_data"
    METADATA = "metadata"

class ProcessingPurpose(Enum):
    """Legal basis for data processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    RESEARCH = "research"
    STATISTICAL = "statistical"

class DataSubjectRight(Enum):
    """Data subject rights under various regulations"""
    ACCESS = "access"  # Right to access personal data
    RECTIFICATION = "rectification"  # Right to correct inaccurate data
    ERASURE = "erasure"  # Right to be forgotten
    PORTABILITY = "portability"  # Right to data portability
    RESTRICTION = "restriction"  # Right to restrict processing
    OBJECTION = "objection"  # Right to object to processing
    WITHDRAW_CONSENT = "withdraw_consent"  # Right to withdraw consent
    OPT_OUT = "opt_out"  # Right to opt out (CCPA)
    DELETE = "delete"  # Right to delete (CCPA)
    KNOW = "know"  # Right to know (CCPA)

@dataclass
class DataSubject:
    """Data subject (individual) information"""
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    jurisdiction: Optional[str] = None  # EU, CA, etc.
    created_at: float = field(default_factory=time.time)
    consent_status: Dict[str, bool] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def has_consent(self, purpose: str) -> bool:
        """Check if subject has given consent for specific purpose"""
        return self.consent_status.get(purpose, False)
    
    def give_consent(self, purpose: str, timestamp: Optional[float] = None):
        """Record consent for specific purpose"""
        self.consent_status[purpose] = True
        self.preferences[f"{purpose}_consent_date"] = timestamp or time.time()
    
    def withdraw_consent(self, purpose: str, timestamp: Optional[float] = None):
        """Withdraw consent for specific purpose"""
        self.consent_status[purpose] = False
        self.preferences[f"{purpose}_withdrawal_date"] = timestamp or time.time()

@dataclass
class DataProcessingRecord:
    """Record of data processing activity"""
    id: str
    data_subject_id: str
    data_categories: List[DataCategory]
    processing_purposes: List[ProcessingPurpose]
    legal_basis: ProcessingPurpose
    processor: str  # Who processed the data
    timestamp: float
    retention_period: Optional[int] = None  # Days
    cross_border_transfer: bool = False
    transfer_countries: List[str] = field(default_factory=list)
    data_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'data_subject_id': self.data_subject_id,
            'data_categories': [cat.value for cat in self.data_categories],
            'processing_purposes': [purpose.value for purpose in self.processing_purposes],
            'legal_basis': self.legal_basis.value,
            'processor': self.processor,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'retention_period': self.retention_period,
            'cross_border_transfer': self.cross_border_transfer,
            'transfer_countries': self.transfer_countries,
            'data_hash': self.data_hash,
            'metadata': self.metadata
        }

@dataclass
class DataSubjectRequest:
    """Data subject rights request"""
    id: str
    data_subject_id: str
    request_type: DataSubjectRight
    description: str
    timestamp: float
    status: str = "pending"  # pending, in_progress, completed, rejected
    response_due_date: Optional[float] = None
    completed_at: Optional[float] = None
    processor: Optional[str] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'data_subject_id': self.data_subject_id,
            'request_type': self.request_type.value,
            'description': self.description,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'status': self.status,
            'response_due_date': self.response_due_date,
            'completed_at': self.completed_at,
            'processor': self.processor,
            'notes': self.notes,
            'metadata': self.metadata
        }

@dataclass
class CompliancePolicy:
    """Compliance policy configuration"""
    name: str
    regulations: List[ComplianceRegulation]
    data_retention_days: int = 365
    consent_required_purposes: List[str] = field(default_factory=list)
    automated_deletion_enabled: bool = True
    cross_border_transfers_allowed: bool = False
    allowed_transfer_countries: List[str] = field(default_factory=list)
    data_protection_officer_email: Optional[str] = None
    breach_notification_required: bool = True
    breach_notification_hours: int = 72
    data_subject_response_days: int = 30
    pseudonymization_enabled: bool = True
    encryption_required: bool = True
    audit_logging_enabled: bool = True
    
    def is_regulation_applicable(self, regulation: ComplianceRegulation) -> bool:
        """Check if regulation applies to this policy"""
        return regulation in self.regulations
    
    def requires_consent(self, purpose: str) -> bool:
        """Check if consent is required for purpose"""
        return purpose in self.consent_required_purposes

class GDPRCompliance:
    """GDPR-specific compliance implementation"""
    
    @staticmethod
    def validate_legal_basis(purpose: ProcessingPurpose, 
                           has_consent: bool = False,
                           is_contract_necessary: bool = False,
                           is_legal_obligation: bool = False,
                           is_vital_interest: bool = False,
                           is_public_task: bool = False,
                           is_legitimate_interest: bool = False) -> bool:
        """Validate GDPR legal basis for processing"""
        
        if purpose == ProcessingPurpose.CONSENT:
            return has_consent
        elif purpose == ProcessingPurpose.CONTRACT:
            return is_contract_necessary
        elif purpose == ProcessingPurpose.LEGAL_OBLIGATION:
            return is_legal_obligation
        elif purpose == ProcessingPurpose.VITAL_INTERESTS:
            return is_vital_interest
        elif purpose == ProcessingPurpose.PUBLIC_TASK:
            return is_public_task
        elif purpose == ProcessingPurpose.LEGITIMATE_INTERESTS:
            return is_legitimate_interest
        
        return False
    
    @staticmethod
    def calculate_response_deadline(request_date: float) -> float:
        """Calculate GDPR response deadline (30 days)"""
        return request_date + (30 * 24 * 3600)
    
    @staticmethod
    def is_cross_border_transfer_allowed(from_country: str, to_country: str) -> bool:
        """Check if cross-border data transfer is allowed under GDPR"""
        # EU/EEA countries
        eu_countries = {
            'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
            'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
            'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'IS', 'LI', 'NO'
        }
        
        # Adequacy decisions
        adequate_countries = {
            'AD', 'AR', 'CA', 'FO', 'GG', 'IL', 'IM', 'JP', 'JE', 'NZ',
            'CH', 'UY', 'GB'  # As of 2021
        }
        
        # Within EU/EEA is always allowed
        if from_country in eu_countries and to_country in eu_countries:
            return True
        
        # To adequate countries is allowed
        if to_country in adequate_countries:
            return True
        
        # Otherwise requires appropriate safeguards
        return False

class CCPACompliance:
    """CCPA-specific compliance implementation"""
    
    @staticmethod
    def is_california_resident(data_subject: DataSubject) -> bool:
        """Check if data subject is California resident"""
        return data_subject.jurisdiction == 'CA'
    
    @staticmethod
    def calculate_response_deadline(request_date: float) -> float:
        """Calculate CCPA response deadline (45 days)"""
        return request_date + (45 * 24 * 3600)
    
    @staticmethod
    def is_personal_information_sale_allowed(data_subject: DataSubject) -> bool:
        """Check if personal information sale is allowed"""
        return not data_subject.preferences.get('opt_out_of_sale', False)

class HIPAACompliance:
    """HIPAA-specific compliance implementation"""
    
    @staticmethod
    def is_phi_data(data_categories: List[DataCategory]) -> bool:
        """Check if data contains Protected Health Information"""
        phi_categories = {
            DataCategory.HEALTH_DATA,
            DataCategory.BIOMETRIC_DATA,
            DataCategory.GENETIC_DATA
        }
        
        return any(cat in phi_categories for cat in data_categories)
    
    @staticmethod
    def requires_business_associate_agreement(processor: str, 
                                            covered_entities: List[str]) -> bool:
        """Check if Business Associate Agreement is required"""
        return processor not in covered_entities
    
    @staticmethod
    def calculate_breach_notification_deadline(breach_date: float) -> float:
        """Calculate HIPAA breach notification deadline (60 days)"""
        return breach_date + (60 * 24 * 3600)

class PDPACompliance:
    """PDPA-specific compliance implementation (Singapore/Thailand)"""
    
    @staticmethod
    def is_singapore_resident(data_subject: DataSubject) -> bool:
        """Check if data subject is Singapore resident"""
        return data_subject.jurisdiction == 'SG'
    
    @staticmethod
    def is_thailand_resident(data_subject: DataSubject) -> bool:
        """Check if data subject is Thailand resident"""
        return data_subject.jurisdiction == 'TH'
    
    @staticmethod
    def calculate_response_deadline(request_date: float, jurisdiction: str = 'SG') -> float:
        """Calculate PDPA response deadline (30 days SG, 30 days TH)"""
        return request_date + (30 * 24 * 3600)
    
    @staticmethod
    def requires_consent_withdrawal_mechanism(data_subject: DataSubject) -> bool:
        """Check if consent withdrawal mechanism is required"""
        return True  # Always required under PDPA
    
    @staticmethod
    def is_cross_border_transfer_allowed(from_country: str, to_country: str) -> bool:
        """Check if cross-border data transfer is allowed under PDPA"""
        # PDPA requires adequate protection or consent
        adequate_countries = {
            'SG', 'EU', 'GB', 'CA', 'AU', 'NZ', 'JP', 'KR'
        }
        
        if to_country in adequate_countries:
            return True
        
        # Otherwise requires consent or contractual safeguards
        return False

class SOC2Compliance:
    """SOC 2 compliance implementation"""
    
    TRUST_CRITERIA = [
        'security',
        'availability', 
        'processing_integrity',
        'confidentiality',
        'privacy'
    ]
    
    @staticmethod
    def validate_security_controls(controls: Dict[str, bool]) -> Tuple[bool, List[str]]:
        """Validate SOC 2 security controls"""
        required_controls = [
            'access_control',
            'encryption_at_rest',
            'encryption_in_transit',
            'vulnerability_management',
            'incident_response',
            'change_management',
            'monitoring_logging'
        ]
        
        missing_controls = []
        for control in required_controls:
            if not controls.get(control, False):
                missing_controls.append(control)
        
        return len(missing_controls) == 0, missing_controls
    
    @staticmethod
    def validate_availability_controls(uptime_percentage: float, 
                                     monitoring_enabled: bool) -> bool:
        """Validate availability controls"""
        return uptime_percentage >= 99.0 and monitoring_enabled
    
    @staticmethod
    def validate_processing_integrity(data_validation: bool,
                                    error_handling: bool,
                                    data_backup: bool) -> bool:
        """Validate processing integrity controls"""
        return all([data_validation, error_handling, data_backup])

class ISO27001Compliance:
    """ISO 27001 compliance implementation"""
    
    CONTROL_DOMAINS = [
        'information_security_policies',
        'organization_of_information_security',
        'human_resource_security',
        'asset_management',
        'access_control',
        'cryptography',
        'physical_environmental_security',
        'operations_security',
        'communications_security',
        'system_acquisition_development_maintenance',
        'supplier_relationships',
        'information_security_incident_management',
        'information_security_business_continuity',
        'compliance'
    ]
    
    @staticmethod
    def validate_isms_controls(controls: Dict[str, Dict[str, bool]]) -> Tuple[bool, List[str]]:
        """Validate Information Security Management System controls"""
        missing_controls = []
        
        for domain in ISO27001Compliance.CONTROL_DOMAINS:
            if domain not in controls:
                missing_controls.append(f"Missing domain: {domain}")
                continue
            
            domain_controls = controls[domain]
            if not all(domain_controls.values()):
                failed_controls = [k for k, v in domain_controls.items() if not v]
                missing_controls.extend([f"{domain}.{control}" for control in failed_controls])
        
        return len(missing_controls) == 0, missing_controls
    
    @staticmethod
    def validate_risk_assessment(risk_register: Dict[str, Any]) -> bool:
        """Validate risk assessment requirements"""
        required_fields = ['threats', 'vulnerabilities', 'impacts', 'likelihood', 'controls']
        return all(field in risk_register for field in required_fields)
    
    @staticmethod
    def validate_incident_management(incident_procedures: Dict[str, bool]) -> bool:
        """Validate incident management procedures"""
        required_procedures = [
            'detection_reporting',
            'classification',
            'response_procedures',
            'escalation_procedures',
            'evidence_collection',
            'recovery_procedures',
            'lessons_learned'
        ]
        
        return all(incident_procedures.get(proc, False) for proc in required_procedures)

class LGPDCompliance:
    """LGPD compliance implementation (Brazil)"""
    
    @staticmethod
    def is_brazil_resident(data_subject: DataSubject) -> bool:
        """Check if data subject is Brazil resident"""
        return data_subject.jurisdiction == 'BR'
    
    @staticmethod
    def calculate_response_deadline(request_date: float) -> float:
        """Calculate LGPD response deadline (15 days)"""
        return request_date + (15 * 24 * 3600)
    
    @staticmethod
    def validate_legal_basis(purpose: ProcessingPurpose,
                           has_consent: bool = False,
                           is_legitimate_interest: bool = False,
                           is_legal_obligation: bool = False) -> bool:
        """Validate LGPD legal basis for processing"""
        # Similar to GDPR but with Brazilian-specific requirements
        if purpose == ProcessingPurpose.CONSENT:
            return has_consent
        elif purpose == ProcessingPurpose.LEGITIMATE_INTERESTS:
            return is_legitimate_interest
        elif purpose == ProcessingPurpose.LEGAL_OBLIGATION:
            return is_legal_obligation
        
        return False

class PIPEDACompliance:
    """PIPEDA compliance implementation (Canada)"""
    
    @staticmethod
    def is_canada_resident(data_subject: DataSubject) -> bool:
        """Check if data subject is Canada resident"""
        return data_subject.jurisdiction == 'CA'
    
    @staticmethod
    def calculate_response_deadline(request_date: float) -> float:
        """Calculate PIPEDA response deadline (30 days)"""
        return request_date + (30 * 24 * 3600)
    
    @staticmethod
    def validate_privacy_principles(principles: Dict[str, bool]) -> Tuple[bool, List[str]]:
        """Validate PIPEDA privacy principles"""
        required_principles = [
            'accountability',
            'identifying_purposes',
            'consent',
            'limiting_collection',
            'limiting_use_disclosure_retention',
            'accuracy',
            'safeguards',
            'openness',
            'individual_access',
            'challenging_compliance'
        ]
        
        missing_principles = []
        for principle in required_principles:
            if not principles.get(principle, False):
                missing_principles.append(principle)
        
        return len(missing_principles) == 0, missing_principles

class ComplianceManager:
    """Main compliance management system"""
    
    def __init__(self, policy: CompliancePolicy):
        self.policy = policy
        
        # Data storage
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.subject_requests: List[DataSubjectRequest] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Compliance implementations
        self.gdpr = GDPRCompliance()
        self.ccpa = CCPACompliance()
        self.hipaa = HIPAACompliance()
        self.pdpa = PDPACompliance()
        self.soc2 = SOC2Compliance()
        self.iso27001 = ISO27001Compliance()
        self.lgpd = LGPDCompliance()
        self.pipeda = PIPEDACompliance()
        
        # Automated processes
        self._start_automated_processes()
        
        logger.info(f"Compliance manager initialized for {[r.value for r in policy.regulations]}")
    
    def _start_automated_processes(self):
        """Start automated compliance processes"""
        if self.policy.automated_deletion_enabled:
            self._start_automated_deletion()
    
    def _start_automated_deletion(self):
        """Start automated data deletion process"""
        def deletion_loop():
            while True:
                try:
                    self._process_automated_deletions()
                    time.sleep(24 * 3600)  # Check daily
                except Exception as e:
                    logger.error(f"Error in automated deletion: {e}")
        
        thread = threading.Thread(target=deletion_loop, daemon=True)
        thread.start()
    
    def register_data_subject(self, 
                            email: str,
                            name: Optional[str] = None,
                            jurisdiction: Optional[str] = None) -> DataSubject:
        """Register new data subject"""
        
        subject_id = str(uuid.uuid4())
        subject = DataSubject(
            id=subject_id,
            email=email,
            name=name,
            jurisdiction=jurisdiction
        )
        
        with self._lock:
            self.data_subjects[subject_id] = subject
        
        logger.info(f"Registered data subject: {subject_id}")
        return subject
    
    def record_data_processing(self,
                             data_subject_id: str,
                             data_categories: List[DataCategory],
                             processing_purposes: List[ProcessingPurpose],
                             legal_basis: ProcessingPurpose,
                             processor: str,
                             data: Optional[bytes] = None,
                             retention_period: Optional[int] = None,
                             cross_border_transfer: bool = False,
                             transfer_countries: Optional[List[str]] = None) -> DataProcessingRecord:
        """Record data processing activity"""
        
        # Validate legal basis
        subject = self.data_subjects.get(data_subject_id)
        if not subject:
            raise ValueError(f"Data subject not found: {data_subject_id}")
        
        # Check consent if required
        for purpose in processing_purposes:
            if self.policy.requires_consent(purpose.value):
                if not subject.has_consent(purpose.value):
                    raise ValueError(f"Consent required for purpose: {purpose.value}")
        
        # Check cross-border transfers
        if cross_border_transfer and not self.policy.cross_border_transfers_allowed:
            raise ValueError("Cross-border transfers not allowed by policy")
        
        if transfer_countries:
            for country in transfer_countries:
                if country not in self.policy.allowed_transfer_countries:
                    # Check regulation-specific rules
                    if ComplianceRegulation.GDPR in self.policy.regulations:
                        if not self.gdpr.is_cross_border_transfer_allowed(subject.jurisdiction or 'EU', country):
                            raise ValueError(f"Cross-border transfer to {country} not allowed under GDPR")
        
        # Check HIPAA requirements
        if ComplianceRegulation.HIPAA in self.policy.regulations:
            if self.hipaa.is_phi_data(data_categories):
                logger.info("Processing PHI data - additional HIPAA safeguards required")
        
        # Create processing record
        record_id = str(uuid.uuid4())
        data_hash = None
        
        if data:
            data_hash = hashlib.sha256(data).hexdigest()
        
        record = DataProcessingRecord(
            id=record_id,
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            legal_basis=legal_basis,
            processor=processor,
            timestamp=time.time(),
            retention_period=retention_period or self.policy.data_retention_days,
            cross_border_transfer=cross_border_transfer,
            transfer_countries=transfer_countries or [],
            data_hash=data_hash
        )
        
        with self._lock:
            self.processing_records.append(record)
        
        logger.info(f"Recorded data processing: {record_id}")
        return record
    
    def handle_subject_request(self,
                             data_subject_id: str,
                             request_type: DataSubjectRight,
                             description: str) -> DataSubjectRequest:
        """Handle data subject rights request"""
        
        subject = self.data_subjects.get(data_subject_id)
        if not subject:
            raise ValueError(f"Data subject not found: {data_subject_id}")
        
        # Calculate response deadline based on applicable regulations
        response_due_date = None
        current_time = time.time()
        
        if ComplianceRegulation.GDPR in self.policy.regulations:
            response_due_date = self.gdpr.calculate_response_deadline(current_time)
        elif ComplianceRegulation.CCPA in self.policy.regulations:
            response_due_date = self.ccpa.calculate_response_deadline(current_time)
        else:
            # Default to policy setting
            response_due_date = current_time + (self.policy.data_subject_response_days * 24 * 3600)
        
        request_id = str(uuid.uuid4())
        request = DataSubjectRequest(
            id=request_id,
            data_subject_id=data_subject_id,
            request_type=request_type,
            description=description,
            timestamp=current_time,
            response_due_date=response_due_date
        )
        
        with self._lock:
            self.subject_requests.append(request)
        
        # Process certain requests automatically
        if request_type == DataSubjectRight.ACCESS:
            self._process_access_request(request)
        elif request_type == DataSubjectRight.ERASURE:
            self._process_erasure_request(request)
        elif request_type == DataSubjectRight.PORTABILITY:
            self._process_portability_request(request)
        
        logger.info(f"Created subject request: {request_id} ({request_type.value})")
        return request
    
    def _process_access_request(self, request: DataSubjectRequest):
        """Process data subject access request"""
        try:
            # Find all processing records for this subject
            subject_records = [
                record for record in self.processing_records
                if record.data_subject_id == request.data_subject_id
            ]
            
            # Compile data access report
            access_data = {
                'data_subject_id': request.data_subject_id,
                'personal_data_categories': list(set(
                    cat.value for record in subject_records 
                    for cat in record.data_categories
                )),
                'processing_purposes': list(set(
                    purpose.value for record in subject_records
                    for purpose in record.processing_purposes
                )),
                'processors': list(set(record.processor for record in subject_records)),
                'retention_periods': [record.retention_period for record in subject_records],
                'cross_border_transfers': any(record.cross_border_transfer for record in subject_records),
                'transfer_countries': list(set(
                    country for record in subject_records
                    for country in record.transfer_countries
                )),
                'processing_records': [record.to_dict() for record in subject_records]
            }
            
            # Update request
            request.status = "completed"
            request.completed_at = time.time()
            request.metadata['access_data'] = access_data
            
        except Exception as e:
            request.status = "failed"
            request.notes = f"Error processing access request: {e}"
            logger.error(f"Error processing access request {request.id}: {e}")
    
    def _process_erasure_request(self, request: DataSubjectRequest):
        """Process data subject erasure request"""
        try:
            # Check if erasure is possible (no legal obligation to retain)
            subject_records = [
                record for record in self.processing_records
                if record.data_subject_id == request.data_subject_id
            ]
            
            # Check for legal obligations to retain data
            cannot_erase_reasons = []
            
            for record in subject_records:
                if ProcessingPurpose.LEGAL_OBLIGATION in record.processing_purposes:
                    cannot_erase_reasons.append(f"Legal obligation: {record.id}")
                
                # Check if within retention period for legitimate interests
                if (record.legal_basis == ProcessingPurpose.LEGITIMATE_INTERESTS and
                    record.retention_period and
                    time.time() - record.timestamp < record.retention_period * 24 * 3600):
                    cannot_erase_reasons.append(f"Within retention period: {record.id}")
            
            if cannot_erase_reasons:
                request.status = "rejected"
                request.notes = f"Cannot erase data: {'; '.join(cannot_erase_reasons)}"
            else:
                # Perform erasure
                self._erase_subject_data(request.data_subject_id)
                request.status = "completed"
                request.completed_at = time.time()
            
        except Exception as e:
            request.status = "failed"
            request.notes = f"Error processing erasure request: {e}"
            logger.error(f"Error processing erasure request {request.id}: {e}")
    
    def _process_portability_request(self, request: DataSubjectRequest):
        """Process data portability request"""
        try:
            # Find all personal data for this subject
            subject_records = [
                record for record in self.processing_records
                if record.data_subject_id == request.data_subject_id
                and record.legal_basis in [ProcessingPurpose.CONSENT, ProcessingPurpose.CONTRACT]
            ]
            
            # Compile portable data in structured format
            portable_data = {
                'data_subject_id': request.data_subject_id,
                'export_timestamp': time.time(),
                'data_format': 'JSON',
                'personal_data': [record.to_dict() for record in subject_records]
            }
            
            request.status = "completed"
            request.completed_at = time.time()
            request.metadata['portable_data'] = portable_data
            
        except Exception as e:
            request.status = "failed"
            request.notes = f"Error processing portability request: {e}"
            logger.error(f"Error processing portability request {request.id}: {e}")
    
    def _erase_subject_data(self, data_subject_id: str):
        """Erase all data for a data subject"""
        with self._lock:
            # Remove processing records
            self.processing_records = [
                record for record in self.processing_records
                if record.data_subject_id != data_subject_id
            ]
            
            # Remove data subject record
            if data_subject_id in self.data_subjects:
                del self.data_subjects[data_subject_id]
        
        logger.info(f"Erased all data for subject: {data_subject_id}")
    
    def _process_automated_deletions(self):
        """Process automated data deletions based on retention policies"""
        current_time = time.time()
        deleted_count = 0
        
        with self._lock:
            records_to_delete = []
            
            for record in self.processing_records:
                # Check if retention period has expired
                if record.retention_period:
                    expiry_time = record.timestamp + (record.retention_period * 24 * 3600)
                    
                    if current_time > expiry_time:
                        # Check if there's a legal obligation to retain
                        if ProcessingPurpose.LEGAL_OBLIGATION not in record.processing_purposes:
                            records_to_delete.append(record)
            
            # Remove expired records
            for record in records_to_delete:
                self.processing_records.remove(record)
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Automatically deleted {deleted_count} expired data records")
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        current_time = time.time()
        
        with self._lock:
            # Calculate statistics
            total_subjects = len(self.data_subjects)
            total_processing_records = len(self.processing_records)
            total_requests = len(self.subject_requests)
            
            # Request statistics
            pending_requests = [r for r in self.subject_requests if r.status == "pending"]
            overdue_requests = [
                r for r in pending_requests
                if r.response_due_date and current_time > r.response_due_date
            ]
            
            # Data categories
            data_categories = defaultdict(int)
            for record in self.processing_records:
                for category in record.data_categories:
                    data_categories[category.value] += 1
            
            # Cross-border transfers
            cross_border_count = sum(1 for r in self.processing_records if r.cross_border_transfer)
            
            # Consent statistics
            consent_stats = defaultdict(int)
            for subject in self.data_subjects.values():
                for purpose, has_consent in subject.consent_status.items():
                    consent_stats[f"{purpose}_{'granted' if has_consent else 'withdrawn'}"] += 1
        
        return {
            'timestamp': current_time,
            'applicable_regulations': [r.value for r in self.policy.regulations],
            'data_subjects': {
                'total': total_subjects,
                'by_jurisdiction': self._count_by_jurisdiction()
            },
            'processing_records': {
                'total': total_processing_records,
                'by_category': dict(data_categories),
                'cross_border_transfers': cross_border_count
            },
            'subject_requests': {
                'total': total_requests,
                'pending': len(pending_requests),
                'overdue': len(overdue_requests),
                'by_type': self._count_requests_by_type(),
                'by_status': self._count_requests_by_status()
            },
            'consent_statistics': dict(consent_stats),
            'policy_compliance': {
                'data_retention_days': self.policy.data_retention_days,
                'automated_deletion_enabled': self.policy.automated_deletion_enabled,
                'encryption_required': self.policy.encryption_required,
                'audit_logging_enabled': self.policy.audit_logging_enabled
            }
        }
    
    def _count_by_jurisdiction(self) -> Dict[str, int]:
        """Count data subjects by jurisdiction"""
        counts = defaultdict(int)
        for subject in self.data_subjects.values():
            jurisdiction = subject.jurisdiction or 'unknown'
            counts[jurisdiction] += 1
        return dict(counts)
    
    def _count_requests_by_type(self) -> Dict[str, int]:
        """Count requests by type"""
        counts = defaultdict(int)
        for request in self.subject_requests:
            counts[request.request_type.value] += 1
        return dict(counts)
    
    def _count_requests_by_status(self) -> Dict[str, int]:
        """Count requests by status"""
        counts = defaultdict(int)
        for request in self.subject_requests:
            counts[request.status] += 1
        return dict(counts)
    
    def export_compliance_data(self, file_path: str, include_personal_data: bool = False) -> bool:
        """Export compliance data to file"""
        try:
            export_data = {
                'export_timestamp': time.time(),
                'policy': asdict(self.policy),
                'compliance_report': self.get_compliance_report()
            }
            
            if include_personal_data:
                with self._lock:
                    export_data['data_subjects'] = [
                        asdict(subject) for subject in self.data_subjects.values()
                    ]
                    export_data['processing_records'] = [
                        record.to_dict() for record in self.processing_records
                    ]
                    export_data['subject_requests'] = [
                        request.to_dict() for request in self.subject_requests
                    ]
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Compliance data exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting compliance data: {e}")
            return False
    
    def validate_compliance(self) -> Tuple[bool, List[str]]:
        """Validate current compliance status"""
        issues = []
        
        # Check overdue requests
        current_time = time.time()
        overdue_requests = [
            r for r in self.subject_requests
            if r.status == "pending" and r.response_due_date and current_time > r.response_due_date
        ]
        
        if overdue_requests:
            issues.append(f"{len(overdue_requests)} overdue data subject requests")
        
        # Check data retention compliance
        expired_records = []
        for record in self.processing_records:
            if record.retention_period:
                expiry_time = record.timestamp + (record.retention_period * 24 * 3600)
                if current_time > expiry_time:
                    # Check if there's a legal reason to retain
                    if ProcessingPurpose.LEGAL_OBLIGATION not in record.processing_purposes:
                        expired_records.append(record)
        
        if expired_records:
            issues.append(f"{len(expired_records)} records past retention period")
        
        # Check consent validity
        consent_issues = 0
        for record in self.processing_records:
            if record.legal_basis == ProcessingPurpose.CONSENT:
                subject = self.data_subjects.get(record.data_subject_id)
                if subject:
                    for purpose in record.processing_purposes:
                        if not subject.has_consent(purpose.value):
                            consent_issues += 1
        
        if consent_issues > 0:
            issues.append(f"{consent_issues} processing activities lack valid consent")
        
        return len(issues) == 0, issues

# Global compliance manager
_global_compliance_manager: Optional[ComplianceManager] = None

def get_global_compliance_manager() -> Optional[ComplianceManager]:
    """Get global compliance manager instance"""
    return _global_compliance_manager

def initialize_compliance(policy: CompliancePolicy) -> ComplianceManager:
    """Initialize global compliance manager"""
    global _global_compliance_manager
    _global_compliance_manager = ComplianceManager(policy)
    return _global_compliance_manager

# Decorators for compliance
def require_consent(purpose: str):
    """Decorator to require consent for data processing"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This would need integration with the compliance manager
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_data_processing(data_categories: List[DataCategory], 
                       purposes: List[ProcessingPurpose],
                       legal_basis: ProcessingPurpose):
    """Decorator to log data processing activities"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This would need integration with the compliance manager
            result = func(*args, **kwargs)
            
            if _global_compliance_manager:
                # Log the processing activity
                pass
            
            return result
        return wrapper
    return decorator

# Convenience functions
def create_gdpr_policy() -> CompliancePolicy:
    """Create GDPR compliance policy"""
    return CompliancePolicy(
        name="GDPR Policy",
        regulations=[ComplianceRegulation.GDPR],
        data_retention_days=365,
        consent_required_purposes=['research', 'marketing'],
        automated_deletion_enabled=True,
        cross_border_transfers_allowed=True,
        allowed_transfer_countries=['US', 'CA', 'JP'],  # Adequacy decisions
        breach_notification_required=True,
        breach_notification_hours=72,
        data_subject_response_days=30,
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_ccpa_policy() -> CompliancePolicy:
    """Create CCPA compliance policy"""
    return CompliancePolicy(
        name="CCPA Policy",
        regulations=[ComplianceRegulation.CCPA],
        data_retention_days=365,
        consent_required_purposes=[],  # CCPA is opt-out based
        automated_deletion_enabled=True,
        cross_border_transfers_allowed=True,
        breach_notification_required=False,  # No breach notification requirement
        data_subject_response_days=45,
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_hipaa_policy() -> CompliancePolicy:
    """Create HIPAA compliance policy"""
    return CompliancePolicy(
        name="HIPAA Policy",
        regulations=[ComplianceRegulation.HIPAA],
        data_retention_days=2555,  # 7 years
        consent_required_purposes=['research', 'marketing'],
        automated_deletion_enabled=False,  # Manual review required
        cross_border_transfers_allowed=False,
        breach_notification_required=True,
        breach_notification_hours=1440,  # 60 days in hours
        data_subject_response_days=60,
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_pdpa_policy() -> CompliancePolicy:
    """Create PDPA compliance policy (Singapore/Thailand)"""
    return CompliancePolicy(
        name="PDPA Policy",
        regulations=[ComplianceRegulation.PDPA],
        data_retention_days=365,
        consent_required_purposes=['marketing', 'analytics'],
        automated_deletion_enabled=True,
        cross_border_transfers_allowed=True,
        allowed_transfer_countries=['SG', 'TH', 'EU', 'GB', 'CA', 'AU', 'NZ', 'JP', 'KR'],
        breach_notification_required=True,
        breach_notification_hours=72,  # 3 days
        data_subject_response_days=30,
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_soc2_policy() -> CompliancePolicy:
    """Create SOC 2 compliance policy"""
    return CompliancePolicy(
        name="SOC 2 Policy",
        regulations=[ComplianceRegulation.SOC2],
        data_retention_days=2555,  # 7 years for audit purposes
        consent_required_purposes=[],  # SOC 2 focuses on controls, not consent
        automated_deletion_enabled=False,  # Manual review for audit trails
        cross_border_transfers_allowed=True,
        breach_notification_required=True,
        breach_notification_hours=24,  # 1 day
        data_subject_response_days=30,
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_iso27001_policy() -> CompliancePolicy:
    """Create ISO 27001 compliance policy"""
    return CompliancePolicy(
        name="ISO 27001 Policy",
        regulations=[ComplianceRegulation.ISO_27001],
        data_retention_days=2555,  # 7 years for security records
        consent_required_purposes=[],  # ISO 27001 focuses on security controls
        automated_deletion_enabled=False,  # Manual review for security logs
        cross_border_transfers_allowed=True,
        breach_notification_required=True,
        breach_notification_hours=24,  # 1 day
        data_subject_response_days=30,
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_lgpd_policy() -> CompliancePolicy:
    """Create LGPD compliance policy (Brazil)"""
    return CompliancePolicy(
        name="LGPD Policy",
        regulations=[ComplianceRegulation.LGPD],
        data_retention_days=365,
        consent_required_purposes=['marketing', 'analytics'],
        automated_deletion_enabled=True,
        cross_border_transfers_allowed=True,
        allowed_transfer_countries=['BR', 'EU', 'GB', 'CA', 'US'],  # Countries with adequate protection
        breach_notification_required=True,
        breach_notification_hours=72,  # Similar to GDPR
        data_subject_response_days=15,  # Stricter than GDPR
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_pipeda_policy() -> CompliancePolicy:
    """Create PIPEDA compliance policy (Canada)"""
    return CompliancePolicy(
        name="PIPEDA Policy",
        regulations=[ComplianceRegulation.PIPEDA],
        data_retention_days=365,
        consent_required_purposes=['marketing', 'analytics'],
        automated_deletion_enabled=True,
        cross_border_transfers_allowed=True,
        allowed_transfer_countries=['CA', 'EU', 'GB', 'US'],
        breach_notification_required=True,
        breach_notification_hours=72,  # As soon as feasible
        data_subject_response_days=30,
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )

def create_multi_region_policy() -> CompliancePolicy:
    """Create multi-region compliance policy covering major regulations"""
    return CompliancePolicy(
        name="Multi-Region Policy",
        regulations=[
            ComplianceRegulation.GDPR,
            ComplianceRegulation.CCPA,
            ComplianceRegulation.PDPA,
            ComplianceRegulation.LGPD,
            ComplianceRegulation.PIPEDA,
            ComplianceRegulation.SOC2,
            ComplianceRegulation.ISO_27001
        ],
        data_retention_days=365,  # Conservative approach
        consent_required_purposes=['marketing', 'analytics', 'research'],
        automated_deletion_enabled=True,
        cross_border_transfers_allowed=True,
        allowed_transfer_countries=['EU', 'GB', 'CA', 'AU', 'NZ', 'JP', 'KR', 'SG'],
        breach_notification_required=True,
        breach_notification_hours=24,  # Most stringent requirement
        data_subject_response_days=15,  # Most stringent requirement (LGPD)
        pseudonymization_enabled=True,
        encryption_required=True,
        audit_logging_enabled=True
    )
