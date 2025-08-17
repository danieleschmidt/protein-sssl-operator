"""
Compliance Monitoring and Alerting System for protein-sssl-operator
Provides real-time monitoring of compliance violations, regional requirements,
data residency compliance, and automated alerting for legal and regulatory issues.
"""

import json
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    DASHBOARD = "dashboard"

class ComplianceViolationType(Enum):
    """Types of compliance violations"""
    DATA_RETENTION = "data_retention"
    CONSENT_VIOLATION = "consent_violation"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"
    DATA_SOVEREIGNTY = "data_sovereignty"
    ACCESS_CONTROL = "access_control"
    ENCRYPTION_FAILURE = "encryption_failure"
    AUDIT_LOG_FAILURE = "audit_log_failure"
    EXPORT_CONTROL = "export_control"
    BREACH_NOTIFICATION = "breach_notification"
    GDPR_VIOLATION = "gdpr_violation"
    CCPA_VIOLATION = "ccpa_violation"
    HIPAA_VIOLATION = "hipaa_violation"
    SOC2_CONTROL_FAILURE = "soc2_control_failure"

@dataclass
class ComplianceAlert:
    """Compliance alert"""
    alert_id: str
    violation_type: ComplianceViolationType
    severity: AlertSeverity
    title: str
    description: str
    affected_resources: List[str]
    regulation: str
    jurisdiction: str
    remediation_steps: List[str]
    automated_remediation: bool = False
    escalation_required: bool = False
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    assignee: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringRule:
    """Compliance monitoring rule"""
    rule_id: str
    name: str
    description: str
    violation_type: ComplianceViolationType
    regulation: str
    check_function: str  # Name of function to execute
    check_interval_seconds: int
    severity: AlertSeverity
    enabled: bool = True
    alert_channels: List[AlertChannel] = field(default_factory=list)
    remediation_function: Optional[str] = None
    escalation_threshold_hours: int = 24
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertChannel:
    """Alert delivery channel configuration"""
    channel_id: str
    channel_type: AlertChannel
    name: str
    configuration: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)

class ComplianceMetricsCollector:
    """Collects compliance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = defaultdict(dict)
        self._lock = threading.RLock()
    
    def record_compliance_check(self, rule_id: str, result: bool, 
                               duration_ms: float, metadata: Optional[Dict[str, Any]] = None):
        """Record compliance check result"""
        with self._lock:
            timestamp = time.time()
            
            if rule_id not in self.metrics:
                self.metrics[rule_id] = {
                    'total_checks': 0,
                    'passed_checks': 0,
                    'failed_checks': 0,
                    'avg_duration_ms': 0,
                    'last_check': 0,
                    'last_result': None,
                    'failure_rate': 0.0,
                    'history': deque(maxlen=100)
                }
            
            rule_metrics = self.metrics[rule_id]
            rule_metrics['total_checks'] += 1
            rule_metrics['last_check'] = timestamp
            rule_metrics['last_result'] = result
            
            if result:
                rule_metrics['passed_checks'] += 1
            else:
                rule_metrics['failed_checks'] += 1
            
            # Update average duration
            total_checks = rule_metrics['total_checks']
            rule_metrics['avg_duration_ms'] = (
                (rule_metrics['avg_duration_ms'] * (total_checks - 1) + duration_ms) / total_checks
            )
            
            # Update failure rate
            rule_metrics['failure_rate'] = rule_metrics['failed_checks'] / total_checks
            
            # Add to history
            rule_metrics['history'].append({
                'timestamp': timestamp,
                'result': result,
                'duration_ms': duration_ms,
                'metadata': metadata or {}
            })
    
    def get_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)"""
        with self._lock:
            if not self.metrics:
                return 100.0
            
            total_weight = 0
            weighted_score = 0
            
            for rule_id, rule_metrics in self.metrics.items():
                if rule_metrics['total_checks'] > 0:
                    # Weight by number of checks (more frequent checks have higher weight)
                    weight = min(rule_metrics['total_checks'], 100)
                    score = (1 - rule_metrics['failure_rate']) * 100
                    
                    weighted_score += score * weight
                    total_weight += weight
            
            if total_weight == 0:
                return 100.0
            
            return weighted_score / total_weight
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get compliance metrics summary"""
        with self._lock:
            summary = {
                'overall_compliance_score': self.get_compliance_score(),
                'total_rules': len(self.metrics),
                'rule_metrics': {}
            }
            
            for rule_id, rule_metrics in self.metrics.items():
                summary['rule_metrics'][rule_id] = {
                    'total_checks': rule_metrics['total_checks'],
                    'failure_rate': rule_metrics['failure_rate'],
                    'avg_duration_ms': rule_metrics['avg_duration_ms'],
                    'last_check': rule_metrics['last_check'],
                    'last_result': rule_metrics['last_result']
                }
            
            return summary

class AlertNotificationManager:
    """Manages alert notifications across different channels"""
    
    def __init__(self):
        self.channels: Dict[str, AlertChannel] = {}
        self.notification_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    def add_channel(self, channel: AlertChannel):
        """Add notification channel"""
        with self._lock:
            self.channels[channel.channel_id] = channel
        
        logger.info(f"Added alert channel: {channel.name} ({channel.channel_type.value})")
    
    def send_alert(self, alert: ComplianceAlert, channels: Optional[List[str]] = None) -> bool:
        """Send alert through specified channels"""
        if channels is None:
            channels = list(self.channels.keys())
        
        success_count = 0
        
        for channel_id in channels:
            channel = self.channels.get(channel_id)
            if not channel or not channel.enabled:
                continue
            
            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue
            
            try:
                if self._send_to_channel(alert, channel):
                    success_count += 1
                    
                    # Record notification
                    with self._lock:
                        self.notification_history.append({
                            'alert_id': alert.alert_id,
                            'channel_id': channel_id,
                            'timestamp': time.time(),
                            'success': True
                        })
                        
            except Exception as e:
                logger.error(f"Failed to send alert to channel {channel_id}: {e}")
                
                with self._lock:
                    self.notification_history.append({
                        'alert_id': alert.alert_id,
                        'channel_id': channel_id,
                        'timestamp': time.time(),
                        'success': False,
                        'error': str(e)
                    })
        
        return success_count > 0
    
    def _send_to_channel(self, alert: ComplianceAlert, channel: AlertChannel) -> bool:
        """Send alert to specific channel"""
        if channel.channel_type == AlertChannel.EMAIL:
            return self._send_email(alert, channel)
        elif channel.channel_type == AlertChannel.WEBHOOK:
            return self._send_webhook(alert, channel)
        elif channel.channel_type == AlertChannel.SLACK:
            return self._send_slack(alert, channel)
        elif channel.channel_type == AlertChannel.PAGERDUTY:
            return self._send_pagerduty(alert, channel)
        else:
            logger.warning(f"Unsupported channel type: {channel.channel_type}")
            return False
    
    def _send_email(self, alert: ComplianceAlert, channel: AlertChannel) -> bool:
        """Send email notification"""
        try:
            config = channel.configuration
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = config.get('from_email', 'compliance@company.com')
            msg['To'] = config.get('to_email')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create HTML body
            body = f"""
            <html>
            <body>
                <h2>Compliance Alert: {alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Violation Type:</strong> {alert.violation_type.value}</p>
                <p><strong>Regulation:</strong> {alert.regulation}</p>
                <p><strong>Jurisdiction:</strong> {alert.jurisdiction}</p>
                <p><strong>Time:</strong> {datetime.fromtimestamp(alert.timestamp).isoformat()}</p>
                
                <h3>Description</h3>
                <p>{alert.description}</p>
                
                <h3>Affected Resources</h3>
                <ul>
                {"".join(f"<li>{resource}</li>" for resource in alert.affected_resources)}
                </ul>
                
                <h3>Remediation Steps</h3>
                <ol>
                {"".join(f"<li>{step}</li>" for step in alert.remediation_steps)}
                </ol>
                
                <p><strong>Alert ID:</strong> {alert.alert_id}</p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(config.get('smtp_server', 'localhost'), config.get('smtp_port', 587))
            if config.get('use_tls', True):
                server.starttls()
            
            if config.get('username'):
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_webhook(self, alert: ComplianceAlert, channel: AlertChannel) -> bool:
        """Send webhook notification"""
        try:
            config = channel.configuration
            
            payload = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'violation_type': alert.violation_type.value,
                'title': alert.title,
                'description': alert.description,
                'regulation': alert.regulation,
                'jurisdiction': alert.jurisdiction,
                'affected_resources': alert.affected_resources,
                'remediation_steps': alert.remediation_steps,
                'timestamp': alert.timestamp,
                'metadata': alert.metadata
            }
            
            response = requests.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=config.get('timeout', 30)
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _send_slack(self, alert: ComplianceAlert, channel: AlertChannel) -> bool:
        """Send Slack notification"""
        try:
            config = channel.configuration
            
            # Color based on severity
            colors = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ffac33",   # Orange
                AlertSeverity.HIGH: "#ff6b35",     # Red-orange
                AlertSeverity.CRITICAL: "#dc3545"  # Red
            }
            
            payload = {
                "username": "Compliance Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": colors.get(alert.severity, "#cccccc"),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Regulation", "value": alert.regulation, "short": True},
                        {"title": "Jurisdiction", "value": alert.jurisdiction, "short": True},
                        {"title": "Violation Type", "value": alert.violation_type.value, "short": True},
                        {"title": "Affected Resources", "value": ", ".join(alert.affected_resources), "short": False}
                    ],
                    "footer": f"Alert ID: {alert.alert_id}",
                    "ts": int(alert.timestamp)
                }]
            }
            
            response = requests.post(
                config['webhook_url'],
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_pagerduty(self, alert: ComplianceAlert, channel: AlertChannel) -> bool:
        """Send PagerDuty notification"""
        try:
            config = channel.configuration
            
            payload = {
                "routing_key": config['routing_key'],
                "event_action": "trigger",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": alert.title,
                    "source": "protein-sssl-compliance",
                    "severity": alert.severity.value,
                    "component": "compliance",
                    "group": alert.regulation,
                    "class": alert.violation_type.value,
                    "custom_details": {
                        "description": alert.description,
                        "jurisdiction": alert.jurisdiction,
                        "affected_resources": alert.affected_resources,
                        "remediation_steps": alert.remediation_steps,
                        "alert_id": alert.alert_id
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

class ComplianceMonitor:
    """Main compliance monitoring system"""
    
    def __init__(self):
        self.rules: Dict[str, MonitoringRule] = {}
        self.alerts: List[ComplianceAlert] = []
        self.metrics_collector = ComplianceMetricsCollector()
        self.notification_manager = AlertNotificationManager()
        self._monitoring_threads: Dict[str, threading.Thread] = {}
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        # Load default monitoring rules
        self._load_default_rules()
        
        logger.info("Compliance monitor initialized")
    
    def _load_default_rules(self):
        """Load default compliance monitoring rules"""
        
        # Data retention monitoring
        retention_rule = MonitoringRule(
            rule_id="data_retention_check",
            name="Data Retention Compliance",
            description="Monitor data retention policy compliance",
            violation_type=ComplianceViolationType.DATA_RETENTION,
            regulation="GDPR,CCPA",
            check_function="check_data_retention",
            check_interval_seconds=3600,  # Every hour
            severity=AlertSeverity.HIGH,
            alert_channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        # Cross-border transfer monitoring
        transfer_rule = MonitoringRule(
            rule_id="cross_border_transfer_check",
            name="Cross-Border Data Transfer Compliance",
            description="Monitor unauthorized cross-border data transfers",
            violation_type=ComplianceViolationType.CROSS_BORDER_TRANSFER,
            regulation="GDPR,PDPA",
            check_function="check_cross_border_transfers",
            check_interval_seconds=300,  # Every 5 minutes
            severity=AlertSeverity.CRITICAL,
            alert_channels=[AlertChannel.EMAIL, AlertChannel.PAGERDUTY]
        )
        
        # Encryption monitoring
        encryption_rule = MonitoringRule(
            rule_id="encryption_check",
            name="Encryption Compliance",
            description="Monitor encryption requirements",
            violation_type=ComplianceViolationType.ENCRYPTION_FAILURE,
            regulation="SOC2,ISO27001",
            check_function="check_encryption",
            check_interval_seconds=600,  # Every 10 minutes
            severity=AlertSeverity.HIGH,
            alert_channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        # Consent monitoring
        consent_rule = MonitoringRule(
            rule_id="consent_check",
            name="Consent Compliance",
            description="Monitor consent requirements and withdrawals",
            violation_type=ComplianceViolationType.CONSENT_VIOLATION,
            regulation="GDPR,CCPA",
            check_function="check_consent_compliance",
            check_interval_seconds=1800,  # Every 30 minutes
            severity=AlertSeverity.MEDIUM,
            alert_channels=[AlertChannel.EMAIL]
        )
        
        # Export control monitoring
        export_rule = MonitoringRule(
            rule_id="export_control_check",
            name="Export Control Compliance",
            description="Monitor export control violations",
            violation_type=ComplianceViolationType.EXPORT_CONTROL,
            regulation="EAR,ITAR",
            check_function="check_export_control",
            check_interval_seconds=900,  # Every 15 minutes
            severity=AlertSeverity.CRITICAL,
            alert_channels=[AlertChannel.EMAIL, AlertChannel.PAGERDUTY]
        )
        
        with self._lock:
            self.rules[retention_rule.rule_id] = retention_rule
            self.rules[transfer_rule.rule_id] = transfer_rule
            self.rules[encryption_rule.rule_id] = encryption_rule
            self.rules[consent_rule.rule_id] = consent_rule
            self.rules[export_rule.rule_id] = export_rule
    
    def add_rule(self, rule: MonitoringRule):
        """Add monitoring rule"""
        with self._lock:
            self.rules[rule.rule_id] = rule
        
        # Start monitoring thread for this rule
        if rule.enabled:
            self._start_rule_monitoring(rule)
        
        logger.info(f"Added monitoring rule: {rule.name}")
    
    def start_monitoring(self):
        """Start all monitoring threads"""
        with self._lock:
            for rule in self.rules.values():
                if rule.enabled:
                    self._start_rule_monitoring(rule)
        
        logger.info(f"Started monitoring {len(self.rules)} compliance rules")
    
    def stop_monitoring(self):
        """Stop all monitoring threads"""
        self._shutdown_event.set()
        
        with self._lock:
            for thread in self._monitoring_threads.values():
                if thread.is_alive():
                    thread.join(timeout=5)
        
        logger.info("Stopped compliance monitoring")
    
    def _start_rule_monitoring(self, rule: MonitoringRule):
        """Start monitoring thread for a specific rule"""
        def monitor_rule():
            while not self._shutdown_event.wait(rule.check_interval_seconds):
                try:
                    self._execute_rule_check(rule)
                except Exception as e:
                    logger.error(f"Error executing rule {rule.rule_id}: {e}")
        
        thread = threading.Thread(target=monitor_rule, daemon=True)
        thread.start()
        
        with self._lock:
            self._monitoring_threads[rule.rule_id] = thread
    
    def _execute_rule_check(self, rule: MonitoringRule):
        """Execute compliance check for a rule"""
        start_time = time.time()
        
        try:
            # Execute the check function
            result, violations = self._call_check_function(rule.check_function)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics_collector.record_compliance_check(
                rule.rule_id, result, duration_ms
            )
            
            # Handle violations
            if not result and violations:
                for violation in violations:
                    self._create_alert(rule, violation)
            
        except Exception as e:
            logger.error(f"Error in compliance check {rule.rule_id}: {e}")
            
            # Record failed check
            duration_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_compliance_check(
                rule.rule_id, False, duration_ms, {'error': str(e)}
            )
    
    def _call_check_function(self, function_name: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Call compliance check function"""
        # This would integrate with actual compliance check implementations
        # For now, we'll simulate some checks
        
        if function_name == "check_data_retention":
            return self._check_data_retention()
        elif function_name == "check_cross_border_transfers":
            return self._check_cross_border_transfers()
        elif function_name == "check_encryption":
            return self._check_encryption()
        elif function_name == "check_consent_compliance":
            return self._check_consent_compliance()
        elif function_name == "check_export_control":
            return self._check_export_control()
        else:
            logger.warning(f"Unknown check function: {function_name}")
            return True, []
    
    def _check_data_retention(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check data retention compliance"""
        # Simplified implementation - would integrate with actual data systems
        violations = []
        
        # Simulate finding expired data
        if time.time() % 100 < 5:  # 5% chance of violation
            violations.append({
                'resource': 'data-store-1',
                'issue': 'Data retained beyond policy limit',
                'details': 'Found 150 records older than 365 days'
            })
        
        return len(violations) == 0, violations
    
    def _check_cross_border_transfers(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check cross-border transfer compliance"""
        violations = []
        
        # Simulate finding unauthorized transfers
        if time.time() % 200 < 3:  # 1.5% chance of violation
            violations.append({
                'resource': 'transfer-service',
                'issue': 'Unauthorized cross-border data transfer',
                'details': 'Transfer from EU to non-adequate country without safeguards'
            })
        
        return len(violations) == 0, violations
    
    def _check_encryption(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check encryption compliance"""
        violations = []
        
        # Simulate finding unencrypted data
        if time.time() % 150 < 2:  # ~1.3% chance of violation
            violations.append({
                'resource': 'database-1',
                'issue': 'Unencrypted sensitive data',
                'details': 'Found unencrypted personal data in database'
            })
        
        return len(violations) == 0, violations
    
    def _check_consent_compliance(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check consent compliance"""
        violations = []
        
        # Simulate finding consent violations
        if time.time() % 300 < 5:  # ~1.7% chance of violation
            violations.append({
                'resource': 'user-data-processor',
                'issue': 'Processing without valid consent',
                'details': 'Found 25 users with withdrawn consent still being processed'
            })
        
        return len(violations) == 0, violations
    
    def _check_export_control(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check export control compliance"""
        violations = []
        
        # Simulate finding export control violations
        if time.time() % 500 < 1:  # 0.2% chance of violation
            violations.append({
                'resource': 'export-service',
                'issue': 'Export to restricted country',
                'details': 'Attempted technology transfer to embargoed country'
            })
        
        return len(violations) == 0, violations
    
    def _create_alert(self, rule: MonitoringRule, violation: Dict[str, Any]):
        """Create compliance alert"""
        alert = ComplianceAlert(
            alert_id=str(uuid.uuid4()),
            violation_type=rule.violation_type,
            severity=rule.severity,
            title=f"{rule.name} Violation",
            description=f"{violation['issue']}: {violation['details']}",
            affected_resources=[violation['resource']],
            regulation=rule.regulation,
            jurisdiction="GLOBAL",  # Would be determined based on actual context
            remediation_steps=self._get_remediation_steps(rule.violation_type),
            metadata={
                'rule_id': rule.rule_id,
                'violation_details': violation
            }
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # Send notifications
        self.notification_manager.send_alert(alert)
        
        logger.warning(f"Compliance alert created: {alert.alert_id} - {alert.title}")
    
    def _get_remediation_steps(self, violation_type: ComplianceViolationType) -> List[str]:
        """Get remediation steps for violation type"""
        remediation_steps = {
            ComplianceViolationType.DATA_RETENTION: [
                "Identify expired data records",
                "Verify no legal obligation to retain",
                "Execute secure data deletion",
                "Update retention policies if needed"
            ],
            ComplianceViolationType.CROSS_BORDER_TRANSFER: [
                "Stop unauthorized transfers immediately",
                "Review transfer agreements",
                "Implement proper safeguards",
                "Notify data protection authorities if required"
            ],
            ComplianceViolationType.ENCRYPTION_FAILURE: [
                "Identify unencrypted data",
                "Apply encryption immediately",
                "Review encryption policies",
                "Audit other systems for similar issues"
            ],
            ComplianceViolationType.CONSENT_VIOLATION: [
                "Stop processing for affected users",
                "Update consent management system",
                "Contact users to re-obtain consent if appropriate",
                "Review consent collection processes"
            ],
            ComplianceViolationType.EXPORT_CONTROL: [
                "Block the export immediately",
                "Review export control lists",
                "Contact legal counsel",
                "Report to relevant authorities if required"
            ]
        }
        
        return remediation_steps.get(violation_type, ["Contact compliance team immediately"])
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        with self._lock:
            active_rules = len([r for r in self.rules.values() if r.enabled])
            total_alerts = len(self.alerts)
            open_alerts = len([a for a in self.alerts if not a.resolved])
            critical_alerts = len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved])
        
        metrics_summary = self.metrics_collector.get_metrics_summary()
        
        return {
            'monitoring_active': not self._shutdown_event.is_set(),
            'active_rules': active_rules,
            'total_rules': len(self.rules),
            'total_alerts': total_alerts,
            'open_alerts': open_alerts,
            'critical_alerts': critical_alerts,
            'compliance_score': metrics_summary['overall_compliance_score'],
            'last_update': time.time()
        }
    
    def resolve_alert(self, alert_id: str, resolver: str, notes: Optional[str] = None) -> bool:
        """Resolve compliance alert"""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    alert.assignee = resolver
                    if notes:
                        alert.metadata['resolution_notes'] = notes
                    
                    logger.info(f"Alert resolved: {alert_id} by {resolver}")
                    return True
        
        return False

# Global compliance monitor
_global_compliance_monitor: Optional[ComplianceMonitor] = None

def get_compliance_monitor() -> Optional[ComplianceMonitor]:
    """Get global compliance monitor"""
    return _global_compliance_monitor

def initialize_compliance_monitoring() -> ComplianceMonitor:
    """Initialize global compliance monitoring"""
    global _global_compliance_monitor
    _global_compliance_monitor = ComplianceMonitor()
    return _global_compliance_monitor

def start_compliance_monitoring():
    """Start global compliance monitoring"""
    if _global_compliance_monitor:
        _global_compliance_monitor.start_monitoring()

def stop_compliance_monitoring():
    """Stop global compliance monitoring"""
    if _global_compliance_monitor:
        _global_compliance_monitor.stop_monitoring()

# Convenience functions
def add_alert_channel(channel_type: AlertChannel, name: str, configuration: Dict[str, Any]) -> str:
    """Add alert notification channel"""
    if _global_compliance_monitor:
        channel_id = str(uuid.uuid4())
        channel = AlertChannel(
            channel_id=channel_id,
            channel_type=channel_type,
            name=name,
            configuration=configuration
        )
        _global_compliance_monitor.notification_manager.add_channel(channel)
        return channel_id
    return ""

def get_compliance_status() -> Dict[str, Any]:
    """Get current compliance monitoring status"""
    if _global_compliance_monitor:
        return _global_compliance_monitor.get_monitoring_status()
    return {"error": "Compliance monitoring not initialized"}