"""
Accessibility Framework for protein-sssl-operator
Provides WCAG 2.1 AA compliance, screen reader compatibility,
keyboard navigation, high contrast themes, and inclusive design features.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class WCAGLevel(Enum):
    """WCAG conformance levels"""
    A = "A"
    AA = "AA"
    AAA = "AAA"

class WCAGPrinciple(Enum):
    """WCAG 2.1 principles"""
    PERCEIVABLE = "perceivable"
    OPERABLE = "operable"
    UNDERSTANDABLE = "understandable"
    ROBUST = "robust"

class AccessibilityFeature(Enum):
    """Accessibility features"""
    SCREEN_READER = "screen_reader"
    KEYBOARD_NAVIGATION = "keyboard_navigation"
    HIGH_CONTRAST = "high_contrast"
    LARGE_TEXT = "large_text"
    REDUCED_MOTION = "reduced_motion"
    FOCUS_INDICATORS = "focus_indicators"
    ALT_TEXT = "alt_text"
    ARIA_LABELS = "aria_labels"
    COLOR_BLIND_FRIENDLY = "color_blind_friendly"
    VOICE_COMMANDS = "voice_commands"

class ContrastLevel(Enum):
    """Color contrast levels"""
    NORMAL = "normal"          # 4.5:1 for normal text
    LARGE = "large"           # 3:1 for large text
    ENHANCED = "enhanced"     # 7:1 for enhanced (AAA)

class FontSize(Enum):
    """Font size categories"""
    SMALL = "small"      # 12-14px
    NORMAL = "normal"    # 16-18px
    LARGE = "large"      # 20-24px
    EXTRA_LARGE = "extra_large"  # 28px+

class MotionPreference(Enum):
    """Motion preference settings"""
    NO_PREFERENCE = "no_preference"
    REDUCE = "reduce"
    ELIMINATE = "eliminate"

@dataclass
class ColorContrast:
    """Color contrast configuration"""
    foreground: str  # Hex color
    background: str  # Hex color
    ratio: float
    level: ContrastLevel
    passes_aa: bool
    passes_aaa: bool

@dataclass
class AccessibilityProfile:
    """User accessibility profile"""
    user_id: str
    enabled_features: Set[AccessibilityFeature]
    font_size: FontSize
    contrast_preference: ContrastLevel
    motion_preference: MotionPreference
    screen_reader_type: Optional[str] = None
    keyboard_only: bool = False
    high_contrast_theme: bool = False
    use_system_preferences: bool = True
    custom_css: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class WCAGGuideline:
    """WCAG 2.1 guideline"""
    guideline_id: str
    title: str
    level: WCAGLevel
    principle: WCAGPrinciple
    description: str
    success_criteria: List[str]
    implementation_notes: str
    testing_procedure: str
    is_compliant: bool = False

@dataclass
class AccessibilityAuditResult:
    """Accessibility audit result"""
    audit_id: str
    timestamp: float
    component_name: str
    wcag_violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    compliance_score: float  # 0-100
    level_achieved: WCAGLevel
    details: Dict[str, Any] = field(default_factory=dict)

class ColorContrastCalculator:
    """Calculate color contrast ratios"""
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def get_relative_luminance(rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance"""
        def gamma_correct(value: float) -> float:
            if value <= 0.03928:
                return value / 12.92
            else:
                return ((value + 0.055) / 1.055) ** 2.4
        
        r, g, b = [c / 255.0 for c in rgb]
        r_gamma = gamma_correct(r)
        g_gamma = gamma_correct(g)
        b_gamma = gamma_correct(b)
        
        return 0.2126 * r_gamma + 0.7152 * g_gamma + 0.0722 * b_gamma
    
    @staticmethod
    def calculate_contrast_ratio(color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors"""
        rgb1 = ColorContrastCalculator.hex_to_rgb(color1)
        rgb2 = ColorContrastCalculator.hex_to_rgb(color2)
        
        lum1 = ColorContrastCalculator.get_relative_luminance(rgb1)
        lum2 = ColorContrastCalculator.get_relative_luminance(rgb2)
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    @staticmethod
    def validate_contrast(foreground: str, background: str, 
                         large_text: bool = False) -> ColorContrast:
        """Validate color contrast against WCAG standards"""
        ratio = ColorContrastCalculator.calculate_contrast_ratio(foreground, background)
        
        # WCAG AA requirements
        aa_threshold = 3.0 if large_text else 4.5
        # WCAG AAA requirements  
        aaa_threshold = 4.5 if large_text else 7.0
        
        passes_aa = ratio >= aa_threshold
        passes_aaa = ratio >= aaa_threshold
        
        # Determine level
        if large_text:
            level = ContrastLevel.LARGE
        elif ratio >= aaa_threshold:
            level = ContrastLevel.ENHANCED
        else:
            level = ContrastLevel.NORMAL
        
        return ColorContrast(
            foreground=foreground,
            background=background,
            ratio=ratio,
            level=level,
            passes_aa=passes_aa,
            passes_aaa=passes_aaa
        )

class ScreenReaderSupport:
    """Screen reader compatibility utilities"""
    
    COMMON_SCREEN_READERS = [
        "NVDA",      # Windows
        "JAWS",      # Windows
        "ORCA",      # Linux
        "VoiceOver", # macOS/iOS
        "TalkBack",  # Android
        "Dragon",    # Voice control
    ]
    
    @staticmethod
    def generate_aria_label(element_type: str, content: str, 
                          context: Optional[str] = None) -> str:
        """Generate appropriate ARIA label"""
        if element_type == "button":
            return f"Button: {content}"
        elif element_type == "link":
            return f"Link: {content}"
        elif element_type == "input":
            return f"Input field: {content}"
        elif element_type == "table":
            return f"Table: {content}"
        elif element_type == "chart":
            return f"Chart: {content}. {context or ''}"
        elif element_type == "protein_structure":
            return f"Protein structure visualization: {content}. {context or ''}"
        else:
            return content
    
    @staticmethod
    def generate_alt_text(image_type: str, content: str, 
                         data: Optional[Dict[str, Any]] = None) -> str:
        """Generate descriptive alt text"""
        if image_type == "protein_structure":
            structure_type = data.get("structure_type", "unknown") if data else "unknown"
            confidence = data.get("confidence", 0) if data else 0
            return f"Protein structure prediction showing {structure_type} with {confidence:.1%} confidence. {content}"
        
        elif image_type == "graph":
            x_axis = data.get("x_axis", "X") if data else "X"
            y_axis = data.get("y_axis", "Y") if data else "Y"
            return f"Graph showing {y_axis} vs {x_axis}. {content}"
        
        elif image_type == "heatmap":
            return f"Heatmap visualization. {content}"
        
        elif image_type == "logo":
            return f"Logo: {content}"
        
        else:
            return content
    
    @staticmethod
    def create_aria_describedby(descriptions: List[str]) -> str:
        """Create aria-describedby content"""
        return " ".join(descriptions)

class KeyboardNavigationManager:
    """Keyboard navigation support"""
    
    @staticmethod
    def get_tab_order(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate logical tab order for elements"""
        # Sort by position and importance
        def sort_key(element):
            # Primary sort: tab index (if specified)
            tab_index = element.get("tabindex", 0)
            if tab_index < 0:
                return (999, 0, 0)  # Send to end
            
            # Secondary sort: vertical position
            y_pos = element.get("y_position", 0)
            
            # Tertiary sort: horizontal position
            x_pos = element.get("x_position", 0)
            
            return (tab_index, y_pos, x_pos)
        
        return sorted(elements, key=sort_key)
    
    @staticmethod
    def generate_skip_links(sections: List[str]) -> List[Dict[str, str]]:
        """Generate skip navigation links"""
        skip_links = []
        
        for section in sections:
            skip_links.append({
                "text": f"Skip to {section}",
                "href": f"#{section.lower().replace(' ', '_')}",
                "aria_label": f"Skip to {section} section"
            })
        
        return skip_links
    
    @staticmethod
    def validate_keyboard_accessibility(elements: List[Dict[str, Any]]) -> List[str]:
        """Validate keyboard accessibility of elements"""
        issues = []
        
        for element in elements:
            element_type = element.get("type", "unknown")
            
            # Check if interactive elements are focusable
            if element_type in ["button", "link", "input", "select", "textarea"]:
                if not element.get("focusable", True):
                    issues.append(f"{element_type} element is not focusable")
                
                # Check for keyboard event handlers
                if not element.get("keyboard_events", False):
                    issues.append(f"{element_type} element missing keyboard event handlers")
            
            # Check for focus indicators
            if element.get("focusable", False) and not element.get("focus_indicator", False):
                issues.append(f"Element missing focus indicator")
        
        return issues

class AccessibilityThemeManager:
    """Manage accessibility themes and preferences"""
    
    def __init__(self):
        self.themes = self._create_default_themes()
        self.current_theme = "default"
    
    def _create_default_themes(self) -> Dict[str, Dict[str, Any]]:
        """Create default accessibility themes"""
        return {
            "default": {
                "name": "Default",
                "colors": {
                    "primary": "#0066cc",
                    "secondary": "#6c757d", 
                    "background": "#ffffff",
                    "text": "#212529",
                    "link": "#0066cc",
                    "focus": "#0066cc",
                    "border": "#dee2e6"
                },
                "fonts": {
                    "family": "system-ui, -apple-system, sans-serif",
                    "size_base": "16px",
                    "line_height": "1.5"
                },
                "spacing": {
                    "padding": "0.75rem",
                    "margin": "1rem",
                    "border_radius": "0.25rem"
                }
            },
            
            "high_contrast": {
                "name": "High Contrast",
                "colors": {
                    "primary": "#ffffff",
                    "secondary": "#ffff00",
                    "background": "#000000",
                    "text": "#ffffff",
                    "link": "#ffff00",
                    "focus": "#ff0000",
                    "border": "#ffffff"
                },
                "fonts": {
                    "family": "Arial, sans-serif",
                    "size_base": "18px",
                    "line_height": "1.6",
                    "weight": "bold"
                },
                "spacing": {
                    "padding": "1rem",
                    "margin": "1.25rem",
                    "border_radius": "0"
                }
            },
            
            "dark_mode": {
                "name": "Dark Mode",
                "colors": {
                    "primary": "#0d7377",
                    "secondary": "#14a085",
                    "background": "#1a1a1a",
                    "text": "#e9ecef",
                    "link": "#14a085",
                    "focus": "#0d7377",
                    "border": "#495057"
                },
                "fonts": {
                    "family": "system-ui, -apple-system, sans-serif",
                    "size_base": "16px",
                    "line_height": "1.5"
                },
                "spacing": {
                    "padding": "0.75rem",
                    "margin": "1rem",
                    "border_radius": "0.25rem"
                }
            },
            
            "large_text": {
                "name": "Large Text",
                "colors": {
                    "primary": "#0066cc",
                    "secondary": "#6c757d",
                    "background": "#ffffff", 
                    "text": "#212529",
                    "link": "#0066cc",
                    "focus": "#0066cc",
                    "border": "#dee2e6"
                },
                "fonts": {
                    "family": "Arial, sans-serif",
                    "size_base": "22px",
                    "line_height": "1.8",
                    "weight": "500"
                },
                "spacing": {
                    "padding": "1.25rem",
                    "margin": "1.5rem",
                    "border_radius": "0.5rem"
                }
            },
            
            "dyslexia_friendly": {
                "name": "Dyslexia Friendly",
                "colors": {
                    "primary": "#0066cc",
                    "secondary": "#6c757d",
                    "background": "#f8f9fa",
                    "text": "#495057",
                    "link": "#0066cc",
                    "focus": "#0066cc",
                    "border": "#dee2e6"
                },
                "fonts": {
                    "family": "OpenDyslexic, Arial, sans-serif",
                    "size_base": "18px",
                    "line_height": "1.8",
                    "letter_spacing": "0.05em"
                },
                "spacing": {
                    "padding": "1rem",
                    "margin": "1.5rem",
                    "border_radius": "0.5rem"
                }
            }
        }
    
    def get_theme(self, theme_name: str) -> Optional[Dict[str, Any]]:
        """Get theme configuration"""
        return self.themes.get(theme_name)
    
    def apply_user_preferences(self, theme_name: str, 
                             profile: AccessibilityProfile) -> Dict[str, Any]:
        """Apply user preferences to theme"""
        theme = self.get_theme(theme_name).copy()
        if not theme:
            theme = self.get_theme("default").copy()
        
        # Apply font size preference
        font_sizes = {
            FontSize.SMALL: "14px",
            FontSize.NORMAL: "16px", 
            FontSize.LARGE: "20px",
            FontSize.EXTRA_LARGE: "28px"
        }
        
        if profile.font_size in font_sizes:
            theme["fonts"]["size_base"] = font_sizes[profile.font_size]
        
        # Apply high contrast if enabled
        if profile.high_contrast_theme:
            high_contrast = self.get_theme("high_contrast")
            if high_contrast:
                theme["colors"] = high_contrast["colors"]
        
        # Apply custom CSS
        if profile.custom_css:
            theme["custom_css"] = profile.custom_css
        
        return theme
    
    def generate_css(self, theme: Dict[str, Any]) -> str:
        """Generate CSS from theme configuration"""
        css = []
        
        # Root variables
        css.append(":root {")
        
        # Colors
        colors = theme.get("colors", {})
        for name, value in colors.items():
            css.append(f"  --color-{name.replace('_', '-')}: {value};")
        
        # Fonts
        fonts = theme.get("fonts", {})
        for name, value in fonts.items():
            css.append(f"  --font-{name.replace('_', '-')}: {value};")
        
        # Spacing
        spacing = theme.get("spacing", {})
        for name, value in spacing.items():
            css.append(f"  --spacing-{name.replace('_', '-')}: {value};")
        
        css.append("}")
        
        # Base styles
        css.extend([
            "",
            "body {",
            "  font-family: var(--font-family);",
            "  font-size: var(--font-size-base);",
            "  line-height: var(--font-line-height);",
            "  color: var(--color-text);",
            "  background-color: var(--color-background);",
            "}",
            "",
            ".focus-visible {",
            "  outline: 2px solid var(--color-focus);",
            "  outline-offset: 2px;",
            "}",
            "",
            ".sr-only {",
            "  position: absolute !important;",
            "  width: 1px !important;",
            "  height: 1px !important;",
            "  padding: 0 !important;",
            "  margin: -1px !important;",
            "  overflow: hidden !important;",
            "  clip: rect(0, 0, 0, 0) !important;",
            "  white-space: nowrap !important;",
            "  border: 0 !important;",
            "}",
            "",
            "@media (prefers-reduced-motion: reduce) {",
            "  * {",
            "    animation-duration: 0.01ms !important;",
            "    animation-iteration-count: 1 !important;",
            "    transition-duration: 0.01ms !important;",
            "  }",
            "}"
        ])
        
        # Custom CSS
        if "custom_css" in theme:
            css.extend(["", theme["custom_css"]])
        
        return "\n".join(css)

class WCAGAuditor:
    """WCAG 2.1 compliance auditor"""
    
    def __init__(self):
        self.guidelines = self._load_wcag_guidelines()
    
    def _load_wcag_guidelines(self) -> Dict[str, WCAGGuideline]:
        """Load WCAG 2.1 guidelines"""
        guidelines = {}
        
        # Sample guidelines (in practice, load from comprehensive database)
        sample_guidelines = [
            WCAGGuideline(
                guideline_id="1.1.1",
                title="Non-text Content",
                level=WCAGLevel.A,
                principle=WCAGPrinciple.PERCEIVABLE,
                description="All non-text content has text alternative",
                success_criteria=["Images have alt text", "Form controls have labels"],
                implementation_notes="Use alt attributes, aria-label, or aria-labelledby",
                testing_procedure="Check all images, icons, and form controls for text alternatives"
            ),
            WCAGGuideline(
                guideline_id="1.4.3",
                title="Contrast (Minimum)",
                level=WCAGLevel.AA,
                principle=WCAGPrinciple.PERCEIVABLE,
                description="Text has contrast ratio of at least 4.5:1",
                success_criteria=["Normal text: 4.5:1", "Large text: 3:1"],
                implementation_notes="Use color contrast tools to verify ratios",
                testing_procedure="Test all text/background color combinations"
            ),
            WCAGGuideline(
                guideline_id="2.1.1",
                title="Keyboard",
                level=WCAGLevel.A,
                principle=WCAGPrinciple.OPERABLE,
                description="All functionality available via keyboard",
                success_criteria=["Tab navigation works", "Enter/Space activate controls"],
                implementation_notes="Ensure focusable elements have keyboard handlers",
                testing_procedure="Navigate entire interface using only keyboard"
            ),
            WCAGGuideline(
                guideline_id="2.4.3",
                title="Focus Order",
                level=WCAGLevel.A,
                principle=WCAGPrinciple.OPERABLE,
                description="Focus order follows logical sequence",
                success_criteria=["Tab order matches visual order", "Skip links available"],
                implementation_notes="Use tabindex carefully, provide skip links",
                testing_procedure="Tab through interface and verify logical order"
            ),
            WCAGGuideline(
                guideline_id="3.1.1",
                title="Language of Page",
                level=WCAGLevel.A,
                principle=WCAGPrinciple.UNDERSTANDABLE,
                description="Page language is programmatically determined",
                success_criteria=["HTML lang attribute set", "Language changes marked"],
                implementation_notes="Set lang attribute on html element",
                testing_procedure="Check HTML source for lang attributes"
            ),
            WCAGGuideline(
                guideline_id="4.1.1",
                title="Parsing",
                level=WCAGLevel.A,
                principle=WCAGPrinciple.ROBUST,
                description="Content parses without errors",
                success_criteria=["Valid HTML markup", "Unique IDs", "Proper nesting"],
                implementation_notes="Validate HTML, use unique IDs",
                testing_procedure="Run HTML validator, check for parsing errors"
            )
        ]
        
        for guideline in sample_guidelines:
            guidelines[guideline.guideline_id] = guideline
        
        return guidelines
    
    def audit_component(self, component_data: Dict[str, Any]) -> AccessibilityAuditResult:
        """Audit component for WCAG compliance"""
        audit_id = str(uuid.uuid4())
        component_name = component_data.get("name", "unknown")
        violations = []
        warnings = []
        recommendations = []
        
        # Check images for alt text
        images = component_data.get("images", [])
        for image in images:
            if not image.get("alt_text"):
                violations.append("1.1.1: Image missing alt text")
        
        # Check color contrast
        colors = component_data.get("colors", {})
        if "foreground" in colors and "background" in colors:
            contrast = ColorContrastCalculator.validate_contrast(
                colors["foreground"], colors["background"]
            )
            if not contrast.passes_aa:
                violations.append(f"1.4.3: Insufficient color contrast ({contrast.ratio:.1f}:1)")
        
        # Check keyboard accessibility
        interactive_elements = component_data.get("interactive_elements", [])
        for element in interactive_elements:
            if not element.get("keyboard_accessible", False):
                violations.append("2.1.1: Element not keyboard accessible")
            
            if not element.get("focus_indicator", False):
                warnings.append("2.4.7: Element missing visible focus indicator")
        
        # Check form labels
        form_controls = component_data.get("form_controls", [])
        for control in form_controls:
            if not control.get("label") and not control.get("aria_label"):
                violations.append("3.3.2: Form control missing label")
        
        # Check heading structure
        headings = component_data.get("headings", [])
        if headings:
            levels = [h.get("level", 1) for h in headings]
            if levels and levels[0] != 1:
                warnings.append("1.3.1: Page doesn't start with h1")
            
            for i in range(1, len(levels)):
                if levels[i] > levels[i-1] + 1:
                    warnings.append("1.3.1: Heading levels skip (h1 to h3)")
        
        # Check language
        if not component_data.get("language"):
            violations.append("3.1.1: Language not specified")
        
        # Calculate compliance score
        total_checks = len(self.guidelines)
        violation_count = len(violations)
        compliance_score = max(0, (total_checks - violation_count) / total_checks * 100)
        
        # Determine achieved level
        if violation_count == 0:
            level_achieved = WCAGLevel.AA
        elif violation_count <= 2:
            level_achieved = WCAGLevel.A
        else:
            level_achieved = WCAGLevel.A  # Failed
        
        # Generate recommendations
        if violations:
            recommendations.append("Address all WCAG violations before deployment")
        if warnings:
            recommendations.append("Consider addressing warnings for better accessibility")
        
        recommendations.extend([
            "Test with actual screen readers",
            "Conduct user testing with disabled users",
            "Implement automated accessibility testing",
            "Provide accessibility training for developers"
        ])
        
        return AccessibilityAuditResult(
            audit_id=audit_id,
            timestamp=time.time(),
            component_name=component_name,
            wcag_violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            compliance_score=compliance_score,
            level_achieved=level_achieved,
            details={
                "total_guidelines_checked": total_checks,
                "images_checked": len(images),
                "interactive_elements_checked": len(interactive_elements),
                "form_controls_checked": len(form_controls),
                "headings_checked": len(headings)
            }
        )

class AccessibilityManager:
    """Main accessibility management system"""
    
    def __init__(self):
        self.profiles: Dict[str, AccessibilityProfile] = {}
        self.theme_manager = AccessibilityThemeManager()
        self.auditor = WCAGAuditor()
        self.contrast_calculator = ColorContrastCalculator()
        self.screen_reader_support = ScreenReaderSupport()
        self.keyboard_nav = KeyboardNavigationManager()
        self._lock = threading.RLock()
        
        logger.info("Accessibility manager initialized")
    
    def create_profile(self, user_id: str, 
                      preferences: Optional[Dict[str, Any]] = None) -> AccessibilityProfile:
        """Create accessibility profile for user"""
        
        profile = AccessibilityProfile(
            user_id=user_id,
            enabled_features=set(),
            font_size=FontSize.NORMAL,
            contrast_preference=ContrastLevel.NORMAL,
            motion_preference=MotionPreference.NO_PREFERENCE
        )
        
        if preferences:
            # Apply user preferences
            if "font_size" in preferences:
                profile.font_size = FontSize(preferences["font_size"])
            
            if "high_contrast" in preferences:
                profile.high_contrast_theme = preferences["high_contrast"]
                if preferences["high_contrast"]:
                    profile.enabled_features.add(AccessibilityFeature.HIGH_CONTRAST)
            
            if "screen_reader" in preferences:
                profile.screen_reader_type = preferences["screen_reader"]
                profile.enabled_features.add(AccessibilityFeature.SCREEN_READER)
            
            if "keyboard_only" in preferences:
                profile.keyboard_only = preferences["keyboard_only"]
                if preferences["keyboard_only"]:
                    profile.enabled_features.add(AccessibilityFeature.KEYBOARD_NAVIGATION)
            
            if "reduced_motion" in preferences:
                if preferences["reduced_motion"]:
                    profile.motion_preference = MotionPreference.REDUCE
                    profile.enabled_features.add(AccessibilityFeature.REDUCED_MOTION)
        
        with self._lock:
            self.profiles[user_id] = profile
        
        logger.info(f"Created accessibility profile for user: {user_id}")
        return profile
    
    def get_profile(self, user_id: str) -> Optional[AccessibilityProfile]:
        """Get user accessibility profile"""
        return self.profiles.get(user_id)
    
    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user accessibility profile"""
        with self._lock:
            profile = self.profiles.get(user_id)
            if not profile:
                return False
            
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.updated_at = time.time()
            
        logger.info(f"Updated accessibility profile for user: {user_id}")
        return True
    
    def get_user_theme(self, user_id: str) -> Dict[str, Any]:
        """Get personalized theme for user"""
        profile = self.get_profile(user_id)
        if not profile:
            return self.theme_manager.get_theme("default")
        
        # Determine base theme
        if profile.high_contrast_theme:
            base_theme = "high_contrast"
        elif profile.font_size == FontSize.LARGE or profile.font_size == FontSize.EXTRA_LARGE:
            base_theme = "large_text"
        else:
            base_theme = "default"
        
        return self.theme_manager.apply_user_preferences(base_theme, profile)
    
    def generate_user_css(self, user_id: str) -> str:
        """Generate personalized CSS for user"""
        theme = self.get_user_theme(user_id)
        return self.theme_manager.generate_css(theme)
    
    def audit_accessibility(self, component_data: Dict[str, Any]) -> AccessibilityAuditResult:
        """Perform accessibility audit"""
        return self.auditor.audit_component(component_data)
    
    def get_accessibility_report(self) -> Dict[str, Any]:
        """Generate accessibility compliance report"""
        
        with self._lock:
            total_users = len(self.profiles)
            
            # Feature usage statistics
            feature_usage = defaultdict(int)
            font_size_distribution = defaultdict(int)
            contrast_preferences = defaultdict(int)
            motion_preferences = defaultdict(int)
            
            for profile in self.profiles.values():
                for feature in profile.enabled_features:
                    feature_usage[feature.value] += 1
                
                font_size_distribution[profile.font_size.value] += 1
                contrast_preferences[profile.contrast_preference.value] += 1
                motion_preferences[profile.motion_preference.value] += 1
        
        return {
            "timestamp": time.time(),
            "total_users": total_users,
            "feature_usage": dict(feature_usage),
            "font_size_distribution": dict(font_size_distribution),
            "contrast_preferences": dict(contrast_preferences),
            "motion_preferences": dict(motion_preferences),
            "available_themes": list(self.theme_manager.themes.keys()),
            "wcag_guidelines_supported": len(self.auditor.guidelines),
            "screen_readers_supported": ScreenReaderSupport.COMMON_SCREEN_READERS
        }

# Global accessibility manager
_global_accessibility_manager: Optional[AccessibilityManager] = None

def get_accessibility_manager() -> Optional[AccessibilityManager]:
    """Get global accessibility manager"""
    return _global_accessibility_manager

def initialize_accessibility() -> AccessibilityManager:
    """Initialize global accessibility manager"""
    global _global_accessibility_manager
    _global_accessibility_manager = AccessibilityManager()
    return _global_accessibility_manager

# Convenience functions
def create_user_profile(user_id: str, preferences: Optional[Dict[str, Any]] = None) -> Optional[AccessibilityProfile]:
    """Create accessibility profile for user"""
    if _global_accessibility_manager:
        return _global_accessibility_manager.create_profile(user_id, preferences)
    return None

def get_user_css(user_id: str) -> str:
    """Get personalized CSS for user"""
    if _global_accessibility_manager:
        return _global_accessibility_manager.generate_user_css(user_id)
    return ""

def audit_component(component_data: Dict[str, Any]) -> Optional[AccessibilityAuditResult]:
    """Audit component accessibility"""
    if _global_accessibility_manager:
        return _global_accessibility_manager.audit_accessibility(component_data)
    return None

def validate_color_contrast(foreground: str, background: str, large_text: bool = False) -> ColorContrast:
    """Validate color contrast"""
    return ColorContrastCalculator.validate_contrast(foreground, background, large_text)