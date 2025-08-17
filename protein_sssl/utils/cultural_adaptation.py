"""
Cultural Adaptation Framework for protein-sssl-operator
Provides cultural considerations for scientific data, local scientific notation standards,
regional regulatory compliance, cultural color and symbol preferences,
and regional communication styles.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class CulturalRegion(Enum):
    """Cultural regions with distinct scientific traditions"""
    WESTERN_EUROPE = "western_europe"
    EASTERN_EUROPE = "eastern_europe"
    NORTH_AMERICA = "north_america"
    EAST_ASIA = "east_asia"
    SOUTHEAST_ASIA = "southeast_asia"
    SOUTH_ASIA = "south_asia"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    LATIN_AMERICA = "latin_america"
    OCEANIA = "oceania"

class ScientificNotation(Enum):
    """Scientific notation systems"""
    DECIMAL_POINT = "decimal_point"      # 1.234
    DECIMAL_COMMA = "decimal_comma"      # 1,234
    SCIENTIFIC_E = "scientific_e"        # 1.234e+05
    SCIENTIFIC_X10 = "scientific_x10"    # 1.234 × 10⁵
    ENGINEERING = "engineering"          # 123.4 × 10³

class UnitSystem(Enum):
    """Measurement unit systems"""
    SI_METRIC = "si_metric"              # Metric system (meters, grams, etc.)
    IMPERIAL = "imperial"                # Imperial system (feet, pounds, etc.)
    CGS = "cgs"                         # Centimeter-gram-second
    PLANCK = "planck"                   # Planck units
    ATOMIC = "atomic"                   # Atomic units

class ColorCulture(Enum):
    """Cultural color associations"""
    WESTERN = "western"                  # Red=danger, Green=success
    EAST_ASIAN = "east_asian"           # Red=luck/prosperity, White=mourning
    ISLAMIC = "islamic"                 # Green=sacred, Blue=protection
    AFRICAN = "african"                 # Earth tones, Red=life force
    INDIAN = "indian"                   # Saffron=sacred, Colors=chakras

class CommunicationStyle(Enum):
    """Regional communication preferences"""
    DIRECT = "direct"                   # Direct, explicit communication
    HIGH_CONTEXT = "high_context"       # Implicit, context-dependent
    HIERARCHICAL = "hierarchical"       # Formal, respectful of authority
    EGALITARIAN = "egalitarian"         # Informal, peer-to-peer
    COLLECTIVIST = "collectivist"       # Group-oriented decisions
    INDIVIDUALIST = "individualist"     # Individual-focused decisions

@dataclass
class CulturalPreferences:
    """Cultural preferences for a region"""
    region: CulturalRegion
    languages: List[str]
    scientific_notation: ScientificNotation
    unit_system: UnitSystem
    color_culture: ColorCulture
    communication_style: CommunicationStyle
    decimal_separator: str
    thousands_separator: str
    currency_position: str  # "before", "after"
    date_format: str
    time_format: str
    number_grouping: int  # 3 for thousands, 4 for myriads
    rtl_script: bool = False
    
    # Scientific traditions
    amino_acid_code: str = "single"  # "single", "three", "both"
    protein_naming: str = "uniprot"  # "uniprot", "local", "traditional"
    structure_representation: str = "cartoon"  # "cartoon", "space_fill", "wireframe"
    
    # Cultural colors
    success_color: str = "#28a745"
    warning_color: str = "#ffc107"
    danger_color: str = "#dc3545"
    info_color: str = "#17a2b8"
    primary_color: str = "#007bff"
    
    # Typography preferences
    font_family: str = "system-ui"
    font_weight: str = "normal"
    line_spacing: float = 1.5
    
    # Layout preferences
    reading_direction: str = "ltr"  # "ltr", "rtl"
    content_alignment: str = "left"  # "left", "right", "center"
    image_placement: str = "left"   # "left", "right", "center"

@dataclass
class ScientificConvention:
    """Scientific conventions for a region"""
    region: CulturalRegion
    amino_acid_codes: Dict[str, str]  # Three-letter to single-letter mapping
    protein_databases: List[str]      # Preferred protein databases
    structure_formats: List[str]      # Preferred structure formats
    citation_style: str              # Citation format preference
    measurement_precision: int       # Default decimal places
    significant_figures: int         # Default significant figures
    scientific_symbols: Dict[str, str]  # Symbol preferences
    nomenclature_system: str         # Naming convention system
    
    # Research paper conventions
    abstract_length: int = 250       # Preferred abstract length
    keyword_count: int = 6           # Preferred keyword count
    reference_style: str = "apa"     # Citation style
    
    # Data presentation
    graph_style: str = "line"        # Default graph type
    color_scheme: str = "colorblind" # Default color scheme
    axis_labels: str = "full"        # "full", "abbreviated"

@dataclass
class LocalizedContent:
    """Localized content for scientific terms"""
    term: str
    region: CulturalRegion
    translation: str
    context: str
    scientific_accuracy: float  # 0-1 rating
    usage_frequency: float      # 0-1 rating
    alternatives: List[str] = field(default_factory=list)
    etymology: Optional[str] = None
    cultural_notes: Optional[str] = None

class ScientificNotationConverter:
    """Convert between different scientific notation systems"""
    
    @staticmethod
    def format_number(value: float, 
                     notation: ScientificNotation,
                     decimal_sep: str = ".",
                     thousands_sep: str = ",",
                     precision: int = 3) -> str:
        """Format number according to notation system"""
        
        if notation == ScientificNotation.DECIMAL_POINT:
            # Standard decimal notation
            formatted = f"{value:.{precision}f}"
            
        elif notation == ScientificNotation.DECIMAL_COMMA:
            # European notation with comma as decimal separator
            formatted = f"{value:.{precision}f}".replace(".", ",")
            
        elif notation == ScientificNotation.SCIENTIFIC_E:
            # Scientific notation with 'e'
            formatted = f"{value:.{precision}e}"
            
        elif notation == ScientificNotation.SCIENTIFIC_X10:
            # Scientific notation with × 10^n
            exp = int(f"{value:.0e}".split('e')[1])
            mantissa = value / (10 ** exp)
            formatted = f"{mantissa:.{precision}f} × 10^{exp}"
            
        elif notation == ScientificNotation.ENGINEERING:
            # Engineering notation (powers of 1000)
            exp = int(f"{value:.0e}".split('e')[1])
            eng_exp = (exp // 3) * 3
            mantissa = value / (10 ** eng_exp)
            formatted = f"{mantissa:.{precision}f} × 10^{eng_exp}"
            
        else:
            formatted = str(value)
        
        # Apply separators
        if thousands_sep and "." in formatted:
            integer_part, decimal_part = formatted.split(".")
            if len(integer_part) > 3:
                # Add thousands separators
                reversed_int = integer_part[::-1]
                grouped = [reversed_int[i:i+3] for i in range(0, len(reversed_int), 3)]
                integer_part = thousands_sep.join(grouped)[::-1]
            formatted = f"{integer_part}{decimal_sep}{decimal_part}"
        
        return formatted
    
    @staticmethod
    def format_scientific_value(value: float,
                              unit: str,
                              preferences: CulturalPreferences) -> str:
        """Format scientific value with unit according to cultural preferences"""
        
        formatted_number = ScientificNotationConverter.format_number(
            value,
            preferences.scientific_notation,
            preferences.decimal_separator,
            preferences.thousands_separator
        )
        
        return f"{formatted_number} {unit}"

class ProteinNomenclatureManager:
    """Manage protein nomenclature across cultures"""
    
    def __init__(self):
        self.nomenclature_systems = self._load_nomenclature_systems()
        self.amino_acid_mappings = self._load_amino_acid_mappings()
    
    def _load_nomenclature_systems(self) -> Dict[str, Dict[str, Any]]:
        """Load different protein nomenclature systems"""
        return {
            "uniprot": {
                "name": "UniProt",
                "description": "Universal Protein Database naming",
                "regions": [CulturalRegion.WESTERN_EUROPE, CulturalRegion.NORTH_AMERICA],
                "format": "P{accession}",
                "example": "P04637 (p53 tumor suppressor)"
            },
            "japanese": {
                "name": "Japanese Protein Names",
                "description": "Traditional Japanese protein naming",
                "regions": [CulturalRegion.EAST_ASIA],
                "format": "Traditional kanji + function",
                "example": "蛋白質 p53 (tanpaku-shitsu p53)"
            },
            "chinese": {
                "name": "Chinese Protein Names", 
                "description": "Chinese scientific naming convention",
                "regions": [CulturalRegion.EAST_ASIA],
                "format": "Chinese characters + latin",
                "example": "蛋白质 p53 (dànbáizhì p53)"
            },
            "arabic": {
                "name": "Arabic Scientific Names",
                "description": "Arabic scientific terminology",
                "regions": [CulturalRegion.MIDDLE_EAST],
                "format": "Arabic script with transliteration",
                "example": "بروتين p53 (brutin p53)"
            }
        }
    
    def _load_amino_acid_mappings(self) -> Dict[CulturalRegion, Dict[str, str]]:
        """Load amino acid code preferences by region"""
        return {
            CulturalRegion.WESTERN_EUROPE: {
                "Alanine": "A", "Glycine": "G", "Leucine": "L",
                "preference": "single_letter"
            },
            CulturalRegion.EAST_ASIA: {
                "Alanine": "Ala", "Glycine": "Gly", "Leucine": "Leu", 
                "preference": "three_letter"
            },
            CulturalRegion.MIDDLE_EAST: {
                "Alanine": "A (آل)", "Glycine": "G (غلي)", "Leucine": "L (لو)",
                "preference": "bilingual"
            }
        }
    
    def get_protein_name(self, protein_id: str, 
                        region: CulturalRegion,
                        system: str = "uniprot") -> str:
        """Get culturally appropriate protein name"""
        
        system_info = self.nomenclature_systems.get(system, {})
        
        if region in system_info.get("regions", []):
            # Use preferred system for this region
            return f"{protein_id} ({system_info.get('name', 'Standard')})"
        else:
            # Fall back to UniProt with localization note
            return f"{protein_id} (UniProt standard)"
    
    def get_amino_acid_representation(self, amino_acid: str,
                                    region: CulturalRegion) -> str:
        """Get culturally appropriate amino acid representation"""
        
        mapping = self.amino_acid_mappings.get(region, {})
        return mapping.get(amino_acid, amino_acid)

class CulturalColorManager:
    """Manage culturally appropriate colors"""
    
    COLOR_MEANINGS = {
        ColorCulture.WESTERN: {
            "red": "danger, error, stop",
            "green": "success, go, safe",
            "yellow": "warning, caution",
            "blue": "information, trust",
            "purple": "luxury, creativity",
            "orange": "energy, enthusiasm",
            "black": "sophistication, formality",
            "white": "purity, cleanliness"
        },
        ColorCulture.EAST_ASIAN: {
            "red": "luck, prosperity, celebration",
            "gold": "wealth, honor",
            "green": "growth, harmony",
            "blue": "immortality, healing",
            "white": "mourning, death",
            "black": "mystery, formality",
            "purple": "nobility, spirituality",
            "pink": "femininity, love"
        },
        ColorCulture.ISLAMIC: {
            "green": "paradise, nature, Islam",
            "blue": "protection, spirituality",
            "white": "purity, peace",
            "gold": "wealth, divine",
            "black": "elegance, formality",
            "red": "courage, strength",
            "purple": "luxury, royalty",
            "brown": "earth, stability"
        },
        ColorCulture.AFRICAN: {
            "red": "life force, vitality",
            "black": "maturity, masculinity",
            "white": "purity, truth",
            "green": "fertility, growth",
            "yellow": "wealth, status",
            "blue": "love, peace",
            "brown": "earth, stability",
            "orange": "creativity, healing"
        },
        ColorCulture.INDIAN: {
            "saffron": "sacred, courage",
            "red": "power, passion",
            "green": "nature, peace",
            "blue": "divine, infinite",
            "yellow": "knowledge, learning",
            "white": "purity, truth",
            "purple": "spirituality, luxury",
            "pink": "compassion, femininity"
        }
    }
    
    @staticmethod
    def get_culturally_appropriate_colors(culture: ColorCulture,
                                        purpose: str) -> Dict[str, str]:
        """Get colors appropriate for culture and purpose"""
        
        base_colors = {
            "primary": "#007bff",
            "success": "#28a745", 
            "warning": "#ffc107",
            "danger": "#dc3545",
            "info": "#17a2b8"
        }
        
        if culture == ColorCulture.EAST_ASIAN:
            # Red is auspicious, avoid white for UI elements
            base_colors.update({
                "success": "#d4a574",  # Gold instead of green
                "primary": "#c8102e",  # Red for primary
                "warning": "#ff6b35",  # Orange instead of yellow
                "info": "#4a90a4"     # Muted blue
            })
            
        elif culture == ColorCulture.ISLAMIC:
            # Green is sacred, use respectfully
            base_colors.update({
                "primary": "#2e8b57", # Sea green
                "success": "#228b22", # Forest green  
                "info": "#4682b4",   # Steel blue
                "warning": "#daa520"  # Goldenrod
            })
            
        elif culture == ColorCulture.AFRICAN:
            # Earth tones and vibrant colors
            base_colors.update({
                "primary": "#8b4513",  # Saddle brown
                "success": "#32cd32",  # Lime green
                "warning": "#ff8c00",  # Dark orange
                "info": "#20b2aa"     # Light sea green
            })
            
        elif culture == ColorCulture.INDIAN:
            # Rich, vibrant colors
            base_colors.update({
                "primary": "#ff6600",  # Saffron
                "success": "#00b300",  # Green
                "warning": "#ffcc00",  # Golden yellow
                "info": "#0066cc"     # Royal blue
            })
        
        return base_colors
    
    @staticmethod
    def validate_color_appropriateness(color: str, 
                                     culture: ColorCulture,
                                     context: str) -> Tuple[bool, str]:
        """Validate if color is culturally appropriate"""
        
        # This is a simplified validation - in practice would be more comprehensive
        warnings = []
        
        if culture == ColorCulture.EAST_ASIAN:
            if color.lower() in ["white", "#ffffff", "#fff"] and context in ["background", "primary"]:
                warnings.append("White may have negative connotations in East Asian cultures")
        
        elif culture == ColorCulture.ISLAMIC:
            if color.lower() in ["green", "#008000", "#00ff00"] and context == "error":
                warnings.append("Green is sacred in Islamic culture, avoid for negative contexts")
        
        elif culture == ColorCulture.INDIAN:
            if color.lower() in ["black", "#000000", "#000"] and context in ["celebration", "success"]:
                warnings.append("Black may not be appropriate for positive contexts in Indian culture")
        
        return len(warnings) == 0, "; ".join(warnings)

class RegionalCommunicationManager:
    """Manage communication styles across regions"""
    
    COMMUNICATION_GUIDELINES = {
        CommunicationStyle.DIRECT: {
            "tone": "Clear and explicit",
            "error_messages": "Direct problem statement with solution",
            "instructions": "Step-by-step, imperative mood",
            "feedback": "Specific and immediate",
            "example": "Error: Invalid input. Please enter a valid protein sequence."
        },
        
        CommunicationStyle.HIGH_CONTEXT: {
            "tone": "Polite and contextual",
            "error_messages": "Gentle explanation with context",
            "instructions": "Guided suggestions with reasoning",
            "feedback": "Considerate and constructive",
            "example": "It appears the protein sequence may need adjustment for optimal results."
        },
        
        CommunicationStyle.HIERARCHICAL: {
            "tone": "Formal and respectful",
            "error_messages": "Formal notification with deference",
            "instructions": "Respectful requests and suggestions",
            "feedback": "Formal acknowledgment",
            "example": "We respectfully suggest reviewing the protein sequence format."
        },
        
        CommunicationStyle.EGALITARIAN: {
            "tone": "Casual and collaborative",
            "error_messages": "Friendly problem-solving approach",
            "instructions": "Collaborative suggestions",
            "feedback": "Peer-to-peer communication",
            "example": "Looks like we need to check the protein sequence format together."
        },
        
        CommunicationStyle.COLLECTIVIST: {
            "tone": "Group-oriented",
            "error_messages": "Focus on collective success",
            "instructions": "Emphasize team benefits",
            "feedback": "Group achievement focus",
            "example": "Let's work together to ensure the protein sequence is formatted correctly."
        },
        
        CommunicationStyle.INDIVIDUALIST: {
            "tone": "Personal achievement focused",
            "error_messages": "Individual control emphasis",
            "instructions": "Personal responsibility",
            "feedback": "Individual accomplishment",
            "example": "You can improve your results by adjusting the protein sequence format."
        }
    }
    
    @staticmethod
    def adapt_message(message: str,
                     style: CommunicationStyle,
                     message_type: str = "info") -> str:
        """Adapt message to communication style"""
        
        guidelines = RegionalCommunicationManager.COMMUNICATION_GUIDELINES.get(style, {})
        
        if message_type == "error":
            # Apply error message style
            if style == CommunicationStyle.HIGH_CONTEXT:
                return f"We notice that {message.lower()}. Perhaps reviewing the input might help."
            elif style == CommunicationStyle.HIERARCHICAL:
                return f"We respectfully inform you that {message.lower()}."
            elif style == CommunicationStyle.COLLECTIVIST:
                return f"Our system encountered: {message}. Let's resolve this together."
            else:
                return message
        
        elif message_type == "success":
            if style == CommunicationStyle.COLLECTIVIST:
                return f"Great work! {message}"
            elif style == CommunicationStyle.HIERARCHICAL:
                return f"We are pleased to confirm: {message}"
            else:
                return message
        
        return message

class CulturalAdaptationManager:
    """Main cultural adaptation management system"""
    
    def __init__(self):
        self.cultural_preferences = self._load_cultural_preferences()
        self.scientific_conventions = self._load_scientific_conventions()
        self.nomenclature_manager = ProteinNomenclatureManager()
        self.color_manager = CulturalColorManager()
        self.communication_manager = RegionalCommunicationManager()
        self.localized_content: Dict[str, List[LocalizedContent]] = defaultdict(list)
        self._lock = threading.RLock()
        
        logger.info("Cultural adaptation manager initialized")
    
    def _load_cultural_preferences(self) -> Dict[CulturalRegion, CulturalPreferences]:
        """Load cultural preferences for each region"""
        return {
            CulturalRegion.WESTERN_EUROPE: CulturalPreferences(
                region=CulturalRegion.WESTERN_EUROPE,
                languages=["en", "fr", "de", "es", "it"],
                scientific_notation=ScientificNotation.DECIMAL_COMMA,
                unit_system=UnitSystem.SI_METRIC,
                color_culture=ColorCulture.WESTERN,
                communication_style=CommunicationStyle.DIRECT,
                decimal_separator=",",
                thousands_separator=".",
                currency_position="after",
                date_format="DD/MM/YYYY",
                time_format="24h",
                number_grouping=3
            ),
            
            CulturalRegion.NORTH_AMERICA: CulturalPreferences(
                region=CulturalRegion.NORTH_AMERICA,
                languages=["en", "es", "fr"],
                scientific_notation=ScientificNotation.DECIMAL_POINT,
                unit_system=UnitSystem.SI_METRIC,
                color_culture=ColorCulture.WESTERN,
                communication_style=CommunicationStyle.INDIVIDUALIST,
                decimal_separator=".",
                thousands_separator=",",
                currency_position="before",
                date_format="MM/DD/YYYY",
                time_format="12h",
                number_grouping=3
            ),
            
            CulturalRegion.EAST_ASIA: CulturalPreferences(
                region=CulturalRegion.EAST_ASIA,
                languages=["ja", "ko", "zh"],
                scientific_notation=ScientificNotation.SCIENTIFIC_X10,
                unit_system=UnitSystem.SI_METRIC,
                color_culture=ColorCulture.EAST_ASIAN,
                communication_style=CommunicationStyle.HIGH_CONTEXT,
                decimal_separator=".",
                thousands_separator=",",
                currency_position="before",
                date_format="YYYY/MM/DD",
                time_format="24h",
                number_grouping=4,  # Myriads system
                amino_acid_code="three",
                protein_naming="local"
            ),
            
            CulturalRegion.MIDDLE_EAST: CulturalPreferences(
                region=CulturalRegion.MIDDLE_EAST,
                languages=["ar", "fa", "tr", "he"],
                scientific_notation=ScientificNotation.DECIMAL_POINT,
                unit_system=UnitSystem.SI_METRIC,
                color_culture=ColorCulture.ISLAMIC,
                communication_style=CommunicationStyle.HIERARCHICAL,
                decimal_separator=".",
                thousands_separator=",",
                currency_position="after",
                date_format="DD/MM/YYYY",
                time_format="12h",
                number_grouping=3,
                rtl_script=True,
                reading_direction="rtl",
                content_alignment="right"
            ),
            
            CulturalRegion.SOUTH_ASIA: CulturalPreferences(
                region=CulturalRegion.SOUTH_ASIA,
                languages=["hi", "bn", "ta", "te", "ur"],
                scientific_notation=ScientificNotation.SCIENTIFIC_E,
                unit_system=UnitSystem.SI_METRIC,
                color_culture=ColorCulture.INDIAN,
                communication_style=CommunicationStyle.HIERARCHICAL,
                decimal_separator=".",
                thousands_separator=",",
                currency_position="before",
                date_format="DD/MM/YYYY",
                time_format="12h",
                number_grouping=3,
                success_color="#ff6600",  # Saffron
                primary_color="#0066cc"
            )
        }
    
    def _load_scientific_conventions(self) -> Dict[CulturalRegion, ScientificConvention]:
        """Load scientific conventions for each region"""
        return {
            CulturalRegion.WESTERN_EUROPE: ScientificConvention(
                region=CulturalRegion.WESTERN_EUROPE,
                amino_acid_codes={"ALA": "A", "GLY": "G", "LEU": "L"},
                protein_databases=["UniProt", "PDB", "EBI"],
                structure_formats=["PDB", "mmCIF", "PDBx"],
                citation_style="nature",
                measurement_precision=3,
                significant_figures=4,
                scientific_symbols={"angstrom": "Å", "degree": "°"},
                nomenclature_system="iupac"
            ),
            
            CulturalRegion.EAST_ASIA: ScientificConvention(
                region=CulturalRegion.EAST_ASIA,
                amino_acid_codes={"ALA": "Ala", "GLY": "Gly", "LEU": "Leu"},
                protein_databases=["PDBj", "UniProt", "KEGG"],
                structure_formats=["PDB", "mmCIF"],
                citation_style="japanese",
                measurement_precision=4,
                significant_figures=5,
                scientific_symbols={"angstrom": "Å", "degree": "°"},
                nomenclature_system="local_japanese",
                graph_style="bar",
                color_scheme="traditional"
            )
        }
    
    def get_cultural_preferences(self, region: CulturalRegion) -> Optional[CulturalPreferences]:
        """Get cultural preferences for region"""
        return self.cultural_preferences.get(region)
    
    def get_scientific_conventions(self, region: CulturalRegion) -> Optional[ScientificConvention]:
        """Get scientific conventions for region"""
        return self.scientific_conventions.get(region)
    
    def adapt_content_for_region(self, content: Dict[str, Any], 
                               region: CulturalRegion) -> Dict[str, Any]:
        """Adapt content for cultural region"""
        
        preferences = self.get_cultural_preferences(region)
        conventions = self.get_scientific_conventions(region)
        
        if not preferences:
            return content
        
        adapted_content = content.copy()
        
        # Adapt colors
        if "colors" in content:
            appropriate_colors = self.color_manager.get_culturally_appropriate_colors(
                preferences.color_culture, "interface"
            )
            adapted_content["colors"] = appropriate_colors
        
        # Adapt number formatting
        if "numbers" in content:
            for key, value in content["numbers"].items():
                if isinstance(value, (int, float)):
                    adapted_content["numbers"][key] = ScientificNotationConverter.format_number(
                        value, preferences.scientific_notation,
                        preferences.decimal_separator,
                        preferences.thousands_separator
                    )
        
        # Adapt protein names
        if "proteins" in content:
            for i, protein in enumerate(content["proteins"]):
                if "name" in protein:
                    adapted_content["proteins"][i]["name"] = self.nomenclature_manager.get_protein_name(
                        protein["name"], region
                    )
        
        # Adapt messages
        if "messages" in content:
            for message_type, messages in content["messages"].items():
                if isinstance(messages, list):
                    adapted_messages = []
                    for message in messages:
                        adapted_message = self.communication_manager.adapt_message(
                            message, preferences.communication_style, message_type
                        )
                        adapted_messages.append(adapted_message)
                    adapted_content["messages"][message_type] = adapted_messages
        
        # Adapt layout for RTL if needed
        if preferences.rtl_script:
            adapted_content["layout"] = {
                "direction": "rtl",
                "text_align": "right",
                "float_default": "right"
            }
        
        return adapted_content
    
    def get_localized_scientific_term(self, term: str, 
                                    region: CulturalRegion) -> str:
        """Get localized scientific term"""
        
        # Look for existing localization
        for content in self.localized_content[term]:
            if content.region == region:
                return content.translation
        
        # Return original term if no localization found
        return term
    
    def add_localized_content(self, content: LocalizedContent):
        """Add localized content"""
        with self._lock:
            self.localized_content[content.term].append(content)
        
        logger.info(f"Added localized content for '{content.term}' in {content.region.value}")
    
    def validate_cultural_appropriateness(self, content: Dict[str, Any],
                                        region: CulturalRegion) -> Tuple[bool, List[str]]:
        """Validate content for cultural appropriateness"""
        
        issues = []
        preferences = self.get_cultural_preferences(region)
        
        if not preferences:
            return True, []
        
        # Check colors
        if "colors" in content:
            for color_name, color_value in content["colors"].items():
                is_appropriate, warning = self.color_manager.validate_color_appropriateness(
                    color_value, preferences.color_culture, color_name
                )
                if not is_appropriate:
                    issues.append(f"Color issue: {warning}")
        
        # Check text direction for RTL languages
        if preferences.rtl_script and content.get("layout", {}).get("direction") != "rtl":
            issues.append("Content should use RTL layout for this region")
        
        # Check communication style
        if "tone" in content:
            expected_tone = self.communication_manager.COMMUNICATION_GUIDELINES.get(
                preferences.communication_style, {}
            ).get("tone", "")
            if expected_tone and expected_tone.lower() not in content["tone"].lower():
                issues.append(f"Communication style should be: {expected_tone}")
        
        return len(issues) == 0, issues
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Generate cultural adaptation report"""
        
        with self._lock:
            total_regions = len(CulturalRegion)
            supported_regions = len(self.cultural_preferences)
            
            localization_stats = {}
            for term, contents in self.localized_content.items():
                localization_stats[term] = {
                    "total_localizations": len(contents),
                    "regions": [c.region.value for c in contents]
                }
        
        return {
            "timestamp": time.time(),
            "total_regions": total_regions,
            "supported_regions": supported_regions,
            "cultural_preferences_loaded": list(self.cultural_preferences.keys()),
            "scientific_conventions_loaded": list(self.scientific_conventions.keys()),
            "localized_terms": len(self.localized_content),
            "localization_statistics": localization_stats,
            "color_cultures_supported": [c.value for c in ColorCulture],
            "communication_styles_supported": [s.value for s in CommunicationStyle],
            "scientific_notations_supported": [n.value for n in ScientificNotation]
        }

# Global cultural adaptation manager
_global_cultural_manager: Optional[CulturalAdaptationManager] = None

def get_cultural_manager() -> Optional[CulturalAdaptationManager]:
    """Get global cultural adaptation manager"""
    return _global_cultural_manager

def initialize_cultural_adaptation() -> CulturalAdaptationManager:
    """Initialize global cultural adaptation manager"""
    global _global_cultural_manager
    _global_cultural_manager = CulturalAdaptationManager()
    return _global_cultural_manager

# Convenience functions
def adapt_content(content: Dict[str, Any], region: CulturalRegion) -> Dict[str, Any]:
    """Adapt content for cultural region"""
    if _global_cultural_manager:
        return _global_cultural_manager.adapt_content_for_region(content, region)
    return content

def format_scientific_number(value: float, region: CulturalRegion) -> str:
    """Format number according to regional scientific notation"""
    if _global_cultural_manager:
        preferences = _global_cultural_manager.get_cultural_preferences(region)
        if preferences:
            return ScientificNotationConverter.format_number(
                value, preferences.scientific_notation,
                preferences.decimal_separator,
                preferences.thousands_separator
            )
    return str(value)

def get_appropriate_colors(region: CulturalRegion) -> Dict[str, str]:
    """Get culturally appropriate colors for region"""
    if _global_cultural_manager:
        preferences = _global_cultural_manager.get_cultural_preferences(region)
        if preferences:
            return CulturalColorManager.get_culturally_appropriate_colors(
                preferences.color_culture, "interface"
            )
    return {"primary": "#007bff", "success": "#28a745", "warning": "#ffc107", "danger": "#dc3545"}