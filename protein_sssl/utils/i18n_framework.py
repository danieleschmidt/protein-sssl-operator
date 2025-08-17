"""
Comprehensive Internationalization (i18n) Framework for protein-sssl-operator
Provides multi-language support, dynamic language switching, localized error messages,
timezone handling, currency formatting, and RTL language support.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
from pathlib import Path
import locale
import gettext
import re
from collections import defaultdict
import functools

logger = logging.getLogger(__name__)

class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    ARABIC = "ar"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    KOREAN = "ko"
    HINDI = "hi"
    DUTCH = "nl"

class Currency(Enum):
    """Supported currencies"""
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    CHF = "CHF"  # Swiss Franc
    KRW = "KRW"  # South Korean Won
    INR = "INR"  # Indian Rupee
    BRL = "BRL"  # Brazilian Real
    RUB = "RUB"  # Russian Ruble

class NumberFormat(Enum):
    """Number formatting styles"""
    DECIMAL_POINT = "decimal_point"  # 1,234.56
    DECIMAL_COMMA = "decimal_comma"  # 1.234,56
    SPACE_COMMA = "space_comma"      # 1 234,56
    APOSTROPHE_POINT = "apostrophe_point"  # 1'234.56

class DateFormat(Enum):
    """Date formatting styles"""
    MDY = "mdy"  # MM/DD/YYYY (US)
    DMY = "dmy"  # DD/MM/YYYY (Europe)
    YMD = "ymd"  # YYYY/MM/DD (Asia)
    ISO = "iso"  # YYYY-MM-DD (ISO 8601)

@dataclass
class LocaleConfig:
    """Configuration for a specific locale"""
    language: Language
    country_code: str
    currency: Currency
    timezone: str
    number_format: NumberFormat
    date_format: DateFormat
    is_rtl: bool = False
    decimal_separator: str = "."
    thousands_separator: str = ","
    date_separator: str = "/"
    time_format_24h: bool = True
    first_day_of_week: int = 1  # 1=Monday, 0=Sunday
    
    def __post_init__(self):
        """Set format-specific separators"""
        if self.number_format == NumberFormat.DECIMAL_COMMA:
            self.decimal_separator = ","
            self.thousands_separator = "."
        elif self.number_format == NumberFormat.SPACE_COMMA:
            self.decimal_separator = ","
            self.thousands_separator = " "
        elif self.number_format == NumberFormat.APOSTROPHE_POINT:
            self.decimal_separator = "."
            self.thousands_separator = "'"
        
        if self.date_format == DateFormat.ISO:
            self.date_separator = "-"

@dataclass
class TranslationEntry:
    """Single translation entry"""
    key: str
    value: str
    context: Optional[str] = None
    plurals: Optional[Dict[str, str]] = None
    variables: Optional[List[str]] = None

@dataclass
class LocalizedError:
    """Localized error message"""
    code: str
    message_key: str
    severity: str = "error"
    category: str = "general"
    variables: Optional[Dict[str, Any]] = None

class RTLSupport:
    """Right-to-left language support utilities"""
    
    RTL_LANGUAGES = {
        Language.ARABIC,
    }
    
    @staticmethod
    def is_rtl_language(language: Language) -> bool:
        """Check if language is right-to-left"""
        return language in RTLSupport.RTL_LANGUAGES
    
    @staticmethod
    def apply_rtl_formatting(text: str, language: Language) -> str:
        """Apply RTL formatting to text"""
        if not RTLSupport.is_rtl_language(language):
            return text
        
        # Add RTL markers
        return f"\u202B{text}\u202C"  # RLE + text + PDF
    
    @staticmethod
    def format_mixed_content(text: str, language: Language) -> str:
        """Format text with mixed LTR/RTL content"""
        if not RTLSupport.is_rtl_language(language):
            return text
        
        # Handle mixed content (e.g., Arabic text with English numbers/terms)
        # This is a simplified implementation
        return text

class NumberFormatter:
    """Number formatting utilities"""
    
    @staticmethod
    def format_number(number: Union[int, float], 
                     locale_config: LocaleConfig,
                     decimal_places: Optional[int] = None) -> str:
        """Format number according to locale"""
        
        if decimal_places is not None:
            formatted = f"{number:.{decimal_places}f}"
        else:
            formatted = str(number)
        
        # Split into integer and decimal parts
        if '.' in formatted:
            integer_part, decimal_part = formatted.split('.')
        else:
            integer_part, decimal_part = formatted, ""
        
        # Add thousands separators
        if len(integer_part) > 3:
            # Add separators from right to left
            reversed_int = integer_part[::-1]
            groups = [reversed_int[i:i+3] for i in range(0, len(reversed_int), 3)]
            integer_part = locale_config.thousands_separator.join(groups)[::-1]
        
        # Combine with decimal separator
        if decimal_part:
            return f"{integer_part}{locale_config.decimal_separator}{decimal_part}"
        else:
            return integer_part
    
    @staticmethod
    def format_currency(amount: float,
                       currency: Currency,
                       locale_config: LocaleConfig) -> str:
        """Format currency amount"""
        
        # Format the number
        formatted_amount = NumberFormatter.format_number(amount, locale_config, 2)
        
        # Currency symbols and positions
        currency_symbols = {
            Currency.USD: "$",
            Currency.EUR: "€",
            Currency.GBP: "£",
            Currency.JPY: "¥",
            Currency.CNY: "¥",
            Currency.CAD: "CA$",
            Currency.AUD: "AU$",
            Currency.CHF: "CHF",
            Currency.KRW: "₩",
            Currency.INR: "₹",
            Currency.BRL: "R$",
            Currency.RUB: "₽"
        }
        
        symbol = currency_symbols.get(currency, currency.value)
        
        # Position based on locale
        if locale_config.language in [Language.ENGLISH]:
            return f"{symbol}{formatted_amount}"
        elif locale_config.language in [Language.FRENCH, Language.GERMAN]:
            return f"{formatted_amount} {symbol}"
        else:
            return f"{symbol} {formatted_amount}"
    
    @staticmethod
    def format_percentage(value: float, locale_config: LocaleConfig) -> str:
        """Format percentage"""
        formatted = NumberFormatter.format_number(value * 100, locale_config, 1)
        return f"{formatted}%"

class DateTimeFormatter:
    """Date and time formatting utilities"""
    
    @staticmethod
    def format_date(dt: datetime, locale_config: LocaleConfig) -> str:
        """Format date according to locale"""
        
        if locale_config.date_format == DateFormat.MDY:
            return dt.strftime(f"%m{locale_config.date_separator}%d{locale_config.date_separator}%Y")
        elif locale_config.date_format == DateFormat.DMY:
            return dt.strftime(f"%d{locale_config.date_separator}%m{locale_config.date_separator}%Y")
        elif locale_config.date_format == DateFormat.YMD:
            return dt.strftime(f"%Y{locale_config.date_separator}%m{locale_config.date_separator}%d")
        elif locale_config.date_format == DateFormat.ISO:
            return dt.strftime("%Y-%m-%d")
        else:
            return dt.strftime("%Y-%m-%d")
    
    @staticmethod
    def format_time(dt: datetime, locale_config: LocaleConfig) -> str:
        """Format time according to locale"""
        
        if locale_config.time_format_24h:
            return dt.strftime("%H:%M:%S")
        else:
            return dt.strftime("%I:%M:%S %p")
    
    @staticmethod
    def format_datetime(dt: datetime, locale_config: LocaleConfig) -> str:
        """Format date and time"""
        date_str = DateTimeFormatter.format_date(dt, locale_config)
        time_str = DateTimeFormatter.format_time(dt, locale_config)
        return f"{date_str} {time_str}"
    
    @staticmethod
    def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """Convert datetime between timezones"""
        try:
            import pytz
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
            
            # Localize if naive
            if dt.tzinfo is None:
                dt = from_timezone.localize(dt)
            
            return dt.astimezone(to_timezone)
        except ImportError:
            logger.warning("pytz not available, timezone conversion disabled")
            return dt

class TranslationManager:
    """Manages translations and localization"""
    
    def __init__(self, translations_dir: str = "translations"):
        self.translations_dir = Path(translations_dir)
        self.translations: Dict[Language, Dict[str, TranslationEntry]] = {}
        self.fallback_language = Language.ENGLISH
        self._lock = threading.RLock()
        
        # Load all translations
        self._load_translations()
    
    def _load_translations(self):
        """Load all translation files"""
        if not self.translations_dir.exists():
            logger.warning(f"Translations directory not found: {self.translations_dir}")
            self._create_default_translations()
            return
        
        for lang in Language:
            translation_file = self.translations_dir / f"{lang.value}.json"
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    translations = {}
                    for key, value in data.items():
                        if isinstance(value, dict):
                            translations[key] = TranslationEntry(
                                key=key,
                                value=value.get('value', ''),
                                context=value.get('context'),
                                plurals=value.get('plurals'),
                                variables=value.get('variables', [])
                            )
                        else:
                            translations[key] = TranslationEntry(key=key, value=str(value))
                    
                    self.translations[lang] = translations
                    logger.info(f"Loaded {len(translations)} translations for {lang.value}")
                    
                except Exception as e:
                    logger.error(f"Error loading translations for {lang.value}: {e}")
    
    def _create_default_translations(self):
        """Create default English translations"""
        default_translations = {
            # General
            "app.name": "Protein SSSL Operator",
            "app.description": "Self-Supervised Learning for Protein Structure Prediction",
            
            # Navigation
            "nav.dashboard": "Dashboard",
            "nav.models": "Models",
            "nav.data": "Data",
            "nav.analysis": "Analysis",
            "nav.settings": "Settings",
            
            # Actions
            "action.save": "Save",
            "action.cancel": "Cancel",
            "action.delete": "Delete",
            "action.edit": "Edit",
            "action.create": "Create",
            "action.upload": "Upload",
            "action.download": "Download",
            
            # Status
            "status.loading": "Loading...",
            "status.saving": "Saving...",
            "status.success": "Success",
            "status.error": "Error",
            "status.warning": "Warning",
            "status.info": "Information",
            
            # Errors
            "error.generic": "An error occurred",
            "error.network": "Network error",
            "error.validation": "Validation error",
            "error.auth": "Authentication error",
            "error.permission": "Permission denied",
            "error.not_found": "Not found",
            "error.server": "Server error",
            
            # Scientific terms
            "science.protein": "Protein",
            "science.sequence": "Sequence",
            "science.structure": "Structure",
            "science.fold": "Fold",
            "science.domain": "Domain",
            "science.residue": "Residue",
            "science.amino_acid": "Amino Acid",
            "science.alpha_helix": "Alpha Helix",
            "science.beta_sheet": "Beta Sheet",
            "science.confidence": "Confidence",
            "science.prediction": "Prediction",
            
            # Units
            "unit.angstrom": "Å",
            "unit.kilodalton": "kDa",
            "unit.percent": "%",
            "unit.degree": "°",
            
            # Privacy and compliance
            "privacy.consent": "Consent",
            "privacy.data_protection": "Data Protection",
            "privacy.cookies": "Cookies",
            "privacy.terms": "Terms of Service",
            "privacy.policy": "Privacy Policy",
            
            # Time
            "time.second": {"value": "second", "plurals": {"one": "second", "other": "seconds"}},
            "time.minute": {"value": "minute", "plurals": {"one": "minute", "other": "minutes"}},
            "time.hour": {"value": "hour", "plurals": {"one": "hour", "other": "hours"}},
            "time.day": {"value": "day", "plurals": {"one": "day", "other": "days"}},
            "time.week": {"value": "week", "plurals": {"one": "week", "other": "weeks"}},
            "time.month": {"value": "month", "plurals": {"one": "month", "other": "months"}},
            "time.year": {"value": "year", "plurals": {"one": "year", "other": "years"}},
        }
        
        # Create translations directory
        self.translations_dir.mkdir(exist_ok=True, parents=True)
        
        # Save English translations
        en_file = self.translations_dir / "en.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        # Load into memory
        translations = {}
        for key, value in default_translations.items():
            if isinstance(value, dict):
                translations[key] = TranslationEntry(
                    key=key,
                    value=value.get('value', ''),
                    context=value.get('context'),
                    plurals=value.get('plurals'),
                    variables=value.get('variables', [])
                )
            else:
                translations[key] = TranslationEntry(key=key, value=str(value))
        
        self.translations[Language.ENGLISH] = translations
        logger.info("Created default English translations")
    
    def get_translation(self, 
                       key: str, 
                       language: Language,
                       variables: Optional[Dict[str, Any]] = None,
                       count: Optional[int] = None,
                       fallback: Optional[str] = None) -> str:
        """Get translation for key"""
        
        with self._lock:
            # Try requested language
            if language in self.translations:
                if key in self.translations[language]:
                    entry = self.translations[language][key]
                    return self._format_translation(entry, variables, count)
            
            # Try fallback language
            if (self.fallback_language in self.translations and 
                key in self.translations[self.fallback_language]):
                entry = self.translations[self.fallback_language][key]
                return self._format_translation(entry, variables, count)
            
            # Return fallback or key
            return fallback or key
    
    def _format_translation(self, 
                          entry: TranslationEntry,
                          variables: Optional[Dict[str, Any]] = None,
                          count: Optional[int] = None) -> str:
        """Format translation with variables and plurals"""
        
        # Handle plurals
        if count is not None and entry.plurals:
            if count == 1 and 'one' in entry.plurals:
                text = entry.plurals['one']
            elif 'other' in entry.plurals:
                text = entry.plurals['other']
            else:
                text = entry.value
        else:
            text = entry.value
        
        # Handle variables
        if variables:
            try:
                text = text.format(**variables)
            except KeyError as e:
                logger.warning(f"Missing variable in translation '{entry.key}': {e}")
        
        return text
    
    def add_translation(self, 
                       language: Language,
                       key: str,
                       value: str,
                       context: Optional[str] = None,
                       plurals: Optional[Dict[str, str]] = None):
        """Add or update translation"""
        
        with self._lock:
            if language not in self.translations:
                self.translations[language] = {}
            
            entry = TranslationEntry(
                key=key,
                value=value,
                context=context,
                plurals=plurals
            )
            
            self.translations[language][key] = entry
    
    def get_missing_translations(self, target_language: Language) -> List[str]:
        """Get list of missing translation keys for language"""
        
        if Language.ENGLISH not in self.translations:
            return []
        
        english_keys = set(self.translations[Language.ENGLISH].keys())
        
        if target_language not in self.translations:
            return list(english_keys)
        
        target_keys = set(self.translations[target_language].keys())
        return list(english_keys - target_keys)

class I18nManager:
    """Main internationalization manager"""
    
    def __init__(self, default_locale: Optional[LocaleConfig] = None):
        self.current_locale = default_locale or self._get_default_locale()
        self.translation_manager = TranslationManager()
        self.number_formatter = NumberFormatter()
        self.datetime_formatter = DateTimeFormatter()
        self.rtl_support = RTLSupport()
        
        # Thread-local storage for user-specific locales
        self._local = threading.local()
        
        logger.info(f"I18n manager initialized with locale: {self.current_locale.language.value}")
    
    def _get_default_locale(self) -> LocaleConfig:
        """Get default locale configuration"""
        return LocaleConfig(
            language=Language.ENGLISH,
            country_code="US",
            currency=Currency.USD,
            timezone="UTC",
            number_format=NumberFormat.DECIMAL_POINT,
            date_format=DateFormat.MDY,
            is_rtl=False
        )
    
    def set_locale(self, locale_config: LocaleConfig):
        """Set current locale"""
        self.current_locale = locale_config
        
        # Set thread-local locale
        self._local.locale = locale_config
        
        logger.info(f"Locale changed to: {locale_config.language.value}")
    
    def get_current_locale(self) -> LocaleConfig:
        """Get current locale (thread-safe)"""
        # Try thread-local first
        if hasattr(self._local, 'locale'):
            return self._local.locale
        
        return self.current_locale
    
    def translate(self, 
                 key: str,
                 variables: Optional[Dict[str, Any]] = None,
                 count: Optional[int] = None,
                 fallback: Optional[str] = None) -> str:
        """Translate text"""
        locale = self.get_current_locale()
        translated = self.translation_manager.get_translation(
            key, locale.language, variables, count, fallback
        )
        
        # Apply RTL formatting if needed
        if locale.is_rtl:
            translated = self.rtl_support.apply_rtl_formatting(translated, locale.language)
        
        return translated
    
    def format_number(self, 
                     number: Union[int, float],
                     decimal_places: Optional[int] = None) -> str:
        """Format number according to current locale"""
        locale = self.get_current_locale()
        return self.number_formatter.format_number(number, locale, decimal_places)
    
    def format_currency(self, amount: float) -> str:
        """Format currency according to current locale"""
        locale = self.get_current_locale()
        return self.number_formatter.format_currency(amount, locale.currency, locale)
    
    def format_percentage(self, value: float) -> str:
        """Format percentage according to current locale"""
        locale = self.get_current_locale()
        return self.number_formatter.format_percentage(value, locale)
    
    def format_date(self, dt: datetime) -> str:
        """Format date according to current locale"""
        locale = self.get_current_locale()
        return self.datetime_formatter.format_date(dt, locale)
    
    def format_time(self, dt: datetime) -> str:
        """Format time according to current locale"""
        locale = self.get_current_locale()
        return self.datetime_formatter.format_time(dt, locale)
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to current locale"""
        locale = self.get_current_locale()
        return self.datetime_formatter.format_datetime(dt, locale)
    
    def get_localized_error(self, error_code: str, **kwargs) -> str:
        """Get localized error message"""
        error_key = f"error.{error_code}"
        return self.translate(error_key, variables=kwargs, fallback=f"Error: {error_code}")
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages"""
        return list(self.translation_manager.translations.keys())
    
    def get_locale_info(self) -> Dict[str, Any]:
        """Get current locale information"""
        locale = self.get_current_locale()
        return {
            'language': locale.language.value,
            'country_code': locale.country_code,
            'currency': locale.currency.value,
            'timezone': locale.timezone,
            'number_format': locale.number_format.value,
            'date_format': locale.date_format.value,
            'is_rtl': locale.is_rtl,
            'decimal_separator': locale.decimal_separator,
            'thousands_separator': locale.thousands_separator,
            'date_separator': locale.date_separator,
            'time_format_24h': locale.time_format_24h,
            'first_day_of_week': locale.first_day_of_week
        }

# Global i18n manager
_global_i18n_manager: Optional[I18nManager] = None

def get_i18n_manager() -> Optional[I18nManager]:
    """Get global i18n manager"""
    return _global_i18n_manager

def initialize_i18n(locale: Optional[LocaleConfig] = None) -> I18nManager:
    """Initialize global i18n manager"""
    global _global_i18n_manager
    _global_i18n_manager = I18nManager(locale)
    return _global_i18n_manager

# Convenience functions
def translate(key: str, **kwargs) -> str:
    """Global translate function"""
    if _global_i18n_manager:
        return _global_i18n_manager.translate(key, **kwargs)
    return key

def format_number(number: Union[int, float], decimal_places: Optional[int] = None) -> str:
    """Global number formatting function"""
    if _global_i18n_manager:
        return _global_i18n_manager.format_number(number, decimal_places)
    return str(number)

def format_currency(amount: float) -> str:
    """Global currency formatting function"""
    if _global_i18n_manager:
        return _global_i18n_manager.format_currency(amount)
    return f"${amount:.2f}"

def format_date(dt: datetime) -> str:
    """Global date formatting function"""
    if _global_i18n_manager:
        return _global_i18n_manager.format_date(dt)
    return dt.strftime("%Y-%m-%d")

# Decorators
def localized(func):
    """Decorator to ensure function uses localized formatting"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # Apply localization to result if it's a string
        if isinstance(result, str) and _global_i18n_manager:
            # This could be enhanced to auto-detect translation keys
            pass
        return result
    return wrapper

# Common locale configurations
LOCALE_CONFIGS = {
    "en-US": LocaleConfig(
        language=Language.ENGLISH,
        country_code="US",
        currency=Currency.USD,
        timezone="America/New_York",
        number_format=NumberFormat.DECIMAL_POINT,
        date_format=DateFormat.MDY,
        time_format_24h=False
    ),
    "en-GB": LocaleConfig(
        language=Language.ENGLISH,
        country_code="GB",
        currency=Currency.GBP,
        timezone="Europe/London",
        number_format=NumberFormat.DECIMAL_POINT,
        date_format=DateFormat.DMY,
        time_format_24h=True
    ),
    "es-ES": LocaleConfig(
        language=Language.SPANISH,
        country_code="ES",
        currency=Currency.EUR,
        timezone="Europe/Madrid",
        number_format=NumberFormat.DECIMAL_COMMA,
        date_format=DateFormat.DMY
    ),
    "fr-FR": LocaleConfig(
        language=Language.FRENCH,
        country_code="FR",
        currency=Currency.EUR,
        timezone="Europe/Paris",
        number_format=NumberFormat.SPACE_COMMA,
        date_format=DateFormat.DMY
    ),
    "de-DE": LocaleConfig(
        language=Language.GERMAN,
        country_code="DE",
        currency=Currency.EUR,
        timezone="Europe/Berlin",
        number_format=NumberFormat.DECIMAL_COMMA,
        date_format=DateFormat.DMY
    ),
    "ja-JP": LocaleConfig(
        language=Language.JAPANESE,
        country_code="JP",
        currency=Currency.JPY,
        timezone="Asia/Tokyo",
        number_format=NumberFormat.DECIMAL_POINT,
        date_format=DateFormat.YMD
    ),
    "zh-CN": LocaleConfig(
        language=Language.CHINESE_SIMPLIFIED,
        country_code="CN",
        currency=Currency.CNY,
        timezone="Asia/Shanghai",
        number_format=NumberFormat.DECIMAL_POINT,
        date_format=DateFormat.YMD
    ),
    "ar-SA": LocaleConfig(
        language=Language.ARABIC,
        country_code="SA",
        currency=Currency.USD,  # Often used in international contexts
        timezone="Asia/Riyadh",
        number_format=NumberFormat.DECIMAL_POINT,
        date_format=DateFormat.DMY,
        is_rtl=True
    ),
    "pt-BR": LocaleConfig(
        language=Language.PORTUGUESE,
        country_code="BR",
        currency=Currency.BRL,
        timezone="America/Sao_Paulo",
        number_format=NumberFormat.DECIMAL_COMMA,
        date_format=DateFormat.DMY
    ),
    "it-IT": LocaleConfig(
        language=Language.ITALIAN,
        country_code="IT",
        currency=Currency.EUR,
        timezone="Europe/Rome",
        number_format=NumberFormat.DECIMAL_COMMA,
        date_format=DateFormat.DMY
    ),
    "ru-RU": LocaleConfig(
        language=Language.RUSSIAN,
        country_code="RU",
        currency=Currency.RUB,
        timezone="Europe/Moscow",
        number_format=NumberFormat.SPACE_COMMA,
        date_format=DateFormat.DMY
    )
}

def get_locale_config(locale_code: str) -> Optional[LocaleConfig]:
    """Get locale configuration by code"""
    return LOCALE_CONFIGS.get(locale_code)

def get_available_locales() -> List[str]:
    """Get list of available locale codes"""
    return list(LOCALE_CONFIGS.keys())