from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum, IntEnum, StrEnum
from functools import lru_cache
from typing import get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field


class AgentSpecialization(StrEnum):
    API = "api"
    DATABASE = "database"
    ENTERPRISE = "enterprise"
    GENERAL = "general"
    PERFORMANCE = "performance"
    SECURITY = "security"


class BusinessImpact(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"


class Category(StrEnum):
    ACCOUNT_MANAGEMENT = "Account Management"
    DATA_ISSUE = "Data Issue"
    FEATURE_REQUEST = "Feature Request"
    SECURITY = "Security"
    TECHNICAL_ISSUE = "Technical Issue"


class SubCategory(StrEnum):
    UI_UX = "UI/UX"
    API = "API"
    ENHANCEMENT = "Enhancement"
    DOCUMENTATION = "Documentation"
    NEW_FEATURE = "New Feature"
    SUBSCRIPTION = "Subscription"
    LICENSE = "License"
    BILLING = "Billing"
    UPGRADE = "Upgrade"
    ACCESS_CONTROL = "Access Control"
    AUTHENTICATION = "Authentication"
    ENCRYPTION = "Encryption"
    COMPLIANCE = "Compliance"
    AUTHORIZATION = "Authorization"
    VULNERABILITY = "Vulnerability"
    IMPORT_EXPORT = "Import/Export"
    DATA_LOSS = "Data Loss"
    SYNC_ERROR = "Sync Error"
    CORRUPTION = "Corruption"
    VALIDATION = "Validation"
    INTEGRATION = "Integration"
    PERFORMANCE = "Performance"
    COMPATIBILITY = "Compatibility"
    BUG = "Bug"
    CONFIGURATION = "Configuration"


class Channel(StrEnum):
    API = "api"
    CHAT = "chat"
    EMAIL = "email"
    PHONE = "phone"
    PORTAL = "portal"
    SLACK = "slack"


class CustomerSentiment(StrEnum):
    ANGRY = "angry"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    GRATEFUL = "grateful"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


class CustomerTier(StrEnum):
    ENTERPRISE = "enterprise"
    FREE = "free"
    PREMIUM = "premium"
    PROFESSIONAL = "professional"
    STARTER = "starter"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    SANDBOX = "sandbox"
    STAGING = "staging"
    TEST = "test"


class EscalationReason(StrEnum):
    COMPLEX_TECHNICAL_ISSUE_REQUIRING_SPECIALIST = "Complex technical issue requiring specialist"
    CUSTOMER_REQUESTED_ESCALATION = "Customer requested escalation"
    HIGH_PRIORITY_CUSTOMER = "High priority customer"
    MULTIPLE_FAILED_RESOLUTION_ATTEMPTS = "Multiple failed resolution attempts"
    SLA_BREACH_RISK = "SLA breach risk"


class Language(StrEnum):
    DE = "de"
    EN = "en"
    ES = "es"
    FR = "fr"
    IT = "it"
    JA = "ja"
    PT = "pt"
    ZH = "zh"


class Priority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"


class Product(StrEnum):
    API_GATEWAY = "API Gateway"
    ANALYTICS_DASHBOARD = "Analytics Dashboard"
    CLOUDBACKUP_ENTERPRISE = "CloudBackup Enterprise"
    DATASYNC_PRO = "DataSync Pro"
    STREAMPROCESSOR = "StreamProcessor"


class ProductModule(StrEnum):
    API_CONNECTOR = "api_connector"
    AUTH_SERVICE = "auth_service"
    BACKUP_SERVICE = "backup_service"
    BATCH_PROCESSOR = "batch_processor"
    CACHE_LAYER = "cache_layer"
    COMPRESSION_ENGINE = "compression_engine"
    DATA_AGGREGATOR = "data_aggregator"
    DATA_VALIDATOR = "data_validator"
    ENCRYPTION_LAYER = "encryption_layer"
    ERROR_HANDLER = "error_handler"
    EVENT_HANDLER = "event_handler"
    EXPORT_MODULE = "export_module"
    MONITORING = "monitoring"
    RATE_LIMITER = "rate_limiter"
    REPORT_BUILDER = "report_builder"
    REQUEST_ROUTER = "request_router"
    RESTORE_MODULE = "restore_module"
    SCHEDULER = "scheduler"
    SYNC_ENGINE = "sync_engine"
    VISUALIZATION = "visualization"


class Region(StrEnum):
    APAC = "APAC"
    EU = "EU"
    LATAM = "LATAM"
    MEA = "MEA"
    NA = "NA"


class ResolutionCode(StrEnum):
    BUG_FIX = "BUG_FIX"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    DATA_REPAIR = "DATA_REPAIR"
    DUPLICATE = "DUPLICATE"
    ENVIRONMENT_ISSUE = "ENVIRONMENT_ISSUE"
    ESCALATED = "ESCALATED"
    FEATURE_ADDED = "FEATURE_ADDED"
    PATCH_APPLIED = "PATCH_APPLIED"
    RESTART_REQUIRED = "RESTART_REQUIRED"
    USER_EDUCATION = "USER_EDUCATION"
    WONT_FIX = "WONT_FIX"
    WORKAROUND = "WORKAROUND"


class Severity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class SATISFACTION_SCORE(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class ATTACHMENTS_COUNT(IntEnum):
    NONE = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class Ticket(BaseModel):
    """Normalized representation of a support ticket."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime
    customer_id: str
    organization_id: str
    customer_tier: CustomerTier = Field(..., json_schema_extra={"feature": True})
    product: Product = Field(..., json_schema_extra={"feature": True})
    product_version: str
    product_module: ProductModule = Field(..., json_schema_extra={"feature": True})
    category: Category = Field(..., json_schema_extra={"target": True})
    subcategory: SubCategory = Field(..., json_schema_extra={"target": True})
    priority: Priority = Field(..., json_schema_extra={"feature": True})
    severity: Severity = Field(..., json_schema_extra={"feature": True})
    channel: Channel = Field(..., json_schema_extra={"feature": True})
    subject: str = Field(..., json_schema_extra={"feature": True})
    description: str = Field(..., json_schema_extra={"feature": True})
    error_logs: str = Field(..., json_schema_extra={"feature": True})
    stack_trace: str = Field(..., json_schema_extra={"feature": True})
    customer_sentiment: CustomerSentiment = Field(..., json_schema_extra={"target": True})
    previous_tickets: int = Field(ge=0, json_schema_extra={"feature": True})
    resolution: str = Field(..., json_schema_extra={"feature": True})
    resolution_code: ResolutionCode
    resolution_time_hours: float = Field(ge=0, json_schema_extra={"feature": True})
    resolution_attempts: int = Field(ge=1, json_schema_extra={"feature": True})
    agent_id: str
    agent_experience_months: int = Field(ge=0, json_schema_extra={"feature": True})
    agent_specialization: AgentSpecialization = Field(..., json_schema_extra={"feature": True})
    agent_actions: list[str] = Field(..., json_schema_extra={"feature": True})
    escalated: bool = Field(..., json_schema_extra={"feature": True})
    escalation_reason: EscalationReason | None = None
    transferred_count: int = Field(ge=0, le=10, json_schema_extra={"feature": True})
    satisfaction_score: SATISFACTION_SCORE = Field(..., json_schema_extra={"feature": True})
    feedback_text: str = Field(..., json_schema_extra={"feature": True})
    resolution_helpful: bool = Field(..., json_schema_extra={"feature": True})
    tags: list[str] = Field(..., json_schema_extra={"feature": True})
    related_tickets: list[str] = Field(..., json_schema_extra={"feature": True})
    kb_articles_viewed: list[str] = Field(..., json_schema_extra={"feature": True})
    kb_articles_helpful: list[str] = Field(..., json_schema_extra={"feature": True})
    environment: Environment = Field(..., json_schema_extra={"feature": True})
    account_age_days: int = Field(ge=0, json_schema_extra={"feature": True})
    account_monthly_value: int = Field(ge=0, json_schema_extra={"feature": True})
    similar_issues_last_30_days: int = Field(ge=0, json_schema_extra={"feature": True})
    product_version_age_days: int = Field(ge=0, json_schema_extra={"feature": True})
    known_issue: bool = Field(..., json_schema_extra={"feature": True})
    bug_report_filed: bool = Field(..., json_schema_extra={"feature": True})
    resolution_template_used: str | None = None
    auto_suggested_solutions: list[str] = Field(..., json_schema_extra={"feature": True})
    auto_suggestion_accepted: bool = Field(..., json_schema_extra={"feature": True})
    ticket_text_length: int = Field(ge=0, json_schema_extra={"feature": True})
    response_count: int = Field(ge=1, json_schema_extra={"feature": True})
    attachments_count: ATTACHMENTS_COUNT = Field(..., json_schema_extra={"feature": True})
    contains_error_code: bool = Field(..., json_schema_extra={"feature": True})
    contains_stack_trace: bool = Field(..., json_schema_extra={"feature": True})
    business_impact: BusinessImpact = Field(..., json_schema_extra={"feature": True})
    affected_users: int = Field(ge=0, json_schema_extra={"feature": True})
    weekend_ticket: bool = Field(..., json_schema_extra={"feature": True})
    after_hours: bool = Field(..., json_schema_extra={"feature": True})
    language: Language = Field(..., json_schema_extra={"feature": True})
    region: Region = Field(..., json_schema_extra={"feature": True})


@lru_cache(maxsize=1)
def get_text_features() -> list[str]:
    res = []
    for name, field in Ticket.model_fields.items():
        attr = field.json_schema_extra
        if (attr is None) or (not isinstance(attr, Mapping)):
            continue
        t = field.annotation
        if t is None:
            continue
        if (t is str) and attr.get("feature", False):
            res.append(name)
    return res


@lru_cache(maxsize=1)
def get_text_list_features() -> list[str]:
    res = []
    for name, field in Ticket.model_fields.items():
        attr = field.json_schema_extra
        if (attr is None) or (not isinstance(attr, Mapping)):
            continue
        t = field.annotation
        origin = get_origin(t)
        if origin is None:
            continue
        args = get_args(t)
        if args is None:
            continue
        if (origin is list) and (args[0] is str):
            res.append(name)
    return res


@lru_cache(maxsize=1)
def get_cat_features() -> list[str]:
    res = []
    for name, field in Ticket.model_fields.items():
        attr = field.json_schema_extra
        if (attr is None) or (not isinstance(attr, Mapping)):
            continue
        t = field.annotation
        if t is None:
            continue
        if issubclass(t, Enum) and attr.get("feature", False):
            res.append(name)
    return res


@lru_cache(maxsize=1)
def get_bool_features() -> list[str]:
    res = []
    for name, field in Ticket.model_fields.items():
        attr = field.json_schema_extra
        if (attr is None) or (not isinstance(attr, Mapping)):
            continue
        t = field.annotation
        if t is None:
            continue
        if (t is bool) and attr.get("feature", False):
            res.append(name)
    return res


@lru_cache(maxsize=1)
def get_num_features() -> list[str]:
    res = []
    for name, field in Ticket.model_fields.items():
        attr = field.json_schema_extra
        if (attr is None) or (not isinstance(attr, Mapping)):
            continue
        t = field.annotation
        if t is None:
            continue
        if (t is int) or (t is float) and attr.get("feature", False):
            res.append(name)
    return res


@lru_cache(maxsize=1)
def get_target() -> list[str]:
    res = []
    for name, field in Ticket.model_fields.items():
        attr = field.json_schema_extra
        if (attr is None) or (not isinstance(attr, Mapping)):
            continue
        if attr.get("target", False):
            res.append(name)
    return res


CAT_MAPPING = {
    Category.FEATURE_REQUEST: [
        SubCategory.UI_UX,
        SubCategory.API,
        SubCategory.ENHANCEMENT,
        SubCategory.DOCUMENTATION,
        SubCategory.NEW_FEATURE,
    ],
    Category.ACCOUNT_MANAGEMENT: [
        SubCategory.SUBSCRIPTION,
        SubCategory.LICENSE,
        SubCategory.BILLING,
        SubCategory.UPGRADE,
        SubCategory.ACCESS_CONTROL,
    ],
    Category.SECURITY: [
        SubCategory.AUTHENTICATION,
        SubCategory.ENCRYPTION,
        SubCategory.COMPLIANCE,
        SubCategory.AUTHORIZATION,
        SubCategory.VULNERABILITY,
    ],
    Category.DATA_ISSUE: [
        SubCategory.IMPORT_EXPORT,
        SubCategory.DATA_LOSS,
        SubCategory.SYNC_ERROR,
        SubCategory.CORRUPTION,
        SubCategory.VALIDATION,
    ],
    Category.TECHNICAL_ISSUE: [
        SubCategory.INTEGRATION,
        SubCategory.PERFORMANCE,
        SubCategory.COMPATIBILITY,
        SubCategory.BUG,
        SubCategory.CONFIGURATION,
    ],
}

TEXT_FEATURES = get_text_features()
BOOL_FEATURES = get_bool_features()
TEXT_LIST_FEATURES = get_text_list_features()
CAT_FEATURES = get_cat_features()
NUM_FEATURES = get_num_features()
TARGETS = get_target()
