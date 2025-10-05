from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AgentSpecialization(str, Enum):
    API = "api"
    DATABASE = "database"
    ENTERPRISE = "enterprise"
    GENERAL = "general"
    PERFORMANCE = "performance"
    SECURITY = "security"


class BusinessImpact(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"


class Category(str, Enum):
    ACCOUNT_MANAGEMENT = "Account Management"
    DATA_ISSUE = "Data Issue"
    FEATURE_REQUEST = "Feature Request"
    SECURITY = "Security"
    TECHNICAL_ISSUE = "Technical Issue"


class Channel(str, Enum):
    API = "api"
    CHAT = "chat"
    EMAIL = "email"
    PHONE = "phone"
    PORTAL = "portal"
    SLACK = "slack"


class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    GRATEFUL = "grateful"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


class CustomerTier(str, Enum):
    ENTERPRISE = "enterprise"
    FREE = "free"
    PREMIUM = "premium"
    PROFESSIONAL = "professional"
    STARTER = "starter"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    SANDBOX = "sandbox"
    STAGING = "staging"
    TEST = "test"


class EscalationReason(str, Enum):
    COMPLEX_TECHNICAL_ISSUE_REQUIRING_SPECIALIST = (
        "Complex technical issue requiring specialist"
    )
    CUSTOMER_REQUESTED_ESCALATION = "Customer requested escalation"
    HIGH_PRIORITY_CUSTOMER = "High priority customer"
    MULTIPLE_FAILED_RESOLUTION_ATTEMPTS = "Multiple failed resolution attempts"
    SLA_BREACH_RISK = "SLA breach risk"


class Language(str, Enum):
    DE = "de"
    EN = "en"
    ES = "es"
    FR = "fr"
    IT = "it"
    JA = "ja"
    PT = "pt"
    ZH = "zh"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"


class Product(str, Enum):
    API_GATEWAY = "API Gateway"
    ANALYTICS_DASHBOARD = "Analytics Dashboard"
    CLOUDBACKUP_ENTERPRISE = "CloudBackup Enterprise"
    DATASYNC_PRO = "DataSync Pro"
    STREAMPROCESSOR = "StreamProcessor"


class ProductModule(str, Enum):
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


class Region(str, Enum):
    APAC = "APAC"
    EU = "EU"
    LATAM = "LATAM"
    MEA = "MEA"
    NA = "NA"


class ResolutionCode(str, Enum):
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


class Severity(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Ticket(BaseModel):
    """Normalized representation of a support ticket."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime
    customer_id: str
    organization_id: str
    customer_tier: CustomerTier
    product: Product
    product_version: str
    product_module: ProductModule
    category: Category
    subcategory: str
    priority: Priority
    severity: Severity
    channel: Channel
    subject: str
    description: str
    error_logs: str
    stack_trace: str
    customer_sentiment: CustomerSentiment
    previous_tickets: int = Field(ge=0)
    resolution: str
    resolution_code: ResolutionCode
    resolution_time_hours: float = Field(ge=0)
    resolution_attempts: int = Field(ge=1)
    agent_id: str
    agent_experience_months: int = Field(ge=0)
    agent_specialization: AgentSpecialization
    agent_actions: list[str]
    escalated: bool
    escalation_reason: EscalationReason | None = None
    transferred_count: int = Field(ge=0, le=3)
    satisfaction_score: int = Field(ge=1, le=5)
    feedback_text: str
    resolution_helpful: bool
    tags: list[str]
    related_tickets: list[str]
    kb_articles_viewed: list[str]
    kb_articles_helpful: list[str]
    environment: Environment
    account_age_days: int = Field(ge=0)
    account_monthly_value: int = Field(ge=0)
    similar_issues_last_30_days: int = Field(ge=0)
    product_version_age_days: int = Field(ge=0)
    known_issue: bool
    bug_report_filed: bool
    resolution_template_used: str | None = None
    auto_suggested_solutions: list[str]
    auto_suggestion_accepted: bool
    ticket_text_length: int = Field(ge=0)
    response_count: int = Field(ge=1)
    attachments_count: int = Field(ge=0, le=5)
    contains_error_code: bool
    contains_stack_trace: bool
    business_impact: BusinessImpact
    affected_users: int = Field(ge=0)
    weekend_ticket: bool
    after_hours: bool
    language: Language
    region: Region

