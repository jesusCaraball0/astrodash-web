from pydantic_settings import BaseSettings
from pydantic import Field, AnyUrl, field_validator
from typing import Optional, List, Dict
import os

class Settings(BaseSettings):
    # General
    app_name: str = Field("AstroDash API", env="APP_NAME")
    environment: str = Field("production", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")

    # API
    api_prefix: str = Field("/api/v1", env="API_PREFIX")
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS")  # Allow all hosts for API usage
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")    # Allow all origins for API usage

    # Security Settings
    secret_key: str = Field("supersecret", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(60 * 24, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst_limit: int = Field(10, env="RATE_LIMIT_BURST_LIMIT")

    # Security Headers
    enable_hsts: bool = Field(True, env="ENABLE_HSTS")
    enable_csp: bool = Field(True, env="ENABLE_CSP")
    enable_permissions_policy: bool = Field(True, env="ENABLE_PERMISSIONS_POLICY")

    # Input Validation
    max_request_size: int = Field(100 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 100MB
    max_file_size: int = Field(50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB

    # Session Security
    session_cookie_secure: bool = Field(True, env="SESSION_COOKIE_SECURE")
    session_cookie_httponly: bool = Field(True, env="SESSION_COOKIE_HTTPONLY")
    session_cookie_samesite: str = Field("strict", env="SESSION_COOKIE_SAMESITE")

    # Database
    db_url: Optional[AnyUrl] = Field(None, env="DATABASE_URL")
    db_echo: bool = Field(False, env="DB_ECHO")

    # Storage
    storage_dir: str = Field("storage", env="STORAGE_DIR")
    user_model_dir: str = Field("app/astrodash_models/user_uploaded", env="USER_MODEL_DIR")

    # ML Model Paths
    dash_model_path: str = Field("app/astrodash_models/zeroZ/pytorch_model.pth", env="DASH_MODEL_PATH")
    transformer_model_path: str = Field("app/astrodash_models/yuqing_models/TF_wiserep_v6.pt", env="TRANSFORMER_MODEL_PATH")

    # Template and Line List Paths
    template_path: str = Field("app/astrodash_models/sn_and_host_templates.npz", env="TEMPLATE_PATH")
    line_list_path: str = Field("app/astrodash_models/sneLineList.txt", env="LINE_LIST_PATH")

    # ML Configuration Parameters
    # DASH model parameters
    nw: int = Field(1024, env="NW")  # Number of wavelength bins
    w0: float = Field(3500.0, env="W0")  # Minimum wavelength in Angstroms
    w1: float = Field(10000.0, env="W1")  # Maximum wavelength in Angstroms

    # Transformer model parameters
    label_mapping: Dict[str, int] = Field(
        {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4},
        env="LABEL_MAPPING"
    )

    # Logging
    log_dir: str = Field("logs", env="LOG_DIR")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Other
    osc_api_url: str = Field("https://api.astrocats.space", env="OSC_API_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from environment

    @field_validator("allowed_hosts", "cors_origins", mode="before")
    @classmethod
    def split_str(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @field_validator("label_mapping", mode="before")
    @classmethod
    def parse_label_mapping(cls, v):
        if isinstance(v, str):
            # Parse JSON string if provided as environment variable
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Fallback to default if parsing fails
                return {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4}
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if v == "supersecret" and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("SECRET_KEY must be set to a secure value in production")
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed_environments = ["development", "staging", "production", "test"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v

    @field_validator("session_cookie_samesite")
    @classmethod
    def validate_session_cookie_samesite(cls, v):
        allowed_values = ["strict", "lax", "none"]
        if v not in allowed_values:
            raise ValueError(f"SESSION_COOKIE_SAMESITE must be one of: {allowed_values}")
        return v

def get_settings() -> Settings:
    return Settings()
