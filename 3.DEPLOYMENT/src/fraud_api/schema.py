"""HTTP contract v1. Unknown fields are rejected, so label or incumbent columns can never reach the model.

Structural groups (card, device, basket, last_login) mirror notebook 03: a group is either fully present or
absent, and absence is what the model learned as "no card / no device profile / no basket / no prior login".
Optional scalar fields may be null: the service fills them with the train-fitted notebook-03 values and reports
them in `defaults_applied`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import constants as C

IST = timezone(timedelta(minutes=C.IST_OFFSET_MINUTES))
_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _clean_str(v: Any) -> Any:
    """Notebook 02 `sclean`: blanks and sentinel tokens mean 'no value'."""
    if isinstance(v, str):
        s = " ".join(v.split())
        if s == "" or s.upper() in C.SENTINELS or s.lower() == "nan":
            return None
        return s
    return v


class _Base(BaseModel):
    model_config = _STRICT

    @field_validator("*", mode="before")
    @classmethod
    def _sentinels(cls, v):
        return _clean_str(v)


class Card(_Base):
    token_age_h: float = Field(ge=0, description="hours since the card token was first seen")
    network: str
    issuing_country: str = Field(pattern=r"^[A-Z]{2}$")
    issuer: str | None = None
    product_type: str | None = None


class Device(_Base):
    device_age_h: float = Field(ge=0, description="hours since the device was first seen")
    device_type: str
    is_emulator: bool
    browser_family: str | None = None


class Basket(_Base):
    n_categories: int = Field(ge=1)
    max_unit_price: float = Field(gt=0)
    total_qty: int = Field(ge=1)


class LastLogin(_Base):
    hours_since: float = Field(ge=0, description="hours between the last prior login and this payment")
    new_device: bool
    unusual_location: bool
    failed_attempts: int | None = Field(default=None, ge=0, le=C.FAILED_ATTEMPTS_SENTINEL)
    risk_score: float | None = Field(default=None, ge=0, le=100)


class ScoreRequest(_Base):
    payment_id: str | None = Field(default=None, max_length=64)
    payment_ts: datetime = Field(description="payment time; timezone-aware values are converted to IST")
    payment_method: str
    payment_gateway: str | None = Field(default=None, description="null for Cash on Delivery")
    payment_amount: float = Field(gt=0)
    processing_fee: float = Field(ge=0)
    discount_amount: float | None = Field(default=None, ge=0)
    item_count: int | None = Field(default=None, ge=1)
    attempt_seq_in_session: int | None = Field(default=None, ge=1)
    is_guest_checkout: bool
    is_3ds_attempted: bool
    is_3ds_success: bool
    coupon_applied: bool
    address_match: bool | None = None
    shipping_speed: str | None = None
    shipping_addr_age_hours: float | None = Field(default=None, ge=0)
    delivery_type: str | None = None
    kyc_level: int | None = Field(default=None, ge=0, le=2)
    city_tier: int | None = Field(default=None, ge=1, le=3)
    email_domain_class: str | None = None
    acquisition_channel: str | None = None
    account_age_days: float | None = Field(default=None, ge=0, lt=C.ACCOUNT_AGE_SENTINEL_DAYS)
    prior_return_rate: float | None = Field(default=None, ge=0, le=1)
    prior_orders_12m: int | None = Field(default=None, ge=0)
    merchant_category: str | None = None
    avg_ticket_size: float = Field(gt=0, description="merchant average ticket; required (training never saw a gap)")
    trailing_chargeback_rate_bps: float | None = Field(default=None, ge=0)
    ip_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    ip_region_code: int | None = None
    home_region_code: int | None = None
    asn_type: str | None = None
    asn_country: str | None = None
    reputation_score: float | None = Field(default=None, ge=0)
    ip_n_customers: int = Field(ge=1, description="as-of distinct customers on this IP (see features.py)")
    device_n_customers: int | None = Field(default=None, ge=1, description="null only when there is no device id")
    card: Card | None = None
    device: Device | None = None
    basket: Basket | None = None
    last_login: LastLogin | None = None
    payment_risk_score: float | None = Field(default=None, ge=0,
                                             description="incumbent score; used ONLY by the opt-in fallback")

    @field_validator("payment_ts")
    @classmethod
    def _to_ist_naive(cls, v: datetime) -> datetime:
        return v.astimezone(IST).replace(tzinfo=None) if v.tzinfo is not None else v

    @field_validator("ip_region_code", "home_region_code")
    @classmethod
    def _region_sentinel(cls, v):
        return None if v in C.REGION_SENTINELS else v


class Contribution(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    prediction_id: str
    payment_id: str | None
    source: str
    model_version: str | None
    contract_version: str
    fraud_probability: float | None
    score_raw: float | None
    decision: bool
    threshold_raw: float | None
    operating_point_provisional: bool
    gate_status: str | None
    defaults_applied: list[str]
    explanation: list[Contribution] | None = None
    explanation_bias: float | None = None
