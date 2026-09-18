"""Contract v1: what is rejected, what is normalised, and how structural groups map."""
import pytest
from pydantic import ValidationError

from fraud_api.predict import check_levels, to_source_frame
from fraud_api.schema import ScoreRequest
from fraud_api.transform import ContractError, derive


def _req(request_dicts, **kw):
    d = dict(request_dicts[0])
    d.update(kw)
    return d


@pytest.mark.parametrize("field,value", [("is_fraud", 1), ("label_source", "chargeback"),
                                         ("sample_weight", 1.0), ("alerted_flag", "Y"), ("customer_id", "C1")])
def test_label_incumbent_and_unknown_fields_are_rejected(request_dicts, field, value):
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(_req(request_dicts, **{field: value}))


@pytest.mark.parametrize("field", ["payment_ts", "payment_method", "payment_amount", "processing_fee",
                                   "is_guest_checkout", "is_3ds_attempted", "is_3ds_success", "coupon_applied",
                                   "avg_ticket_size", "ip_n_customers"])
def test_required_fields(request_dicts, field):
    d = _req(request_dicts)
    d.pop(field)
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(d)


@pytest.mark.parametrize("kw", [{"payment_amount": 0}, {"payment_amount": -1.0}, {"kyc_level": 3},
                                {"city_tier": 0}, {"prior_return_rate": 1.5}, {"ip_country": "India"},
                                {"account_age_days": 36500}, {"avg_ticket_size": 0}, {"is_3ds_success": None}])
def test_out_of_range_values_are_rejected(request_dicts, kw):
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(_req(request_dicts, **kw))


def test_structural_groups_are_all_or_nothing(request_dicts):
    card = {"token_age_h": 10.0, "network": "Visa"}                    # issuing_country missing
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(_req(request_dicts, card=card))
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(_req(request_dicts, device={"device_type": "Mac", "is_emulator": False}))


def test_unknown_category_level_is_a_contract_error(request_dicts, p06):
    r = ScoreRequest.model_validate(_req(request_dicts, payment_method="Crypto"))
    with pytest.raises(ContractError) as e:
        check_levels([r], p06)
    assert e.value.field == "payment_method"


def test_sentinels_become_null_exactly_like_notebook_02(request_dicts):
    r = ScoreRequest.model_validate(_req(request_dicts, ip_country="XX", payment_gateway="  unknown ",
                                         ip_region_code=-1, home_region_code=999, delivery_type=""))
    assert r.ip_country is None and r.payment_gateway is None and r.delivery_type is None
    assert r.ip_region_code is None and r.home_region_code is None


def test_failed_attempts_overflow_is_a_gap(request_dicts):
    lg = {"hours_since": 5.0, "new_device": False, "unusual_location": False, "failed_attempts": 255}
    src = to_source_frame([ScoreRequest.model_validate(_req(request_dicts, last_login=lg))])
    assert src.last_login_failed_attempts.isna().iloc[0]
    assert src.has_prior_login.iloc[0] == 1


def test_timezone_aware_timestamps_are_converted_to_ist(request_dicts):
    r = ScoreRequest.model_validate(_req(request_dicts, payment_ts="2025-01-04T20:00:00Z"))   # Sat 20:00 UTC
    d = derive(to_source_frame([r]))
    assert (d.hour_of_day.iloc[0], d.day_of_week.iloc[0]) == (1, 6)        # Sun 01:30 IST
    assert d.is_night.iloc[0] == 1 and d.is_weekend.iloc[0] == 1


def test_absent_groups_map_to_the_trained_structural_gaps(request_dicts):
    r = ScoreRequest.model_validate(_req(request_dicts, card=None, device=None, basket=None, last_login=None))
    s = to_source_frame([r]).iloc[0]
    assert (s.has_card, s.has_device_profile, s.has_basket, s.has_prior_login) == (0, 0, 0, 0)
    for c in ["card_token_age_h", "network", "issuing_country", "device_age_h", "device_type", "is_emulator",
              "n_categories", "max_unit_price", "total_qty", "hours_since_last_login"]:
        assert s[c] is None or s[c] != s[c], c                                # None or NaN
