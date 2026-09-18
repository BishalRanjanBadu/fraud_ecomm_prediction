#!/usr/bin/env python3
"""
Synthetic fraud-detection dataset generator.

Deterministic under SEED. Regenerates the full 11-file dataset from scratch.

Causal order: latent fraud is assigned FIRST, then observable attributes are
mutated to carry each mechanism's signature. The incumbent rules engine then
reads a strict subset of the true drivers, which is what produces the biased
alert stream and the selective-labels problem the dataset exists to pose.

  python generate.py --out ./out            # dataset only
  python generate.py --out ./out --emit-truth   # + ground_truth.csv (evaluation only)

The latent truth is NOT part of the dataset. Labels are observable only through
fraud_events (biased), audit_sample (unbiased, 12k) and chargebacks (delayed).
"""
import numpy as np, pandas as pd, argparse, os, json

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SEED           = 42
N_PAY          = 500_000
N_CUST         = 199_000
N_MERCH        = 2_000
N_DEV          = 140_000
N_LOGIN        = 180_000
N_AUDIT        = 12_000
START          = pd.Timestamp('2024-01-01 00:00:00')
END            = pd.Timestamp('2025-12-28 23:59:00')
FRAUD_RATE     = 0.009           # ~4,500 true fraud
WEAK_FRAC      = 0.25            # fraud with near-absent signature
FALSE_SIG_FRAC = 0.012           # legit rows mimicking a full fraud mechanism
MIMIC_STRENGTH = 0.70            # how completely a mimic expresses that mechanism
DRIFT_START    = pd.Timestamp('2025-08-01')
TARGET_RECALL  = 0.55            # incumbent rules engine
TARGET_PRECIS  = 0.35
REVIEW_FN      = 0.10            # alert reviewer: true fraud called legitimate
REVIEW_FP      = 0.004           # alert reviewer: legitimate called fraud
AUDIT_FN       = 0.06            # audit reviewer (more careful)
AUDIT_FP       = 0.001

MECH = ['Stolen Card', 'Account Takeover', 'Identity Mismatch',
        'Promotion Abuse', 'Multiple Account Abuse', 'Return Abuse']
MECH_P = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05])

# signature -> (background rate in legit population)
SIGS = {
    'ip_country_mismatch': 0.005, 'card_fresh': 0.020, 'threeds_fail': 0.012,
    'threeds_skip': 0.090, 'high_amount': 0.015, 'express_ship': 0.120,
    'addr_mismatch': 0.012, 'new_device': 0.060, 'unusual_login': 0.030,
    'addr_changed': 0.045, 'amount_escalation': 0.050, 'guest_checkout': 0.080,
    'disposable_email': 0.030, 'new_account': 0.008, 'kyc0': 0.120,
    'ip_region_mismatch': 0.060, 'device_shared': 0.006, 'ip_shared': 0.004,
    'coupon': 0.220, 'low_value': 0.150, 'single_category': 0.380,
    'high_return_rate': 0.100, 'hosting_asn': 0.030, 'burst_timing': 0.030,
    'long_tenure': 0.250,
}

# mechanism -> {signature: expression probability at full strength}
MECH_SIGS = {
    'Stolen Card': {'ip_country_mismatch': .74, 'card_fresh': .80, 'threeds_fail': .66,
                    'threeds_skip': .50, 'high_amount': .70, 'express_ship': .70,
                    'addr_mismatch': .76, 'hosting_asn': .40},
    'Account Takeover': {'new_device': .82, 'unusual_login': .80, 'addr_changed': .72,
                         'amount_escalation': .70, 'ip_region_mismatch': .68,
                         'threeds_fail': .72, 'express_ship': .55,
                         'hosting_asn': .25, 'addr_mismatch': .70,
                         'ip_country_mismatch': .30, 'card_fresh': .38},
    'Identity Mismatch': {'guest_checkout': .84, 'disposable_email': .74, 'new_account': .80,
                          'kyc0': .86, 'ip_region_mismatch': .70, 'addr_mismatch': .80,
                          'card_fresh': .62},
    'Promotion Abuse': {'device_shared': .90, 'ip_shared': .88, 'coupon': .88,
                        'low_value': .80, 'new_account': .84, 'single_category': .76,
                        'kyc0': .60},
    'Multiple Account Abuse': {'device_shared': .92, 'ip_shared': .92, 'new_account': .66,
                               'addr_mismatch': .56, 'burst_timing': .78, 'kyc0': .58},
    'Return Abuse': {'high_return_rate': .86, 'long_tenure': .80, 'high_amount': .80,
                     'single_category': .70, 'express_ship': .42},
}

CATEGORIES = ['Electronics', 'Apparel', 'Home & Kitchen', 'Beauty', 'Grocery',
              'Sports', 'Books', 'Toys', 'Jewellery', 'Mobiles']
CAT_P      = np.array([.16, .20, .14, .10, .12, .07, .06, .06, .04, .05])
CAT_PRICE  = {'Electronics': 8.2, 'Apparel': 6.6, 'Home & Kitchen': 7.0, 'Beauty': 6.2,
              'Grocery': 5.6, 'Sports': 7.0, 'Books': 5.8, 'Toys': 6.4,
              'Jewellery': 9.0, 'Mobiles': 9.4}
REGION_CODES = [11, 12, 14, 16, 20, 22, 24, 26, 30, 34, 36, 38, 40, 42, 44,
                45, 46, 50, 52, 56, 57, 58, 60, 62, 67, 70, 72, 75, 78, 80, 82, 85]
NETWORKS  = ['Visa', 'Mastercard', 'RuPay', 'Amex']
NET_P     = np.array([.38, .32, .27, .03])
ISSUERS   = ['HDFC Bank', 'ICICI Bank', 'State Bank of India', 'Axis Bank', 'Kotak Mahindra',
             'IndusInd Bank', 'Yes Bank', 'IDFC First', 'Federal Bank', 'Citibank NA',
             'Standard Chartered', 'Bank of Baroda', 'Punjab National Bank', 'AU Small Finance']
FOREIGN_ISSUERS = ['Chase', 'Capital One', 'Barclays', 'Santander', 'DBS', 'Maybank']
PROD_TYPES = ['Classic', 'Gold', 'Platinum', 'Signature', 'Business', 'Prepaid']
PT_P       = np.array([.34, .24, .20, .10, .08, .04])
GATEWAYS   = ['Razorpay', 'PayU', 'Cashfree', 'BillDesk', 'Amazon Pay Gateway']
METHODS    = ['UPI', 'Credit Card', 'Debit Card', 'Cash on Delivery', 'Wallet', 'Net Banking', 'EMI']
CARD_METHODS = {'Credit Card', 'Debit Card', 'EMI'}
FEE_RATE   = {'UPI': .0035, 'Credit Card': .0195, 'Debit Card': .0090, 'Cash on Delivery': .0,
              'Wallet': .0150, 'Net Banking': .0110, 'EMI': .0240}
SHIP_SPEED = ['Standard', 'Express', 'Same Day', 'Scheduled']
EMAIL_CLS  = ['mainstream', 'corporate', 'niche', 'disposable']
MCC_MAP    = {'Electronics': 5732, 'Apparel': 5651, 'Home & Kitchen': 5719, 'Beauty': 5977,
              'Grocery': 5411, 'Sports': 5941, 'Books': 5942, 'Toys': 5945,
              'Jewellery': 5944, 'Mobiles': 4812}
CITIES = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune',
          'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Indore', 'Kochi', 'Chandigarh']
FRAUD_RULE_CODES = {'R01': 'HIGH_AMOUNT_VS_MERCHANT', 'R02': 'IP_COUNTRY_MISMATCH',
                    'R03': 'BILLING_SHIPPING_MISMATCH', 'R04': 'THREEDS_FAILED'}
QUEUES = ['CARD_RISK_Q', 'ACCOUNT_RISK_Q', 'GENERAL_Q']


def ts_str(a):
    s = pd.to_datetime(pd.Series(np.asarray(a)))
    return s.dt.strftime('%Y-%m-%dT%H:%M:%S').to_numpy()


def main(outdir, emit_truth, clean=False):
    rng = np.random.default_rng(SEED)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'contracts'), exist_ok=True)

    # ========================================================================
    # 1. PAYMENT TIMESTAMPS (volume seasonality, festive Oct/Nov peak)
    # ========================================================================
    months = pd.date_range(START, END, freq='MS')
    shape = np.array([1.00, .85, .99, .98, .99, .99, .98, 1.00, 1.15, 1.42, 1.84, 1.54])
    w = np.concatenate([shape, shape * 1.08])[:len(months)]
    w = w / w.sum()
    per_month = rng.multinomial(N_PAY, w)
    ts_parts = []
    hour_w = np.array([.9, .5, .3, .2, .2, .3, .6, 1.2, 1.8, 2.4, 2.8, 3.0, 3.1, 3.0,
                       2.9, 2.8, 2.9, 3.1, 3.4, 3.8, 4.0, 3.4, 2.4, 1.5])
    hour_w = hour_w / hour_w.sum()
    for m, n in zip(months, per_month):
        m_end = min(m + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59), END)
        days = (m_end - m).days + 1
        dow = (m + pd.to_timedelta(np.arange(days), 'D')).dayofweek.to_numpy()
        dw = np.array([1.05, 1.02, 1.00, 1.03, 1.12, 1.24, 1.16])[dow]
        d = rng.choice(days, n, p=dw / dw.sum())
        h = rng.choice(24, n, p=hour_w)
        mi = rng.integers(0, 60, n)
        s = rng.integers(0, 60, n)
        t = m + pd.to_timedelta(d, 'D') + pd.to_timedelta(h, 'h') + \
            pd.to_timedelta(mi, 'm') + pd.to_timedelta(s, 's')
        ts_parts.append(t[t <= END])
    pay_ts = np.concatenate([p.values for p in ts_parts])
    rng.shuffle(pay_ts)
    pay_ts = pd.to_datetime(pay_ts[:N_PAY])
    if len(pay_ts) < N_PAY:  # top up
        extra = START + pd.to_timedelta(rng.integers(0, int((END - START).total_seconds()),
                                                     N_PAY - len(pay_ts)), 's')
        pay_ts = pd.DatetimeIndex(np.concatenate([pay_ts.values, extra.values]))
    P = pd.DataFrame({'payment_timestamp': pay_ts})
    P['payment_id'] = ['PAY%07d' % (i + 1) for i in range(N_PAY)]
    P['order_id'] = ['O%07d' % (i + 1) for i in range(N_PAY)]

    # ========================================================================
    # 2. CUSTOMERS (assignment first; signup back-dated so tenure is coherent)
    # ========================================================================
    cust_ids = np.array(['C%06d' % (i + 1) for i in range(N_CUST)])
    N_RESERVED = 12_000                       # single-use accounts (genuinely new + fraud rings)
    est_ids = cust_ids[:N_CUST - N_RESERVED]
    act = rng.lognormal(0, .55, len(est_ids)); act /= act.sum()
    P['customer_id'] = rng.choice(est_ids, N_PAY, p=act)

    # ========================================================================
    # 3. LATENT FRAUD ASSIGNMENT
    # ========================================================================
    n_fraud = int(round(N_PAY * FRAUD_RATE))
    fraud_idx = rng.choice(N_PAY, n_fraud, replace=False)
    is_fraud = np.zeros(N_PAY, bool); is_fraud[fraud_idx] = True
    mech = np.array([''] * N_PAY, dtype=object)
    mech[fraud_idx] = rng.choice(MECH, n_fraud, p=MECH_P)
    strength = np.zeros(N_PAY)
    weak = rng.random(n_fraud) < WEAK_FRAC
    strength[fraud_idx] = np.where(weak, rng.uniform(.05, .25, n_fraud),
                                   rng.uniform(.75, 1.0, n_fraud))

    # ---- express signatures -------------------------------------------------
    S = {k: (rng.random(N_PAY) < v) for k, v in SIGS.items()}   # background
    # legit rows that mimic an entire mechanism (traveller on a new device shipping to a
    # new address, bargain hunter stacking coupons, heavy returner). Weighted away from
    # Stolen Card so the mimics cost the MODEL precision without gifting the rules engine
    # extra false positives.
    false_sig = (rng.random(N_PAY) < FALSE_SIG_FRAC) & ~is_fraud
    mimic_m = rng.choice(MECH, N_PAY, p=[.04, .16, .10, .34, .10, .26])
    for m_name, sigmap in MECH_SIGS.items():
        rows = false_sig & (mimic_m == m_name)
        if not rows.any():
            continue
        for sname, p_expr in sigmap.items():
            S[sname] = S[sname] | (rows & (rng.random(N_PAY) < p_expr * MIMIC_STRENGTH))

    drift = (P.payment_timestamp.values >= DRIFT_START.to_datetime64())
    for m_name, sigmap in MECH_SIGS.items():
        rows = (mech == m_name)
        if not rows.any():
            continue
        for s, p_expr in sigmap.items():
            p = np.full(N_PAY, p_expr)
            if m_name == 'Account Takeover':          # 2025-H2 concept drift
                if s == 'new_device':
                    p = np.where(drift, .10, p_expr)
                elif s in ('hosting_asn', 'threeds_fail'):
                    p = np.where(drift, .85, p_expr)
            hit = rows & (rng.random(N_PAY) < p * strength)
            S[s] = S[s] | hit

    na = np.where(S['new_account'])[0]
    assert len(na) <= N_RESERVED, f"reserved pool too small: need {len(na)}"
    P.loc[P.index[na], 'customer_id'] = cust_ids[N_CUST - N_RESERVED:][:len(na)]

    # ========================================================================
    # 4. CUSTOMER MASTER
    # ========================================================================
    first_pay = P.groupby('customer_id').payment_timestamp.min()
    newacct_cust = set(P.loc[S['new_account'], 'customer_id'])
    gap = pd.to_timedelta(rng.exponential(220, len(first_pay)).clip(1, 2000), 'D')
    signup = first_pay - gap
    is_new = first_pay.index.isin(list(newacct_cust))
    signup = signup.where(~is_new,
                          first_pay - pd.to_timedelta(rng.uniform(.2, 2.0, len(first_pay)), 'D'))
    signup = signup.clip(lower=pd.Timestamp('2021-06-01'))
    cust = pd.DataFrame({'customer_id': cust_ids})
    cust = cust.merge(signup.rename('signup_timestamp'), left_on='customer_id',
                      right_index=True, how='left')
    miss = cust.signup_timestamp.isna()
    cust.loc[miss, 'signup_timestamp'] = START - pd.to_timedelta(
        rng.uniform(30, 1200, miss.sum()), 'D')
    nC = len(cust)
    cust['home_region_code'] = rng.choice(REGION_CODES, nC)
    cust['home_pincode'] = (cust.home_region_code * 10000 + rng.integers(1, 9999, nC)).astype(str)
    cust['kyc_level'] = rng.choice([0, 1, 2], nC, p=[.12, .48, .40])
    cust['email_domain_class'] = rng.choice(EMAIL_CLS, nC, p=[.70, .16, .12, .02])
    cust['acquisition_channel'] = rng.choice(
        ['Organic', 'Paid Search', 'Social', 'Referral', 'Affiliate', 'Email'],
        nC, p=[.30, .22, .18, .12, .10, .08])
    cust['city'] = rng.choice(CITIES, nC)
    cust['city_tier'] = rng.choice([1, 2, 3], nC, p=[.45, .35, .20])
    cust['prior_return_rate'] = np.round(rng.beta(2, 12, nC), 4)
    cust['prior_orders_12m'] = rng.poisson(3.2, nC)
    cidx = pd.Series(np.arange(nC), index=cust.customer_id)
    ci = cidx.reindex(P.customer_id).to_numpy()

    # per-payment customer attribute overrides driven by signatures
    kyc = cust.kyc_level.to_numpy()[ci].copy()
    kyc[S['kyc0']] = 0
    email_cls = cust.email_domain_class.to_numpy()[ci].copy()
    email_cls[S['disposable_email']] = 'disposable'
    prr = cust.prior_return_rate.to_numpy()[ci].copy()
    prr[S['high_return_rate']] = np.round(rng.beta(7, 4, S['high_return_rate'].sum()), 4)
    cust['kyc_level'] = pd.Series(kyc, index=P.index).groupby(P.customer_id).min() \
        .reindex(cust.customer_id).fillna(pd.Series(cust.kyc_level.values,
                                                    index=cust.customer_id)).astype(int).values
    ecl = pd.Series(email_cls, index=P.index).groupby(P.customer_id).agg(
        lambda s: 'disposable' if (s == 'disposable').any() else s.iloc[0])
    cust['email_domain_class'] = ecl.reindex(cust.customer_id).fillna(
        pd.Series(cust.email_domain_class.values, index=cust.customer_id)).values
    prr_c = pd.Series(prr, index=P.index).groupby(P.customer_id).max()
    cust['prior_return_rate'] = prr_c.reindex(cust.customer_id).fillna(
        pd.Series(cust.prior_return_rate.values, index=cust.customer_id)).values
    # long tenure signature
    lt_cust = P.loc[S['long_tenure'], 'customer_id'].unique()
    m_lt = cust.customer_id.isin(lt_cust)
    cust.loc[m_lt, 'signup_timestamp'] = cust.loc[m_lt, 'signup_timestamp'].where(
        cust.loc[m_lt, 'signup_timestamp'] < START - pd.Timedelta('400D'),
        START - pd.to_timedelta(rng.uniform(420, 1300, m_lt.sum()), 'D'))

    signup_arr = pd.to_datetime(cust.signup_timestamp).to_numpy()[ci]
    P['account_age_days_at_txn'] = np.round(
        (P.payment_timestamp.to_numpy() - signup_arr) / np.timedelta64(1, 'D'), 3).clip(0.01)

    # ========================================================================
    # 5. MERCHANTS
    # ========================================================================
    mcat = rng.choice(CATEGORIES, N_MERCH, p=CAT_P)
    merch = pd.DataFrame({
        'merchant_id': ['M%05d' % (i + 1) for i in range(N_MERCH)],
        'merchant_name': ['Merchant_%05d' % (i + 1) for i in range(N_MERCH)],
        'category': mcat,
        'mcc': [MCC_MAP[c] for c in mcat],
        'city': rng.choice(CITIES, N_MERCH),
        'onboarded_date': (START - pd.to_timedelta(rng.uniform(30, 1800, N_MERCH), 'D')).normalize(),
        'avg_ticket_size': np.nan,   # set from realised payments below
        'trailing_chargeback_rate_bps': np.round(rng.gamma(2, 9, N_MERCH), 1),
    })
    mw = rng.lognormal(0, .8, N_MERCH); mw /= mw.sum()
    P['merchant_id'] = rng.choice(merch.merchant_id.values, N_PAY, p=mw)
    midx = pd.Series(np.arange(N_MERCH), index=merch.merchant_id)
    mi = midx.reindex(P.merchant_id).to_numpy()
    merch_cat = merch.category.to_numpy()[mi]
    merch_tkt = merch.avg_ticket_size.to_numpy()[mi]

    # ========================================================================
    # 6. ORDERS + ORDER ITEMS
    # ========================================================================
    item_count = rng.choice([1, 2, 3, 4, 5, 6], N_PAY, p=[.25, .28, .20, .13, .08, .06])
    item_count = np.where(S['single_category'] & (item_count > 3), 2, item_count)
    scale = np.array([1.22, 1.00, 0.83])[cust.city_tier.to_numpy()[ci] - 1]
    scale[S['high_amount']] *= rng.uniform(3.0, 9.0, S['high_amount'].sum())
    scale[S['amount_escalation']] *= rng.uniform(2.5, 6.0, S['amount_escalation'].sum())
    scale[S['low_value']] *= rng.uniform(.12, .35, S['low_value'].sum())

    tot = int(item_count.sum())
    oid_rep = np.repeat(P.order_id.values, item_count)
    cat_rep = np.where(rng.random(tot) < .70,
                       np.repeat(merch_cat, item_count),
                       rng.choice(CATEGORIES, tot, p=CAT_P))
    single = np.repeat(S['single_category'], item_count)
    cat_rep = np.where(single, np.repeat(merch_cat, item_count), cat_rep)
    mu = np.array([CAT_PRICE[c] for c in cat_rep])
    unit = np.exp(rng.normal(mu, .45)) * np.repeat(scale, item_count)
    u_ = rng.random(tot)
    unit = np.where(u_ < .44, np.floor(unit) + 0.99,
           np.where(u_ < .70, np.round(unit / 10) * 10.0,
           np.where(u_ < .78, np.floor(unit) + 0.50, np.round(unit, 2))))
    unit = np.round(unit, 2).clip(29, None)
    qty = rng.choice([1, 2, 3], tot, p=[.78, .17, .05])
    items = pd.DataFrame({
        'order_item_id': ['OI%08d' % (i + 1) for i in range(tot)],
        'order_id': oid_rep, 'sku': ['SKU%06d' % x for x in rng.integers(1, 60000, tot)],
        'category': cat_rep, 'quantity': qty, 'unit_price': unit,
        'line_amount': np.round(unit * qty, 2)})
    ov = items.groupby('order_id', sort=False).line_amount.sum()
    order_value = ov.reindex(P.order_id).to_numpy()

    coupon = S['coupon']
    disc_pct = np.where(coupon, rng.uniform(.05, .40, N_PAY), 0.0)
    discount = np.round(order_value * disc_pct, 2)
    ship_speed = np.array(SHIP_SPEED)[rng.choice(4, N_PAY, p=[.62, .18, .08, .12])]
    ship_speed[S['express_ship']] = 'Express'
    ship_charge = np.round(np.where(ship_speed == 'Standard', 0.0,
                           np.where(ship_speed == 'Express', 79.0,
                           np.where(ship_speed == 'Same Day', 149.0, 49.0))), 2)
    home_pin = cust.home_pincode.to_numpy()[ci]
    home_reg = cust.home_region_code.to_numpy()[ci]
    billing_pin = home_pin
    ship_reg = np.where(S['addr_mismatch'], rng.choice(REGION_CODES, N_PAY), home_reg)
    ship_pin = np.where(S['addr_mismatch'],
                        (ship_reg * 10000 + rng.integers(1, 9999, N_PAY)).astype(str), home_pin)
    addr_age = np.where(S['addr_changed'], rng.uniform(0.5, 72, N_PAY),
                        rng.uniform(200, 20000, N_PAY))
    orders = pd.DataFrame({
        'order_id': P.order_id, 'customer_id': P.customer_id,
        'order_timestamp': (P.payment_timestamp - pd.to_timedelta(
            rng.uniform(30, 900, N_PAY), 's')),
        'item_count': item_count, 'order_value': np.round(order_value, 2),
        'discount_amount': discount, 'primary_category': merch_cat,
        'coupon_code': np.where(coupon, ['CPN%04d' % x for x in rng.integers(1, 400, N_PAY)], ''),
        'shipping_speed': ship_speed, 'shipping_charge': ship_charge,
        'shipping_pincode': ship_pin, 'billing_pincode': billing_pin,
        'address_match_flag': (~S['addr_mismatch']).astype(int),
        'shipping_addr_age_hours': np.round(addr_age, 2),
        'delivery_type': rng.choice(['Home', 'Pickup Point', 'Locker'], N_PAY, p=[.86, .09, .05]),
    })
    P['payment_amount'] = np.round((order_value - discount + ship_charge).clip(29), 2)

    # ========================================================================
    # 7. DEVICES / CARDS / IP
    # ========================================================================
    nd = S['new_device']
    n_fresh = int(nd.sum())
    # pool layout: [0,nC) one primary device per customer | [nC,nC+N_RING) abuse-ring devices
    #              | next N_SHARED household/office devices | last n_fresh single-use devices
    N_RING, N_SHARED = 150, 20_000
    b_ring, b_shared, b_fresh = nC, nC + N_RING, nC + N_RING + N_SHARED
    N_DEV = b_fresh + n_fresh
    dev = pd.DataFrame({
        'device_id': ['DEV%08d' % (i + 1) for i in range(N_DEV)],
        'device_type': rng.choice(['Android Phone', 'iPhone', 'Windows Desktop', 'Mac', 'Tablet'],
                                  N_DEV, p=[.52, .19, .18, .06, .05]),
        'os_family': '', 'browser_family': rng.choice(
            ['Chrome Mobile', 'Chrome', 'Safari', 'Firefox', 'Edge', 'WebView'],
            N_DEV, p=[.42, .25, .16, .06, .06, .05]),
        'is_emulator': (rng.random(N_DEV) < .012).astype(int),
        'first_seen_timestamp': START - pd.to_timedelta(rng.uniform(1, 1400, N_DEV), 'D')})
    dev['os_family'] = np.select(
        [dev.device_type == 'Android Phone', dev.device_type == 'iPhone',
         dev.device_type == 'Windows Desktop', dev.device_type == 'Mac'],
        ['Android', 'iOS', 'Windows', 'macOS'], 'Android')
    dev_i = np.arange(nC)[ci].copy()                                 # primary device, 1:1
    swap = rng.random(N_PAY) < .03                                   # legit household sharing
    dev_i[swap] = b_shared + rng.integers(0, N_SHARED, swap.sum())
    shared = S['device_shared']
    dev_i[shared] = b_ring + rng.integers(0, N_RING, shared.sum())
    dev_i[nd] = np.arange(b_fresh, N_DEV)                            # one fresh device each
    P['device_id'] = dev.device_id.to_numpy()[dev_i]
    fs = dev.first_seen_timestamp.to_numpy().copy()
    fs[b_fresh:N_DEV] = (P.payment_timestamp.to_numpy()[nd] -
                         pd.to_timedelta(rng.uniform(0.05, 20, n_fresh), 'h').to_numpy())
    e_d = pd.Series(P.payment_timestamp.to_numpy()).groupby(pd.Series(dev_i)).min()
    fs[e_d.index] = np.minimum(fs[e_d.index], e_d.to_numpy() - np.timedelta64(60, 's'))
    dev['first_seen_timestamp'] = fs

    n_asn_res, n_asn_host = 220, 18
    asn_res = 9000 + rng.choice(29000, n_asn_res, replace=False)      # 9000-37999
    asn_host = 39000 + rng.choice(9000, n_asn_host, replace=False)    # 39000-47999
    asn_pool = np.concatenate([asn_res, asn_host])
    asn_type = np.concatenate([rng.choice(['Residential', 'Mobile'], n_asn_res, p=[.55, .45]),
                               rng.choice(['Hosting', 'VPN'], n_asn_host, p=[.6, .4])])
    ipr = pd.DataFrame({
        'ip_asn': 'AS' + pd.Series(asn_pool).astype(str),
        'asn_name': ['ASN_%05d' % a for a in asn_pool],
        'asn_type': asn_type,
        'is_hosting': np.isin(asn_type, ['Hosting', 'VPN']).astype(int),
        'asn_country': np.where(np.isin(asn_type, ['Hosting', 'VPN']),
                                rng.choice(['US', 'NL', 'SG', 'DE', 'RU'], len(asn_pool)), 'IN'),
        'reputation_score': np.round(np.where(np.isin(asn_type, ['Hosting', 'VPN']),
                                              rng.uniform(35, 90, len(asn_pool)),
                                              rng.uniform(0, 35, len(asn_pool))), 2)})
    cust_asn = rng.choice(asn_res, nC)
    asn = cust_asn[ci].copy()
    asn[S['hosting_asn']] = rng.choice(asn_host, S['hosting_asn'].sum())
    P['ip_asn'] = 'AS' + pd.Series(asn).astype(str)
    ipc = np.full(N_PAY, 'IN', dtype=object)
    foreign = ['US', 'GB', 'AE', 'SG', 'NG', 'RU', 'CN', 'DE', 'BR', 'VN']
    ipc[S['ip_country_mismatch']] = rng.choice(foreign, S['ip_country_mismatch'].sum())
    P['ip_country'] = ipc
    ipreg = np.where(S['ip_region_mismatch'] | S['ip_country_mismatch'],
                     rng.choice(REGION_CODES, N_PAY), home_reg)
    P['ip_region_code'] = ipreg
    cust_ip_b = rng.integers(0, 256, (nC, 2))
    ipb = cust_ip_b[ci]
    ip_shared_pool = rng.integers(1, 255, (300, 4))
    ip = np.stack([np.where(asn >= 39000, 45, 103), ipb[:, 0], ipb[:, 1],
                   rng.integers(1, 255, N_PAY)], 1)
    sh = S['ip_shared']
    ip[sh] = ip_shared_pool[rng.choice(300, sh.sum())]
    P['ip_address'] = pd.Series(['.'.join(map(str, r)) for r in ip])

    _band = np.digitize(P.payment_amount.to_numpy(), [800, 5000, 30000])
    _mp = np.array([  # UPI  Credit  Debit   COD  Wallet NetBank  EMI
        [.42, .06, .14, .18, .17, .03, .00],   # < Rs 800
        [.36, .16, .19, .15, .07, .07, .00],   # Rs 800-5k
        [.24, .28, .18, .08, .02, .14, .06],   # Rs 5k-30k
        [.10, .37, .09, .02, .01, .16, .25]])  # > Rs 30k
    _u = rng.random(N_PAY)
    method = np.array(METHODS)[(_mp[_band].cumsum(1) < _u[:, None]).sum(1).clip(0, 6)]
    sc_mech = (mech == 'Stolen Card')
    method[sc_mech] = np.where(rng.random(sc_mech.sum()) < .65, 'Credit Card', 'Debit Card')
    upi_boost = (P.payment_timestamp.dt.year == 2025).to_numpy() & (rng.random(N_PAY) < .06)
    method[upi_boost] = 'UPI'
    is_card = np.isin(method, list(CARD_METHODS))
    n_cards_pool = 160_000
    cards_bin = rng.integers(400000, 659999, n_cards_pool)
    card_net = np.array(NETWORKS)[rng.choice(4, n_cards_pool, p=NET_P)]
    card_foreign = rng.random(n_cards_pool) < .035
    cards = pd.DataFrame({
        'card_token': ['CRD%08d' % (i + 1) for i in range(n_cards_pool)],
        'bin': cards_bin, 'network': card_net,
        'issuer': np.where(card_foreign, rng.choice(FOREIGN_ISSUERS, n_cards_pool),
                           rng.choice(ISSUERS, n_cards_pool)),
        'product_type': np.array(PROD_TYPES)[rng.choice(6, n_cards_pool, p=PT_P)],
        'issuing_country': np.where(card_foreign, rng.choice(
            ['US', 'GB', 'AE', 'SG', 'DE'], n_cards_pool), 'IN'),
        'token_first_seen_timestamp': START - pd.to_timedelta(
            rng.uniform(1, 1200, n_cards_pool), 'D')})
    cards['is_prepaid'] = (cards.product_type == 'Prepaid').astype(int)
    fresh = S['card_fresh'] & is_card
    n_freshcard = int(fresh.sum())
    b_freshcard = n_cards_pool - n_freshcard          # tail block = single-use fresh tokens
    cust_card = rng.integers(0, b_freshcard, nC)
    card_i = np.where(is_card, cust_card[ci], -1)
    card_i[fresh] = np.arange(b_freshcard, n_cards_pool)
    # stolen-card mechanism: foreign issuer over-represented
    sc = (mech == 'Stolen Card') & is_card & (rng.random(N_PAY) < .5 * strength)
    foreign_cards = np.where(cards.issuing_country.values != 'IN')[0]
    if sc.sum() and len(foreign_cards):
        card_i[sc] = rng.choice(foreign_cards, sc.sum())
    P['card_token'] = np.where(card_i >= 0, cards.card_token.to_numpy()[card_i.clip(0)], '')
    tfs = np.where(card_i >= 0, cards.token_first_seen_timestamp.to_numpy()[card_i.clip(0)],
                   np.datetime64('NaT'))
    tfs = pd.to_datetime(tfs)
    tfs_new = P.payment_timestamp - pd.to_timedelta(rng.uniform(.05, 1.0, N_PAY), 'h')
    tfs = pd.Series(np.where(fresh, tfs_new, tfs))
    tf = cards.token_first_seen_timestamp.to_numpy().copy()
    tf[b_freshcard:n_cards_pool] = np.minimum(
        tfs[fresh].to_numpy(),
        P.payment_timestamp.to_numpy()[fresh] - np.timedelta64(60, 's'))
    ok_c = card_i >= 0
    earliest = pd.Series(P.payment_timestamp.to_numpy()[ok_c]).groupby(
        pd.Series(card_i[ok_c])).min()
    tf[earliest.index] = np.minimum(tf[earliest.index],
                                    earliest.to_numpy() - np.timedelta64(60, 's'))
    cards['token_first_seen_timestamp'] = tf
    tfs = pd.Series(np.where(ok_c, tf[card_i.clip(0)], np.datetime64('NaT')))
    tfs = pd.to_datetime(tfs)

    # ========================================================================
    # 8. PAYMENT ATTRIBUTES
    # ========================================================================
    P['payment_method'] = method
    P['payment_gateway'] = np.where(method == 'Cash on Delivery', '',
                                    np.array(GATEWAYS)[rng.choice(5, N_PAY,
                                                       p=[.28, .22, .20, .16, .14])])
    _base = np.array([FEE_RATE[m] for m in method]) * rng.uniform(.88, 1.14, N_PAY)
    _waiver = rng.random(N_PAY) < .034                  # promo / negotiated fee waivers
    P['processing_fee'] = np.round(P.payment_amount.to_numpy() * np.where(_waiver, 0.0, _base), 2)
    tds_att = np.where(is_card, rng.random(N_PAY) > .09, False)
    tds_att = np.where(S['threeds_skip'] & is_card, False, tds_att)
    tds_ok = tds_att & (rng.random(N_PAY) > .03)
    tds_ok = np.where(S['threeds_fail'] & tds_att, False, tds_ok)
    P['is_3ds_attempted'] = tds_att.astype(int)
    P['is_3ds_success'] = tds_ok.astype(int)
    P['is_guest_checkout'] = S['guest_checkout'].astype(int)
    P['attempt_seq_in_session'] = np.where(
        S['burst_timing'], rng.integers(2, 7, N_PAY),
        rng.choice([1, 2, 3], N_PAY, p=[.88, .09, .03]))
    P['session_id'] = ['SES%08d' % (i + 1) for i in range(N_PAY)]
    failed = rng.random(N_PAY) < .030
    P['payment_status'] = np.where(failed, 'Failed', 'Success')
    P['failure_reason'] = np.where(failed, rng.choice(
        ['Insufficient Funds', 'Gateway Timeout', 'OTP Failed', 'Bank Declined',
         'UPI Timeout', 'Card Expired'], N_PAY), '')

    # ========================================================================
    # 8b. LOGINS (built before the engine: one rule reads the prior login) (ATO precursors + background)
    # ========================================================================
    ato = np.where(S['unusual_login'])[0]
    n_bg = max(N_LOGIN - len(ato), 0)
    # 48% of logins immediately precede a purchase (real sessions are purchase-driven);
    # the rest are ambient browsing/session checks scattered through the window
    n_pre = int(n_bg * .48)
    pre_pay = rng.choice(N_PAY, n_pre, replace=False)
    pre_cust = P.customer_id.to_numpy()[pre_pay]
    pre_ts = (P.payment_timestamp.to_numpy()[pre_pay] -
              pd.to_timedelta(rng.uniform(1, 115, n_pre), 'm').to_numpy())
    n_amb = n_bg - n_pre
    amb_cust = rng.choice(est_ids, n_amb, p=act)
    amb_ts = START + pd.to_timedelta(rng.uniform(0, (END - START).total_seconds(), n_amb), 's')
    bg_cust = np.concatenate([pre_cust, amb_cust])
    bg_ts = pd.DatetimeIndex(np.concatenate([pre_ts, amb_ts.to_numpy()]))
    bg_dev = cidx.reindex(bg_cust).to_numpy().copy()   # primary device index == customer index
    swap2 = rng.random(n_bg) < .06
    bg_dev[swap2] = b_shared + rng.integers(0, N_SHARED, swap2.sum())
    L = pd.DataFrame({
        'customer_id': np.concatenate([P.customer_id.to_numpy()[ato], bg_cust]),
        'login_timestamp': np.concatenate([
            (P.payment_timestamp.to_numpy()[ato] -
             pd.to_timedelta(rng.uniform(.5, 24, len(ato)), 'h').to_numpy()),
            bg_ts.to_numpy()]),
        'device_id': np.concatenate([P.device_id.to_numpy()[ato], dev.device_id.to_numpy()[bg_dev]]),
        'ip_address': np.concatenate([P.ip_address.to_numpy()[ato],
                                      pd.Series(['.'.join(map(str, r)) for r in
                                      np.stack([np.full(n_bg, 103), cust_ip_b[
                                          cidx.reindex(bg_cust).to_numpy()][:, 0],
                                          cust_ip_b[cidx.reindex(bg_cust).to_numpy()][:, 1],
                                          rng.integers(1, 255, n_bg)], 1)]).to_numpy()]),
        'ip_country': np.concatenate([P.ip_country.to_numpy()[ato], np.full(n_bg, 'IN')]),
        'ip_asn': np.concatenate([P.ip_asn.to_numpy()[ato],
                                  'AS' + pd.Series(cust_asn[cidx.reindex(bg_cust).to_numpy()]
                                                   ).astype(str).to_numpy()]),
        'is_ato': np.concatenate([np.ones(len(ato), bool), np.zeros(n_bg, bool)])})
    L = L.sample(frac=1, random_state=SEED).reset_index(drop=True)
    L['login_id'] = ['L%07d' % (i + 1) for i in range(len(L))]
    L['session_id'] = ['LSE%08d' % (i + 1) for i in range(len(L))]
    nL = len(L)
    L['new_device_flag'] = np.where(L.is_ato, (rng.random(nL) < .72).astype(int),
                                    (rng.random(nL) < .095).astype(int))
    L['unusual_location_flag'] = np.where(L.is_ato, (rng.random(nL) < .70).astype(int),
                                          (rng.random(nL) < .033).astype(int))
    L['failed_attempt_count'] = np.where(L.is_ato, rng.choice([0, 1, 2, 3, 4], nL,
                                                              p=[.18, .34, .28, .14, .06]),
                                         rng.choice([0, 1, 2, 3], nL, p=[.79, .17, .035, .005]))
    L['login_channel'] = rng.choice(['App', 'Web', 'mWeb'], nL, p=[.58, .28, .14])
    L['auth_method'] = rng.choice(['Password', 'OTP', 'Biometric', 'SSO'], nL,
                                  p=[.44, .34, .16, .06])
    L['login_risk_score'] = np.round(
        (12 * L.new_device_flag + 16 * L.unusual_location_flag +
         6 * L.failed_attempt_count + rng.gamma(2, 4, nL)).clip(0, 100), 2)
    L = L.drop(columns='is_ato')

    _ls = L[['customer_id', 'login_timestamp', 'session_id']].copy()
    _ls['login_timestamp'] = pd.to_datetime(_ls.login_timestamp).astype('datetime64[ns]')
    _ls = _ls.sort_values('login_timestamp')
    _pp = P[['payment_id', 'customer_id', 'payment_timestamp']].copy()
    _pp['payment_timestamp'] = pd.to_datetime(_pp.payment_timestamp).astype('datetime64[ns]')
    _pp = _pp.sort_values('payment_timestamp')
    _m = pd.merge_asof(_pp, _ls, left_on='payment_timestamp', right_on='login_timestamp',
                       by='customer_id', allow_exact_matches=False)
    _within = ((_m.payment_timestamp - _m.login_timestamp).dt.total_seconds() <= 7200)
    _sess = _m.session_id.where(_within).set_axis(_m.payment_id).reindex(P.payment_id)
    P['session_id'] = np.where(_sess.notna().to_numpy(), _sess.to_numpy(),
                               np.array(['SES%08d' % (i + 1) for i in range(N_PAY)]))
    _lg = L[['customer_id', 'login_timestamp', 'unusual_location_flag',
             'failed_attempt_count']].copy()
    _lg['login_timestamp'] = pd.to_datetime(_lg.login_timestamp).astype('datetime64[ns]')
    _lg = _lg.sort_values('login_timestamp')
    _p = P[['payment_id', 'customer_id', 'payment_timestamp']].copy()
    _p['payment_timestamp'] = pd.to_datetime(_p.payment_timestamp).astype('datetime64[ns]')
    _p = _p.sort_values('payment_timestamp')
    _a = pd.merge_asof(_p, _lg, left_on='payment_timestamp', right_on='login_timestamp',
                       by='customer_id', allow_exact_matches=False)
    _h = (_a.payment_timestamp - _a.login_timestamp).dt.total_seconds() / 3600
    _risk = ((_h <= 24) & ((_a.unusual_location_flag == 1) |
                           (_a.failed_attempt_count >= 2))).to_numpy()
    risky_login = pd.Series(_risk, index=_a.payment_id).reindex(P.payment_id).fillna(
        False).to_numpy()

    mt = P.groupby('merchant_id').payment_amount.median()
    merch['avg_ticket_size'] = np.round(
        mt.reindex(merch.merchant_id).fillna(mt.median()).to_numpy(), 2)
    merch_tkt = merch.avg_ticket_size.to_numpy()[mi]

    # ========================================================================
    # 9. INCUMBENT RULES ENGINE (reads 4 of ~24 true drivers, + noise)
    # ========================================================================
    dev_n_cust = pd.Series(P.device_id).map(P.groupby('device_id').customer_id.nunique()).to_numpy()
    m_p98 = P.groupby('merchant_id').payment_amount.quantile(.98)
    r01 = (P.payment_amount.to_numpy() >
           m_p98.reindex(P.merchant_id).to_numpy())              # amount vs merchant p98
    r02 = (P.ip_country.to_numpy() != 'IN')                        # foreign IP
    r03 = (orders.address_match_flag.to_numpy() == 0)              # billing/shipping mismatch
    r04 = (tds_att & ~tds_ok)                                      # 3DS failed
    r05 = (P.account_age_days_at_txn.to_numpy() < 2.0)             # brand-new account
    r06 = (dev_n_cust >= 3)                                        # device shared across accounts
    card_age_h = (P.payment_timestamp - tfs).dt.total_seconds().to_numpy() / 3600
    r07 = np.nan_to_num(card_age_h, nan=1e9) < 24.0                # freshly tokenised card
    r08 = risky_login                                              # risky login within 24h
    ip_n_cust = pd.Series(P.ip_address).map(
        P.groupby('ip_address').customer_id.nunique()).to_numpy()
    r09 = ip_n_cust >= 5                                           # IP shared across accounts
    rules = np.stack([r01, r02, r03, r04, r05, r06, r07, r08, r09])
    W = np.array([1.45, 1.75, 1.40, 1.55, 1.25, 1.40, 1.50, 1.55, 1.50])
    z = -4.60 + (W[:, None] * rules).sum(0) + rng.normal(0, 0.15, N_PAY)
    score = 100 / (1 + np.exp(-z))
    P['payment_risk_score'] = np.round(score, 2)
    # solve the operating point for the precision target; recall is then whatever the
    # engine's (deliberately partial) rule coverage earns it
    order = np.argsort(-score)
    tp = np.cumsum(is_fraud[order])
    k = np.arange(1, N_PAY + 1)
    prec = tp / k
    cand = np.arange(1000, 80000)
    d = np.abs(prec[cand] - TARGET_PRECIS)
    n_alert = int(cand[np.where(d == d.min())[0][-1]])   # largest K at the target precision
    thr = score[order][n_alert - 1]
    alert = score >= thr
    P['alerted_flag'] = np.where(alert, 'Y', 'N')
    codes = np.array(['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'R09'])
    rule_codes = np.array(['|'.join(codes[rules[:, i]]) for i in range(N_PAY)], dtype=object)
    FRAUD_RULE_CODES['R05'] = 'NEW_ACCOUNT'; FRAUD_RULE_CODES['R06'] = 'DEVICE_SHARED'
    FRAUD_RULE_CODES['R07'] = 'CARD_TOKEN_FRESH'
    FRAUD_RULE_CODES['R08'] = 'RISKY_LOGIN_24H'
    FRAUD_RULE_CODES['R09'] = 'IP_SHARED_ACCOUNTS'

    # ========================================================================
    # 11. LABEL SOURCES
    # ========================================================================
    u = rng.random(N_PAY)
    conf_alert = np.where(is_fraud, u > REVIEW_FN, u < REVIEW_FP)
    ai = np.where(alert)[0]
    alert_lat = pd.to_timedelta(rng.uniform(2, 96, len(ai)), 'h')
    a_ts = P.payment_timestamp.to_numpy()[ai] + pd.to_timedelta(
        rng.uniform(1, 45, len(ai)), 'm').to_numpy()
    q_ts = a_ts + pd.to_timedelta(rng.uniform(5, 240, len(ai)), 'm').to_numpy()
    d_ts = a_ts + alert_lat.to_numpy()
    cf = conf_alert[ai]
    ub = rng.random(len(ai))
    blocked = np.where(cf, ub < .62, ub < .085)     # some legitimate payments get blocked
    ur = rng.random(len(ai))
    released = np.where(cf, ur < .07, ur < .74)     # some confirmed fraud is released in error
    released = released & ~blocked
    action = np.where(blocked, 'Blocked', np.where(released, 'Released', 'Manual Review'))
    loss = np.where(cf & ~blocked,
                    np.round(P.payment_amount.to_numpy()[ai] * rng.uniform(.55, 1.0, len(ai)), 2),
                    0.0)
    FE = pd.DataFrame({
        'fraud_event_id': ['F%06d' % (i + 1) for i in range(len(ai))],
        'payment_id': P.payment_id.to_numpy()[ai], 'order_id': P.order_id.to_numpy()[ai],
        'customer_id': P.customer_id.to_numpy()[ai],
        'alert_timestamp': ts_str(a_ts), 'queued_timestamp': ts_str(q_ts),
        'decision_timestamp': ts_str(d_ts),
        'queue_name': np.where(r02[ai] | r04[ai], 'CARD_RISK_Q',
                               np.where(r03[ai], 'ACCOUNT_RISK_Q', 'GENERAL_Q')),
        'reviewer_id': ['RV%03d' % x for x in rng.integers(1, 61, len(ai))],
        'alert_rule_codes': rule_codes[ai],
        'manual_review_flag': np.where(action == 'Manual Review', 'Y', 'N'),
        'action_status': action,
        'confirmed_fraud_flag': np.where(cf, 'Y', 'N'),
        'fraud_loss_amount': loss})

    nonalert = np.where(~alert)[0]
    aud = rng.choice(nonalert, N_AUDIT, replace=False)
    ua = rng.random(N_AUDIT)
    av = np.where(is_fraud[aud], ua > AUDIT_FN, ua < AUDIT_FP)
    s_ts = P.payment_timestamp.to_numpy()[aud] + pd.to_timedelta(
        rng.uniform(1, 6, N_AUDIT), 'D').to_numpy()
    AU = pd.DataFrame({
        'audit_id': ['AUD%06d' % (i + 1) for i in range(N_AUDIT)],
        'payment_id': P.payment_id.to_numpy()[aud],
        'sampled_timestamp': ts_str(s_ts),
        'review_completed_timestamp': ts_str(s_ts + pd.to_timedelta(
            rng.uniform(4, 8, N_AUDIT), 'D').to_numpy()),
        'auditor_id': ['AU%03d' % x for x in rng.integers(1, 25, N_AUDIT)],
        'audit_verdict': np.where(av, 'Fraud', 'Legitimate'),
        'audit_notes_code': np.where(av, rng.choice(['NC01', 'NC02', 'NC03'], N_AUDIT),
                                     rng.choice(['NC10', 'NC11'], N_AUDIT))})

    card_mech = np.isin(mech, ['Stolen Card', 'Account Takeover', 'Identity Mismatch'])
    missed = is_fraud & ~alert & card_mech & is_card
    rel_fraud = np.zeros(N_PAY, bool); rel_fraud[ai[cf & (action == 'Released')]] = True
    cb_true = (missed | (rel_fraud & is_card)) & (rng.random(N_PAY) < .72)
    friendly = (~is_fraud) & is_card & (rng.random(N_PAY) < .0105)
    cb_idx = np.where(cb_true | friendly)[0]
    raised = P.payment_timestamp.to_numpy()[cb_idx] + pd.to_timedelta(
        rng.uniform(15, 75, len(cb_idx)), 'D').to_numpy()
    keep = raised <= END.to_datetime64()
    cb_idx, raised = cb_idx[keep], raised[keep]
    res = raised + pd.to_timedelta(rng.uniform(30, 90, len(cb_idx)), 'D').to_numpy()
    res_s = np.where(res <= END.to_datetime64(), ts_str(res), '')
    CB = pd.DataFrame({
        'chargeback_id': ['CB%06d' % (i + 1) for i in range(len(cb_idx))],
        'payment_id': P.payment_id.to_numpy()[cb_idx],
        'card_token': P.card_token.to_numpy()[cb_idx],
        'raised_timestamp': ts_str(raised), 'resolution_timestamp': res_s,
        'reason_code': np.where(rng.random(len(cb_idx)) <
                                np.where(cb_true[cb_idx], .86, .19),
                                rng.choice(['10.4', '10.5', '4837'], len(cb_idx)),
                                rng.choice(['13.1', '13.3', '12.6'], len(cb_idx))),
        'disputed_amount': P.payment_amount.to_numpy()[cb_idx],
        'outcome': np.where(res <= END.to_datetime64(),
                            np.where(rng.random(len(cb_idx)) <
                                     np.where(cb_true[cb_idx], .78, .26),
                                     'Issuer Won', 'Merchant Won'), 'Pending')})

    # ========================================================================
    # 12. WRITE
    # ========================================================================
    P['payment_timestamp'] = ts_str(P.payment_timestamp)
    orders['order_timestamp'] = ts_str(orders.order_timestamp)
    L['login_timestamp'] = ts_str(L.login_timestamp)
    cust['signup_timestamp'] = ts_str(cust.signup_timestamp)
    cust['home_region_code'] = cust.home_region_code.astype(int)
    dev['first_seen_timestamp'] = ts_str(dev.first_seen_timestamp)
    cards['token_first_seen_timestamp'] = ts_str(cards.token_first_seen_timestamp)
    merch['onboarded_date'] = pd.to_datetime(merch.onboarded_date).dt.strftime('%Y-%m-%d')

    PAY_COLS = ['payment_id', 'order_id', 'customer_id', 'merchant_id', 'payment_timestamp',
                'payment_method', 'payment_gateway', 'card_token', 'payment_amount',
                'processing_fee', 'payment_status', 'failure_reason', 'device_id',
                'ip_address', 'ip_country', 'ip_asn', 'ip_region_code', 'session_id',
                'attempt_seq_in_session', 'is_guest_checkout', 'is_3ds_attempted',
                'is_3ds_success', 'account_age_days_at_txn', 'payment_risk_score',
                'alerted_flag']
    LOG_COLS = ['login_id', 'customer_id', 'login_timestamp', 'session_id', 'device_id',
                'ip_address', 'ip_country', 'ip_asn', 'login_channel', 'auth_method',
                'new_device_flag', 'unusual_location_flag', 'failed_attempt_count',
                'login_risk_score']
    used_d = set(P.device_id); keep_d = dev.device_id.isin(used_d) | (rng.random(len(dev)) < .035)
    dev = dev[keep_d].reset_index(drop=True)
    used_c = set(P.card_token[P.card_token != ''])
    cards = cards[cards.card_token.isin(used_c) |
                  (rng.random(len(cards)) < .03)].reset_index(drop=True)

    out = {'customers.csv': cust, 'merchants.csv': merch, 'cards.csv': cards,
           'devices.csv': dev, 'ip_reputation.csv': ipr, 'orders.csv': orders,
           'order_items.csv': items,
           'payments.csv': P[PAY_COLS], 'account_logins.csv': L[LOG_COLS],
           'fraud_events.csv': FE, 'audit_sample.csv': AU, 'chargebacks.csv': CB}
    if not clean:
        from messify import messify
        out, defects = messify(out, rng, END)
        defects.to_csv(os.path.join(outdir, 'KNOWN_DEFECTS.csv'), index=False)
    for name, df in out.items():
        df.to_csv(os.path.join(outdir, name), index=False)

    truth = pd.DataFrame({'payment_id': P.payment_id, 'is_fraud': is_fraud.astype(int),
                          'fraud_mechanism': mech, 'signature_strength': np.round(strength, 3)})
    if emit_truth:
        truth.to_csv(os.path.join(outdir, 'ground_truth.csv'), index=False)
    return out, truth, {'alert': alert, 'is_fraud': is_fraud, 'S': S, 'mech': mech,
                        'orders': orders, 'merch_tkt': merch_tkt, 'is_card': is_card}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./out')
    ap.add_argument('--emit-truth', action='store_true')
    ap.add_argument('--clean', action='store_true',
                    help='skip the raw-landing degradation stage (analysis-ready output)')
    a = ap.parse_args()
    o, t, _ = main(a.out, a.emit_truth, a.clean)
    for k, v in o.items():
        print(f'{k:24s} {len(v):>9,} rows  {v.shape[1]:>2} cols')
    print(f'true fraud: {int(t.is_fraud.sum()):,} ({100*t.is_fraud.mean():.3f}%)')
