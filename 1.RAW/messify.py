"""
messify.py — turns the clean generated frames into a realistic RAW LANDING extract.

Everything here is recoverable by a correct Phase-1 cleaning pipeline. Nothing destroys
signal that a competent notebook 02/03 cannot restore. The point is that notebooks 02
(cleaning), 03 (missing/outliers) and 06 (collinearity) must have real work to do.

Every injection below is catalogued in DEFECT_LOG so the dataset can be scored objectively.
"""
import numpy as np, pandas as pd

DEFECT_LOG = []


def _log(table, kind, col, n, note=''):
    DEFECT_LOG.append({'table': table, 'defect': kind, 'column': col, 'rows': int(n),
                       'note': note})


def _pad(s, rng, p):
    """random leading/trailing whitespace on a string column"""
    m = rng.random(len(s)) < p
    out = s.astype(object).copy()
    pads = rng.choice([' ', '  ', '\t', ' '], m.sum())
    out[m] = pd.Series(np.where(rng.random(m.sum()) < .5,
                                pads + s[m].astype(str),
                                s[m].astype(str) + pads), index=s.index[m])
    return out, int(m.sum())


def _spell(s, rng, variants, p):
    """inconsistent categorical spellings, as produced by several upstream systems"""
    out = s.astype(object).copy()
    total = 0
    for canon, alts in variants.items():
        idx = np.where(s.astype(str) == canon)[0]
        if not len(idx):
            continue
        hit = idx[rng.random(len(idx)) < p]
        if len(hit):
            out.iloc[hit] = rng.choice(alts, len(hit))
            total += len(hit)
    return out, total


def _nullify(df, col, rng, p, sentinels=None, table=''):
    """missing values, expressed the way real extracts express them: blanks AND sentinels"""
    n = len(df)
    m = rng.random(n) < p
    vals = df[col].astype(object).copy()
    if sentinels:
        pick = rng.choice(len(sentinels) + 1, m.sum())
        chosen = np.array([('' if i == len(sentinels) else sentinels[i]) for i in pick],
                          dtype=object)
        vals[m] = chosen
    else:
        vals[m] = np.nan
    df[col] = vals
    _log(table, 'missing/sentinel', col, m.sum(),
         'sentinels: ' + ','.join(map(repr, sentinels)) if sentinels else 'blank')
    return df


def messify(out, rng, END):
    pay = out['payments.csv']; od = out['orders.csv']; cu = out['customers.csv']
    lg = out['account_logins.csv']; cd = out['cards.csv']; dv = out['devices.csv']
    it = out['order_items.csv']; fe = out['fraud_events.csv']

    # ---------------------------------------------------------------- 1. derived and
    # collinear columns, so notebook 06 has genuine VIF / structural-exclusion work
    amt = pd.to_numeric(pay.payment_amount)
    pay['amount_usd'] = np.round(amt / 83.2, 2)                  # exact linear dependence
    pay['amount_paise'] = (amt * 100).round().astype('int64')    # exact linear dependence
    pay['fee_pct'] = np.round(pd.to_numeric(pay.processing_fee) / amt.replace(0, np.nan), 6)
    od['order_value_inr'] = od.order_value                       # exact duplicate, renamed
    od['net_payable'] = np.round(od.order_value - od.discount_amount + od.shipping_charge, 2)
    _log('payments.csv', 'collinear-by-construction', 'amount_usd/amount_paise/fee_pct',
         len(pay), 'amount_usd = payment_amount/83.2 exactly; amount_paise = *100')
    _log('orders.csv', 'collinear-by-construction', 'order_value_inr/net_payable', len(od),
         'order_value_inr duplicates order_value; net_payable = value-discount+shipping')

    # ---------------------------------------------------------------- 2. mixed timestamp
    # formats, including an ambiguous DD/MM/YYYY slice
    def mix_dates(s, rng, table, col):
        t = pd.to_datetime(s)
        u = rng.random(len(t))
        out_ = t.dt.strftime('%Y-%m-%dT%H:%M:%S').astype(object)
        m2 = u < .12
        out_[m2] = t[m2].dt.strftime('%Y-%m-%d %H:%M:%S')
        m3 = (u >= .12) & (u < .17)
        out_[m3] = t[m3].dt.strftime('%d/%m/%Y %H:%M')           # ambiguous with US format
        m4 = (u >= .17) & (u < .19)
        out_[m4] = t[m4].dt.strftime('%Y-%m-%dT%H:%M:%S') + '+05:30'
        _log(table, 'mixed datetime formats', col, (m2 | m3 | m4).sum(),
             'ISO / space-separated / DD-MM-YYYY / ISO+offset')
        return out_
    pay['payment_timestamp'] = mix_dates(pay.payment_timestamp, rng, 'payments.csv',
                                         'payment_timestamp')
    lg['login_timestamp'] = mix_dates(lg.login_timestamp, rng, 'account_logins.csv',
                                      'login_timestamp')

    # ---------------------------------------------------------------- 3. categorical
    # spelling drift across upstream systems
    pay['payment_method'], n = _spell(pay.payment_method, rng, {
        'Credit Card': ['credit card', 'CREDIT_CARD', 'Credit  Card', 'CC'],
        'Debit Card': ['debit card', 'DEBIT_CARD', 'DC'],
        'Cash on Delivery': ['COD', 'cash on delivery', 'Cash On Delivery'],
        'Net Banking': ['netbanking', 'NET_BANKING', 'Net banking'],
        'UPI': ['upi', 'Upi']}, .11)
    _log('payments.csv', 'inconsistent categories', 'payment_method', n)
    pay['payment_gateway'], n = _spell(pay.payment_gateway.fillna(''), rng, {
        'Razorpay': ['RazorPay', 'razorpay', 'RAZORPAY'],
        'PayU': ['Payu', 'PAYU', 'payu'],
        'Cashfree': ['CashFree', 'cashfree'],
        'BillDesk': ['Billdesk', 'BILLDESK']}, .13)
    _log('payments.csv', 'inconsistent categories', 'payment_gateway', n)
    od['shipping_speed'], n = _spell(od.shipping_speed, rng, {
        'Express': ['EXPRESS', 'express', 'Exp'],
        'Standard': ['STANDARD', 'standard', 'Std'],
        'Same Day': ['SAME_DAY', 'same day', 'SameDay']}, .12)
    _log('orders.csv', 'inconsistent categories', 'shipping_speed', n)
    cu['city'], n = _spell(cu.city, rng, {
        'Bengaluru': ['Bangalore', 'BENGALURU', 'bengaluru', 'Bengaluru '],
        'Mumbai': ['Bombay', 'MUMBAI', 'mumbai'],
        'Kolkata': ['Calcutta', 'KOLKATA'],
        'Chennai': ['Madras', 'CHENNAI'],
        'Hyderabad': ['HYDERABAD', 'hyderabad']}, .16)
    _log('customers.csv', 'inconsistent categories', 'city', n,
         'includes legacy city names (Bombay/Calcutta/Madras/Bangalore)')

    # ---------------------------------------------------------------- 4. whitespace
    for df, col, tbl in [(pay, 'ip_country', 'payments.csv'),
                         (cu, 'email_domain_class', 'customers.csv'),
                         (od, 'delivery_type', 'orders.csv')]:
        df[col], n = _pad(df[col].astype(str), rng, .025)
        _log(tbl, 'leading/trailing whitespace', col, n)

    # ---------------------------------------------------------------- 5. missing values
    # and sentinels. NOTE 'NA' in ip_country is genuinely ambiguous: it is also Namibia.
    pay = _nullify(pay, 'ip_country', rng, .021, ['NA ', 'unknown', 'XX'], 'payments.csv')
    pay = _nullify(pay, 'device_id', rng, .014, None, 'payments.csv')
    pay = _nullify(pay, 'ip_asn', rng, .018, ['UNKNOWN', 'AS0'], 'payments.csv')
    pay = _nullify(pay, 'ip_region_code', rng, .012, ['-1', '99'], 'payments.csv')
    pay = _nullify(pay, 'account_age_days_at_txn', rng, .009, ['-1', 'NaN '], 'payments.csv')
    pay = _nullify(pay, 'attempt_seq_in_session', rng, .006, ['', 'null'], 'payments.csv')
    cu = _nullify(cu, 'home_pincode', rng, .032, ['000000', 'NA'], 'customers.csv')
    cu = _nullify(cu, 'kyc_level', rng, .017, ['-1', '999'], 'customers.csv')
    cu = _nullify(cu, 'acquisition_channel', rng, .019, ['UNKNOWN', 'other'], 'customers.csv')
    cu = _nullify(cu, 'prior_return_rate', rng, .022, None, 'customers.csv')
    od = _nullify(od, 'shipping_addr_age_hours', rng, .046, ['NaN ', '-1'], 'orders.csv')
    od = _nullify(od, 'billing_pincode', rng, .028, [''], 'orders.csv')
    cd = _nullify(cd, 'issuer', rng, .029, ['UNKNOWN BANK', 'N/A '], 'cards.csv')
    cd = _nullify(cd, 'product_type', rng, .015, ['UNKNOWN'], 'cards.csv')
    lg = _nullify(lg, 'login_risk_score', rng, .031, None, 'account_logins.csv')
    dv = _nullify(dv, 'browser_family', rng, .024, ['Unknown', 'other'], 'devices.csv')

    # MAR, not MCAR: geo lookup fails more often behind hosting/VPN infrastructure
    host = pay.ip_asn.astype(str).str[2:].str.isdigit()
    hostmask = host & pd.to_numeric(pay.ip_asn.astype(str).str[2:], errors='coerce').ge(39000)
    extra = hostmask & (rng.random(len(pay)) < .09)
    pay.loc[extra, 'ip_country'] = ''
    _log('payments.csv', 'MAR missingness', 'ip_country', extra.sum(),
         'geo lookup fails more often on hosting/VPN ASNs — missingness is informative')

    # ---------------------------------------------------------------- 6. type damage
    pay['payment_amount'] = pay.payment_amount.astype(object)
    m = rng.random(len(pay)) < .006
    pay.loc[m, 'payment_amount'] = [f"{v:,.2f}" for v in
                                    pd.to_numeric(pay.loc[m, 'payment_amount'])]                                   # thousands separators
    _log('payments.csv', 'numeric stored as formatted string', 'payment_amount', m.sum(),
         'e.g. "1,234.50" — forces object dtype on the whole column')
    for c in ['is_3ds_attempted', 'is_3ds_success', 'is_guest_checkout']:
        v = pay[c].astype(object)
        u = rng.random(len(pay))
        v[u < .18] = np.where(pd.to_numeric(pay[c])[u < .18] == 1, 'Y', 'N')
        v[(u >= .18) & (u < .22)] = np.where(pd.to_numeric(pay[c])[(u >= .18) & (u < .22)] == 1,
                                             'TRUE', 'FALSE')
        pay[c] = v
        _log('payments.csv', 'mixed boolean encoding', c, int((u < .22).sum()),
             '1/0 mixed with Y/N and TRUE/FALSE')
    cd['bin'] = cd['bin'].astype(float)
    _log('cards.csv', 'integer stored as float', 'bin', len(cd), 'BIN reads as 412345.0')

    # ---------------------------------------------------------------- 7. impossible values
    # and genuine extremes (notebook 03 must tell these apart)
    n = len(pay)
    idx = rng.choice(n, 240, replace=False)
    pay.loc[pay.index[idx], 'payment_amount'] = list(np.round(-pd.to_numeric(
        pay.loc[pay.index[idx], 'payment_amount'], errors='coerce').abs().to_numpy(), 2))
    _log('payments.csv', 'impossible value', 'payment_amount', 240,
         'negative amounts — refund rows mis-ingested into the payment stream')
    idx = rng.choice(n, 150, replace=False)
    pay.loc[pay.index[idx], 'payment_amount'] = 0.0
    _log('payments.csv', 'impossible value', 'payment_amount', 150, 'zero-value payments')
    idx = rng.choice(n, 60, replace=False)
    pay.loc[pay.index[idx], 'payment_amount'] = list(np.round(rng.uniform(5e6, 2e7, 60), 2))
    pay.loc[pay.index[idx], 'customer_id'] = 'C000001'
    _log('payments.csv', 'test transactions', 'payment_amount', 60,
         'internal test rows: absurd amounts on a single sentinel customer')
    pay['account_age_days_at_txn'] = pay.account_age_days_at_txn.astype(object)
    idx = rng.choice(n, 420, replace=False)
    pay.loc[pay.index[idx], 'account_age_days_at_txn'] = list(rng.choice([36500, 99999], 420))
    _log('payments.csv', 'impossible value', 'account_age_days_at_txn', 420,
         '100-year-old accounts')
    idx = rng.choice(len(lg), 300, replace=False)
    lg.loc[lg.index[idx], 'failed_attempt_count'] = 255
    _log('account_logins.csv', 'impossible value', 'failed_attempt_count', 300,
         'uint8 overflow sentinel')
    idx = rng.choice(len(cu), 800, replace=False)
    cu.loc[cu.index[idx], 'prior_orders_12m'] = rng.integers(400, 900, 800)
    _log('customers.csv', 'outlier', 'prior_orders_12m', 800,
         'wholesale/reseller accounts — genuine, must NOT be capped away')

    # ---------------------------------------------------------------- 8. duplicates
    ex = pay.sample(frac=.004, random_state=7)
    pay = pd.concat([pay, ex], ignore_index=True)
    _log('payments.csv', 'exact duplicate rows', '*', len(ex),
         'double-submitted webhook deliveries — safe to drop')
    near = pay.sample(n=1400, random_state=11).copy()
    near['payment_id'] = ['PAYR%06d' % i for i in range(len(near))]
    near['payment_status'] = 'Failed'
    near['failure_reason'] = 'Gateway Timeout'
    pay = pd.concat([pay, near], ignore_index=True)
    _log('payments.csv', 'near-duplicate rows', 'payment_id', len(near),
         'genuine retries: same customer/amount seconds apart, distinct payment_id. '
         'Dropping these blindly destroys the retry signal')
    dup_it = it.sample(n=2600, random_state=13)
    it = pd.concat([it, dup_it], ignore_index=True)
    _log('order_items.csv', 'duplicate line items', '*', len(dup_it))
    dup_cu = cu.sample(n=600, random_state=17).copy()
    dup_cu['customer_id'] = dup_cu.customer_id + '_DUP'
    cu = pd.concat([cu, dup_cu], ignore_index=True)
    _log('customers.csv', 'duplicate identities', 'customer_id', len(dup_cu),
         'same person re-registered; suffixed id, otherwise identical attributes')

    # ---------------------------------------------------------------- 9. referential gaps
    orphan = rng.choice(len(pay), 900, replace=False)
    pay.loc[pay.index[orphan], 'device_id'] = ['DEVX%07d' % i for i in range(900)]
    _log('payments.csv', 'referential gap', 'device_id', 900,
         'device ids absent from devices.csv — fingerprint service outage window')
    drop_orders = rng.choice(od.order_id.to_numpy(), 380, replace=False)
    it = it[~it.order_id.isin(drop_orders)]
    _log('order_items.csv', 'referential gap', 'order_id', 380,
         'orders with zero line items')

    # ---------------------------------------------------------------- 10. row order
    pay = pay.sample(frac=1, random_state=23).reset_index(drop=True)
    _log('payments.csv', 'unsorted', '*', 0,
         'rows are not in timestamp or id order — a naive index split is not a temporal split')

    out['payments.csv'] = pay; out['orders.csv'] = od; out['customers.csv'] = cu
    out['account_logins.csv'] = lg; out['cards.csv'] = cd; out['devices.csv'] = dv
    out['order_items.csv'] = it; out['fraud_events.csv'] = fe
    return out, pd.DataFrame(DEFECT_LOG)
