"""Reference cleaning pipeline + quality gate, run against the RAW landing extract."""
import pandas as pd, numpy as np, sys, os, re
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
D = sys.argv[1] if len(sys.argv) > 1 else './raw'
R = lambda f: pd.read_csv(os.path.join(D, f), low_memory=False)
SENT = {'', 'NA', 'N/A', 'NAN', 'NAN ', 'NULL', 'NONE', 'UNKNOWN', 'XX', '-1', '999',
        '999999', '000000', 'AS0', 'OTHER', 'UNKNOWN BANK', 'NIL', '?'}


def sclean(s):
    o = s.astype('string').str.strip()
    return o.mask(o.str.upper().isin(SENT))


def parse_dt(s):
    x = s.astype('string').str.strip().str.replace(r'\+05:30$', '', regex=True)
    out = pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    m = out.isna()
    out[m] = pd.to_datetime(x[m], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    m = out.isna()
    out[m] = pd.to_datetime(x[m], format='%d/%m/%Y %H:%M', errors='coerce')   # DD/MM, not MM/DD
    m = out.isna()
    out[m] = pd.to_datetime(x[m], format='mixed', dayfirst=True, errors='coerce')
    return out


def num(s):
    return pd.to_numeric(sclean(s).str.replace(',', '', regex=False), errors='coerce')


def boolc(s):
    x = sclean(s).str.upper()
    return x.map({'1': 1, '0': 0, 'Y': 1, 'N': 0, 'TRUE': 1, 'FALSE': 0, '1.0': 1,
                  '0.0': 0}).astype('Float64')


CANON_METHOD = {'CREDIT CARD': 'Credit Card', 'CREDIT_CARD': 'Credit Card', 'CC': 'Credit Card',
                'DEBIT CARD': 'Debit Card', 'DEBIT_CARD': 'Debit Card', 'DC': 'Debit Card',
                'COD': 'Cash on Delivery', 'CASH ON DELIVERY': 'Cash on Delivery',
                'NETBANKING': 'Net Banking', 'NET_BANKING': 'Net Banking',
                'NET BANKING': 'Net Banking', 'UPI': 'UPI', 'WALLET': 'Wallet', 'EMI': 'EMI'}
CANON_CITY = {'BANGALORE': 'Bengaluru', 'BOMBAY': 'Mumbai', 'CALCUTTA': 'Kolkata',
              'MADRAS': 'Chennai'}


def canon(s, extra=None):
    x = sclean(s).str.replace(r'\s+', ' ', regex=True).str.strip()
    u = x.str.upper()
    if extra:
        x = pd.Series(np.where(u.isin(extra.keys()), u.map(extra), x), index=x.index,
                      dtype='string')
        u = x.astype('string').str.upper()
    return x.str.title().mask(x.isna())


print("=" * 82); print("RAW LANDING — WHAT THE CLEANER HAD TO DEAL WITH"); print("=" * 82)
kd = R('KNOWN_DEFECTS.csv')
print(kd.groupby('defect').rows.agg(['count', 'sum']).rename(
    columns={'count': 'injections', 'sum': 'rows'}).to_string())

pay = R('payments.csv'); od = R('orders.csv'); cu = R('customers.csv'); cd = R('cards.csv')
dv = R('devices.csv'); mr = R('merchants.csv'); it = R('order_items.csv')
lg = R('account_logins.csv'); ipr = R('ip_reputation.csv')
fe = R('fraud_events.csv'); au = R('audit_sample.csv'); cb = R('chargebacks.csv')
raw_rows = len(pay)

# ---------------------------------------------------------------- clean payments
pay['payment_timestamp'] = parse_dt(pay.payment_timestamp)
pay['payment_amount'] = num(pay.payment_amount)
pay['processing_fee'] = num(pay.processing_fee)
pay['account_age_days_at_txn'] = num(pay.account_age_days_at_txn)
pay['attempt_seq_in_session'] = num(pay.attempt_seq_in_session)
for c in ['is_3ds_attempted', 'is_3ds_success', 'is_guest_checkout']:
    pay[c] = boolc(pay[c])
pay['payment_method'] = canon(pay.payment_method, CANON_METHOD)
pay['payment_gateway'] = canon(pay.payment_gateway)
for c in ['ip_country', 'ip_asn', 'device_id', 'card_token', 'merchant_id', 'customer_id']:
    pay[c] = sclean(pay[c])
n0 = len(pay); pay = pay.drop_duplicates(); n_exact = n0 - len(pay)
n1 = len(pay); pay = pay.drop_duplicates('payment_id', keep='first')
n_pk = n1 - len(pay)   # same id, non-identical row: keep first, flag for review
retry = pay.payment_id.astype(str).str.startswith('PAYR')
n_retry = int(retry.sum())
pay['is_retry'] = retry.astype(int)                     # keep: retries are signal, not noise
bad_amt = pay.payment_amount.le(0) | pay.payment_amount.isna()
test_rows = pay.payment_amount.gt(3e6)
pay.loc[pay.account_age_days_at_txn.ge(20000) | pay.account_age_days_at_txn.lt(0),
        'account_age_days_at_txn'] = np.nan
pay = pay[~bad_amt & ~test_rows].copy()
print(f"\ncleaner: {raw_rows:,} raw -> {len(pay):,} rows "
      f"(-{n_exact:,} exact dupes, -{n_pk:,} duplicate payment_id, -{int(bad_amt.sum()):,} non-positive amounts, "
      f"-{int(test_rows.sum()):,} test rows; {n_retry:,} retries KEPT and flagged)")
print(f"unparsed timestamps: {pay.payment_timestamp.isna().sum():,}  "
      f"| payment_method levels after canonicalisation: {pay.payment_method.nunique()}  "
      f"| gateway levels: {pay.payment_gateway.nunique()}")

od['shipping_speed'] = canon(od.shipping_speed)
od['shipping_addr_age_hours'] = num(od.shipping_addr_age_hours)
od['billing_pincode'] = sclean(od.billing_pincode)
cu['city'] = canon(cu.city, CANON_CITY)
cu['kyc_level'] = num(cu.kyc_level).where(lambda x: x.between(0, 2))
cu['home_pincode'] = sclean(cu.home_pincode)
cu['prior_return_rate'] = num(cu.prior_return_rate)
cu['email_domain_class'] = canon(cu.email_domain_class)
cu['signup_timestamp'] = parse_dt(cu.signup_timestamp)
cu = cu[~cu.customer_id.astype(str).str.endswith('_DUP')]
cd['issuer'] = sclean(cd.issuer); cd['bin'] = num(cd.bin).astype('Int64')
cd['token_first_seen_timestamp'] = parse_dt(cd.token_first_seen_timestamp)
dv['first_seen_timestamp'] = parse_dt(dv.first_seen_timestamp)
dv['browser_family'] = sclean(dv.browser_family)
lg['login_timestamp'] = parse_dt(lg.login_timestamp)
lg['login_risk_score'] = num(lg.login_risk_score)
lg.loc[lg.failed_attempt_count.ge(100), 'failed_attempt_count'] = np.nan
it = it.drop_duplicates()
print(f"customers {len(cu):,} (dupes removed) | cards {len(cd):,} | devices {len(dv):,} "
      f"| items {len(it):,} after dedup")
print(f"city levels after canonicalisation: {cu.city.nunique()}")

# ---------------------------------------------------------------- features
X = pay.merge(od.drop(columns='customer_id'), on='order_id', how='left') \
       .merge(cu, on='customer_id', how='left') \
       .merge(mr.rename(columns={'category': 'mcat'}).drop(columns=['city', 'merchant_name']),
              on='merchant_id', how='left') \
       .merge(cd, on='card_token', how='left').merge(dv, on='device_id', how='left') \
       .merge(ipr, on='ip_asn', how='left')
b = it.groupby('order_id').agg(n_lines=('order_item_id', 'size'), n_cat=('category', 'nunique'),
                               max_unit=('unit_price', 'max'), tot_qty=('quantity', 'sum'))
X = X.merge(b, on='order_id', how='left')
X['card_age_h'] = (X.payment_timestamp - X.token_first_seen_timestamp).dt.total_seconds() / 3600
X['dev_age_h'] = (X.payment_timestamp - X.first_seen_timestamp).dt.total_seconds() / 3600
X['amt_vs_merch'] = X.payment_amount / X.avg_ticket_size
X['ipc_mismatch'] = (X.ip_country.fillna('IN') != 'IN').astype(int)
X['ip_country_missing'] = X.ip_country.isna().astype(int)      # MAR: missingness is signal
_hr = pd.to_numeric(X.home_pincode, errors='coerce').floordiv(10000)
X['ipr_mismatch'] = np.where(_hr.isna(), np.nan,
                             (X.ip_region_code.astype('float64') != _hr).astype(float))
X['iss_foreign'] = (X.issuing_country.fillna('IN') != 'IN').astype(int)
X['bill_ship_mismatch'] = 1 - X.address_match_flag
X['has_coupon'] = X.coupon_code.notna().astype(int)
X['disp_email'] = X.email_domain_class.eq('Disposable').astype(int)
X = X.merge(pay.groupby('device_id').customer_id.nunique().rename('dev_n'), on='device_id',
            how='left')
X = X.merge(pay.groupby('ip_address').customer_id.nunique().rename('ip_n'), on='ip_address',
            how='left')
_l = lg[['customer_id', 'login_timestamp', 'new_device_flag', 'unusual_location_flag',
         'failed_attempt_count', 'login_risk_score']].dropna(subset=['login_timestamp']).copy()
_l['customer_id'] = _l.customer_id.astype(str)
_l = _l.sort_values('login_timestamp')
_p = X[['payment_id', 'customer_id', 'payment_timestamp']].dropna(
    subset=['payment_timestamp']).copy()
_p['customer_id'] = _p.customer_id.astype(str)
_p = _p.sort_values('payment_timestamp')
a = pd.merge_asof(_p, _l, left_on='payment_timestamp', right_on='login_timestamp',
                  by='customer_id', allow_exact_matches=False)
a['h_since_login'] = (a.payment_timestamp - a.login_timestamp).dt.total_seconds() / 3600
X = X.merge(a[['payment_id', 'new_device_flag', 'unusual_location_flag', 'failed_attempt_count',
               'login_risk_score', 'h_since_login']], on='payment_id', how='left')
for c in ['payment_method', 'shipping_speed', 'mcat', 'network', 'product_type', 'device_type',
          'browser_family', 'email_domain_class', 'asn_type', 'acquisition_channel',
          'payment_gateway', 'city']:
    X[c] = X[c].astype(str).astype('category')
F = ['payment_amount', 'processing_fee', 'item_count', 'order_value', 'discount_amount',
     'shipping_charge', 'shipping_addr_age_hours', 'address_match_flag',
     'account_age_days_at_txn', 'attempt_seq_in_session', 'is_guest_checkout',
     'is_3ds_attempted', 'is_3ds_success', 'kyc_level', 'prior_return_rate', 'prior_orders_12m',
     'city_tier', 'avg_ticket_size', 'trailing_chargeback_rate_bps', 'bin', 'is_prepaid',
     'is_emulator', 'n_lines', 'n_cat', 'max_unit', 'tot_qty', 'card_age_h', 'dev_age_h',
     'amt_vs_merch', 'ipc_mismatch', 'ip_country_missing', 'ipr_mismatch', 'iss_foreign',
     'bill_ship_mismatch', 'has_coupon', 'disp_email', 'dev_n', 'ip_n', 'is_hosting',
     'reputation_score', 'new_device_flag', 'unusual_location_flag', 'failed_attempt_count',
     'login_risk_score', 'h_since_login', 'is_retry', 'payment_method', 'shipping_speed',
     'mcat', 'network', 'product_type', 'device_type', 'browser_family', 'email_domain_class',
     'asn_type', 'acquisition_channel']
for c in F:
    if str(X[c].dtype) not in ('category',):
        X[c] = pd.to_numeric(X[c], errors='coerce').astype('float64')

truth = pd.read_csv(os.path.join(D, 'ground_truth.csv'))
y = truth.set_index('payment_id').is_fraud.reindex(X.payment_id)
keep = y.notna().to_numpy() & X.payment_timestamp.notna().to_numpy()
X = X[keep].reset_index(drop=True); y = y[keep].to_numpy().astype(int)

print("\n" + "=" * 82); print("GATE"); print("=" * 82)
print(f"modelling frame: {len(X):,} rows, {len(F)} features, {y.sum():,} fraud "
      f"({100*y.mean():.3f}%)")
uni = {}
for c in F:
    s = X[c].cat.codes if str(X[c].dtype) == 'category' else X[c]
    s = pd.to_numeric(s, errors='coerce').fillna(-999)
    v = roc_auc_score(y, s); uni[c] = max(v, 1 - v)
top = pd.Series(uni).sort_values(ascending=False)
print(f"max univariate AUC = {top.max():.4f} ({top.index[0]})  "
      f"-> {'PASS' if top.max() < .80 else 'FAIL'}")
al = (pay.set_index('payment_id').alerted_flag.reindex(X.payment_id) == 'Y').to_numpy()
print(f"incumbent: alerts={al.sum():,} recall={(al&(y==1)).sum()/y.sum():.3f} "
      f"precision={(al&(y==1)).sum()/max(al.sum(),1):.3f}")
cut = pd.Timestamp('2025-07-01')
tr = (X.payment_timestamp < cut).to_numpy(); te = ~tr
m = lgb.LGBMClassifier(n_estimators=600, learning_rate=.05, num_leaves=48,
                       min_child_samples=60, subsample=.9, colsample_bytree=.8,
                       reg_lambda=1.0, random_state=42, n_jobs=4, verbose=-1)
m.fit(X.loc[tr, F], y[tr]); p = m.predict_proba(X.loc[te, F])[:, 1]
roc = roc_auc_score(y[te], p); pr = average_precision_score(y[te], p)
print(f"model (ground-truth label): ROC={roc:.4f} {'PASS' if .88<=roc<=.93 else 'OUT OF BAND'}"
      f" | PR={pr:.4f} {'PASS' if .30<=pr<=.50 else 'OUT OF BAND'}")
print("top features:", ', '.join(pd.Series(m.feature_importances_, index=F)
                                 .sort_values(ascending=False).head(8).index))

print("\n" + "=" * 82); print("FEASIBILITY — training on OBSERVABLE labels only"); print("=" * 82)
conf = set(fe.loc[fe.confirmed_fraud_flag == 'Y', 'payment_id'])
aud_f = set(au.loc[au.audit_verdict == 'Fraud', 'payment_id'])
cb_f = set(cb.loc[cb.outcome == 'Issuer Won', 'payment_id'])
pid = X.payment_id
k = int(al.sum() * te.sum() / len(X))
for nm, pos in [('alerts only', conf), ('alerts + audit', conf | aud_f),
                ('alerts + audit + chargebacks', conf | aud_f | cb_f)]:
    yl = pid.isin(pos).astype(int).to_numpy()
    mm = lgb.LGBMClassifier(n_estimators=500, learning_rate=.05, num_leaves=48,
                            min_child_samples=40, random_state=42, n_jobs=4, verbose=-1)
    mm.fit(X.loc[tr, F], yl[tr]); pp = mm.predict_proba(X.loc[te, F])[:, 1]
    topk = np.zeros(int(te.sum()), bool); topk[np.argsort(-pp)[:k]] = True
    print(f"  {nm:30s} train_pos={int(yl[tr].sum()):>6,}  ROC={roc_auc_score(y[te],pp):.4f} "
          f"PR={average_precision_score(y[te],pp):.4f}  recall@{k:,}={(topk&(y[te]==1)).sum()/y[te].sum():.3f}")
te_al = al[te]
print(f"  {'INCUMBENT rules engine':30s} {'':>13s}  ROC="
      f"{roc_auc_score(y[te], pd.to_numeric(pay.set_index('payment_id').payment_risk_score.reindex(X.payment_id))[te]):.4f} "
      f"PR={average_precision_score(y[te], pd.to_numeric(pay.set_index('payment_id').payment_risk_score.reindex(X.payment_id))[te]):.4f}"
      f"  recall@{int(te_al.sum()):,}={(te_al&(y[te]==1)).sum()/y[te].sum():.3f}")
