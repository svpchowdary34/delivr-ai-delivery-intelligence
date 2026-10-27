# DELIVR AI — Delivery Performance & Operations Intelligence | ONE Google Colab cell
# Find the delay. Fix the process.
import subprocess, sys, os, re, zipfile, html, math, warnings
warnings.filterwarnings("ignore")
try:
    import gradio, plotly, sklearn
    assert int(gradio.__version__.split(".")[0]) in (4, 5)
except Exception:
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scikit-learn>=1.2", "pandas", "numpy"])
import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix

C = dict(fg="#193B37", sg="#BCE8B0", am="#E8B96A", iv="#F5F2E9", ch="#263B37", li="#D6C8F0", co="#D98276", gg="#A5B5AD")
G = {}; TESTS = []; _DEMO = {}
S0 = dict(t_hold=2.0, t_proc=4.0, t_wait=4.0, exp_f=0.6, hmin=36.0, thr_med=.15, thr_hi=.30, min_n=30)
BLANK = ["", "nan", "none", "n/a", "na", "-", "null", "nat"]
esc = lambda x: html.escape(str(x)); nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
nf = lambda v: "—" if nul(v) else format(v, ",.0f"); fx = lambda v, d=1: "—" if nul(v) else format(v, "." + str(d) + "f"); pc = lambda v, d=0: "—" if nul(v) else format(v, "." + str(d) + "%")
dts = lambda v: "—" if nul(v) else pd.Timestamp(v).strftime("%d %b %Y %H:%M"); money = lambda v: "—" if nul(v) else "₹" + format(v, ",.0f")
hrs = lambda a, b: (a - b).dt.total_seconds() / 3600
RK = {"High": ("▲", C["co"]), "Medium": ("●", C["am"]), "Low": ("✓", C["sg"])}
def pill(s, col=None): g, c = RK.get(s, ("", C["gg"])); return "<span class='pill' style='background:%s'>%s %s</span>" % (col or c, g, esc(s))
def note(t, k="info"): return "<div class='note' style='border-left:4px solid %s'>%s</div>" % ({"info": C["gg"], "ok": C["fg"], "warn": C["am"], "err": C["co"], "ai": C["li"]}[k], t)
def card(l, v, d, c=C["fg"]): return "<div class='kc'><div class='kl'>%s</div><div class='kv' style='color:%s'>%s</div><div class='kd'>%s</div></div>" % (l, c, v, d)
SEC = lambda t: "<div class='sec'>%s</div>" % t
def tbl(d, cols, fmt=None):
    fmt = fmt or {}; h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows(): h += "<tr>" + "".join("<td>%s</td>" % (fmt[k](r[k]) if k in fmt else ("—" if nul(r[k]) else esc(r[k]))) for k, _n in cols) + "</tr>"
    return h + "</table></div>"
def sty(f, h=340, t=""):
    f.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["ch"]), height=h, title=dict(text=t, x=.02), margin=dict(l=55, r=25, t=50, b=45), legend=dict(orientation="h", y=-.25), colorway=[C["fg"], C["am"], C["li"], C["co"], C["gg"]]); return f
def banner(): return note("<b>Illustrative synthetic data.</b> Fictional orders, warehouses and carriers with planted patterns; not a real company's performance.", "warn") if G.get("demo") else ""
def safe(n):
    def deco(fn):
        def w(*a):
            if "X" not in G: return note("Load data first.") if n == 1 else (note("Load data first."),) + (None,) * (n - 1)
            try: return fn(*a)
            except Exception as e: m = note("Could not render this page: " + esc(e), "err"); return m if n == 1 else (m,) + (None,) * (n - 1)
        return w
    return deco
def clean_str(s): t = s.astype(str).str.strip(); return t.where(~(s.isna() | t.str.lower().isin(BLANK)), np.nan)
def to_num(s): return s.astype(float) if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s.astype(str).str.replace(r"[₹$,%\s]", "", regex=True), errors="coerce")
def to_dt(s):
    if pd.api.types.is_datetime64_any_dtype(s): return s
    try: return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception: return pd.to_datetime(s, errors="coerce")
def read_csv_any(f):
    p = f if isinstance(f, str) else getattr(f, "name", f)
    try: d = pd.read_csv(p)
    except Exception: d = pd.read_csv(p, sep=None, engine="python", encoding_errors="replace")
    if d.empty: raise ValueError("A file contains no rows.")
    return d

# ------------------------------ FICTIONAL synthetic operations data (fixed seed) ------------------------------
START = pd.Timestamp("2026-07-28"); END = pd.Timestamp("2026-10-26 18:00"); ND = 91
WH = ["WH1", "WH2", "WH3", "WH4"]; WF = dict(WH1=1.0, WH2=1.05, WH3=.95, WH4=1.0); ZONES = {"Metro": 24, "Tier-2": 48, "Rural": 72, "North-East": 96, "Hill": 84}; CARS = ["CAR1", "CAR2", "CAR3", "CAR4"]
CATS = {"Electronics": 10, "Fashion": 15, "Home": 12, "Grocery": 10, "Beauty": 8, "Books": 5}; CUT = dict(WH1=[10, 14, 18, 22], WH2=[11, 15, 19], WH3=[12, 18], WH4=[9, 13, 17, 21])
STAFF = dict(WH1=(3, 3, 1), WH2=(3, 3, 1), WH3=(3, 2, 1), WH4=(3, 3, 1)); FEST = set(range(40, 46)); SH = ["Morning", "Evening", "Night"]
HW = np.array([.3, .2, .15, .15, .2, .4, 1, 2, 3.5, 5, 6.5, 7.5, 7, 6, 5.5, 5, 5.5, 6.5, 7.5, 8, 6.5, 4.5, 2.5, 1.0]); HW = HW / HW.sum()
rid = lambda w, z: "%s-%s" % (w, z.replace("-", "")[:3].upper())
def make_data(seed=5):
    r = np.random.default_rng(seed); days = pd.date_range(START, periods=ND); T = ND * 24 + 240; skus = []; scat = {}
    for c, n in CATS.items():
        for _ in range(n): s = "SKU%03d" % (len(skus) + 1); skus.append(s); scat[s] = c
    routes = pd.DataFrame([dict(route_id=rid(w, z), warehouse_id=w, zone=z, planned_transit_h=round(pt * WF[w], 1), distance_km=int(pt * 8.5 * WF[w])) for w in WH for z, pt in ZONES.items()]); parts = []
    for w in WH:
        cnt = np.array([r.poisson(26 * (1.8 if i in FEST else 1.0) * (1.15 if d.dayofweek >= 5 else 1.0)) for i, d in enumerate(days)]); n = int(cnt.sum())
        ts = days[np.repeat(np.arange(ND), cnt)] + pd.to_timedelta(r.choice(24, n, p=HW) * 60 + r.uniform(0, 60, n), unit="m")
        zone = r.choice(list(ZONES), n, p=[.4, .3, .12, .08, .1]); car = r.choice(CARS, n, p=[.3, .3, .2, .2]); mode = np.where(r.random(n) < .2, "Express", "Standard"); sku = r.choice(skus, n); cat = np.array([scat[s] for s in sku])
        held = r.random(n) < np.where((w == "WH2") & (cat == "Electronics"), .28, .03); hold = .1 + r.exponential(.15, n) + held * r.uniform(6, 30, n); rel = ts + pd.to_timedelta(hold, unit="h")
        hi_ = np.minimum(np.asarray((rel - START).total_seconds() / 3600).astype(int), T - 1); arr = np.bincount(hi_, minlength=T).astype(float); hod = np.arange(T) % 24; st = STAFF[w]
        cap = np.select([(hod >= 6) & (hod < 14), (hod >= 14) & (hod < 22)], [st[0], st[1]], st[2]) * .85 / (S0["hmin"] / 60); bl = np.zeros(T); b = 0.0
        for t in range(T): bl[t] = b; b = max(0.0, b + arr[t] - cap[t])
        qw = np.minimum(bl[hi_] / cap[hi_] + .3, 60); exc = r.random(n) < (.06 if w == "WH2" else .04); svc = .6 * r.lognormal(0, .3, n) + exc * r.uniform(2, 5, n)
        ps = rel + pd.to_timedelta(qw, unit="h"); pe = ps + pd.to_timedelta(svc, unit="h"); ct = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in pd.date_range(START - pd.Timedelta(days=1), periods=ND + 8) for h in CUT[w]]).values.astype("datetime64[ns]").astype("int64")
        ix = np.minimum(np.searchsorted(ct, pe.values.astype("datetime64[ns]").astype("int64")), len(ct) - 1); disp = pd.to_datetime(ct[ix])
        pt_ = {z: ZONES[z] * WF[w] for z in ZONES}; pl = np.array([pt_[z] for z in zone]); m = np.where(mode == "Express", .6, 1.0); ne = zone == "North-East"; c3 = car == "CAR3"
        mult = np.where(c3 & ne, 1.5, np.where(c3, 1.05, np.where((car == "CAR4") & (zone == "Rural"), 1.2, 1.0))); sg = np.where(c3 & ne, .5, .18); split = r.random(n) < np.where(held, .15, .05)
        pf = .05 + .1 * np.isin(zone, ["Rural", "Hill"]) + .2 * (c3 & ne); fails = (r.random(n) < pf).astype(int) + (r.random(n) < pf / 3).astype(int)
        trans = pl * m * mult * r.lognormal(-sg ** 2 / 2, sg) + 24 * fails + 12 * split; deliv = disp + pd.to_timedelta(trans, unit="h"); prom = ts + pd.to_timedelta(pl * m + np.where(mode == "Express", 18, 22), unit="h"); rh = np.asarray(rel.hour); canc = r.random(n) < .02
        parts.append(pd.DataFrame(dict(order_ts=ts, warehouse_id=w, zone=zone, route_id=[rid(w, z) for z in zone], carrier_id=car, shipping_mode=mode, sku=sku, category=cat, quantity=r.integers(1, 4, n), priority_customer=r.random(n) < .08, customer_id=["C%04d" % i for i in r.integers(1, 3500, n)],
                                    promised_ts=prom, release_ts=rel, pick_start=ps, pack_end=pe, dispatch_ts=disp, delivery_ts=deliv, stock_hold=held, pick_exception=exc, split_shipment=split, failed_attempts=fails.astype(float), cancelled=canc, returned=(~canc) & (r.random(n) < .02), shift=np.select([(rh >= 6) & (rh < 14), (rh >= 14) & (rh < 22)], ["Morning", "Evening"], "Night"))))
    O = pd.concat(parts).sort_values("order_ts").reset_index(drop=True); O.insert(0, "order_id", ["ORD%05d" % (i + 1) for i in range(len(O))])
    for c in ["release_ts", "pick_start", "pack_end", "dispatch_ts", "delivery_ts"]: O.loc[O[c] > END, c] = pd.NaT
    O = O[O.order_ts <= END].reset_index(drop=True); O.loc[O.cancelled, ["pick_start", "pack_end", "dispatch_ts", "delivery_ts"]] = pd.NaT; O.loc[O.delivery_ts.isna(), "failed_attempts"] = np.nan
    O["status"] = np.select([O.cancelled.values, (O.returned & O.delivery_ts.notna()).values, O.delivery_ts.notna().values, O.dispatch_ts.notna().values, O.pack_end.notna().values, O.release_ts.notna().values], ["Cancelled", "Returned", "Delivered", "In transit", "Awaiting dispatch", "Processing"], "On hold (stock)")
    D = O[O.delivery_ts.notna()]; lh = hrs(D.delivery_ts, D.promised_ts).values; p = np.select([lh <= 0, lh < 12, lh < 24], [.01, .08, .2], .4); k = r.random(len(D)) < p; kk = int(k.sum())
    comp = pd.DataFrame(dict(complaint_id=["CMP%05d" % (i + 1) for i in range(kk)], order_id=D.order_id.values[k], complaint_ts=pd.to_datetime(D.delivery_ts.values[k]) + pd.to_timedelta(r.uniform(1, 48, kk), unit="h"), category=np.where(lh[k] > 0, "Late delivery", "Product issue")))
    inv = pd.DataFrame([dict(warehouse_id=w, sku=s, available_units=int(r.integers(0, 4) if (w == "WH2" and scat[s] == "Electronics") else r.integers(10, 90))) for w in WH for s in skus])
    stf = pd.DataFrame([dict(warehouse_id=w, date=d, shift=sh, staff_scheduled=STAFF[w][j], productive_hours_per_staff=6.8) for w in WH for d in days for j, sh in enumerate(SH)])
    oc = ["order_id", "customer_id", "order_ts", "warehouse_id", "zone", "route_id", "carrier_id", "shipping_mode", "category", "sku", "quantity", "priority_customer", "promised_ts", "delivery_ts", "status", "failed_attempts", "shift"]
    return dict(orders=O[oc], events=O[["order_id", "release_ts", "pick_start", "pack_end", "dispatch_ts", "stock_hold", "pick_exception", "split_shipment"]], inv=inv, routes=routes, staff=stf, comp=comp)
def get_demo_raw(seed=5):
    if ("raw", seed) not in _DEMO: _DEMO[("raw", seed)] = make_data(seed)
    return _DEMO[("raw", seed)]

# ------------------------------ ingestion and validation ------------------------------
def _sp(s):
    o = {}
    for it in s.split(","):
        p = it.strip().split(":"); o[p[0]] = (p[1], p[2] == "1", p[3].split("|") if len(p) > 3 and p[3] else [])
    return o
SPEC = dict(orders=_sp("order_id:s:1:id|order,customer_id:s:0:customer,order_ts:d:1:order_time|created_at|order_date,warehouse_id:s:1:warehouse|fulfilment_center,zone:s:0:region,route_id:s:0:route,carrier_id:s:0:carrier|courier,shipping_mode:s:0:mode,category:s:0:product_category,sku:s:0,quantity:n:0:qty,priority_customer:s:0:priority,promised_ts:d:1:promised_date|promised_delivery,delivery_ts:d:0:delivered_at|delivery_date|actual_delivery,status:s:0:order_status,failed_attempts:n:0,shift:s:0"),
 events=_sp("order_id:s:1,release_ts:d:0:released_at,pick_start:d:0,pack_end:d:0:packed_at,dispatch_ts:d:0:dispatched_at,stock_hold:s:0,pick_exception:s:0,split_shipment:s:0"), inv=_sp("warehouse_id:s:1:warehouse,sku:s:1,available_units:n:1:available|stock"),
 routes=_sp("route_id:s:1:route,warehouse_id:s:0,zone:s:0,planned_transit_h:n:1:planned_transit,distance_km:n:0"), staff=_sp("warehouse_id:s:1:warehouse,date:d:1,shift:s:1,staff_scheduled:n:1:staff,productive_hours_per_staff:n:0"), comp=_sp("complaint_id:s:0,order_id:s:1,complaint_ts:d:0,category:s:0"))
TN = dict(orders="orders", events="warehouse_events", inv="inventory", routes="routes", staff="staffing", comp="complaints"); REQ = ("orders", "events")
KEY = dict(orders=["order_id"], events=["order_id"], inv=["warehouse_id", "sku"], routes=["route_id"], staff=["warehouse_id", "date", "shift"], comp=["complaint_id"])
nm_ = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
def parse_map(text):
    ov = {}
    for p in re.split(r"[,;\n]", text or ""):
        if "=" in p: k, v = p.split("=", 1); ov[nm_(k)] = nm_(v)
    return ov
def map_cols(raw, t, ov, iss):
    d = raw.copy(); d.columns = [nm_(c) for c in d.columns]; ren = {}
    for f, (k, req, al) in SPEC[t].items():
        c = ov.get(t + "_" + f) or ov.get(f); c = c if c in d.columns else next((a for a in [f] + [nm_(x) for x in al] if a in d.columns and a not in ren), None)
        if c is None or c in ren:
            if req: iss.append(("error", "%s: required field '%s' not found. Columns found: %s. Use the mapping box, e.g. %s=your_column." % (TN[t], f, ", ".join(map(str, raw.columns[:12])), f)))
            continue
        if c != f and f in d.columns: d = d.drop(columns=[f])
        ren[c] = f
    d = d.rename(columns=ren); return d.loc[:, ~d.columns.duplicated()]
def ctab(raw, t, ov, iss, ex):
    d = map_cols(raw, t, ov, iss)
    if any(l == "error" for l, _ in iss): return d
    d = d.copy(); d["_row"] = d.index + 2; bad = pd.Series(False, index=d.index)
    for f, (k, req, al) in SPEC[t].items():
        if f not in d.columns: d[f] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]") if k == "d" else np.nan
        elif k == "n": d[f] = to_num(d[f])
        elif k == "d": d[f] = to_dt(d[f])
        else: d[f] = clean_str(d[f])
    def flag(m, why):
        nonlocal bad
        m = m.fillna(False) & ~bad
        if m.any(): ex.append((TN[t], why, int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: %s (rows %s)" % (TN[t], m.sum(), why, ", ".join(str(int(i)) for i in d["_row"][m].values[:6])))); bad = bad | m
    for f, (k, req, al) in SPEC[t].items():
        if req: flag(d[f].isna(), "missing or invalid " + f)
        if k == "n": flag(d[f] < 0, "negative " + f)
    flag(d.duplicated(KEY[t], keep="first") & d[KEY[t]].notna().all(axis=1), "duplicate " + "+".join(KEY[t])); return d[~bad].drop(columns="_row").reset_index(drop=True)
def load_all(raw, mapping=""):
    iss, ex, ov, D = [], [], parse_map(mapping), {}; rows_in = {t: len(raw[t]) for t in SPEC if raw.get(t) is not None}
    for t in REQ:
        if raw.get(t) is None: iss.append(("error", "Missing required table: %s." % TN[t]))
    if any(l == "error" for l, _ in iss): raise ValueError(" | ".join(m for l, m in iss if l == "error"))
    for t in rows_in: D[t] = ctab(raw[t], t, ov, iss, ex)
    errs = [m for l, m in iss if l == "error"]
    if errs: raise ValueError(" | ".join(errs))
    for t in SPEC:
        if t not in D: D[t] = pd.DataFrame({f: pd.Series(dtype="datetime64[ns]" if k == "d" else float if k == "n" else object) for f, (k, _, _) in SPEC[t].items()}); iss.append(("info", "No %s table: analyses that need it are downgraded or unavailable." % TN[t]))
    if D["orders"].empty: raise ValueError("No valid orders remain after validation.")
    for t in ("events", "comp"):
        m = ~D[t].order_id.isin(D["orders"].order_id)
        if m.any(): ex.append((TN[t], "order_id not in orders", int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: order_id not found in orders." % (TN[t], m.sum()))); D[t] = D[t][~m].reset_index(drop=True)
    O = D["orders"].merge(D["events"], on="order_id", how="left"); bad = pd.Series(False, index=O.index)
    for why, m in [("delivery before order time", hrs(O.delivery_ts, O.order_ts) < 0), ("promised time before order time", hrs(O.promised_ts, O.order_ts) < 0), ("release before order time", hrs(O.release_ts, O.order_ts) < 0), ("pack end before release", hrs(O.pack_end, O.release_ts) < 0), ("dispatch before pack end", hrs(O.dispatch_ts, O.pack_end) < 0), ("delivery before dispatch", hrs(O.delivery_ts, O.dispatch_ts) < 0)]:
        m = m.fillna(False) & ~bad
        if m.any(): ex.append(("orders", "impossible duration: " + why, int(m.sum()))); iss.append(("warn", "orders: %d orders excluded: impossible duration (%s), e.g. %s." % (m.sum(), why, ", ".join(O.order_id[m].head(3))))); bad = bad | m
    ids = set(O.order_id[bad])
    if ids:
        for t in ("orders", "events", "comp"): D[t] = D[t][~D[t].order_id.isin(ids)].reset_index(drop=True)
    if (~D["orders"].order_id.isin(D["events"].order_id)).any(): iss.append(("info", "%d orders have no warehouse events: their stage durations stay missing (never filled)." % int((~D["orders"].order_id.isin(D["events"].order_id)).sum())))
    if len(D["orders"]) < 300: iss.append(("warn", "Only %d orders: statistical and predictive results are unreliable." % len(D["orders"])))
    D.update(issues=iss, excluded=ex, rows_in=rows_in); return D

# ------------------------------ preprocessing and KPIs ------------------------------
DIMS = {"Warehouse": "warehouse_id", "Zone": "zone", "Route": "route_id", "Carrier": "carrier_id", "Shift": "shift", "Category": "category", "Shipping mode": "shipping_mode"}
def backlog(O):
    out = pd.Series(np.nan, index=O.index); ns = lambda s: s.values.astype("datetime64[ns]").astype("int64")
    for w, g in O[O.release_ts.notna() & ~O.canc].groupby("warehouse_id"):
        rel = np.sort(ns(g.release_ts)); dsp = np.sort(ns(g.dispatch_ts.dropna())); t = ns(g.release_ts); out.loc[g.index] = np.searchsorted(rel, t, "right") - np.searchsorted(dsp, t, "right") - 1
    return out
def prep(D, S):
    O = D["orders"].merge(D["events"], on="order_id", how="left")
    O = O.merge(D["routes"][["route_id", "planned_transit_h"]].drop_duplicates("route_id"), on="route_id", how="left") if len(D["routes"]) else O.assign(planned_transit_h=np.nan)
    tb = lambda c: O[c].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    for c, n in (("stock_hold", "hold_f"), ("pick_exception", "exc_f"), ("split_shipment", "split_f"), ("priority_customer", "prio_f")): O[n] = tb(c)
    rh = O.release_ts.dt.hour; O["shift"] = O["shift"].fillna(pd.Series(np.select([rh.between(6, 13), rh.between(14, 21), rh.notna()], ["Morning", "Evening", "Night"], "Unknown"), index=O.index))
    for c in ("warehouse_id", "zone", "route_id", "carrier_id", "shipping_mode", "category", "shift"): O[c] = O[c].fillna("n/a").astype(str)
    st = O.status.fillna("").astype(str).str.lower(); O["canc"] = st.str.contains("cancel"); O["deliv"] = O.delivery_ts.notna() & ~O.canc; O["open"] = ~O["deliv"] & ~O.canc
    asof = pd.concat([O[c] for c in ("order_ts", "release_ts", "pack_end", "dispatch_ts", "delivery_ts")]).max(); O["late"] = np.where(O["deliv"], (O.delivery_ts > O.promised_ts).astype(float), np.nan)
    O["delay_h"] = np.where(O["deliv"], np.maximum(hrs(O.delivery_ts, O.promised_ts), 0), np.nan); O["breach_open"] = O["open"] & (asof > O.promised_ts)
    O["hold_h"] = hrs(O.release_ts, O.order_ts); O["proc_h"] = hrs(O.pack_end, O.release_ts); O["wait_h"] = hrs(O.dispatch_ts, O.pack_end); O["transit_h"] = hrs(O.delivery_ts, O.dispatch_ts)
    O["tgt_transit"] = O.planned_transit_h * np.where(O.shipping_mode.str.lower().str.contains("express"), S["exp_f"], 1.0)
    for s, t in (("hold", "t_hold"), ("proc", "t_proc"), ("wait", "t_wait")): O["x_" + s] = np.maximum(O[s + "_h"] - S[t], 0)
    O["x_transit"] = np.maximum(O.transit_h - O.tgt_transit, 0); O["fail"] = O.failed_attempts; O["backlog"] = backlog(O); O["comp"] = O.order_id.isin(D["comp"].order_id)
    O["comp_late"] = O.order_id.isin(D["comp"].order_id[D["comp"].category.astype(str).str.lower().str.contains("late")]); O["quantity"] = O.quantity.fillna(1); O["rel_hour"] = O.release_ts.dt.hour; return O, asof
def kpis(O):
    d = O[O["deliv"]]; l = d.late == 1; n = len(d)
    return dict(total=len(O), deliv=n, open=int(O["open"].sum()), canc=int(O.canc.sum()), ontime=1 - d.late.mean() if n else np.nan, late_rate=d.late.mean() if n else np.nan, late_n=int(l.sum()), avg_delay=d.delay_h[l].mean() if l.any() else np.nan, med_delay=d.delay_h[l].median() if l.any() else np.nan,
                breaches=int(l.sum() + O.breach_open.sum()), hold=O.hold_h.mean(), proc=O.proc_h.mean(), wait=O.wait_h.mean(), transit=O.transit_h.mean(), comp_rate=d.comp.mean() if n else np.nan)
def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; den = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - a) / den, (c + a) / den
def rate_tbl(d, dim, min_n=30):
    g = d.groupby(dim).agg(n=("late", "size"), late_n=("late", "sum"), hold=("hold_h", "mean"), proc=("proc_h", "mean"), wait=("wait_h", "mean"), transit=("transit_h", "mean"), fail=("fail", "mean")).reset_index().rename(columns={dim: "seg"}); g["rate"] = g.late_n / g.n
    w = [wilson(k, n) for k, n in zip(g.late_n, g.n)]; g["lo"] = [a for a, _ in w]; g["hi"] = [b for _, b in w]; g["small"] = g.n < min_n; return g.sort_values("rate", ascending=False)
def assoc(d, n_boot=30):
    X = pd.get_dummies(d[["warehouse_id", "zone", "carrier_id", "shipping_mode", "category", "shift"]], drop_first=True).astype(float); X["stock_hold"] = d.hold_f.astype(float).values; X["split_shipment"] = d.split_f.astype(float).values; X["backlog_per_10_orders"] = (d.backlog.fillna(d.backlog.median()) / 10).values
    y = d.late.values.astype(int)
    if y.sum() < 30 or (1 - y).sum() < 30: return None
    b = LogisticRegression(max_iter=300).fit(X, y).coef_[0]; rng = np.random.default_rng(0); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() >= 5: bs.append(LogisticRegression(max_iter=300).fit(X.values[i], y[i]).coef_[0])
    bs = np.array(bs); t = pd.DataFrame(dict(factor=X.columns, OR=np.exp(b), lo=np.exp(np.percentile(bs, 2.5, 0)), hi=np.exp(np.percentile(bs, 97.5, 0)))); t["sig"] = (t.lo > 1) | (t.hi < 1); return t.assign(a=np.abs(np.log(t.OR))).sort_values("a", ascending=False).drop(columns="a")

# ------------------------------ delivery-risk model (leakage-safe) ------------------------------
FORBID = {"transit_h", "proc_h", "wait_h", "late", "delay_h", "fail", "comp", "delivery_ts", "dispatch_ts", "pack_end", "pick_start", "exc_f", "status"}
def feats(D, cols=None):
    X = pd.DataFrame(dict(rel_hour=D.release_ts.dt.hour, rel_dow=D.release_ts.dt.dayofweek, hold_h=D.hold_h, backlog=D.backlog, planned_transit_h=D.planned_transit_h, lead_h=hrs(D.promised_ts, D.release_ts), quantity=D.quantity, express=D.shipping_mode.str.lower().str.contains("express").astype(float), prio=D.prio_f.astype(float), hold_f=D.hold_f.astype(float), split_f=D.split_f.astype(float)))
    X["slack_h"] = X.lead_h - D.tgt_transit; X = pd.concat([X, pd.get_dummies(D[["warehouse_id", "zone", "carrier_id", "category"]], prefix=["w", "z", "c", "cat"]).astype(float)], axis=1); return X if cols is None else X.reindex(columns=cols, fill_value=0.0)
def rules_score(op, ref):
    slack = hrs(op.promised_ts, op.release_ts) - op.tgt_transit; return (.6 * np.clip((12 - slack.fillna(12)) / 36, 0, 1) + .25 * np.clip(op.backlog.fillna(0) / max(ref, 1), 0, 1) + .15 * op.hold_f.astype(float)).values
def mk_scored(op, p, S, asof, fac, rules=False):
    hi, md = (.35, .2) if rules else (S["thr_hi"], S["thr_med"]); s = op[["order_id", "warehouse_id", "zone", "carrier_id", "shipping_mode", "promised_ts", "hold_h", "backlog"]].copy(); s["p"] = np.asarray(p, float)
    s["band"] = np.select([s.p >= hi, s.p >= md], ["High", "Medium"], "Low"); s["breached"] = asof > s.promised_ts; s["factors"] = pd.Series(fac or {}).reindex(s.index).fillna("") if fac else ""; return s.sort_values("p", ascending=False)
def fit_risk(O, S, asof):
    D = O[O.release_ts.notna() & ~O.canc].copy(); cols = list(feats(D).columns); R = dict(ok=False, mode="rules", cols=cols, scored=pd.DataFrame(), why=""); dd = D[D["deliv"]].sort_values("order_ts"); nl = int((dd.late == 1).sum()); op = D[D["open"]]
    def rules(why):
        R["why"] = why
        if len(op): R["scored"] = mk_scored(op, rules_score(op, D.backlog.quantile(.9) if len(D) else 1), S, asof, None, True)
        return R
    if len(dd) < 300 or nl < 30 or len(dd) - nl < 30: return rules("a supervised model needs 300+ delivered orders with 30+ late and 30+ on-time (found %d delivered, %d late); transparent rules-based scoring is used instead and cannot be validated." % (len(dd), nl))
    cut = dd.order_ts.iloc[int(len(dd) * .7)]; tr = dd[(dd.order_ts < cut) & (dd.delivery_ts <= cut)]; te = dd[dd.order_ts >= cut]
    if len(tr) < 200 or len(te) < 50: return rules("not enough chronologically separated training and test orders.")
    cc = tr.order_ts.quantile(.8); fit = tr[(tr.order_ts < cc) & (tr.delivery_ts <= cc)]; cal = tr[tr.order_ts >= cc]; yf = fit.late.astype(int).values; yt = te.late.astype(int).values
    if yf.sum() < 20 or yt.sum() < 10 or len(yt) - yt.sum() < 10: return rules("too few late orders in the chronological training or test period.")
    prev = yf.mean(); w = np.where(yf == 1, .5 / prev, .5 / (1 - prev)) if prev < .2 else None; mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=.08, max_depth=4, random_state=0).fit(feats(fit, cols), yf, sample_weight=w); pr = lambda df: mdl.predict_proba(feats(df, cols))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pr(cal), cal.late.astype(int)) if len(cal) >= 100 and cal.late.sum() >= 10 else None; praw = pr(te); pcal = iso.predict(praw) if iso is not None else praw
    use_cal = iso is not None and brier_score_loss(yt, pcal) < brier_score_loss(yt, praw); prob = pcal if use_cal else praw; fin = (lambda p: iso.predict(p)) if use_cal else (lambda p: p); hi, md = S["thr_hi"], S["thr_med"]
    base = -feats(te, cols)["slack_h"].fillna(0).values; rows = []
    for nm, t in (("Medium", md), ("High", hi)): pd_ = prob >= t; rows.append(dict(band=nm, thr=t, prec=precision_score(yt, pd_, zero_division=0), rec=recall_score(yt, pd_, zero_division=0), f1=f1_score(yt, pd_, zero_division=0), flagged=pd_.mean()))
    q = pd.qcut(pd.Series(prob), 5, duplicates="drop"); cb = pd.DataFrame(dict(p=prob, y=yt)).groupby(q.values).agg(pm=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    pi = permutation_importance(mdl, feats(te.head(1500), cols), yt[:1500], scoring="average_precision", n_repeats=3, random_state=0); imp = pd.DataFrame(dict(f=cols, v=pi.importances_mean)).sort_values("v", ascending=False).head(8)
    R.update(ok=True, mode="ml", calibrated=bool(use_cal), n_train=len(fit), n_test=len(te), prev=float(yt.mean()), ap=float(average_precision_score(yt, prob)), ap_base=float(average_precision_score(yt, base)), auc=float(roc_auc_score(yt, prob)), brier_raw=float(brier_score_loss(yt, praw)), brier_cal=float(brier_score_loss(yt, pcal)), thr=pd.DataFrame(rows), cm=confusion_matrix(yt, prob >= hi, labels=[0, 1]), calib=cb, imp=imp, weighted=w is not None,
             split=dict(train_max_delivery=tr.delivery_ts.max(), cut=cut, test_min_order=te.order_ts.min()))
    if len(op):
        p = fin(pr(op)); top = op.loc[pd.Series(p, index=op.index).sort_values(ascending=False).head(12).index]; Xt = feats(top, cols); base_p = mdl.predict_proba(Xt)[:, 1]; med = feats(fit, cols).median(); dl = {}
        for c in cols: X2 = Xt.copy(); X2[c] = med[c]; dl[c] = base_p - mdl.predict_proba(X2)[:, 1]
        dl = pd.DataFrame(dl, index=top.index); fac = {i: ", ".join("%s=%s (+%.0f pts)" % (c, fx(Xt.loc[i, c], 1), 100 * dl.loc[i, c]) for c in dl.loc[i].sort_values(ascending=False).head(3).index if dl.loc[i, c] > 0) for i in top.index}
        R["scored"] = mk_scored(op, p, S, asof, fac)
    return R

# ------------------------------ recommendations, forecasting, simulation ------------------------------
STG = dict(hold=("Order-release hold (stock / allocation)", "Inventory & allocation", "Review stock availability and allocation rules for {s}; test alternative fulfilment locations", "Medium"), proc=("Warehouse processing", "Warehouse operations", "Adjust picking and packing shifts or redistribute volume for {s}; review picking priorities", "Medium"),
           wait=("Dispatch waiting", "Dispatch & logistics", "Review dispatch cut-off times for {s}", "Low"), transit=("Transit and final mile", "Carrier management", "Investigate transit delays and failed attempts on {s}; review carrier allocation rules or SLA", "High"))
RC = ["rank", "problem", "segment", "evidence", "action", "team", "urgency", "direction", "effort", "limits", "kpi", "score"]
def recs(V, S, rk=None):
    d = V[V["deliv"]]; out = []
    if len(d) < 60: return pd.DataFrame(columns=RC)
    ov = d.late.mean()
    for nm, col in DIMS.items():
        for r in rate_tbl(d, col, S["min_n"]).itertuples():
            if r.n < S["min_n"] or r.lo <= ov or r.late_n - r.n * ov < 5 or r.seg == "n/a": continue
            sub = d[(d[col] == r.seg) & (d.late == 1)]; xs = sub[["x_hold", "x_proc", "x_wait", "x_transit"]].mean(); k = xs.idxmax()[2:] if xs.notna().any() and xs.max() > .05 else None; pb, team, act, eff = STG[k] if k else ("Mixed stages", "Operations", "Investigate {s}: no single stage dominates the excess time", "Medium"); ex = r.late_n - r.n * ov
            out.append(dict(problem="Late deliveries concentrated in %s %s; main excess time in: %s" % (nm.lower(), r.seg, pb), segment="%s = %s" % (nm, r.seg), evidence="Late rate %s (95%% CI %s–%s, n=%d) vs %s overall: about %.0f more late orders than expected. Mean excess hours over target on late orders: hold %s, processing %s, dispatch wait %s, transit %s." % (pc(r.rate, 1), pc(r.lo, 1), pc(r.hi, 1), r.n, pc(ov, 1), ex, fx(xs["x_hold"]), fx(xs["x_proc"]), fx(xs["x_wait"]), fx(xs["x_transit"])),
                            action=act.format(s=r.seg), team=team, urgency="High" if r.rate >= 2 * ov else "Medium", effort=eff, score=ex, limits="Historical association, not a proven cause; n=%d; weather, demand mix and carrier capacity are only partly observed." % r.n, kpi="Late-delivery rate and mean %s time for %s, tracked weekly" % ({"hold": "hold", "proc": "processing", "wait": "dispatch wait", "transit": "transit"}.get(k, "end-to-end"), r.seg)))
    if rk is not None and len(rk):
        hi = rk[rk.band == "High"]
        if len(hi): out.append(dict(problem="Orders at risk of missing the promise are only visible after they are late", segment="All open released orders", evidence="%d open orders are currently High risk (%s)." % (len(hi), "calibrated model" if G.get("X", {}).get("risk", {}).get("calibrated") else "model score or rules; not a validated probability"), action="Introduce earlier warnings: alert operations at release for High-risk orders and notify priority customers proactively", team="Customer operations", urgency="High" if len(hi) >= 20 else "Medium", effort="Low", score=float(hi.p.sum()) if len(hi) else 0.0, limits="Risk scores are estimates; thresholds must be tuned to team capacity.", kpi="Share of High-risk orders contacted or expedited; late rate of flagged orders"))
    R = pd.DataFrame(out, columns=[c for c in RC if c not in ("rank", "direction")] if out else None)
    if R.empty: return pd.DataFrame(columns=RC)
    R = R.sort_values("score", ascending=False).reset_index(drop=True); R["rank"] = np.arange(1, len(R) + 1); R.loc[R["rank"] <= 3, "urgency"] = "High"; R["direction"] = "Expected to lower the late-delivery rate in this segment; the size is unknown until tested (see Scenario Lab for hypothetical estimates)."; return R[RC]
def vol_forecast(V):
    d = V[V.order_ts < V.order_ts.max().normalize()]; pv = d.groupby([d.order_ts.dt.normalize(), "warehouse_id"]).size().unstack(fill_value=0)
    if len(pv) < 42: return None
    pv = pv.reindex(pd.date_range(pv.index.min(), pv.index.max()), fill_value=0).astype(float); lag = lambda k: pv.shift(k); a = sum(lag(k) for k in (7, 14, 21, 28)) / 4; b = pv.rolling(7).mean().shift(1); te = pv.index[-14:]
    wp = lambda f: float((f.loc[te] - pv.loc[te]).abs().sum().sum() / pv.loc[te].sum().sum()); wa, wb = wp(a), wp(b); best = "Same-weekday mean (4 weeks)" if wa <= wb else "Trailing 7-day mean"; fut = []
    for k in range(1, 8):
        dt = pv.index[-1] + pd.Timedelta(days=k); fut.append((pv.iloc[[-7 + k - 1 + 7 * i for i in (-3, -2, -1, 0)] if False else [len(pv) - 7 * i + (k - 1) - 7 + 7 for i in (4, 3, 2, 1)] if False else []].mean() if False else None))
    wd = [pv[pv.index.dayofweek == (pv.index[-1] + pd.Timedelta(days=k)).dayofweek].tail(4).mean() for k in range(1, 8)]; tm = pv.tail(7).mean(); fc = pd.DataFrame([w_ if wa <= wb else tm for w_ in wd], index=[pv.index[-1] + pd.Timedelta(days=k) for k in range(1, 8)])
    return dict(pv=pv, wa=wa, wb=wb, best=best, fc=fc, ho_act=pv.loc[te].sum(1), ho_pred=(a if wa <= wb else b).loc[te].sum(1))
P0 = dict(staff=0, proc=0, wait=0, hold=0, transit=0, shift=0, src="", dst="", pen=10, rate=250.0, impl=0.0)
def simulate(V, p, staff_h=3.0):
    d = V[V["deliv"]].dropna(subset=["hold_h", "proc_h", "wait_h", "transit_h"]).copy()
    if len(d) < 50: raise ValueError("Fewer than 50 delivered orders with complete stage timestamps: the simulator needs them.")
    hold, proc, wait, trn = (d[c].values.astype(float).copy() for c in ("hold_h", "proc_h", "wait_h", "transit_h")); ph = hrs(d.promised_ts, d.order_ts).values; wh = d.warehouse_id.values; tot0 = hold + proc + wait + trn; late0 = tot0 > ph + 1e-9
    if p["shift"] > 0 and p["src"] and p["dst"] and p["src"] != p["dst"]:
        ref = d[d.warehouse_id == p["dst"]]; mv = (wh == p["src"]) & (np.random.default_rng(1).random(len(d)) < p["shift"] / 100)
        if len(ref) >= 30: hold[mv] = ref.hold_h.median(); proc[mv] = ref.proc_h.median(); wait[mv] = ref.wait_h.median(); trn[mv] = trn[mv] * (1 + p["pen"] / 100)
    peak = d.release_ts.dt.hour.isin(d.release_ts.dt.hour.value_counts().head(4).index).values; proc = np.where(peak, proc / (1 + p["staff"] / 100), proc) * (1 - p["proc"] / 100); wait = wait * (1 - p["wait"] / 100); hold = np.where(d.hold_f.values, hold * (1 - p["hold"] / 100), hold); trn = trn * (1 + p["transit"] / 100)
    tot = hold + proc + wait + trn; late1 = tot > ph + 1e-9; d0 = np.maximum(tot0 - ph, 0); d1 = np.maximum(tot - ph, 0); bk = lambda x: np.select([x <= 1e-9, x < 12, x < 24], [0, 1, 2], 3); cr = pd.Series(d.comp.values.astype(float)).groupby(bk(d0)).mean().reindex(range(4)).fillna(0).values
    idx = np.random.default_rng(0).integers(0, len(d), (200, len(d))); nd = d.order_ts.dt.normalize().nunique(); ci = np.percentile(late1[idx].mean(1), [2.5, 97.5]); c0 = np.percentile(late0[idx].mean(1), [2.5, 97.5])
    return dict(n=len(d), excl=int(V["deliv"].sum()) - len(d), late0=late0.mean(), late1=late1.mean(), ci0=c0, ci1=ci, b0=int(late0.sum()), b1=int(late1.sum()), dl0=d0[late0].mean() if late0.any() else np.nan, dl1=d1[late1].mean() if late1.any() else np.nan, pre0=(d.hold_h + d.proc_h + d.wait_h).mean(), pre1=float((hold + proc + wait).mean()), ce0=float(cr[bk(d0)].sum()), ce1=float(cr[bk(d1)].sum()),
                cost=p["staff"] / 100 * 4 * staff_h * len(set(wh)) * nd * p["rate"] + p["impl"], wh=pd.DataFrame(dict(w=wh, a=late0, b=late1)).groupby("w").mean().reset_index())

# ------------------------------ pages ------------------------------
def vw(wh, zn, cr, d1, d2):
    O = G["X"]["O"]; m = pd.Series(True, index=O.index)
    if wh: m &= O.warehouse_id.isin(wh)
    if zn: m &= O.zone.isin(zn)
    if cr: m &= O.carrier_id.isin(cr)
    a = pd.to_datetime(d1, errors="coerce") if d1 else pd.NaT; b = pd.to_datetime(d2, errors="coerce") if d2 else pd.NaT
    if pd.notna(a): m &= O.order_ts >= a
    if pd.notna(b): m &= O.order_ts < b + pd.Timedelta(days=1)
    if not m.any(): raise ValueError("No orders match the filters.")
    return O[m]
def fsc(wh, zn, cr):
    rk = G["X"]["risk"]["scored"]
    if not len(rk): return rk
    m = pd.Series(True, index=rk.index)
    if wh: m &= rk.warehouse_id.isin(wh)
    if zn: m &= rk.zone.isin(zn)
    if cr: m &= rk.carrier_id.isin(cr)
    return rk[m]
@safe(4)
def render_ct(wh, zn, cr, d1, d2):
    X = G["X"]; S = X["S"]; V = vw(wh, zn, cr, d1, d2); k = kpis(V); rk = fsc(wh, zn, cr); d = V[V["deliv"]]; R = recs(V, S, rk); hi = int((rk.band == "High").sum()) if len(rk) else 0
    h = banner() + "<div class='kg'>" + card("Total orders", nf(k["total"]), "%s delivered · %s open · %s cancelled" % (nf(k["deliv"]), nf(k["open"]), nf(k["canc"]))) + card("On-time rate", pc(k["ontime"], 1), "delivered on or before promise ÷ delivered") + card("Late rate", pc(k["late_rate"], 1), "%s late deliveries" % nf(k["late_n"]), C["co"]) + card("Delay (late orders)", "%s h avg · %s h median" % (fx(k["avg_delay"]), fx(k["med_delay"])), "hours beyond the promise", C["am"]) + card("SLA breaches", nf(k["breaches"]), "late deliveries + open orders past promise", C["co"]) + \
        card("Warehouse processing", fx(k["proc"]) + " h", "release → packed (mean)") + card("Dispatch waiting", fx(k["wait"]) + " h", "packed → dispatched (mean)") + card("Transit", fx(k["transit"]) + " h", "dispatched → delivered (mean)") + card("Complaint rate", pc(k["comp_rate"], 1), "delivered orders with a complaint", C["am"]) + card("High-risk open orders", nf(hi), "▲ need intervention (see Delivery Risk)", C["co"]) + "</div>"
    h += note("<b>Definitions.</b> Orders not yet delivered are <b>censored</b>: they are excluded from on-time and late rates, never counted as on-time; cancelled orders are excluded from rates. Open orders past their promise are counted only in SLA breaches. Stage durations use only orders with both timestamps; nothing is filled in. The latest weeks look better than they are because late orders are still in transit.", "info")
    t = rate_tbl(d, "zone", S["min_n"]); h += SEC("Performance by region (delivery zone)") + tbl(t, [("seg", "Zone"), ("n", "Delivered"), ("rate", "Late rate"), ("lo", "95% low"), ("hi", "95% high"), ("transit", "Mean transit h"), ("small", "Small sample")], dict(n=nf, rate=lambda v: pc(v, 1), lo=lambda v: pc(v, 1), hi=lambda v: pc(v, 1), transit=fx, small=lambda v: "⚠ n < %d" % S["min_n"] if v else ""))
    h += SEC("Highest-priority operational bottlenecks") + (tbl(R.head(5), [("rank", "#"), ("problem", "Problem"), ("urgency", "Urgency"), ("evidence", "Evidence")], dict(urgency=pill)) if len(R) else note("No segment is statistically elevated in this view.", "ok"))
    w = d.groupby(d.order_ts.dt.to_period("W").dt.start_time).agg(n=("late", "size"), r=("late", "mean")).reset_index(); f1 = go.Figure([go.Bar(x=w.order_ts, y=w.n, name="Delivered orders", marker_color=C["gg"], yaxis="y"), go.Scatter(x=w.order_ts, y=w.r, name="Late rate (right axis)", line=dict(color=C["co"], width=3), yaxis="y2")]); f1 = sty(f1, 340, "Weekly volume and late rate (two axes: read each separately)"); f1.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", title="Late rate"), yaxis=dict(title="Orders"))
    tw = rate_tbl(d, "warehouse_id", S["min_n"]); f2 = go.Figure(go.Bar(x=tw.seg, y=tw.rate, marker_color=C["fg"], error_y=dict(type="data", symmetric=False, array=tw.hi - tw.rate, arrayminus=tw.rate - tw.lo))); f2 = sty(f2, 320, "Late rate by warehouse (95% Wilson interval)"); f2.update_yaxes(tickformat=".0%")
    lt = d[d.late == 1]; f3 = go.Figure(go.Histogram(x=lt.delay_h.clip(upper=lt.delay_h.quantile(.99)) if len(lt) else [], nbinsx=30, marker_color=C["am"])); f3 = sty(f3, 320, "Delay distribution of late deliveries (hours beyond promise, capped at P99)"); return h, f1, f2, f3
@safe(3)
def render_rc(wh, zn, cr, d1, d2, dim, val):
    X = G["X"]; S = X["S"]; V = vw(wh, zn, cr, d1, d2); d = V[V["deliv"]]; ov = d.late.mean(); col = DIMS.get(dim, "warehouse_id")
    st = d.groupby("warehouse_id").agg(n=("late", "size"), hold=("hold_h", "mean"), proc=("proc_h", "mean"), wait=("wait_h", "mean"), transit=("transit_h", "mean"), tgt=("tgt_transit", "mean"), xs=("x_proc", "mean")).reset_index()
    h = banner() + note("<b>Where delay accumulates.</b> Targets: hold %s h, processing %s h, dispatch wait %s h, transit = route plan (× %s for Express). Excess = hours above target. <b>Associations are not causes</b>: segments and models below show where delay clusters, not what produces it.", "info") % (S["t_hold"], S["t_proc"], S["t_wait"], S["exp_f"]) if False else banner() + note("<b>Where delay accumulates.</b> Targets: hold %s h, processing %s h, dispatch wait %s h, transit = route plan (× %s for Express). Excess = hours above target. <b>Associations are not causes</b>: the tables show where delay clusters, not what produces it." % (S["t_hold"], S["t_proc"], S["t_wait"], S["exp_f"]), "info")
    h += SEC("Actual vs target stage duration by warehouse (mean hours)") + tbl(st, [("seg" if False else "warehouse_id", "Warehouse"), ("n", "Delivered"), ("hold", "Hold (target %s)" % S["t_hold"]), ("proc", "Processing (target %s)" % S["t_proc"]), ("wait", "Dispatch wait (target %s)" % S["t_wait"]), ("transit", "Transit"), ("tgt", "Transit target")], dict(n=nf, hold=fx, proc=fx, wait=fx, transit=fx, tgt=fx))
    for nm, c in DIMS.items():
        t = rate_tbl(d, c, S["min_n"]); h += SEC("Late rate by " + nm.lower()) + tbl(t.head(12), [("seg", nm), ("n", "Delivered"), ("rate", "Late rate"), ("lo", "95% low"), ("hi", "95% high"), ("proc", "Proc h"), ("wait", "Wait h"), ("transit", "Transit h"), ("fail", "Failed attempts"), ("small", "Note")], dict(n=nf, rate=lambda v: pc(v, 1), lo=lambda v: pc(v, 1), hi=lambda v: pc(v, 1), proc=fx, wait=fx, transit=fx, fail=lambda v: fx(v, 2), small=lambda v: "⚠ small sample" if v else ""))
    cmp_ = pd.DataFrame([(n, d[d.late == 0][c].mean(), d[d.late == 1][c].mean()) for n, c in (("Hold h", "hold_h"), ("Processing h", "proc_h"), ("Dispatch wait h", "wait_h"), ("Transit h", "transit_h"), ("Warehouse backlog at release (orders)", "backlog"), ("Failed attempts", "fail"), ("Stock-hold share", "hold_f"), ("Split-shipment share", "split_f"), ("Pick-exception share", "exc_f"))], columns=["m", "a", "b"]); h += SEC("On-time vs late orders") + tbl(cmp_, [("m", "Measure"), ("a", "On-time orders"), ("b", "Late orders")], dict(a=lambda v: fx(v, 2), b=lambda v: fx(v, 2)))
    key = (len(d), int(d.late.sum()), d.index[0], d.index[-1]); G.setdefault("_as", {}); a = G["_as"].get(key) if key in G["_as"] else assoc(d); G["_as"][key] = a
    h += SEC("Adjusted associations (logistic regression, odds ratios; reference = first level alphabetically)") + (tbl(a.head(10), [("factor", "Factor"), ("OR", "Odds ratio"), ("lo", "95% low (bootstrap)"), ("hi", "95% high"), ("sig", "Interval excludes 1")], dict(OR=lambda v: fx(v, 2), lo=lambda v: fx(v, 2), hi=lambda v: fx(v, 2), sig=lambda v: "yes" if v else "no")) + note("Odds ratios adjust for the other listed factors but not for unobserved confounders (weather, demand mix, carrier capacity). Treat as associations. 30 bootstrap resamples: intervals are rough.", "warn") if a is not None else note("Too few late or on-time orders for an adjusted model.", "warn"))
    if val:
        sub = d[d[col].astype(str) == val.strip()]; l = sub[sub.late == 1].sort_values("delay_h", ascending=False).head(40); h += SEC("Drill-down: %s = %s (%d delivered, late rate %s)" % (esc(dim), esc(val), len(sub), pc(sub.late.mean(), 1) if len(sub) else "—")) + (tbl(l, [("order_id", "Order"), ("order_ts", "Ordered"), ("warehouse_id", "Warehouse"), ("route_id", "Route"), ("carrier_id", "Carrier"), ("delay_h", "Delay h"), ("hold_h", "Hold"), ("proc_h", "Proc"), ("wait_h", "Wait"), ("transit_h", "Transit")], dict(order_ts=dts, delay_h=fx, hold_h=fx, proc_h=fx, wait_h=fx, transit_h=fx)) if len(l) else note("No late orders for this value (check spelling, e.g. WH3).", "info"))
    lt = d[d.late == 1]; ex = lt[["x_hold", "x_proc", "x_wait", "x_transit"]].sum().sort_values(ascending=False); lab = {"x_hold": "Hold", "x_proc": "Processing", "x_wait": "Dispatch wait", "x_transit": "Transit / final mile"}; cum = ex.cumsum() / max(ex.sum(), 1e-9)
    f1 = go.Figure([go.Bar(x=[lab[i] for i in ex.index], y=ex.values, marker_color=C["am"], name="Excess hours (late orders)"), go.Scatter(x=[lab[i] for i in ex.index], y=cum.values, name="Cumulative share (right axis)", yaxis="y2", line=dict(color=C["fg"], width=3))]); f1 = sty(f1, 340, "Pareto: excess hours over target on late orders, by stage"); f1.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]))
    f2 = go.Figure([go.Bar(x=st.warehouse_id, y=st[c], name=n) for c, n in (("hold", "Hold"), ("proc", "Processing"), ("wait", "Dispatch wait"))]); f2.update_layout(barmode="stack"); f2 = sty(f2, 320, "Pre-dispatch time by warehouse (hours; targets %s / %s / %s)" % (S["t_hold"], S["t_proc"], S["t_wait"])); return h, f1, f2
@safe(3)
def render_risk(wh, zn, cr, d1, d2):
    X = G["X"]; R = X["risk"]; S = X["S"]; sc = fsc(wh, zn, cr)
    h = banner() + note("<b>Prediction time = order release</b> (after any stock hold). <b>Target</b> = delivered after the promised time. <b>Features</b> (all known at release): release hour and weekday, hold hours, warehouse backlog at release (counted from events before release), planned transit, promise slack, quantity, Express, priority customer, stock-hold and split flags, warehouse, zone, carrier, category. <b>Excluded</b>: processing, dispatch and transit durations, pick exceptions, failed attempts, delivery times, complaints.", "info"); f1 = f2 = None
    if R["mode"] == "ml":
        sp = R["split"]; t = R["thr"]; cm = R["cm"]
        h += note("<b>Chronological evaluation.</b> Train on orders before %s whose delivery was already known at that date (latest known delivery %s); test on orders from %s onward (%d train, %d test, %s late in test). Class imbalance: %s. Calibration: %s." % (dts(sp["cut"]), dts(sp["train_max_delivery"]), dts(sp["test_min_order"]), R["n_train"], R["n_test"], pc(R["prev"], 1), "balanced sample weights" if R["weighted"] else "not needed", "isotonic on a later training slice (Brier %.3f → %.3f)" % (R["brier_raw"], R["brier_cal"]) if R["calibrated"] else "not validated, so scores are rankings, not probabilities"), "ai")
        h += note("Held-out PR-AUC <b>%.2f</b> (no-skill = %.2f; simple 'promise slack' baseline = %.2f), ROC-AUC <b>%.2f</b>. %s" % (R["ap"], R["prev"], R["ap_base"], R["auc"], "<b>Synthetic labels: these metrics show the workflow, not real-world accuracy.</b>" if G.get("demo") else "Verify on your own data before relying on it."), "ai") + tbl(t, [("band", "Band threshold"), ("thr", "Score ≥"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1"), ("flagged", "Share of orders flagged")], dict(thr=lambda v: fx(v, 2), prec=lambda v: pc(v, 1), rec=lambda v: pc(v, 1), f1=lambda v: fx(v, 2), flagged=lambda v: pc(v, 1))) + note("Confusion matrix at the High threshold: TN %d, FP %d, FN %d, TP %d. A lower threshold catches more late orders (higher recall) but sends more false alarms; set it to what the team can act on." % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]), "info")
        c = R["calib"]; f1 = go.Figure([go.Scatter(x=c.pm, y=c.obs, mode="lines+markers", name="Model", line=dict(color=C["fg"], width=3)), go.Scatter(x=[0, max(c.pm.max(), c.obs.max())], y=[0, max(c.pm.max(), c.obs.max())], name="Perfect calibration", line=dict(color=C["gg"], dash="dash"))]); f1 = sty(f1, 320, "Calibration on the held-out period (5 score bins)"); f1.update_xaxes(title="Mean predicted"); f1.update_yaxes(title="Observed late share")
        i = R["imp"]; f2 = sty(go.Figure(go.Bar(y=i.f, x=i.v, orientation="h", marker_color=C["li"])), 320, "Features the model relies on (permutation importance; not causes)"); f2.update_yaxes(autorange="reversed")
    else: h += note("<b>Supervised model disabled:</b> " + esc(R["why"]) + " Scores below are transparent rules: 0.6 × promise-slack shortfall + 0.25 × backlog + 0.15 × stock hold (bands: ≥ 0.35 High, ≥ 0.20 Medium). They are <b>not probabilities</b>.", "warn")
    if len(sc): cnt = sc.band.value_counts(); h += "<div class='kg'>" + "".join(card(b + " risk", nf(cnt.get(b, 0)), "open released orders", RK[b][1]) for b in ("High", "Medium", "Low")) + "</div>" + SEC("Open orders by risk (top 30; factors = approximate effect of replacing each feature with a typical value)") + tbl(sc.head(30), [("order_id", "Order"), ("warehouse_id", "Warehouse"), ("zone", "Zone"), ("carrier_id", "Carrier"), ("promised_ts", "Promised"), ("backlog", "Backlog at release"), ("p", "Score"), ("band", "Band"), ("breached", "Already past promise"), ("factors", "Top risk factors")], dict(promised_ts=dts, backlog=nf, p=lambda v: fx(v, 2), band=pill, breached=lambda v: "yes" if v else ""))
    else: h += note("No open released orders to score.", "info")
    return h, f1, f2
@safe(4)
def render_wh(wh, zn, cr, d1, d2):
    X = G["X"]; V = vw(wh, zn, cr, d1, d2); asof = X["asof"]; nd = max(V.order_ts.dt.normalize().nunique(), 1); rows = []
    for w, g in V.groupby("warehouse_id"):
        aw = g[g["open"] & g.release_ts.notna() & g.pack_end.isna()]; ad = g[g.pack_end.notna() & g.dispatch_ts.isna() & ~g.canc]; q = g.proc_h.dropna()
        rows.append(dict(w=w, n=len(g), per_day=len(g) / nd, med=q.median(), p90=q.quantile(.9) if len(q) else np.nan, hold=g.hold_h.mean(), wait=g.wait_h.mean(), bl=g.backlog.mean(), sh=g.hold_f.mean(), ex=g.exc_f.mean(), aw=len(aw), ad=len(ad), age=hrs(pd.Series(asof, index=pd.concat([aw, ad]).index), pd.concat([aw, ad]).order_ts).max() if len(aw) + len(ad) else np.nan))
    t = pd.DataFrame(rows); nb = t.bl.mean(); mm = t[t.bl >= 1.5 * nb].w.tolist(); h = banner() + note("<b>Capacity mismatch check.</b> %s Network capacity can look sufficient while one site queues: compare backlog and processing time across sites. Backlog = orders released but not yet dispatched at the moment an order is released. No individual workers are assessed." % (("Warehouses with mean backlog ≥ 1.5× the network average: <b>%s</b> (network mean %s orders)." % (", ".join(mm), fx(nb))) if mm else "No warehouse has a backlog ≥ 1.5× the network average."), "info")
    h += tbl(t, [("w", "Warehouse"), ("n", "Orders"), ("per_day", "Orders/day"), ("med", "Median proc h"), ("p90", "P90 proc h"), ("hold", "Mean hold h"), ("wait", "Mean dispatch wait h"), ("bl", "Mean backlog"), ("sh", "Stock-hold share"), ("ex", "Pick-exception share"), ("aw", "Awaiting processing now"), ("ad", "Awaiting dispatch now"), ("age", "Oldest waiting (h)")], dict(n=nf, per_day=fx, med=fx, p90=fx, hold=fx, wait=fx, bl=fx, sh=pc, ex=pc, aw=nf, ad=nf, age=fx))
    ar = V.assign(hr=V.order_ts.dt.hour).groupby(["warehouse_id", "hr"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0) / nd; f1 = go.Figure(go.Heatmap(z=ar.values, x=list(ar.columns), y=list(ar.index), colorscale=[[0, C["iv"]], [1, C["fg"]]], hovertemplate="%{y} hour %{x}: %{z:.1f} orders/day<extra></extra>")); f1 = sty(f1, 300, "Order arrivals: average orders per day by hour and warehouse"); f1.update_xaxes(title="Hour of day")
    b = V[V.release_ts.notna()].groupby([V.release_ts.dt.normalize(), "warehouse_id"]).backlog.mean().unstack(); f2 = go.Figure([go.Scatter(x=b.index, y=b[c], name=c, mode="lines") for c in b.columns]); f2 = sty(f2, 320, "Backlog over time (daily mean orders awaiting dispatch at release)")
    dd = V[V.dispatch_ts.notna()]; th = dd.groupby([dd.dispatch_ts.dt.normalize(), "warehouse_id"]).size().unstack(fill_value=0); f3 = go.Figure([go.Scatter(x=th.index, y=th[c], name=c, mode="lines") for c in th.columns]); f3 = sty(f3, 300, "Throughput: orders dispatched per day"); h += note("Possible actions are generated from these measurements in the Action Centre (workload redistribution, shift and dispatch cut-off changes, picking priorities, inventory relocation).", "ai"); return h, f1, f2, f3
@safe(3)
def render_rt(wh, zn, cr, d1, d2):
    X = G["X"]; S = X["S"]; V = vw(wh, zn, cr, d1, d2); d = V[V["deliv"]]; ov = d.late.mean(); rt = rate_tbl(d, "route_id", S["min_n"]); m = d.groupby("route_id").agg(plan=("planned_transit_h", "mean"), act=("transit_h", "median")).reset_index().rename(columns={"route_id": "seg"}); rt = rt.merge(m, on="seg"); rt["exc"] = (rt.lo > ov) & (rt.n >= S["min_n"])
    h = banner() + note("Routes flagged as <b>exceptions</b> have a late-rate interval entirely above the overall rate (%s) with n ≥ %d. Transit is dispatch → delivery and includes failed attempts. No live traffic or road routes are used; only historical records." % (pc(ov, 1), S["min_n"]), "info") + SEC("Routes") + tbl(rt.head(25), [("seg", "Route"), ("n", "Delivered"), ("plan", "Planned transit h"), ("act", "Median actual h"), ("rate", "Late rate"), ("lo", "95% low"), ("hi", "95% high"), ("fail", "Failed attempts / order"), ("exc", "Exception")], dict(n=nf, plan=fx, act=fx, rate=lambda v: pc(v, 1), lo=lambda v: pc(v, 1), hi=lambda v: pc(v, 1), fail=lambda v: fx(v, 2), exc=lambda v: "▲ yes" if v else ""))
    for nm, c in (("Carrier", "carrier_id"), ("Shipping mode", "shipping_mode")): h += SEC(nm + " service level") + tbl(rate_tbl(d, c, S["min_n"]), [("seg", nm), ("n", "Delivered"), ("rate", "Late rate"), ("lo", "95% low"), ("hi", "95% high"), ("transit", "Mean transit h"), ("fail", "Failed attempts / order")], dict(n=nf, rate=lambda v: pc(v, 1), lo=lambda v: pc(v, 1), hi=lambda v: pc(v, 1), transit=fx, fail=lambda v: fx(v, 2)))
    cz = rate_tbl(d, "carrier_id", S["min_n"]).iloc[0:0]; zc = d.groupby(["zone", "carrier_id"]).agg(n=("late", "size"), r=("late", "mean")).reset_index(); zc = zc[zc.n >= S["min_n"]].sort_values("r", ascending=False).head(6); h += SEC("Highest late-rate carrier × zone combinations (n ≥ %d)" % S["min_n"]) + tbl(zc, [("zone", "Zone"), ("carrier_id", "Carrier"), ("n", "Delivered"), ("r", "Late rate")], dict(n=nf, r=lambda v: pc(v, 1)))
    t = rt.sort_values("n", ascending=False).head(15); f1 = go.Figure([go.Bar(x=t.seg, y=t.plan, name="Planned", marker_color=C["gg"]), go.Bar(x=t.seg, y=t.act, name="Median actual", marker_color=C["fg"])]); f1.update_layout(barmode="group"); f1 = sty(f1, 340, "Planned vs actual transit hours (15 busiest routes)"); f1.update_xaxes(tickangle=-30)
    hr = d[d.dispatch_ts.notna()].groupby(d.dispatch_ts.dt.hour).agg(r=("late", "mean"), n=("late", "size")).reset_index(); hr = hr[hr.n >= 20]; f2 = sty(go.Figure(go.Bar(x=hr.dispatch_ts, y=hr.r, marker_color=C["am"])), 320, "Late rate by dispatch hour (hours with ≥ 20 orders)"); f2.update_yaxes(tickformat=".0%"); f2.update_xaxes(title="Dispatch hour"); return h, f1, f2
@safe(2)
def render_inv(wh, zn, cr, d1, d2):
    X = G["X"]; D = X["D"]; S = X["S"]; V = vw(wh, zn, cr, d1, d2); d = V[V["deliv"]]; hd = d[d.hold_f]; ov = d.late.mean(); sp = d[d.split_f]
    h = banner() + "<div class='kg'>" + card("Orders with a stock hold", pc(V.hold_f.mean(), 1), "%s orders" % nf(V.hold_f.sum()), C["am"]) + card("Late rate: held vs not", "%s vs %s" % (pc(hd.late.mean(), 1) if len(hd) else "—", pc(d[~d.hold_f].late.mean(), 1)), "delivered orders", C["co"]) + card("Mean hold time", fx(hd.hold_h.mean()) + " h", "held orders") + card("Split shipments", pc(V.split_f.mean(), 1), "late rate %s vs %s without" % (pc(sp.late.mean(), 1) if len(sp) else "—", pc(d[~d.split_f].late.mean(), 1))) + "</div>"
    g = V.groupby(["warehouse_id", "category"]).agg(n=("hold_f", "size"), held=("hold_f", "sum"), hold=("hold_h", "mean")).reset_index(); g["rate"] = g.held / g.n; g = g[g.n >= S["min_n"]].sort_values("rate", ascending=False).head(8)
    h += SEC("Recurring stock gaps (warehouse × category, hold share)") + tbl(g, [("warehouse_id", "Warehouse"), ("category", "Category"), ("n", "Orders"), ("held", "Held"), ("rate", "Hold share"), ("hold", "Mean hold h")], dict(n=nf, held=nf, rate=pc, hold=fx)); inv = D["inv"]; rt = D["routes"]; rows = []
    for (w, s), x in hd.groupby(["warehouse_id", "sku"]):
        if len(x) < 5: continue
        zone = x.zone.mode().iloc[0]; alt = inv[(inv.sku == s) & (inv.warehouse_id != w)].sort_values("available_units", ascending=False) if len(inv) else inv
        if not len(alt) or not len(rt): rows.append(dict(w=w, sku=s, n=len(x), mh=x.hold_h.mean(), alt="—", units=np.nan, extra=np.nan, net=np.nan, verdict="Cannot evaluate: inventory or route data missing")); continue
        a = alt.iloc[0]; cur = rt[(rt.warehouse_id == w) & (rt.zone == zone)].planned_transit_h; nw = rt[(rt.warehouse_id == a.warehouse_id) & (rt.zone == zone)].planned_transit_h
        if cur.empty or nw.empty: rows.append(dict(w=w, sku=s, n=len(x), mh=x.hold_h.mean(), alt=a.warehouse_id, units=a.available_units, extra=np.nan, net=np.nan, verdict="Cannot evaluate: no route time for zone " + zone)); continue
        ex = float(nw.iloc[0] - cur.iloc[0]); net = x.hold_h.mean() - ex; rows.append(dict(w=w, sku=s, n=len(x), mh=x.hold_h.mean(), alt=a.warehouse_id, units=a.available_units, extra=ex, net=net, verdict="Candidate: saves about %.1f h net (cost not evaluated)" % net if (net > 0 and a.available_units >= 20) else "Not supported by the data"))
    h += SEC("Alternative fulfilment locations for repeatedly held SKUs (≥ 5 held orders)") + (tbl(pd.DataFrame(rows).sort_values("n", ascending=False).head(10), [("w", "Held at"), ("sku", "SKU"), ("n", "Held orders"), ("mh", "Mean hold h"), ("alt", "Best alternative"), ("units", "Units available"), ("extra", "Extra transit h (modal zone)"), ("net", "Net hours saved"), ("verdict", "Assessment")], dict(n=nf, mh=fx, units=nf, extra=fx, net=fx)) if rows else note("No SKU has 5+ held orders in this view.", "info")) + note("An alternative is recommended only when stock exists elsewhere and the extra transit is shorter than the hold; shipping cost, split-order effects and demand at the alternative site are <b>not</b> in the data. Verify before reallocating.", "warn")
    f = go.Figure([go.Bar(x=["No stock hold", "Stock hold"], y=[d[~d.hold_f].late.mean(), hd.late.mean() if len(hd) else 0], marker_color=[C["fg"], C["am"]]), ]); f = sty(f, 300, "Late rate with and without a stock hold"); f.update_yaxes(tickformat=".0%"); return h, f
@safe(2)
def render_staff(wh, zn, cr, d1, d2):
    X = G["X"]; D = X["D"]; S = X["S"]; V = vw(wh, zn, cr, d1, d2); st = D["staff"]; hm = S["hmin"] / 60; h = banner(); f = None
    if len(st):
        sg = st.groupby(["warehouse_id", "shift"]).agg(staff=("staff_scheduled", "mean"), ph=("productive_hours_per_staff", "mean")).reset_index(); sg["ph"] = sg.ph.fillna(6.8); sg["cap_h"] = sg.staff * sg.ph; nd = max(V.order_ts.dt.normalize().nunique(), 1); wl = V[V.release_ts.notna()].groupby(["warehouse_id", "shift"]).size().reset_index(name="n"); sg = sg.merge(wl, on=["warehouse_id", "shift"], how="left"); sg["work_h"] = sg.n.fillna(0) / nd * hm; sg["ratio"] = sg.work_h / sg.cap_h
        h += note("<b>Capacity is local in space and time.</b> Workload = orders released per day in the shift × %.0f min handling (assumption, editable in Settings); capacity = scheduled staff × productive hours. Ratio ≥ 1 means the shift cannot keep up on an average day even if other sites or shifts have slack. Peak days are worse: use the Scenario Lab." % S["hmin"], "info") + tbl(sg.sort_values("ratio", ascending=False), [("warehouse_id", "Warehouse"), ("shift", "Shift"), ("staff", "Avg staff"), ("cap_h", "Capacity h/day"), ("work_h", "Workload h/day"), ("ratio", "Workload ÷ capacity")], dict(staff=fx, cap_h=fx, work_h=fx, ratio=lambda v: ("▲ " if v >= 1 else ("● " if v >= .8 else "✓ ")) + fx(v, 2)))
        dayv = V[V.release_ts.notna()].groupby([V.release_ts.dt.normalize(), "warehouse_id"]).size().reset_index(name="n"); capd = sg.groupby("warehouse_id").cap_h.sum(); dayv["ratio"] = dayv.n * hm / dayv.warehouse_id.map(capd); ov_ = dayv[dayv.ratio > 1].groupby("warehouse_id").size().reindex(WH if G.get("demo") else sorted(dayv.warehouse_id.unique())).fillna(0); h += note("Days where daily workload exceeded daily capacity: " + (", ".join("%s %d" % (k, v) for k, v in ov_.items() if v > 0) or "none") + ".", "info")
    else: h += note("No staffing table: workload-to-capacity cannot be computed (headcount alone is not capacity).", "warn")
    F = vol_forecast(V)
    if F is None: h += note("Forecast disabled: fewer than 42 full days of history.", "warn")
    else:
        h += SEC("Order-volume forecast (next 7 days)") + note("Two simple baselines were compared on the last 14 days (WAPE): same-weekday mean %s vs trailing 7-day mean %s → using <b>%s</b>. No ML is used here because 3 months of history cannot support seasonality learning." % (pc(F["wa"], 1), pc(F["wb"], 1), F["best"]), "ai"); fc = F["fc"].copy(); fc.index = [i.strftime("%a %d %b") for i in fc.index]; fc["Total"] = fc.sum(1); h += tbl(fc.reset_index().rename(columns={"index": "Day"}), [("Day", "Day")] + [(c, c) for c in fc.columns], {c: fx for c in fc.columns})
        if len(st): capd = sg.groupby("warehouse_id").cap_h.sum(); bl = {}; hh = ["Projected backlog (workload hours above capacity, cumulative)"]; pr = pd.DataFrame({w: np.maximum.accumulate(np.cumsum(np.maximum(F["fc"][w].values * hm - capd.get(w, np.inf), 0))) for w in F["fc"].columns}); pr.index = fc.index[:7]; h += SEC(hh[0]) + tbl(pr.reset_index().rename(columns={"index": "Day"}), [("Day", "Day")] + [(c, c) for c in pr.columns], {c: fx for c in pr.columns})
        f = go.Figure([go.Scatter(x=F["ho_act"].index, y=F["ho_act"].values, name="Actual", line=dict(color=C["fg"], width=3)), go.Scatter(x=F["ho_pred"].index, y=F["ho_pred"].values, name="Forecast (baseline)", line=dict(color=C["am"], dash="dot", width=3))]); f = sty(f, 320, "Forecast vs actual daily orders, last 14 days (all warehouses)")
    return h, f
@safe(2)
def render_cmp(wh, zn, cr, d1, d2):
    X = G["X"]; S = X["S"]; V = vw(wh, zn, cr, d1, d2); d = V[V["deliv"]].copy(); d["bucket"] = np.select([d.delay_h <= 0, d.delay_h < 12, d.delay_h < 24], ["On time", "Late <12 h", "Late 12–24 h"], "Late >24 h"); order = ["On time", "Late <12 h", "Late 12–24 h", "Late >24 h"]
    b = d.groupby("bucket").agg(n=("comp", "size"), c=("comp", "sum")).reindex(order).reset_index(); b["rate"] = b.c / b.n; h = banner() + note("Complaint rate = delivered orders with a complaint ÷ delivered orders. Complaints are linked to orders by order_id; <b>association only</b>: no churn or revenue loss is claimed.", "info") + tbl(b, [("bucket", "Delay severity"), ("n", "Orders"), ("c", "Complaints"), ("rate", "Complaint rate")], dict(n=nf, c=nf, rate=lambda v: pc(v, 1)))
    cat = X["D"]["comp"]; cat = cat[cat.order_id.isin(V.order_id)].category.value_counts(); h += note("Complaint categories: " + ("; ".join("%s %d" % (k, v) for k, v in cat.items()) if len(cat) else "none (no complaints table or category)") + ". Late-delivery complaints are those whose category contains 'late'.", "info")
    for nm, c in (("Warehouse", "warehouse_id"), ("Carrier", "carrier_id"), ("Shipping mode", "shipping_mode"), ("Category", "category")): t = d.groupby(c).agg(n=("comp", "size"), r=("comp", "mean"), lr=("late", "mean")).reset_index().sort_values("r", ascending=False); h += SEC("Complaint rate by " + nm.lower()) + tbl(t, [(c, nm), ("n", "Delivered"), ("r", "Complaint rate"), ("lr", "Late rate")], dict(n=nf, r=lambda v: pc(v, 1), lr=lambda v: pc(v, 1)))
    if "customer_id" in d and d.customer_id.notna().any(): rep = d[d.late == 1].groupby("customer_id").size(); rep = rep[rep >= 2]; h += SEC("Repeat delay incidents") + note("%d customers experienced 2+ late deliveries in this view%s." % (len(rep), (" (e.g. " + ", ".join(rep.sort_values(ascending=False).head(3).index) + ")") if len(rep) else ""), "warn" if len(rep) else "info")
    if V.prio_f.any(): pr = V[V.prio_f & ((V.late == 1) | V.breach_open)].sort_values("order_ts", ascending=False).head(15); h += SEC("Priority customers with late or breached orders: proactive outreach") + (tbl(pr, [("order_id", "Order"), ("customer_id", "Customer"), ("warehouse_id", "Warehouse"), ("promised_ts", "Promised"), ("status", "Status"), ("comp", "Complaint filed")], dict(promised_ts=dts, comp=lambda v: "yes" if v else "no")) + note("Suggested: contact these customers before they complain; escalate open breached orders to the warehouse lead.", "ai") if len(pr) else note("No priority-customer orders are late or breached.", "ok"))
    else: h += note("No valid priority_customer field: high-priority customer analysis is skipped.", "info")
    f = go.Figure(go.Bar(x=b.bucket, y=b.rate, marker_color=[C["sg"], C["am"], C["co"], C["co"]])); f = sty(f, 320, "Complaint rate by delay severity"); f.update_yaxes(tickformat=".0%"); return h, f
@safe(1)
def render_act(wh, zn, cr, d1, d2):
    X = G["X"]; V = vw(wh, zn, cr, d1, d2); R = recs(V, X["S"], fsc(wh, zn, cr))
    if R.empty: return banner() + note("No segment is statistically elevated and no High-risk orders exist in this view: no action is generated.", "ok")
    h = banner() + note("Actions are generated from the active data and ranked by estimated excess late orders (late orders above what the overall rate predicts). No improvement percentage is promised. Human review is required.", "ai")
    for r in R.head(8).itertuples(): h += "<div class='tw'><table><tr><th colspan='2'>#%d · %s &nbsp; %s</th></tr>" % (r.rank, esc(r.problem), pill(r.urgency)) + "".join("<tr><td style='width:210px'><b>%s</b></td><td>%s</td></tr>" % (a, esc(b)) for a, b in [("Affected segment", r.segment), ("Evidence (measured)", r.evidence), ("Recommended intervention", r.action), ("Responsible team", r.team), ("Expected direction of impact", r.direction), ("Implementation effort", r.effort), ("Data limitations and assumptions", r.limits), ("How to measure success", r.kpi)]) + "</table></div>"
    return h
@safe(3)
def render_scn(wh, zn, cr, d1, d2, st_, pr_, wa_, ho_, tr_, sh_, src, dst, pen, rate, impl):
    X = G["X"]; D = X["D"]; V = vw(wh, zn, cr, d1, d2); p = dict(staff=float(st_ or 0), proc=float(pr_ or 0), wait=float(wa_ or 0), hold=float(ho_ or 0), transit=float(tr_ or 0), shift=float(sh_ or 0), src=src or "", dst=dst or "", pen=float(pen or 0), rate=float(rate or 0), impl=float(impl or 0)); sf = D["staff"]; sh_avg = float(sf.staff_scheduled.mean() * 1) if len(sf) else 3.0
    r = simulate(V, p, sh_avg); G["scn"] = pd.DataFrame([dict(metric=m, baseline=a, scenario=b) for m, a, b in [("late_rate", r["late0"], r["late1"]), ("late_orders", r["b0"], r["b1"]), ("mean_delay_late_h", r["dl0"], r["dl1"]), ("mean_pre_dispatch_h", r["pre0"], r["pre1"]), ("expected_complaints", r["ce0"], r["ce1"]), ("cost", 0.0, r["cost"])]]); rows = [("Late-delivery rate", pc(r["late0"], 1), "%s (95%% bootstrap %s–%s)" % (pc(r["late1"], 1), pc(r["ci1"][0], 1), pc(r["ci1"][1], 1))), ("Late deliveries (SLA breaches among simulated orders)", nf(r["b0"]), nf(r["b1"])), ("Mean delay of late orders (h)", fx(r["dl0"]), fx(r["dl1"])), ("Mean time to dispatch (h) — backlog proxy", fx(r["pre0"]), fx(r["pre1"])), ("Expected complaints (from observed delay-bucket rates)", fx(r["ce0"], 0), fx(r["ce1"], 0)), ("Estimated cost (assumptions below)", money(0), money(r["cost"]))]
    h = banner() + note("<b>Hypothetical, model-based estimate, not an observed result.</b> Each of the %d delivered orders with complete timestamps is replayed with its stage durations changed: processing ÷ (1 + extra staff %%) in the 4 busiest release hours, then × (1 − reduction); dispatch wait and held-order hold reduced as set; transit scaled; moved orders take the target warehouse's median hold, processing and wait and a transit penalty. A new delivery time is compared with the original promise. <b>No queue feedback</b> is modelled (a faster site does not change other orders' waits). %d orders lacking timestamps are excluded." % (r["n"], r["excl"]), "warn")
    h += tbl(pd.DataFrame(rows, columns=["m", "b", "s"]), [("m", "Measure"), ("b", "Baseline (replayed)"), ("s", "Scenario")], dict(b=str, s=str)) + note("Cost = extra staff %% × 4 peak hours × %.1f staff per warehouse (from the staffing table, or 3 if missing) × warehouses × days × %s per staff-hour + implementation cost. Baseline late-rate interval: %s–%s. Improvements shown are estimates under these assumptions, not guaranteed savings." % (sh_avg, money(p["rate"]), pc(r["ci0"][0], 1), pc(r["ci0"][1], 1)), "info")
    f1 = go.Figure([go.Bar(x=["Baseline", "Scenario"], y=[r["late0"], r["late1"]], marker_color=[C["gg"], C["fg"]], error_y=dict(type="data", symmetric=False, array=[r["ci0"][1] - r["late0"], r["ci1"][1] - r["late1"]], arrayminus=[r["late0"] - r["ci0"][0], r["late1"] - r["ci1"][0]]))]); f1 = sty(f1, 320, "Late-delivery rate, baseline vs scenario (hypothetical, 95% bootstrap)"); f1.update_yaxes(tickformat=".0%")
    w = r["wh"]; f2 = go.Figure([go.Bar(x=w.w, y=w.a, name="Baseline", marker_color=C["gg"]), go.Bar(x=w.w, y=w.b, name="Scenario", marker_color=C["fg"])]); f2.update_layout(barmode="group"); f2 = sty(f2, 320, "Late rate by warehouse, baseline vs scenario"); f2.update_yaxes(tickformat=".0%"); return h, f1, f2
@safe(2)
def render_dq():
    X = G["X"]; D = X["D"]; h = banner() + note("Rows read: " + ", ".join("%s %s" % (TN[k], nf(v)) for k, v in D["rows_in"].items()) + ". Raw uploads stay unchanged; a validated copy is analysed. Excluded rows: <b>%d</b> (listed; nothing is dropped silently)." % sum(c for _, _, c in D["excluded"]), "ok")
    h += "".join(note(esc(m), "err" if l == "error" else "warn" if l == "warn" else "info") for l, m in D["issues"]) or note("No validation issues.", "ok")
    if D["excluded"]: h += SEC("Excluded rows by reason") + tbl(pd.DataFrame(D["excluded"], columns=["t", "w", "c"]), [("t", "Table"), ("w", "Reason"), ("c", "Rows")], dict(c=nf))
    return h, SEC("Self-test results (run at start-up)") + "".join(note(("PASS " if ok else "FAIL ") + esc(g + ": " + n) + " — " + esc(d), "ok" if ok else "err") for g, n, ok, d in TESTS)

# ------------------------------ strategist and exports ------------------------------
def ask(q, _=None):
    if "X" not in G: return "Load data first."
    X = G["X"]; O = X["O"]; S = X["S"]; d = O[O["deliv"]]; q = (q or "").strip() or "Where is the biggest bottleneck?"; ql = q.lower(); parts = []; k = kpis(O); R = recs(O, S, X["risk"]["scored"])
    if re.search(r"bottleneck|where|stage|accumul|cause|why", ql): lt = d[d.late == 1]; ex = lt[["x_hold", "x_proc", "x_wait", "x_transit"]].sum().sort_values(ascending=False); lab = {"x_hold": "release hold", "x_proc": "warehouse processing", "x_wait": "dispatch waiting", "x_transit": "transit / final mile"}; parts.append(("stage_analysis", "On late orders the largest excess time over target is in %s (%s h of %s h total excess). These are associations, not proven causes." % (lab[ex.index[0]], nf(ex.iloc[0]), nf(ex.sum()))))
    if re.search(r"warehouse|worst|site", ql): t = rate_tbl(d, "warehouse_id", S["min_n"]); parts.append(("warehouses", "; ".join("%s late %s (95%% CI %s–%s, n=%d)" % (r.seg, pc(r.rate, 1), pc(r.lo, 1), pc(r.hi, 1), r.n) for r in t.itertuples())))
    if re.search(r"carrier|route|transit", ql): t = rate_tbl(d, "carrier_id", S["min_n"]); u = rate_tbl(d, "route_id", S["min_n"]).head(3); parts.append(("carriers_routes", "Carriers: " + "; ".join("%s %s" % (r.seg, pc(r.rate, 1)) for r in t.itertuples()) + ". Worst routes: " + "; ".join("%s %s (n=%d)" % (r.seg, pc(r.rate, 1), r.n) for r in u.itertuples())))
    if re.search(r"risk|at risk|early|predict|warn", ql): sc = X["risk"]["scored"]; parts.append(("delivery_risk", ("%d open orders are High risk (%s). Top: %s." % (int((sc.band == "High").sum()), "model" if X["risk"]["mode"] == "ml" else "rules-based, not validated", ", ".join(sc.head(3).order_id))) if len(sc) else "no open released orders"))
    if re.search(r"complain|customer", ql): parts.append(("complaints", "Complaint rate %s among delivered orders; late orders %s vs on-time %s." % (pc(k["comp_rate"], 1), pc(d[d.late == 1].comp.mean(), 1), pc(d[d.late == 0].comp.mean(), 1))))
    if re.search(r"first|action|recommend|do next|priority|fix", ql): parts.append(("actions", " | ".join("#%d %s → %s" % (r.rank, r.segment, r.action) for r in R.head(3).itertuples()) if len(R) else "no segment is statistically elevated"))
    if re.search(r"limit|caveat|trust", ql): parts.append(("limitations", "Synthetic or uploaded data only; undelivered orders are censored; associations are not causes; risk scores are estimates; scenarios are hypothetical with no queue feedback; no live traffic or tracking."))
    if not parts: parts = [("summary", "%s orders, on-time %s, late %s, %s SLA breaches, processing %s h, dispatch wait %s h, transit %s h." % (nf(k["total"]), pc(k["ontime"], 1), pc(k["late_rate"], 1), nf(k["breaches"]), fx(k["proc"]), fx(k["wait"]), fx(k["transit"])))]
    return "### Answer (deterministic, computed from the loaded data)\n" + "\n".join("- **%s**: %s" % (n, t) for n, t in parts) + "\n\n*Every figure is calculated from the active dataset; scenarios are hypothetical; operations managers must review before acting.*"
def export_all(wh=None, zn=None, cr=None, d1="", d2=""):
    if "X" not in G: return None, note("Load data first.", "info")
    X = G["X"]; S = X["S"]; V = vw(wh, zn, cr, d1, d2); d = V[V["deliv"]]; out = []; sc = fsc(wh, zn, cr)
    def w(df, fn):
        df.to_csv(fn, index=False)
        if os.path.exists(fn): out.append(fn)
    k = kpis(V); w(pd.DataFrame([k]), "delivr_kpis.csv"); w(pd.concat([rate_tbl(d, c, S["min_n"]).assign(dimension=n) for n, c in DIMS.items()]), "delivr_delay_rates.csv"); w(sc, "delivr_risk_scores.csv"); R = recs(V, S, sc); w(R, "delivr_recommendations.csv")
    b = d.assign(bucket=np.select([d.delay_h <= 0, d.delay_h < 12, d.delay_h < 24], ["On time", "Late <12 h", "Late 12-24 h"], "Late >24 h")).groupby("bucket").agg(n=("comp", "size"), complaints=("comp", "sum")).reset_index(); w(b, "delivr_complaints_by_delay.csv")
    if "scn" in G: w(G["scn"], "delivr_scenario_comparison.csv")
    md = ["# DELIVR AI — Executive summary", "", "_Synthetic demo data._" if G.get("demo") else "", "", "- %s orders; on-time %s; late %s; %s SLA breaches (late deliveries plus open orders past promise)." % (nf(k["total"]), pc(k["ontime"], 1), pc(k["late_rate"], 1), nf(k["breaches"])), "- Mean stage times: processing %s h, dispatch wait %s h, transit %s h; complaint rate %s." % (fx(k["proc"]), fx(k["wait"]), fx(k["transit"]), pc(k["comp_rate"], 1)), "- High-risk open orders: %d (%s)." % (int((sc.band == "High").sum()) if len(sc) else 0, X["risk"]["mode"]), "", "## Ranked actions (proposed; human review required)"] + ["- #%d %s: %s (urgency %s). Evidence: %s" % (r.rank, r.segment, r.action, r.urgency, r.evidence) for r in R.head(6).itertuples()] + ["", "## Limitations", "Undelivered orders are censored. Findings are associations, scenarios are hypothetical, risk scores are estimates."]
    open("delivr_executive_summary.md", "w", encoding="utf-8").write("\n".join(md)); out.append("delivr_executive_summary.md"); return [f for f in out if os.path.exists(f)], note("Generated %d files from the active dataset and filters." % len(out), "ok")

# ------------------------------ self-tests ------------------------------
def demo_X(seed=5, S=None):
    S = dict(S or S0); key = ("X", seed, tuple(S.values()))
    if key not in _DEMO:
        D = load_all(get_demo_raw(seed)); O, asof = prep(D, S); _DEMO[key] = dict(D=D, S=S, O=O, asof=asof, risk=fit_risk(O, S, asof))
    return _DEMO[key]
def miniraw():
    t = pd.Timestamp("2026-03-02 08:00"); rows = []; ev = []
    def mk(oid, hold, proc, wait, trn, prom, status="Delivered", dlv=True, od=0.0):
        o = t + pd.Timedelta(hours=od); rel = o + pd.Timedelta(hours=hold); pe = rel + pd.Timedelta(hours=proc); dp = pe + pd.Timedelta(hours=wait) if wait is not None else pd.NaT; dv = dp + pd.Timedelta(hours=trn) if dlv else pd.NaT
        rows.append(dict(order_id=oid, order_ts=o, warehouse_id="W1", promised_ts=o + pd.Timedelta(hours=prom), delivery_ts=dv, status=status)); ev.append(dict(order_id=oid, release_ts=rel, pack_end=pe if status != "Cancelled" else pd.NaT, dispatch_ts=dp))
    mk("o1", 1, 2, 1, 6, 10); mk("o2", 1, 2, 1, 6, 9); mk("o3", 1, 2, 1, 6, 20); mk("o4", 1, 2, None, 0, 5, "Processing", False); mk("o5", 1, 2, 1, 6, 30, "Cancelled", False); mk("o7", 1, 2, 1, 6, 30, od=-20.0); o = pd.DataFrame(rows); o.loc[o.order_id == "o7", "delivery_ts"] = t - pd.Timedelta(hours=1)
    o = pd.concat([o, o.iloc[[0]]], ignore_index=True); return dict(orders=o, events=pd.DataFrame(ev))
def run_selftests(verbose=True):
    if TESTS: return TESTS
    T = []; saved = dict(G)
    def chk(g, n, fn):
        try: ok, d = fn(); T.append((g, n, bool(ok), d))
        except Exception as e: T.append((g, n, False, type(e).__name__ + ": " + str(e)[:90]))
    def mini(): D = load_all(miniraw()); O, a = prep(D, S0); return D, O, a
    def t1():
        r = miniraw(); r["orders"] = r["orders"].drop(columns=["order_ts"])
        try: load_all(r); return False, "no error"
        except ValueError as e: return "order_ts" in str(e), "missing column reported with guidance"
    def t2(): D = load_all(miniraw()); w = " | ".join(m for _, m in D["issues"]); return "duplicate" in w and "delivery before order time" in w and len(D["orders"]) == 5, "duplicate id and delivery-before-order excluded and reported (5 valid orders kept)"
    def t3(): D, O, a = mini(); k = kpis(O); return abs(k["ontime"] - 2 / 3) < 1e-9 and abs(k["late_rate"] - 1 / 3) < 1e-9 and abs(k["avg_delay"] - 1) < 1e-9 and k["breaches"] == 2 and k["deliv"] == 3 and k["open"] == 1 and k["canc"] == 1, "boundary delivery counts as on-time; 3 delivered (1 late by 1 h); the open order is censored, not on-time; breaches = 1 late + 1 open past promise"
    def t4(): D, O, a = mini(); r = O[O.order_id == "o4"].iloc[0]; o1 = O[O.order_id == "o1"].iloc[0]; return pd.isna(r.wait_h) and abs(o1.hold_h - 1) < 1e-9 and abs(o1.proc_h - 2) < 1e-9 and abs(o1.wait_h - 1) < 1e-9 and abs(o1.transit_h - 6) < 1e-9, "missing dispatch gives NaN wait (not 0); stage durations 1/2/1/6 h computed correctly"
    def t5(): lo, hi = wilson(3, 20); return 0 <= lo <= .15 <= hi <= 1 and np.isnan(wilson(0, 0)[0]), "Wilson interval brackets the rate; empty group gives n/a"
    def t6(): X = demo_X(); return not (set(X["risk"]["cols"]) & FORBID) and not any(("deliver" in c or "dispatch" in c or "pack" in c) for c in X["risk"]["cols"]), "no post-release variable among %d model features" % len(X["risk"]["cols"])
    def t7(): X = demo_X(); R = X["risk"]; s = R["split"]; return R["ok"] and s["train_max_delivery"] <= s["cut"] <= s["test_min_order"], "latest training delivery %s ≤ cut-off %s ≤ first test order %s" % (dts(s["train_max_delivery"]), dts(s["cut"]), dts(s["test_min_order"]))
    def t8(): D, O, a = mini(); R = fit_risk(O, S0, a); return R["mode"] == "rules" and "needs 300+" in R["why"], "tiny dataset: supervised model disabled with an explanation, rules fallback"
    def t9(): R = demo_X()["risk"]; return R["ok"] and R["ap"] > R["prev"] and 0 <= R["auc"] <= 1, "demo PR-AUC %.2f vs prevalence %.2f, baseline %.2f, ROC-AUC %.2f (synthetic labels)" % (R["ap"], R["prev"], R["ap_base"], R["auc"])
    def t10(): X = demo_X(); R = recs(X["O"], X["S"], X["risk"]["scored"]); got = " ".join(R.head(8).segment); return len(R) > 0 and (R.score.diff().dropna() <= 1e-9).all() and any(k in got for k in ("WH2", "WH3", "CAR3", "North-East")) and all(c in R.columns for c in RC), "ranked by excess late orders; planted drivers surfaced: " + got[:70]
    def t11(): X = demo_X(); d = X["O"][X["O"]["deliv"]]; t = rate_tbl(d, "warehouse_id"); return len(t) == 4 and (t.lo <= t.rate + 1e-9).all() and (t.rate <= t.hi + 1e-9).all() and (t.n.sum() == len(d)), "4 warehouses compared; intervals bracket rates; counts add up to %d delivered" % len(d)
    def t12(): X = demo_X(); V = X["O"]; a = simulate(V, dict(P0)); b = simulate(V, dict(P0, proc=50, wait=50)); c = simulate(V, dict(P0, proc=50, wait=50, hold=50)); return abs(a["late0"] - a["late1"]) < 1e-12 and b["late1"] <= a["late0"] and c["late1"] <= b["late1"] + 1e-12, "zero levers reproduce the baseline; late rate %.1f%% → %.1f%% → %.1f%% as levers are added" % (100 * a["late0"], 100 * b["late1"], 100 * c["late1"])
    def t13(): X = demo_X(); G.clear(); G.update(X=X, S=X["S"], demo=True); f, _ = export_all(None, None, None, "", ""); n = len(pd.read_csv("delivr_risk_scores.csv")); G.clear(); G.update(saved); return n == len(X["risk"]["scored"]) and "delivr_executive_summary.md" in f, "%d files; risk export has %d rows matching the analysis" % (len(f), n)
    def t14(): X = demo_X(); G.clear(); G.update(X=X, S=X["S"], demo=True); a = ask("Where is the biggest bottleneck?"); G.clear(); G.update(saved); return "deterministic" in a, "strategist answers from calculated results without any API"
    for g, n, f in [("validation", "missing column", t1), ("validation", "duplicates and impossible durations", t2), ("kpi", "SLA boundary, censoring, breaches", t3), ("kpi", "missing timestamps and stage durations", t4), ("kpi", "Wilson intervals", t5), ("prediction", "leakage-free feature set", t6), ("prediction", "chronological split", t7), ("prediction", "small data fallback", t8), ("prediction", "imbalanced demo beats no-skill", t9), ("root-cause", "recommendation ranking and planted drivers", t10), ("root-cause", "warehouse comparison", t11), ("simulation", "scenario recalculation", t12), ("reporting", "CSV export correctness", t13), ("strategist", "works without an API key", t14)]: chk(g, n, f)
    G.clear(); G.update(saved); TESTS[:] = T
    if verbose:
        for g, n, ok, d in T: print(("PASS " if ok else "FAIL ") + g + ": " + n + " :: " + d)
        print("Result: %d of %d self-tests passed." % (sum(1 for t in T if t[2]), len(T)))
    return TESTS

# ------------------------------ project export ------------------------------
METH = """### Methodology
- **KPIs:** on-time = delivered at or before the promise ÷ delivered orders. Undelivered orders are censored (excluded from rates, never on-time); cancelled are excluded; SLA breaches = late deliveries + open orders past promise.
- **Stages:** hold (order→release), processing (release→packed), dispatch wait (packed→dispatch), transit (dispatch→delivery). Targets are editable; transit target = route plan (× Express factor).
- **Root cause:** late rates with 95% Wilson intervals, Pareto of excess hours, on-time vs late comparison, adjusted logistic odds ratios with a rough bootstrap. All are associations, not causes.
- **Delivery-risk model:** prediction at order release; HistGradientBoosting with chronological split, label-known-at-cut-off training, balanced weights when late < 20%, isotonic calibration when it improves Brier, PR-AUC against a promise-slack baseline. Rules fallback when data is too small.
- **Recommendations:** segments whose late-rate interval is above the overall rate, ranked by excess late orders, mapped to the dominant stage.
- **Scenario:** per-order replay of stage durations with documented levers, no queue feedback, bootstrap interval.
- **Not included:** OR-Tools vehicle-routing demo, PDF export, Streamlit (Gradio is used), ML staffing forecast, live traffic or tracking.
"""
LICENSE = "MIT License\n\nCopyright (c) 2026 DELIVR AI contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
README = """# DELIVR AI — Delivery Performance & Operations Intelligence
Find the delay. Fix the process.

Diagnoses rising delivery delays across the order journey (release, picking, dispatch, transit), predicts which open orders will miss their promise, and ranks evidence-based operational actions. Bundled data is **fictional synthetic data** with planted patterns (warehouse backlog, stock holds, carrier-zone variability).

## Architecture
@@@mermaid
flowchart LR
  A[CSV or synthetic data] --> B[Validation + duration checks]
  B --> C[Stage durations + KPIs]
  C --> D[Root cause: rates, Pareto, associations]
  C --> E[Delivery-risk model: chronological, calibrated]
  D --> F[Ranked action queue]
  E --> F
  C --> G[Scenario replay]
  F --> H[Strategist, exports, summary]
  G --> H
@@@

## Run
@@@
pip install -r requirements.txt
python app.py
@@@
In Google Colab paste `app.py` into one cell. The `src/` modules in the brief are consolidated into `app.py`.

## Tests and limits
`pip install pytest && pytest tests` runs the built-in self-tests. See `docs/`. Not included: OR-Tools routing demo, PDF export. Associations are not causes; scenarios are hypothetical; screenshots: add yours to `screenshots/`.
""".replace("@@@", "`" * 3)
TFILE = "from app import run_selftests\n\n\ndef test_%s():\n    res = [r for r in run_selftests(verbose=False) if r[0] in %s]\n    assert res and all(r[2] for r in res), res\n"
def _src():
    try: return open(__file__, encoding="utf-8").read()
    except Exception: pass
    try:
        for s in list(get_ipython().user_ns.get("In") or [])[::-1]:
            if "DELIVR AI — Delivery Performance" in s and "def main" in s: return s
    except Exception: pass
    return None
def write_samples():
    raw = get_demo_raw(5); os.makedirs("sample_data", exist_ok=True); out = []; fn = dict(orders="orders", events="warehouse_events", inv="inventory", routes="routes", staff="staffing", comp="complaints")
    for k, n in fn.items(): p = "sample_data/%s.csv" % n; raw[k].to_csv(p, index=False); out.append(p)
    return out
def build_project(files):
    root = "delivr-ai-delivery-intelligence"
    for d in ["sample_data", "tests", "docs", "screenshots"]: os.makedirs(os.path.join(root, d), exist_ok=True)
    src = _src()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    for f in files: open(os.path.join(root, f), "wb").write(open(f, "rb").read())
    W = {"requirements.txt": "gradio>=4.44,<6\nplotly\nscikit-learn\npandas\nnumpy\n", "README.md": README, "LICENSE": LICENSE, ".gitignore": "__pycache__/\n.ipynb_checkpoints/\n*.zip\ndelivr_*.csv\ndelivr_*.md\n.env\n", "docs/methodology.md": METH,
         "docs/data_dictionary.md": "# Data dictionary\norders (order_id, customer_id, order_ts, warehouse_id, zone, route_id, carrier_id, shipping_mode, category, sku, quantity, priority_customer, promised_ts, delivery_ts, status, failed_attempts, shift); warehouse_events (order_id, release_ts, pick_start, pack_end, dispatch_ts, stock_hold, pick_exception, split_shipment); inventory (warehouse_id, sku, available_units); routes (route_id, warehouse_id, zone, planned_transit_h, distance_km); staffing (warehouse_id, date, shift, staff_scheduled, productive_hours_per_staff); complaints (complaint_id, order_id, complaint_ts, category). Samples are synthetic.\n",
         "docs/assumptions_and_limitations.md": "# Assumptions and limitations\nSynthetic data with planted patterns. Undelivered orders are censored. Findings are associations. The scenario replay has no queue feedback. Risk scores are estimates. No live tracking or traffic. Human review is required before acting.\n", "screenshots/README.md": "# Screenshots\nAdd your own screenshots here. None are included.\n"}
    for nm, g in [("data_validation", "validation"), ("delivery_kpis", "kpi"), ("root_causes", "root-cause"), ("delay_prediction", "prediction"), ("simulation", "simulation")]: W["tests/test_%s.py" % nm] = TFILE % (nm, "('%s',)" % g)
    for fn, tx in W.items(): open(os.path.join(root, fn), "w", encoding="utf-8").write(tx)
    z = "delivr_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

# ------------------------------ UI ------------------------------
CSS = """
.gradio-container{background:#F5F2E9!important;color:#263B37!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1380px!important}
.hdr{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;padding:24px 30px;border-radius:14px;background:#193B37;color:#F5F2E9;margin-bottom:12px}
.brand{display:flex;gap:16px;align-items:center}.logo{width:52px;height:52px;border-radius:12px;background:#BCE8B0;color:#193B37;font-weight:800;font-size:22px;display:flex;align-items:center;justify-content:center}
.hdr h1{margin:0;font-size:28px;letter-spacing:2px;color:#F5F2E9}.hdr h1 span{color:#BCE8B0}.hdr p{margin:4px 0 0;color:#D6C8F0;font-style:italic}.badges span{display:inline-block;margin:3px;padding:4px 11px;border-radius:6px;background:#ffffff1a;color:#F5F2E9;font-size:12px}
button[role=tab]{background:#193B37!important;color:#F5F2E9!important;border-radius:8px 8px 0 0!important;margin-right:2px}button[role=tab][aria-selected=true]{background:#E8B96A!important;color:#193B37!important;font-weight:700}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:10px 0}.kc{background:#fff;border:1px solid #193B3726;border-radius:10px;padding:14px 16px}.kl{color:#5d726c;font-size:11px;text-transform:uppercase;letter-spacing:1.1px}.kv{font-size:22px;font-weight:700;margin:6px 0 4px}.kd{color:#5d726c;font-size:12px}
.note{background:#fff;padding:10px 14px;border-radius:8px;margin:8px 0;color:#263B37;font-size:14px;line-height:1.5;border:1px solid #193B3718}.sec{font-size:15px;font-weight:700;color:#193B37;margin:18px 0 6px;text-transform:uppercase;letter-spacing:1px}
.pill{display:inline-block;border-radius:99px;padding:2px 10px;font-size:11px;font-weight:700;color:#193B37;white-space:nowrap}.tw{overflow-x:auto;border-radius:8px;border:1px solid #193B3726;margin:8px 0;background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#193B37;color:#F5F2E9;text-align:left;padding:9px 10px;font-weight:600}td{padding:8px 10px;border-top:1px solid #193B3712;color:#263B37;vertical-align:top}tr:hover td{background:#BCE8B022}
button.primary{background:#193B37!important;color:#F5F2E9!important;border:none!important;font-weight:700!important}@media(max-width:640px){.hdr h1{font-size:21px}.kv{font-size:19px}}
"""
HEADER = "<div class='hdr'><div class='brand'><div class='logo'>DV</div><div><h1>DELIVR <span>AI</span></h1><p>Find the delay. Fix the process.</p></div></div><div class='badges'><span>Associations, not causes</span><span>Scenarios are hypothetical</span><span>Undelivered orders are censored</span><span>Works offline</span></div></div>"
def mkS(*a): return dict(t_hold=float(a[0]), t_proc=float(a[1]), t_wait=float(a[2]), exp_f=float(a[3]), hmin=float(a[4]), thr_med=float(a[5]), thr_hi=float(a[6]), min_n=int(a[7]))
def run_all(*a):
    try:
        S = mkS(*a[:8]); files = a[8:14]; mapping = a[14]; seed = int(a[15] or 5); demo = files[0] is None and files[1] is None
        if demo and not mapping: X = demo_X(seed, S)
        else: D = load_all(get_demo_raw(seed) if demo else dict(zip(TN.keys(), [read_csv_any(f) if f is not None else None for f in files])), mapping); O, asof = prep(D, S); X = dict(D=D, S=S, O=O, asof=asof, risk=fit_risk(O, S, asof))
        keep = G.get("scn"); G.clear(); G.update(X=X, S=S, demo=demo)
        if keep is not None: G["scn"] = keep
        k = kpis(X["O"]); return note(("Fictional demo data analysed. " if demo else "Your data analysed. ") + "%s orders, late rate %s, %s open orders, risk mode: %s." % (nf(k["total"]), pc(k["late_rate"], 1), nf(k["open"]), X["risk"]["mode"]), "ok")
    except Exception as e: return note("Could not complete the analysis: " + esc(e), "err")
def refresh():
    if "X" not in G: return tuple(gr.update() for _ in range(5))
    O = G["X"]["O"]; w = sorted(O.warehouse_id.unique()); return gr.update(choices=w, value=[]), gr.update(choices=sorted(O.zone.unique()), value=[]), gr.update(choices=sorted(O.carrier_id.unique()), value=[]), gr.update(choices=[("(none)", "")] + [(x, x) for x in w], value=""), gr.update(choices=[("(none)", "")] + [(x, x) for x in w], value="")
def build_ui(zip_path, sfiles):
    with gr.Blocks(css=CSS, title="DELIVR AI", theme=gr.themes.Base(primary_hue="green", neutral_hue="stone")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("Settings: stage targets, assumptions, risk thresholds (press Recalculate after changing)", open=False):
            with gr.Row():
                s1 = gr.Number(2.0, label="Hold target (h)"); s2 = gr.Number(4.0, label="Processing target (h)"); s3 = gr.Number(4.0, label="Dispatch wait target (h)"); s4 = gr.Number(.6, label="Express transit factor"); s5 = gr.Number(36, label="Handling minutes per order (capacity)")
            with gr.Row():
                s6 = gr.Slider(.05, .5, .15, step=.01, label="Medium-risk threshold"); s7 = gr.Slider(.1, .8, .30, step=.01, label="High-risk threshold"); s8 = gr.Slider(10, 100, 30, step=5, label="Minimum sample size for a finding"); seed = gr.Number(5, label="Demo seed"); run_btn = gr.Button("Recalculate", variant="primary"); rg_btn = gr.Button("Regenerate demo dataset")
        with gr.Accordion("Filters (apply to every page)", open=True):
            with gr.Row():
                f_w = gr.Dropdown([], multiselect=True, label="Warehouse"); f_z = gr.Dropdown([], multiselect=True, label="Zone"); f_c = gr.Dropdown([], multiselect=True, label="Carrier"); f_d1 = gr.Textbox(label="From (YYYY-MM-DD)"); f_d2 = gr.Textbox(label="To (YYYY-MM-DD)"); ap_btn = gr.Button("Apply filters", variant="primary")
        status = gr.HTML(note("Loading the fictional logistics data, training the risk model and running self-tests (about 1-2 minutes)...", "info")); FIN = [f_w, f_z, f_c, f_d1, f_d2]
        with gr.Tabs():
            with gr.Tab("Control Tower"):
                ct_h = gr.HTML(); ct1 = gr.Plot()
                with gr.Row():
                    ct2 = gr.Plot(); ct3 = gr.Plot()
            with gr.Tab("Root-Cause Intelligence"):
                with gr.Row():
                    rc_dim = gr.Dropdown(list(DIMS), value="Warehouse", label="Drill-down dimension"); rc_val = gr.Textbox(label="Value (e.g. WH3, CAR3, North-East)"); rc_btn = gr.Button("Drill down", variant="primary")
                rc_h = gr.HTML()
                with gr.Row():
                    rc1 = gr.Plot(); rc2 = gr.Plot()
            with gr.Tab("Delivery Risk"):
                rk_h = gr.HTML()
                with gr.Row():
                    rk1 = gr.Plot(); rk2 = gr.Plot()
            with gr.Tab("Warehouse & Fulfilment"):
                wh_h = gr.HTML(); wh1 = gr.Plot()
                with gr.Row():
                    wh2 = gr.Plot(); wh3 = gr.Plot()
            with gr.Tab("Route & Carrier"):
                rt_h = gr.HTML()
                with gr.Row():
                    rt1 = gr.Plot(); rt2 = gr.Plot()
            with gr.Tab("Inventory & Allocation"):
                iv_h = gr.HTML(); iv1 = gr.Plot()
            with gr.Tab("Staffing & Capacity"):
                sf_h = gr.HTML(); sf1 = gr.Plot()
            with gr.Tab("Complaints"):
                cm_h = gr.HTML(); cm1 = gr.Plot()
            with gr.Tab("Action Centre"):
                ac_h = gr.HTML()
            with gr.Tab("Scenario Lab"):
                gr.Markdown("Change assumptions and run. Results are **hypothetical** estimates from a per-order replay.")
                with gr.Row():
                    c1 = gr.Slider(0, 100, 0, step=5, label="Extra staff in peak hours (%)"); c2 = gr.Slider(0, 60, 0, step=5, label="Processing time reduction (%)"); c3 = gr.Slider(0, 80, 0, step=5, label="Dispatch wait reduction (cut-off review, %)"); c4 = gr.Slider(0, 80, 0, step=5, label="Stock-hold reduction (more inventory, %)"); c5 = gr.Slider(-30, 50, 0, step=5, label="Transit time change (%)")
                with gr.Row():
                    c6 = gr.Slider(0, 60, 0, step=5, label="Share of orders moved between warehouses (%)"); c7 = gr.Dropdown([], label="Move from"); c8 = gr.Dropdown([], label="Move to"); c9 = gr.Slider(0, 30, 10, step=5, label="Transit penalty for moved orders (%)"); c10 = gr.Number(250, label="Cost per staff-hour (₹)"); c11 = gr.Number(0, label="Implementation cost (₹)")
                sc_btn = gr.Button("Run scenario", variant="primary"); sc_h = gr.HTML()
                with gr.Row():
                    sc1 = gr.Plot(); sc2 = gr.Plot()
            with gr.Tab("Reports & Data"):
                gr.Markdown("Upload CSVs (orders and warehouse events required; inventory, routes, staffing and complaints optional). Leave empty for the fictional demo. Raw uploads are never modified.")
                up = []
                with gr.Row():
                    for k in TN: up.append(gr.File(label=TN[k] + ".csv" + ("" if k in REQ else " (optional)"), file_types=[".csv"]))
                mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="order_ts=created_at, carrier_id=courier", lines=2); ld_btn = gr.Button("Validate and load uploaded data", variant="primary")
                gr.File(value=sfiles, label="Sample CSVs (synthetic)", file_count="multiple", interactive=False); dq_h = gr.HTML(); t_h = gr.HTML()
                gr.Markdown("### Exports"); xp_btn = gr.Button("Generate exports and executive summary", variant="primary"); xp_msg = gr.HTML(); xp_files = gr.File(label="Exports", file_count="multiple", interactive=False); gr.File(value=zip_path, label="Download GitHub-ready project (ZIP)", interactive=False)
            with gr.Tab("Methodology"):
                gr.Markdown(METH)
            with gr.Tab("Strategist"):
                gr.Markdown("Try: *Where is the biggest bottleneck?* · *Which warehouse is worst?* · *Which carriers and routes are slow?* · *Which orders are at risk?* · *What do complaints look like?* · *What should we do first?*")
                ag_q = gr.Textbox(label="Ask a question"); ag_btn = gr.Button("Ask", variant="primary"); ag_out = gr.Markdown()
        RUN = [s1, s2, s3, s4, s5, s6, s7, s8] + up + [mp, seed]; SCI = FIN + [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
        def pages(e):
            e = e.then(render_ct, FIN, [ct_h, ct1, ct2, ct3]).then(render_rc, FIN + [rc_dim, rc_val], [rc_h, rc1, rc2]).then(render_risk, FIN, [rk_h, rk1, rk2]).then(render_wh, FIN, [wh_h, wh1, wh2, wh3]).then(render_rt, FIN, [rt_h, rt1, rt2]).then(render_inv, FIN, [iv_h, iv1])
            return e.then(render_staff, FIN, [sf_h, sf1]).then(render_cmp, FIN, [cm_h, cm1]).then(render_act, FIN, ac_h).then(render_scn, SCI, [sc_h, sc1, sc2]).then(render_dq, None, [dq_h, t_h])
        def full(ev): return pages(ev.then(refresh, None, [f_w, f_z, f_c, c7, c8]))
        full(run_btn.click(run_all, RUN, [status], show_progress="full")); full(ld_btn.click(run_all, RUN, [status], show_progress="full")); full(demo.load(run_all, RUN, [status])); full(rg_btn.click(lambda s: (s or 5) + 1, seed, seed).then(run_all, RUN, [status], show_progress="full"))
        pages(ap_btn.click(lambda: None, None, None)); rc_btn.click(render_rc, FIN + [rc_dim, rc_val], [rc_h, rc1, rc2]); sc_btn.click(render_scn, SCI, [sc_h, sc1, sc2], show_progress="full"); xp_btn.click(export_all, FIN, [xp_files, xp_msg], show_progress="full"); ag_btn.click(ask, ag_q, ag_out); ag_q.submit(ask, ag_q, ag_out)
    return demo
def main():
    sf = write_samples(); zp, saved = build_project(sf)
    print("=" * 78); print("DELIVR AI starting. Demo data is FICTIONAL with planted patterns. No API key needed."); print("Building the demo analysis and running self-tests (about 1-2 minutes):")
    demo_X(); run_selftests(verbose=True); demo = build_ui(zp, sf)
    print("Project ZIP:", zp, "| standalone app.py included:", "yes" if saved else "NO (create app.py yourself by pasting this cell into it)")
    print("Not included / not tested: OR-Tools routing demo, PDF export, ML staffing forecast, uploads of your own CSVs. The public Gradio link is open to anyone: do not load confidential data.")
    print("=" * 78); demo.queue().launch(debug=False, show_error=True)
if __name__ == "__main__" or "google.colab" in sys.modules:
    main()