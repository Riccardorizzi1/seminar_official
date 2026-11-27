from octopus_v13_render import renderer as _v13_render, STYLE as _v13_style

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    ROOT / "data" / "silver" /
    "approved_silver_manifest.json"
)

BLUE = "#2675E8"
NAVY = "#153B79"
CYAN = "#14B8D4"
LIGHT_BLUE = "#84B6F4"
GREY = "#8295AD"
RED = "#D36B6B"

A = "agile_unit_rate_p_kwh_inc_vat"
C = "conventional_unit_rate_p_kwh_inc_vat"

CI = "carbon_intensity_gco2_kwh"

DEMAND = "national_demand_mw"
RENEW = "renewable_generation_mw"

SA = "agile_standing_p_day_inc_vat"
SC = "conventional_standing_p_day_inc_vat"

STRATEGIES = [
    "Minimise electricity costs",
    "Minimise associated emissions",
    "Balance costs and emissions"
]

SCENARIOS = [0, 10, 20, 30, 50]

st.set_page_config(
    page_title="Agile Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(_v13_style, unsafe_allow_html=True)


from octopus_v13_components import (
    individual as _v10_individual,
    justice_start as _v10_justice_start,
    justice_transition as _v10_justice_transition,
    plot as _v10_plot,
    theme_control as _v10_theme_control
)
_v10_theme_control()


# ============================================================
# VISUAL IDENTITY
# ============================================================

_v13_render(st.markdown, 'markdown')(
    """
    <style>

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #7099c844;
    }

    [data-testid="stMetric"] {
        border: 1px solid #7099c844;
        border-radius: 12px;
        padding: 15px 18px;
        
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    .hero {
        
        
        padding: 24px;
        border-radius: 15px;
        margin-bottom: 22px;
    }

    .hero h1 {
        
        font-size: 2rem;
        margin: 0;
    }

    .hero p {
        
        margin-top: 8px;
        margin-bottom: 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DATA VALIDATION UTILITIES
# ============================================================

def check(condition, message):

    if not condition:
        raise ValueError(message)


def require(df, columns, label):

    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{label}: missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def file_values(obj):

    if isinstance(obj, dict):

        for value in obj.values():
            yield from file_values(value)

    elif isinstance(obj, list):

        for value in obj:
            yield from file_values(value)

    elif isinstance(obj, str):

        yield obj


def resolve_file(manifest, filename):

    candidates = []

    for value in file_values(
        manifest.get("data_files", {})
    ):

        p = Path(value)

        if p.name.lower() == filename.lower():

            for base in [
                ROOT,
                MANIFEST.parent,
                ROOT / "data"
            ]:

                q = p if p.is_absolute() else base / p

                if q.is_file():
                    candidates.append(q.resolve())

    output = manifest.get("output_folder")

    if isinstance(output, str):

        p = Path(output)

        for base in [ROOT, MANIFEST.parent]:

            q = (
                p if p.is_absolute()
                else base / p
            ) / filename

            if q.is_file():
                candidates.append(q.resolve())

    candidates = list(dict.fromkeys(candidates))

    check(
        len(candidates) == 1,
        f"Expected one approved file for {filename}. "
        f"Found: {candidates}"
    )

    return candidates[0]


# ============================================================
# LOAD APPROVED SILVER DATA
# ============================================================

@st.cache_resource(show_spinner="Loading approved Silver data...", max_entries=1)
def load_data(manifest_text):

    manifest = json.loads(manifest_text)

    check(
        manifest.get("status") == "PASS",
        "Silver manifest status is not PASS."
    )

    check(
        manifest.get("silver_frozen") is True,
        "Silver layer is not frozen."
    )

    check(
        manifest.get("ready_for_step_3") is True,
        "Silver layer is not approved for Step 3."
    )

    def read(filename):

        return pd.read_csv(
            resolve_file(manifest, filename)
        )

    # --------------------------------------------------------
    # Market data
    # --------------------------------------------------------

    market = read("electricity_market_2025.csv")

    require(
        market,
        [
            "target_interval_index",
            "source_interval_index",
            "local_date",
            "local_time",
            A, C, CI, DEMAND, RENEW
        ],
        "Electricity market"
    )

    market = market.sort_values(
        "target_interval_index"
    ).reset_index(drop=True)

    n = len(market)

    check(
        n == 17520,
        f"Unexpected number of market intervals: {n}"
    )

    check(
        np.array_equal(
            market["target_interval_index"].to_numpy(),
            np.arange(n)
        ),
        "Invalid target interval index."
    )

    source_index = market[
        "source_interval_index"
    ].to_numpy(dtype=int)

    check(
        np.array_equal(
            np.sort(source_index),
            np.arange(n)
        ),
        "Invalid calendar mapping."
    )

    market["date"] = pd.to_datetime(
        market["local_date"]
    ).dt.normalize()

    market["slot"] = (
        market["local_time"]
        .astype(str)
        .str.extract(r"(\d{1,2}:\d{2})")[0]
        .str.zfill(5)
    )

    market["hour"] = pd.to_numeric(
        market["slot"].str[:2],
        errors="coerce"
    )

    check(
        market["hour"].notna().all(),
        "Invalid local time."
    )

    market["month"] = (
        market["date"].dt.strftime("%Y-%m")
    )

    market["peak"] = market["hour"].between(16, 18)

    for column in [A, C, CI, DEMAND, RENEW]:

        market[column] = pd.to_numeric(
            market[column],
            errors="raise"
        )

        check(
            np.isfinite(market[column]).all(),
            f"Invalid market values: {column}"
        )

    # --------------------------------------------------------
    # Standing charges
    # --------------------------------------------------------

    standing = read(
        "tariff_standing_charges_2025.csv"
    )

    require(
        standing,
        ["local_date", SA, SC],
        "Standing charges"
    )

    dates = pd.to_datetime(
        standing["local_date"]
    ).dt.normalize()

    check(
        len(dates) == 365
        and dates.is_unique
        and set(dates) == set(market["date"]),
        "Standing charge coverage mismatch."
    )

    for column in [SA, SC]:

        standing[column] = pd.to_numeric(
            standing[column],
            errors="raise"
        )

        check(
            np.isfinite(standing[column]).all(),
            f"Invalid standing charges: {column}"
        )

    stand_a = standing[SA].sum() / 100
    stand_c = standing[SC].sum() / 100

    # --------------------------------------------------------
    # Household registry
    # --------------------------------------------------------

    registry = read("household_registry.csv")

    require(
        registry,
        [
            "household_id",
            "matrix_row",
            "low_consumption_flag",
            "high_consumption_flag"
        ],
        "Household registry"
    )

    registry = registry.sort_values(
        "matrix_row"
    ).reset_index(drop=True)

    check(
        len(registry) == 3955,
        "Unexpected household count."
    )

    check(
        registry["household_id"].is_unique,
        "Duplicate household IDs."
    )

    check(
        np.array_equal(
            registry["matrix_row"],
            np.arange(len(registry))
        ),
        "Invalid household matrix mapping."
    )

    complete = read(
        "complete_household_sample.csv"
    )

    require(
        complete,
        ["household_id"],
        "Complete household sample"
    )

    registry["complete"] = (
        registry["household_id"]
        .isin(complete["household_id"])
    )

    check(
        registry["complete"].sum() == 881,
        "Unexpected complete household count."
    )

    for flag in [
        "low_consumption_flag",
        "high_consumption_flag"
    ]:

        values = (
            registry[flag]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        check(
            values.isin([
                "true", "false",
                "1", "0", "1.0", "0.0"
            ]).all(),
            f"Unrecognised household flag: {flag}"
        )

        registry[flag] = values.isin([
            "true", "1", "1.0"
        ])

    # --------------------------------------------------------
    # Household consumption matrix
    # --------------------------------------------------------

    matrix_path = resolve_file(
        manifest,
        "household_consumption_2013.npy"
    )

    matrix = np.load(
        matrix_path,
        mmap_mode="r",
        allow_pickle=False
    )

    check(
        matrix.shape == (len(registry), n),
        f"Unexpected matrix shape: {matrix.shape}"
    )

    # --------------------------------------------------------
    # Ofgem evidence
    # --------------------------------------------------------

    social = read("social_group_comparisons.csv")

    peak_evidence = read(
        "peak_consumption_evidence.csv"
    )

    consumer = read(
        "consumer_behaviour_evidence.csv"
    )

    return (
        market,
        registry,
        str(matrix_path),
        float(stand_a),
        float(stand_c),
        social,
        peak_evidence,
        consumer,
        manifest
    )


# ============================================================
# HOUSEHOLD BASELINE CALCULATIONS
# ============================================================

@st.cache_resource(
    show_spinner=False,
    max_entries=1
)
def load_gold():

    import hashlib

    path = (

        ROOT

        / "data"

        / "gold"

        / "step_3_dashboard_v3"

        / "dashboard_analytics.npz"

    )

    check(

        path.exists(),

        "Dashboard Gold file not found."

    )

    with np.load(

        path,

        allow_pickle=False

    ) as archive:

        gold = {

            key: archive[key]

            for key in archive.files

        }

    digest = hashlib.sha256(
        MANIFEST.read_bytes()
    ).hexdigest()

    check(

        str(
            gold["manifest_digest"].item()
        ) == digest,

        "Gold and approved Silver manifest mismatch."

    )

    for value in gold.values():

        value.setflags(
            write=False
        )

    return gold


def calculate_baseline(

    market,

    registry,

    matrix_path,

    stand_a,

    stand_c

):

    gold = load_gold()

    check(

        np.array_equal(

            registry["household_id"]
            .astype(str)
            .to_numpy(),

            gold["household_ids"]

        ),

        "Gold household mapping mismatch."

    )

    check(

        len(market)

        == int(
            gold["market_intervals"].item()
        ),

        "Gold market interval mismatch."

    )

    out = registry.copy()

    columns = gold[
        "baseline_columns"
    ].tolist()

    out[columns] = gold[
        "baseline_values"
    ]

    # No large consumption matrix calculations
    # are performed inside Streamlit.

    return (

        out,

        None,

        None,

        None

    )


def safe_corr(x, y):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    valid = (

        np.isfinite(x)

        & np.isfinite(y)

    )

    x = x[valid]

    y = y[valid]

    if len(x) < 2:

        return float("nan")

    x = x - x.mean()

    y = y - y.mean()

    denominator = np.sqrt(

        np.sum(x * x)

        * np.sum(y * y)

    )

    if denominator == 0:

        return float("nan")

    return float(

        np.sum(x * y)

        / denominator

    )



def cached_household_profile(

    matrix_path,

    rows,

    idx

):

    gold = load_gold()

    rows = np.asarray(
        rows,
        dtype=int
    )

    for k in range(4):

        if np.array_equal(

            rows,

            gold[
                f"profile_rows_{k}"
            ]

        ):

            return gold[
                f"profile_{k}"
            ]

    raise ValueError(

        "Selected household profile "

        "does not exist in Gold."

    )




# ============================================================
# FLEXIBILITY OPTIMISATION
# ============================================================

def normalise(values):

    span = np.ptp(values)

    if span == 0:

        return np.zeros_like(values)

    return (
        values - values.min()
    ) / span


@st.cache_data(show_spinner=False, max_entries=9)
def destination_coefficients(
    market,
    receiving_slots,
    strategy
):

    price = market[A].to_numpy() / 100

    carbon = market[CI].to_numpy() / 1000

    peak = market["peak"].to_numpy(dtype=bool)

    groups = [
        np.asarray(v, dtype=int)
        for v in market.groupby(
            "date",
            sort=True
        ).indices.values()
    ]

    destination_price = []

    destination_carbon = []

    destinations = []

    for g in groups:

        off_peak = g[~peak[g]]

        check(
            len(off_peak) >= receiving_slots,
            "Insufficient eligible receiving intervals."
        )

        if strategy == STRATEGIES[0]:

            score = price[off_peak]

        elif strategy == STRATEGIES[1]:

            score = carbon[off_peak]

        else:

            score = (
                0.5 * normalise(price[off_peak])
                + 0.5 * normalise(carbon[off_peak])
            )

        selected = off_peak[
            np.argsort(
                score,
                kind="stable"
            )[:receiving_slots]
        ]

        destination_price.append(
            price[selected].mean()
        )

        destination_carbon.append(
            carbon[selected].mean()
        )

        destinations.append(selected)

    return (
        np.array(destination_price),
        np.array(destination_carbon),
        destinations
    )


def calculate_flexibility_effects(

    market,

    daily_peak,

    daily_peak_cost,

    daily_peak_carbon,

    receiving_slots,

    strategy

):

    gold = load_gold()

    i = [4, 8, 12].index(
        receiving_slots
    )

    j = STRATEGIES.index(
        strategy
    )

    return (

        gold[
            "cost_effects"
        ][i, j],

        gold[
            "carbon_effects"
        ][i, j],

        gold[
            "shiftable_energy"
        ],

        gold[
            f"dest_{i}_{j}"
        ]

    )



def simulate(base, effects, flexibility):

    cost_change, carbon_change, energy, _ = effects

    fraction = flexibility / 100

    out = base.copy()

    out["shifted_kwh"] = energy * fraction

    out["agile_after"] = (
        out["agile_bill"]
        + fraction * cost_change
    )

    out["emissions_after"] = (
        out["emissions_kg"]
        + fraction * carbon_change
    )

    out["extra_savings"] = (
        out["agile_bill"]
        - out["agile_after"]
    )

    out["total_savings"] = (
        out["conventional_bill"]
        - out["agile_after"]
    )

    out["emissions_reduction"] = (
        out["emissions_kg"]
        - out["emissions_after"]
    )

    return out


# ============================================================
# VISUALISATION FUNCTIONS
# ============================================================

def money(value):

    return f"£{value:,.2f}"


def winner_pct(values):

    return (values > 1e-8).mean() * 100


def scatter_preview(
    df,
    *args,
    **kwargs
):

    original_count = len(df)

    max_points = 1000

    if original_count > max_points:

        df = df.sample(
            n=max_points,
            random_state=42
        )

        original_title = kwargs.get(
            "title",
            ""
        )

        kwargs["title"] = (
            original_title
            + f"<br><sup>"
            + f"Displaying {max_points:,} of "
            + f"{original_count:,} observations"
            + "</sup>"
        )

    # SVG rendering avoids accumulating WebGL contexts
    # when switching dashboard sections.

    kwargs["render_mode"] = "svg"

    return px.scatter(
        df,
        *args,
        **kwargs
    )


def plot(fig):

    fig.update_layout(

        template="plotly_white",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        margin=dict(
            l=20,
            r=20,
            t=75,
            b=25
        ),

        height=fig.layout.height or 370,

        font=dict(
            size=12,
            color="#15304F"
        ),

        title=dict(
            font=dict(size=16)
        ),

        legend_title_text="",

        hovermode="closest"
    )

    fig.update_xaxes(
        gridcolor="#E5EDF7",
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="#E5EDF7",
        zeroline=False
    )

    _v10_plot(

        fig,

        width="stretch",

        theme=None,

        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False
        }
    )



def line(df, x, y, title, labels=None):

    return px.line(
        df,
        x=x,
        y=y,
        title=title,
        labels=labels or {},
        color_discrete_sequence=[
            BLUE, CYAN, NAVY, GREY
        ]
    )


def metrics(items):

    columns = st.columns(len(items))

    for col, (label, value) in zip(
        columns,
        items
    ):

        col.metric(label, value)


def section(title, subtitle):

    st.header(title)
    _v13_render(st.caption, 'caption')(subtitle)


# ============================================================
# LOAD DATA
# ============================================================

try:

    manifest_text = MANIFEST.read_text(
        encoding="utf-8-sig"
    )

    (
        market,
        registry,
        matrix_path,
        stand_a,
        stand_c,
        social,
        peak_evidence,
        consumer,
        manifest
    ) = load_data(manifest_text)

except Exception as error:

    _v13_render(st.error, 'error')(f"Silver data loading failed: {error}")

    _v13_render(st.code, 'code')(str(MANIFEST))

    st.stop()


# ============================================================




from plotly.subplots import make_subplots

RESEARCH_LEVELS = [0, 20, 40]

REFERENCE_SLOTS = 8
REFERENCE_STRATEGY = STRATEGIES[0]

SCENARIO_COLOURS = {
    0: "#8295AD",
    20: "#2675E8",
    40: "#153B79"
}

# ------------------------------------------------------------
# VISUAL REFINEMENTS
# ------------------------------------------------------------

_v13_render(st.markdown, 'markdown')(
    """
    <style>

    .scenario-label {
        font-size: 13px;
        font-weight: 600;
        
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .scenario-card {
        
        border: 1px solid #C9DDF5;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        
    }

    .scenario-card h2 {
        
        margin-top: 3px;
        font-size: 30px;
    }

    .scenario-card p {
        
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .scenario-value {
        font-size: 24px;
        font-weight: 650;
        
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# SIMPLIFIED SIDEBAR
# ------------------------------------------------------------

_v13_render(st.sidebar.markdown, 'markdown')("# ⚡ Agile Energy")

_v13_render(st.sidebar.caption, 'caption')(
    "Flexibility · Sustainability · Distributive justice"
)

page = st.sidebar.radio(
    "Dashboard sections",
    [
        "01 · Electricity market",
        "02 · Household outcomes",
        "03 · Flexibility scenarios",
        "04 · Environmental sustainability",
        "05 · Distributive justice"
    ]
)

st.sidebar.divider()

sample = st.sidebar.selectbox(
    "Household sample",
    [
        "All 3,955 profiles",
        "881 fully observed profiles"
    ]
)

exclude_atypical = st.sidebar.checkbox(
    "Exclude atypical household profiles",
    value=False
)

if sample.startswith("881"):

    mask = registry["complete"].to_numpy().copy()

else:

    mask = np.ones(
        len(registry),
        dtype=bool
    )

if exclude_atypical:

    mask &= ~(
        registry["low_consumption_flag"]
        | registry["high_consumption_flag"]
    ).to_numpy()

if not mask.any():

    _v13_render(st.error, 'error')("No households selected.")
    st.stop()

st.sidebar.metric(
    "Selected households",
    f"{mask.sum():,}"
)

st.sidebar.divider()

_v13_render(st.sidebar.markdown, 'markdown')("**Reference research scenarios**")

_v13_render(st.sidebar.caption, 'caption')(
    "0%, 20% and 40% of peak consumption shifted."
)

_v13_render(st.sidebar.caption, 'caption')(
    "Exploratory flexibility settings are available "
    "inside the relevant sections."
)

st.sidebar.divider()

_v13_render(st.sidebar.caption, 'caption')(
    "2025 electricity market · 2013 household "
    "consumption proxy."
)

_v13_render(st.sidebar.caption, 'caption')(
    "Academic research dashboard — V10"
)

# ------------------------------------------------------------
# DASHBOARD HEADER
# ------------------------------------------------------------

_v13_render(st.markdown, 'markdown')(
    """
    <div class="hero">
        <h1>Agile Energy Analytics</h1>
        <p>
            Octopus Energy |
            Flexibility, Sustainability and Distributive Justice
        </p>
        <p>Research Dashboard — Version 13</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SHARED ANALYTICAL UTILITIES
# ============================================================

def pct_of_bill(numerator, denominator):

    numerator = pd.Series(
        numerator,
        index=denominator.index
    )

    return (
        numerator
        .div(denominator.where(denominator > 0))
        * 100
    )


def scenario_result(
    level,
    receiving_slots=REFERENCE_SLOTS,
    strategy=REFERENCE_STRATEGY
):

    effects = calculate_flexibility_effects(
        market,
        None,
        None,
        None,
        receiving_slots,
        strategy
    )

    result = simulate(
        base,
        effects,
        level
    )

    result["relative_total_savings"] = pct_of_bill(
        result["total_savings"],
        result["conventional_bill"]
    )

    result["relative_additional_savings"] = pct_of_bill(
        result["extra_savings"],
        result["conventional_bill"]
    )

    return result.loc[mask].copy()


def local_controls(key):

    left, middle, right = st.columns(3)

    with left:

        flexibility = st.slider(
            "Peak consumption shifted (%)",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            key=f"{key}_flex",
            help=(
                "Percentage of electricity consumed "
                "between 16:00 and 19:00 that is shifted."
            )
        )

    with middle:

        receiving_slots = st.select_slider(
            "Receiving half-hours per day",
            options=[4, 8, 12],
            value=8,
            key=f"{key}_slots",
            help=(
                "Number of off-peak half-hours receiving "
                "redistributed electricity."
            )
        )

    with right:

        strategy = st.selectbox(
            "Optimisation strategy",
            STRATEGIES,
            key=f"{key}_strategy"
        )

    return flexibility, receiving_slots, strategy


def scenario_summary(results):

    rows = []

    for level, result in results.items():

        rows.append({

            "Flexibility (%)": level,

            "Mean shifted consumption (kWh/year)":
                result["shifted_kwh"].mean(),

            "Median shifted consumption (kWh/year)":
                result["shifted_kwh"].median(),

            "Mean Agile bill (£)":
                result["agile_after"].mean(),

            "Median Agile bill (£)":
                result["agile_after"].median(),

            "Mean additional savings (£)":
                result["extra_savings"].mean(),

            "Median additional savings (£)":
                result["extra_savings"].median(),

            "Mean total savings (£)":
                result["total_savings"].mean(),

            "Median total savings (£)":
                result["total_savings"].median(),

            "Median relative total savings (%)":
                result["relative_total_savings"].median(),

            "Winners (%)":
                winner_pct(result["total_savings"]),

            "Mean emissions (kg)":
                result["emissions_after"].mean(),

            "Mean emissions reduction (kg)":
                result["emissions_reduction"].mean()

        })

    return pd.DataFrame(rows)


def research_scenario_caption():

    _v13_render(st.caption, 'caption')(
        "Fixed research assumptions: 0%, 20% and 40% "
        "of peak consumption shifted; peak period "
        "16:00–19:00; eight receiving half-hours; "
        "cost optimisation; identical household sample "
        "across scenarios."
    )


# ============================================================
# DATA-DRIVEN GRAPH INTERPRETATIONS
# ============================================================

# Keep the original lightweight V3 rendering function.

raw_plot = plot


def plot(fig):

    raw_plot(fig)

    title = str(
        fig.layout.title.text or ""
    )

    # --------------------------------------------------------
    # MARKET INTERPRETATIONS
    # --------------------------------------------------------

    if title.startswith("Electricity price evolution"):

        agile_mean = market[A].mean()
        conventional_mean = market[C].mean()

        cheaper = (
            market[A] < market[C]
        ).mean() * 100

        _v13_render(st.caption, 'caption')(
            f"Agile averaged {agile_mean:.2f}p/kWh, "
            f"compared with {conventional_mean:.2f}p/kWh "
            f"under the conventional tariff. Agile was "
            f"cheaper in {cheaper:.1f}% of half-hourly "
            f"intervals, although temporary price spikes "
            f"created additional financial exposure."
        )

    elif title.startswith(
        "Average half-hourly price profile"
    ):

        profile = market.groupby("slot")[A].mean()

        highest = profile.idxmax()
        lowest = profile.idxmin()

        _v13_render(st.caption, 'caption')(
            f"The highest average Agile price occurs "
            f"at {highest} ({profile.max():.2f}p/kWh), "
            f"while the lowest occurs at {lowest} "
            f"({profile.min():.2f}p/kWh). "
            f"The difference creates an incentive "
            f"to redistribute electricity consumption."
        )

    elif title.startswith("Monthly electricity prices"):

        monthly = market.groupby("month")[[A, C]].mean()

        gap = monthly[C] - monthly[A]

        best = pd.Timestamp(
            gap.idxmax() + "-01"
        ).strftime("%B")

        cheaper_months = int((gap > 0).sum())

        _v13_render(st.caption, 'caption')(
            f"Agile had a lower average unit price in "
            f"{cheaper_months} of the 12 months. "
            f"The largest average price advantage "
            f"occurred in {best}, at "
            f"{gap.max():.2f}p/kWh."
        )

    elif title.startswith(
        "Electricity price distribution"
    ):

        p10 = market[A].quantile(0.10)
        p90 = market[A].quantile(0.90)

        expensive = (
            market[A] > market[C]
        ).mean() * 100

        _v13_render(st.caption, 'caption')(
            f"The central 80% of Agile price observations "
            f"lies between {p10:.2f} and {p90:.2f}p/kWh. "
            f"Agile was more expensive than the "
            f"conventional benchmark in {expensive:.1f}% "
            f"of intervals, illustrating the variability "
            f"hidden by the annual average."
        )

    elif title.startswith(
        "Negative price intervals by month"
    ):

        negative = (
            market.assign(negative=market[A] < 0)
            .groupby("month")["negative"]
            .sum()
        )

        best = pd.Timestamp(
            negative.idxmax() + "-01"
        ).strftime("%B")

        total = int(negative.sum())

        _v13_render(st.caption, 'caption')(
            f"There were {total:,} negative-price "
            f"half-hours during 2025, equivalent to "
            f"{total / 2:,.1f} hours. "
            f"{best} recorded the largest number "
            f"of these opportunities."
        )

    elif title.startswith(
        "Agile electricity price calendar heatmap"
    ):

        peak_mean = market.loc[
            market["peak"], A
        ].mean()

        off_peak_mean = market.loc[
            ~market["peak"], A
        ].mean()

        _v13_render(st.caption, 'caption')(
            f"Average prices during the 16:00–19:00 "
            f"peak were {peak_mean:.2f}p/kWh, compared "
            f"with {off_peak_mean:.2f}p/kWh outside "
            f"the peak. The heatmap shows how this "
            f"pattern varies across individual days."
        )

    # --------------------------------------------------------
    # ENVIRONMENTAL INTERPRETATIONS
    # --------------------------------------------------------

    elif title.startswith(
        "Monthly electricity carbon intensity"
    ):

        monthly = market.groupby("month")[CI].mean()

        highest = pd.Timestamp(
            monthly.idxmax() + "-01"
        ).strftime("%B")

        lowest = pd.Timestamp(
            monthly.idxmin() + "-01"
        ).strftime("%B")

        _v13_render(st.caption, 'caption')(
            f"Average carbon intensity was highest "
            f"in {highest} ({monthly.max():.1f} gCO₂/kWh) "
            f"and lowest in {lowest} "
            f"({monthly.min():.1f} gCO₂/kWh). "
            f"The environmental implications of electricity "
            f"consumption therefore vary throughout the year."
        )

    elif title.startswith(
        "Reported electricity generation by source"
    ):

        fuels = [
            c for c in [
                "gas_generation_mw",
                "coal_generation_mw",
                "nuclear_generation_mw",
                "wind_generation_mw",
                "solar_generation_mw",
                "hydro_generation_mw",
                "biomass_generation_mw"
            ]
            if c in market.columns
        ]

        if fuels:

            averages = market[fuels].mean()

            largest = (
                averages.idxmax()
                .replace("_generation_mw", "")
                .replace("_", " ")
                .title()
            )

            _v13_render(st.caption, 'caption')(
                f"{largest} recorded the highest average "
                f"generation among the sources displayed. "
                f"The monthly changes illustrate the "
                f"changing composition of the electricity "
                f"system throughout the year."
            )

    elif title.startswith(
        "National electricity demand (MW)"
    ):

        r = safe_corr(market[DEMAND], market[A])

        direction = (
            "higher" if r >= 0 else "lower"
        )

        _v13_render(st.caption, 'caption')(
            f"The correlation is {r:+.2f}. Higher national "
            f"electricity demand is associated with "
            f"{direction} Agile prices. This describes "
            f"a historical relationship, not a causal "
            f"estimate of demand response."
        )

    elif title.startswith(
        "Renewable generation (MW)"
    ):

        r = safe_corr(market[RENEW], market[A])

        direction = (
            "higher" if r >= 0 else "lower"
        )

        _v13_render(st.caption, 'caption')(
            f"The correlation is {r:+.2f}. Higher renewable "
            f"generation is associated with {direction} "
            f"Agile prices, indicating how financial "
            f"incentives may relate to renewable "
            f"electricity availability."
        )

    elif title.startswith(
        "Carbon intensity (gCO₂/kWh)"
    ):

        r = safe_corr(market[CI], market[A])

        relationship = (
            "higher" if r >= 0 else "lower"
        )

        _v13_render(st.caption, 'caption')(
            f"The correlation is {r:+.2f}. More carbon-intensive "
            f"electricity tends to coincide with "
            f"{relationship} Agile prices. This indicates "
            f"the degree of alignment between financial "
            f"and environmental incentives, without "
            f"establishing causality."
        )

    elif title.startswith(
        "Associated emissions by flexibility scenario"
    ):

        baseline = selected["emissions_kg"].mean()

        after = selected["emissions_after"].mean()

        reduction = baseline - after

        relative = (
            reduction / baseline * 100
            if baseline > 0 else 0
        )

        _v13_render(st.caption, 'caption')(
            f"Under the selected {flexibility}% flexibility "
            f"scenario, average associated emissions change "
            f"from {baseline:.1f} to {after:.1f} kg/year. "
            f"The estimated reduction is "
            f"{reduction:.1f} kg ({relative:.2f}%). "
            f"This is an accounting estimate, not a "
            f"measured system-wide emissions reduction."
        )

    elif title.startswith(
        "Economic and environmental trade-offs"
    ):

        comparison = tradeoffs.loc[
            tradeoffs["Flexibility (%)"] == flexibility
        ]

        if len(comparison):

            best_cost = comparison.loc[
                comparison[
                    "Additional savings (£)"
                ].idxmax()
            ]

            best_carbon = comparison.loc[
                comparison[
                    "Emissions reduction (kg)"
                ].idxmax()
            ]

            _v13_render(st.caption, 'caption')(
                f"At {flexibility}% flexibility, the maximum "
                f"mean additional saving among the strategies "
                f"is £{best_cost['Additional savings (£)']:.2f}, "
                f"while the maximum associated emissions "
                f"reduction is "
                f"{best_carbon['Emissions reduction (kg)']:.2f} kg. "
                f"The graph shows the trade-offs involved "
                f"in prioritising either objective."
            )


# ============================================================
# LOAD HOUSEHOLD BASELINE FROM GOLD
# ============================================================

if not page.startswith("01"):

    (
        base,
        daily_peak,
        daily_peak_cost,
        daily_peak_carbon
    ) = calculate_baseline(
        market,
        registry,
        matrix_path,
        stand_a,
        stand_c
    )

    households = base.loc[mask].copy()

    households["relative_savings"] = pct_of_bill(
        households["savings"],
        households["conventional_bill"]
    )

    check(
        np.allclose(
            base["conventional_bill"] - base["agile_bill"],
            base["savings"]
        ),
        "Gold household savings reconciliation failed."
    )



if page.startswith("01"):
    section('01 — Electricity market overview', 'Understanding the economic incentives and risks of dynamic electricity pricing.')

    _v13_render(st.markdown, 'markdown')("**Methodological note**")
    _v13_render(st.caption, 'caption')('Data and reference period: this analysis uses 17,520 half-hourly electricity market observations covering the full calendar year 2025. Agile prices are compared with the conventional tariff benchmark collected for this project. Electricity prices are expressed in pence per kWh. Average prices are time-weighted across intervals, rather than weighted by household electricity consumption. Consequently, differences in average unit prices should not be interpreted directly as differences in household electricity bills.')
    st.divider()


    metrics([('Agile average price', f'{market[A].mean():.2f}p/kWh'), ('Conventional average price', f'{market[C].mean():.2f}p/kWh'), ('Negative price intervals', f'{(market[A] < 0).sum():,}'), ('Maximum Agile price', f'{market[A].max():.2f}p/kWh')])

    names = {A: 'Agile', C: 'Conventional'}

    resolution = st.selectbox('Time resolution', ['Daily', 'Weekly', 'Monthly'])

    frequency = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'MS'}[resolution]

    prices = market.set_index('date')[[A, C]].resample(frequency).mean().reset_index().rename(columns=names)

    plot(line(prices, 'date', ['Agile', 'Conventional'], 'Electricity price evolution', {'value': 'Price (p/kWh)', 'date': 'Date'}))

    left, right = st.columns(2)

    with left:
        daily = market.groupby('slot')[[A, C]].mean().reset_index().rename(columns=names)
        plot(line(daily, 'slot', ['Agile', 'Conventional'], 'Average half-hourly price profile', {'value': 'Price (p/kWh)', 'slot': 'Local time'}))

    with right:
        monthly = market.groupby('month')[[A, C]].mean().reset_index().rename(columns=names)
        plot(px.bar(monthly, x='month', y=['Agile', 'Conventional'], barmode='group', title='Monthly electricity prices', labels={'month': 'Month', 'value': 'Price (p/kWh)'}, color_discrete_sequence=[BLUE, LIGHT_BLUE]))

    left, right = st.columns(2)

    with left:
        distribution = market[[A, C]].rename(columns=names).melt(var_name='Tariff', value_name='Price')
        plot(px.histogram(distribution, x='Price', color='Tariff', barmode='overlay', opacity=0.65, nbins=80, title='Electricity price distribution', labels={'Price': 'Price (p/kWh)'}, color_discrete_sequence=[BLUE, LIGHT_BLUE]))

    with right:
        negative = market.assign(negative=market[A] < 0).groupby('month')['negative'].sum().reset_index()
        plot(px.bar(negative, x='month', y='negative', title='Negative price intervals by month', labels={'negative': 'Half-hourly intervals', 'month': 'Month'}, color_discrete_sequence=[BLUE]))

    heatmap = market.pivot_table(index='slot', columns='date', values=A, aggfunc='mean')

    fig = go.Figure(go.Heatmap(z=heatmap.values, x=heatmap.columns, y=heatmap.index, colorscale='RdBu_r', zmid=0, colorbar=dict(title='p/kWh')))

    fig.update_layout(title='Agile electricity price calendar heatmap', xaxis_title='Date', yaxis_title='Local time')

    plot(fig)

    _v13_render(st.caption, 'caption')('Price averages are interval-weighted, not household-consumption-weighted. Repeated autumn clock intervals are averaged only for the calendar heatmap.')




    st.divider()

    st.subheader("What the electricity market tells us")

    agile_average = market[A].mean()

    conventional_average = market[C].mean()

    average_gap = conventional_average - agile_average

    relative_gap = (
        average_gap / conventional_average * 100
        if conventional_average > 0
        else float("nan")
    )

    peak_price = market.loc[
        market["peak"], A
    ].mean()

    off_peak_price = market.loc[
        ~market["peak"], A
    ].mean()

    negative_count = int(
        (market[A] < 0).sum()
    )

    _v13_render(st.markdown, 'markdown')(
        f"""
        Across the 17,520 half-hourly intervals of 2025,
        Agile's average unit price was
        **{agile_average:.2f}p/kWh**, compared with
        **{conventional_average:.2f}p/kWh** under the
        conventional benchmark.

        This represents an average unit-price difference
        of **{average_gap:.2f}p/kWh**
        ({relative_gap:.1f}% of the conventional average).

        However, the economic advantage is not constant
        throughout the day. During the 16:00–19:00 peak
        period, the average Agile price was
        **{peak_price:.2f}p/kWh**, compared with
        **{off_peak_price:.2f}p/kWh** outside the peak.

        The {negative_count:,} negative-price intervals
        also illustrate the opportunities available to
        consumers capable of adapting their electricity
        consumption.
        """
    )

    _v13_render(st.caption, 'caption')(
        "These are time-weighted electricity unit prices, "
        "not consumption-weighted household costs. "
        "Actual household expenditure depends on when "
        "electricity is consumed and on standing charges."
    )

elif page.startswith("02"):
    section(
        "02 — Household economic outcomes",
        "Comparing annual electricity expenditure "
        "without changing household consumption behaviour."
    )

    _v13_render(st.markdown, 'markdown')("**Methodological note**")
    _v13_render(st.caption, 'caption')('Household expenditure is simulated by applying 2025 Agile and conventional electricity prices to the same historical 2013 household consumption profiles, using the approved calendar alignment. The complete analytical population contains 3,955 profiles, of which 881 are fully observed. The sample filters allow sensitivity analysis regarding imputed readings and atypical consumption profiles. Annual electricity bills include standing charges. Absolute savings equal the conventional bill minus the Agile bill; relative savings express this difference as a percentage of the conventional bill. No behavioural adaptation is simulated in this section.')
    st.divider()


    n = len(households)

    wins = int(
        (households["savings"] > 1e-8).sum()
    )

    losses = int(
        (households["savings"] < -1e-8).sum()
    )

    ties = n - wins - losses

    # ------------------------------------------------------------
    # A. PRIMARY ECONOMIC KPIs
    # ------------------------------------------------------------

    metrics([

        (
            "Mean annual Agile bill",
            money(households["agile_bill"].mean())
        ),

        (
            "Mean annual conventional bill",
            money(households["conventional_bill"].mean())
        ),

        (
            "Median annual savings",
            money(households["savings"].median())
        ),

        (
            "Households benefiting",
            f"{wins / n * 100:.2f}%"
        )

    ])

    _v13_render(st.caption, 'caption')(
        "All monetary indicators represent annual "
        "electricity expenditure or savings per household, "
        "including standing charges."
    )

    _v13_render(st.markdown, 'markdown')(
        f"""
        Applying both tariffs to identical consumption
        profiles shows that **{wins:,} of {n:,} households**
        spend less under Agile without changing their
        consumption behaviour.

        {losses:,} households spend more and
        {ties:,} break even.
        """
    )

    # ------------------------------------------------------------
    # B. ANNUAL EXPENDITURE COMPARISON
    # ------------------------------------------------------------

    st.subheader("Annual electricity expenditure")

    bills = households[
        ["agile_bill", "conventional_bill"]
    ].rename(columns={

        "agile_bill": "Agile",

        "conventional_bill": "Conventional"

    })

    fig = px.box(

        bills,

        points=False,

        title="Annual household electricity bills",

        labels={
            "value": "Annual expenditure (£/household/year)",
            "variable": "Tariff"
        },

        color_discrete_sequence=[BLUE]

    )

    fig.update_layout(
        showlegend=False
    )

    raw_plot(fig)

    median_agile = households[
        "agile_bill"
    ].median()

    median_conventional = households[
        "conventional_bill"
    ].median()

    mean_savings = households[
        "savings"
    ].mean()

    _v13_render(st.caption, 'caption')(
        f"The median annual household bill is "
        f"{money(median_agile)} under Agile and "
        f"{money(median_conventional)} under the conventional "
        f"tariff. Across the selected households, adopting "
        f"Agile generates average annual savings of "
        f"{money(mean_savings)} without any simulated "
        f"behavioural adaptation."
    )

    _v13_render(st.caption, 'caption')(
        "The central line in each box represents the median. "
        "The box contains the middle 50% of household bills. "
        "Whiskers follow the standard box-plot convention; "
        "individual outlier markers are hidden for readability."
    )

    # ------------------------------------------------------------
    # C. ABSOLUTE AND RELATIVE SAVINGS
    # ------------------------------------------------------------

    st.subheader(
        "How are the economic benefits distributed?"
    )

    left, right = st.columns(2)

    with left:

        absolute = households[
            "savings"
        ].dropna()

        fig = px.histogram(

            households,

            x="savings",

            nbins=45,

            title="Absolute annual savings",

            labels={
                "savings": "Annual savings (£/household/year)"
            },

            color_discrete_sequence=[BLUE]

        )

        fig.update_yaxes(
            title="Number of households"
        )

        fig.add_vline(

            x=absolute.median(),

            line_dash="dash",

            line_color=NAVY,

            annotation_text="Median"

        )

        raw_plot(fig)

        p10 = absolute.quantile(0.10)

        p90 = absolute.quantile(0.90)

        _v13_render(st.caption, 'caption')(
            f"The median household saves "
            f"{money(absolute.median())} annually. "
            f"The central 80% of household savings lies "
            f"between {money(p10)} and {money(p90)}. "
            f"This demonstrates that households benefit "
            f"to different extents even without flexibility."
        )

    with right:

        relative = households[
            "relative_savings"
        ].dropna()

        fig = px.histogram(

            households,

            x="relative_savings",

            nbins=45,

            title="Relative annual savings",

            labels={
                "relative_savings":
                    "Savings (% of conventional annual bill)"
            },

            color_discrete_sequence=[NAVY]

        )

        fig.update_yaxes(
            title="Number of households"
        )

        fig.add_vline(

            x=relative.median(),

            line_dash="dash",

            line_color=BLUE,

            annotation_text="Median"

        )

        raw_plot(fig)

        _v13_render(st.caption, 'caption')(
            f"The median household reduces its annual "
            f"electricity expenditure by "
            f"{relative.median():.2f}%. "
            f"The central 80% of relative savings lies "
            f"between {relative.quantile(0.10):.2f}% "
            f"and {relative.quantile(0.90):.2f}%. "
            f"This allows comparison between households "
            f"with different electricity expenditure levels."
        )

    _v13_render(st.caption, 'caption')(
        "Absolute savings equal the conventional annual "
        "bill minus the Agile annual bill. Relative savings "
        "equal absolute savings divided by the conventional "
        "annual bill, multiplied by 100."
    )

    # ------------------------------------------------------------
    # D. NUMERICAL VALIDATION
    # No additional dashboard section is displayed.
    # ------------------------------------------------------------

    @st.cache_resource(show_spinner=False, max_entries=1)
    def validate_household_bills_v5(
        manifest_text,
        matrix_path,
        idx,
        agile_rates,
        conventional_rates,
        standing_agile,
        standing_conventional,
        validation_rows
    ):

        matrix = np.load(
            matrix_path,
            mmap_mode="r",
            allow_pickle=False
        )

        idx = np.asarray(
            idx,
            dtype=int
        )

        a = np.asarray(
            agile_rates,
            dtype=np.float64
        )

        c = np.asarray(
            conventional_rates,
            dtype=np.float64
        )

        for matrix_row, expected_a, expected_c in validation_rows:

            x = np.asarray(
                matrix[matrix_row, idx],
                dtype=np.float64
            )

            actual_a = (
                np.sum(x * a)
                + standing_agile
            )

            actual_c = (
                np.sum(x * c)
                + standing_conventional
            )

            if not np.allclose(

                [actual_a, actual_c],

                [expected_a, expected_c],

                rtol=1e-9,

                atol=1e-7

            ):

                raise ValueError(
                    f"Household bill verification failed "
                    f"for matrix row {matrix_row}."
                )

        del matrix

        return True


    median_index = (
        base["savings"]
        .sub(base["savings"].median())
        .abs()
        .idxmin()
    )

    test_indices = list(dict.fromkeys([

        base["savings"].idxmin(),

        median_index,

        base["savings"].idxmax()

    ]))

    validation_rows = tuple(

        (

            int(base.loc[i, "matrix_row"]),

            float(base.loc[i, "agile_bill"]),

            float(base.loc[i, "conventional_bill"])

        )

        for i in test_indices

    )

    try:

        validate_household_bills_v5(

            manifest_text,

            matrix_path,

            tuple(
                market["source_interval_index"]
                .astype(int)
                .tolist()
            ),

            tuple(
                (market[A] / 100).tolist()
            ),

            tuple(
                (market[C] / 100).tolist()
            ),

            stand_a,

            stand_c,

            validation_rows

        )

    except Exception as error:

        _v13_render(st.error, 'error')(
            f"Independent bill verification failed: {error}"
        )

        st.stop()

    # ------------------------------------------------------------
    # E. RESEARCH INTERPRETATION
    # ------------------------------------------------------------

    st.divider()

    st.subheader("What these results tell us")

    _v13_render(st.markdown, 'markdown')(
        f"""
        Under the modelled 2025 market conditions,
        **{wins / n * 100:.2f}% of the selected households**
        benefit from Agile without any behavioural adaptation.

        The differences in absolute and proportional savings
        demonstrate that the economic benefits of adopting
        Agile are heterogeneous across consumption profiles.

        These outcomes establish the baseline against which
        the additional economic value of flexibility will
        be evaluated in the next section.
        """
    )

    _v13_render(st.caption, 'caption')(
        f"Annual standing charges: Agile {money(stand_a)}; "
        f"conventional {money(stand_c)}. "
        f"Results are simulated annual bills, not observed "
        f"customer expenditure."
    )

elif page.startswith("03"):
    section(
        "03 — Flexibility scenarios",
        "Estimating the additional economic value "
        "of three fixed flexibility capabilities."
    )

    _v13_render(st.markdown, 'markdown')("**Methodological note**")
    _v13_render(st.caption, 'caption')('The three reference scenarios assume that households redistribute 0%, 20% or 40% of the electricity originally consumed during the 16:00–19:00 peak period. Redistributed electricity is allocated equally across the eight cheapest eligible off-peak half-hours within the same day. These intervals represent four hours in total and need not be consecutive. Total daily and annual electricity consumption remains unchanged. Each scenario is applied to individual household profiles before calculating mean and median annual outcomes. The scenarios are hypothetical and use historical price information; actual household appliance constraints and flexibility capabilities are not observed.')
    st.divider()


    _v13_render(st.markdown, 'markdown')(
        """
        Starting from the original household consumption
        profiles, we compare three hypothetical situations:

        **0% flexibility:** households maintain their original
        electricity consumption behaviour.

        **20% flexibility:** households redistribute 20% of
        the electricity originally consumed during peak hours.

        **40% flexibility:** households redistribute 40% of
        the electricity originally consumed during peak hours.
        """
    )

    research_scenario_caption()

    _v13_render(st.caption, 'caption')(
        "The percentages refer to consumption between "
        "16:00 and 19:00, not total electricity consumption. "
        "Redistributed demand is allocated equally across "
        "the eight cheapest eligible off-peak half-hours "
        "of the same day."
    )

    fixed = {

        level: scenario_result(level)

        for level in RESEARCH_LEVELS

    }

    summary = scenario_summary(fixed)

    # ------------------------------------------------------------
    # A. ANNUAL BILL GRAPH
    # ------------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=summary["Flexibility (%)"],

            y=summary["Mean Agile bill (£)"],

            mode="lines+markers",

            name="Mean annual Agile bill",

            line=dict(
                color=BLUE,
                width=3
            ),

            marker=dict(size=10)

        )

    )

    fig.add_trace(

        go.Scatter(

            x=summary["Flexibility (%)"],

            y=summary["Median Agile bill (£)"],

            mode="lines+markers",

            name="Median annual Agile bill",

            line=dict(
                color=NAVY,
                width=3
            ),

            marker=dict(size=10)

        )

    )

    fig.add_hline(

        y=households["conventional_bill"].mean(),

        line_dash="dash",

        line_color=GREY,

        annotation_text="Mean conventional annual bill"

    )

    fig.add_hline(

        y=households["conventional_bill"].median(),

        line_dash="dot",

        line_color=GREY,

        annotation_text="Median conventional annual bill",

        annotation_position="bottom right"

    )

    fig.update_layout(

        title="Annual Agile bill by flexibility level",

        xaxis_title="Peak electricity consumption shifted",

        yaxis_title="Annual expenditure (£/household/year)"

    )

    fig.update_xaxes(

        tickvals=RESEARCH_LEVELS,

        ticktext=["0%", "20%", "40%"]

    )

    raw_plot(fig)

    mean_zero = fixed[0][
        "agile_after"
    ].mean()

    mean_twenty = fixed[20][
        "agile_after"
    ].mean()

    mean_forty = fixed[40][
        "agile_after"
    ].mean()

    _v13_render(st.caption, 'caption')(
        f"Mean annual Agile expenditure changes from "
        f"{money(mean_zero)} without flexibility to "
        f"{money(mean_twenty)} under the 20% scenario "
        f"and {money(mean_forty)} under the 40% scenario. "
        f"The difference isolates the additional "
        f"economic effect of consumption redistribution."
    )

    _v13_render(st.caption, 'caption')(
        "The mean bill changes linearly with the flexibility "
        "percentage because the model redistributes a "
        "proportional amount of peak consumption. "
        "This relationship is a consequence of the "
        "simulation design, not an empirically estimated "
        "behavioural response."
    )

    # ------------------------------------------------------------
    # B. THREE FIXED SCENARIO CARDS
    # ------------------------------------------------------------

    st.subheader("Reference scenario outcomes")

    _v13_render(st.write, 'write')(
        "The following three scenarios summarise average "
        "household outcomes under fixed assumptions about "
        "the proportion of peak electricity consumption "
        "that can be redistributed."
    )

    scenario_names = {

        0: "No flexibility",

        20: "Moderate flexibility",

        40: "High flexibility"

    }

    columns = st.columns(3)

    for col, level in zip(
        columns,
        RESEARCH_LEVELS
    ):

        result = fixed[level]

        mean_shifted = result[
            "shifted_kwh"
        ].mean()

        mean_bill = result[
            "agile_after"
        ].mean()

        mean_extra = result[
            "extra_savings"
        ].mean()

        median_extra = result[
            "extra_savings"
        ].median()

        with col:

            with st.container(border=True):

                _v13_render(st.markdown, 'markdown')(
                    f"**{scenario_names[level]}**"
                )

                _v13_render(st.markdown, 'markdown')(
                    f"## {level}%"
                )

                st.divider()

                st.metric(

                    "Mean annual electricity shifted",

                    f"{mean_shifted:,.1f} kWh"

                )

                _v13_render(st.caption, 'caption')(

                    f"{mean_shifted / 365:.2f} "
                    "kWh per household per day"

                )

                st.divider()

                st.metric(

                    "Mean annual Agile bill",

                    money(mean_bill)

                )

                _v13_render(st.caption, 'caption')(
                    "Per household per year"
                )

                st.divider()

                st.metric(

                    "Mean additional annual savings",

                    money(mean_extra)

                )

                _v13_render(st.caption, 'caption')(

                    f"Median additional savings: "
                    f"{money(median_extra)}/year"

                )

    _v13_render(st.caption, 'caption')(
        "All monetary outcomes are annual values per household. "
        "Shifted consumption represents electricity moved away "
        "from peak hours, not electricity saved or eliminated."
    )

    _v10_individual(fixed)

    st.subheader("Mean and median household outcomes")

    comparison = []

    for level in RESEARCH_LEVELS:

        result = fixed[level]

        for label, operation in [

            ("Mean", "mean"),

            ("Median", "median")

        ]:

            comparison.append({

                "Flexibility": f"{level}%",

                "Statistic": label,

                "Annual electricity shifted (kWh)":
                    getattr(
                        result["shifted_kwh"],
                        operation
                    )(),

                "Annual Agile bill (£)":
                    getattr(
                        result["agile_after"],
                        operation
                    )(),

                "Additional annual savings (£)":
                    getattr(
                        result["extra_savings"],
                        operation
                    )(),

                "Total annual savings vs conventional (£)":
                    getattr(
                        result["total_savings"],
                        operation
                    )()

            })

    st.dataframe(

        pd.DataFrame(comparison).round(2),

        hide_index=True,

        width="stretch"

    )

    _v13_render(st.caption, 'caption')(
        f"All values refer to one household over one year. "
        f"The mean describes the average outcome across "
        f"{len(households):,} selected profiles. "
        f"The median describes the midpoint of each "
        f"outcome distribution and is less influenced "
        f"by unusually high or low observations."
    )

    _v13_render(st.caption, 'caption')(
        "Each scenario is applied to individual household "
        "profiles before calculating the mean and median. "
        "The median of each indicator is calculated "
        "independently."
    )

    # ------------------------------------------------------------
    # D. INTERPRETATION
    # ------------------------------------------------------------

    st.divider()

    st.subheader("What flexibility adds")

    baseline_savings = fixed[0][
        "total_savings"
    ].mean()

    flexibility_savings = fixed[40][
        "extra_savings"
    ].mean()

    total_savings = fixed[40][
        "total_savings"
    ].mean()

    shifted_pct = (

        fixed[40]["shifted_kwh"].sum()

        / households["annual_kwh"].sum()

        * 100

        if households["annual_kwh"].sum() > 0

        else float("nan")

    )

    _v13_render(st.markdown, 'markdown')(
        f"""
        Without behavioural adaptation, adopting Agile
        generates average annual savings of
        **{money(baseline_savings)} per household**.

        Under the 40% flexibility scenario, consumption
        redistribution generates a further
        **{money(flexibility_savings)} per household/year**.

        The combined average economic benefit is therefore
        **{money(total_savings)} per household/year**
        relative to the conventional tariff.

        Across the selected population, the 40% scenario
        redistributes {shifted_pct:.2f}% of total annual
        electricity consumption, while preserving the
        original total consumption volume.
        """
    )

    _v13_render(st.caption, 'caption')(
        "The simulations represent hypothetical economic "
        "potential under known historical prices. Actual "
        "flexibility capabilities, appliance constraints "
        "and future market responses are not modelled."
    )

elif page.startswith("04"):
    section(
        "04 — Environmental sustainability",
        "Investigating whether the economic incentives "
        "of dynamic electricity pricing are aligned "
        "with environmental sustainability."
    )

    _v13_render(st.markdown, 'markdown')("**Methodological note**")
    _v13_render(st.caption, 'caption')('Environmental indicators use the collected 2025 British electricity system data. Consumption-associated emissions are estimated by multiplying household electricity consumption in each half-hour by the corresponding average grid carbon intensity and summing across the year. Household emissions are reported in kg CO₂ per household per year. Grid carbon intensity is reported in gCO₂/kWh, and electricity generation in MW. The simulation compares cost, emissions and balanced optimisation strategies. Pearson correlations describe historical linear associations rather than causal relationships. Average carbon intensity is not a marginal emissions factor, so estimated reductions should not be interpreted as measured national emissions reductions. The generation timestamp convention and embedded wind reconciliation remain methodological limitations.')
    st.divider()


    # ------------------------------------------------------------
    # A. LOCAL EXPLORATORY CONTROLS
    # ------------------------------------------------------------

    with st.expander(

        "Environmental flexibility assumptions",

        expanded=True

    ):

        flexibility, receiving_slots, strategy = local_controls(
            "environment"
        )

        _v13_render(st.caption, 'caption')(
            "These controls affect only the environmental "
            "simulation. They do not change the three fixed "
            "research scenarios in Sections 03 and 05."
        )

    selected = scenario_result(

        flexibility,

        receiving_slots,

        strategy

    )

    baseline_emissions = selected[
        "emissions_kg"
    ].mean()

    after_emissions = selected[
        "emissions_after"
    ].mean()

    emissions_reduction = (
        baseline_emissions - after_emissions
    )

    relative_reduction = (

        emissions_reduction / baseline_emissions * 100

        if baseline_emissions > 0

        else float("nan")

    )

    # ------------------------------------------------------------
    # B. PRIMARY ENVIRONMENTAL INDICATORS
    # ------------------------------------------------------------

    metrics([

        (
            "Mean grid carbon intensity",
            f"{market[CI].mean():.1f} gCO₂/kWh"
        ),

        (
            "Mean annual household emissions — baseline",
            f"{baseline_emissions:,.1f} kg CO₂"
        ),

        (
            "Mean annual household emissions — after shifting",
            f"{after_emissions:,.1f} kg CO₂"
        ),

        (
            "Mean annual household emissions reduction",
            f"{emissions_reduction:,.1f} kg CO₂"
        )

    ])

    _v13_render(st.caption, 'caption')(
        "Grid carbon intensity is the mean across the "
        "17,520 half-hourly electricity market intervals "
        "of 2025. All other indicators are average annual "
        "consumption-associated emissions per household."
    )

    _v13_render(st.markdown, 'markdown')(
        f"""
        Under the selected **{flexibility}% flexibility**
        scenario, average annual household-associated
        emissions change from **{baseline_emissions:.1f}**
        to **{after_emissions:.1f} kg CO₂**.

        The estimated reduction is
        **{emissions_reduction:.1f} kg CO₂ per household/year**
        ({relative_reduction:.2f}% of baseline emissions).
        A negative reduction would indicate an increase
        in associated emissions.
        """
    )

    # ============================================================
    # GRAPH 1 - MONTHLY CARBON INTENSITY
    # ============================================================

    st.subheader("Environmental characteristics of electricity")

    monthly_carbon = (
        market.groupby("month")[CI]
        .mean()
        .reset_index()
    )

    fig = line(

        monthly_carbon,

        "month",

        CI,

        "Monthly mean electricity carbon intensity",

        {

            "month": "Month of 2025",

            CI: "Mean carbon intensity (gCO₂/kWh)"

        }

    )

    raw_plot(fig)

    monthly_series = monthly_carbon.set_index(
        "month"
    )[CI]

    highest = pd.Timestamp(

        monthly_series.idxmax() + "-01"

    ).strftime("%B")

    lowest = pd.Timestamp(

        monthly_series.idxmin() + "-01"

    ).strftime("%B")

    _v13_render(st.caption, 'caption')(
        f"The average carbon intensity of electricity "
        f"was highest in {highest} "
        f"({monthly_series.max():.1f} gCO₂/kWh) "
        f"and lowest in {lowest} "
        f"({monthly_series.min():.1f} gCO₂/kWh). "
        f"Identical electricity consumption volumes "
        f"can therefore have different associated emissions "
        f"depending on when electricity is consumed."
    )

    # ============================================================
    # GRAPH 2 - ELECTRICITY GENERATION MIX
    # ============================================================

    possible_fuels = [

        "gas_generation_mw",

        "coal_generation_mw",

        "nuclear_generation_mw",

        "wind_generation_mw",

        "solar_generation_mw",

        "hydro_generation_mw",

        "biomass_generation_mw"

    ]

    fuels = [

        column

        for column in possible_fuels

        if column in market.columns

    ]

    if fuels:

        generation = (

            market.groupby("month")[fuels]

            .mean()

            .reset_index()

        )

        fuel_names = {

            column: column.replace(
                "_generation_mw",
                ""
            ).replace(
                "_",
                " "
            ).title()

            for column in fuels

        }

        generation = generation.rename(
            columns=fuel_names
        )

        fuel_colours = {

            "Gas": "#153B79",

            "Coal": "#596A80",

            "Nuclear": "#5263AF",

            "Wind": "#2675E8",

            "Solar": "#93C5FD",

            "Hydro": "#14A5CA",

            "Biomass": "#7D9AB4"

        }

        fig = px.bar(

            generation,

            x="month",

            y=list(generation.columns[1:]),

            barmode="stack",

            title="Monthly mean electricity generation by source",

            labels={

                "month": "Month of 2025",

                "value": "Mean generation (MW)",

                "variable": "Generation source"

            },

            color_discrete_map=fuel_colours

        )

        raw_plot(fig)

        fuel_means = market[
            fuels
        ].mean()

        largest = fuel_means.idxmax()

        largest_name = fuel_names[largest]

        _v13_render(st.caption, 'caption')(
            f"Among the generation sources displayed, "
            f"{largest_name} recorded the highest "
            f"average power output in 2025, at "
            f"{fuel_means.max():,.0f} MW. "
            f"Changes in the composition of electricity "
            f"generation provide context for the "
            f"observed variations in grid carbon intensity."
        )

        _v13_render(st.caption, 'caption')(
            "The stacked bars represent monthly mean "
            "power generation in MW, not total monthly "
            "electricity production in MWh. "
            "Only the reported generation sources shown "
            "in the chart are included."
        )

    else:

        _v13_render(st.warning, 'warning')(
            "Detailed electricity generation columns "
            "are unavailable in the expected schema."
        )


    # ============================================================
    # COMPACT CORRELATION INDICATORS
    # Replaces two separate scatter plots.
    # ============================================================

    st.subheader("Electricity prices and the energy system")

    demand_r = safe_corr(
        market[DEMAND],
        market[A]
    )

    renewable_r = safe_corr(
        market[RENEW],
        market[A]
    )

    carbon_r = safe_corr(
        market[CI],
        market[A]
    )

    metrics([

        (

            "National demand vs Agile price",

            f"{demand_r:+.2f}"

        ),

        (

            "Renewable generation vs Agile price",

            f"{renewable_r:+.2f}"

        ),

        (

            "Carbon intensity vs Agile price",

            f"{carbon_r:+.2f}"

        )

    ])

    def correlation_description(value):

        if not np.isfinite(value):

            return "cannot be evaluated"

        if value > 0:

            return "is positively associated"

        if value < 0:

            return "is negatively associated"

        return "has no measured linear association"


    _v13_render(st.caption, 'caption')(
        f"National electricity demand "
        f"{correlation_description(demand_r)} "
        f"with Agile prices, while renewable generation "
        f"{correlation_description(renewable_r)} "
        f"with Agile prices. These Pearson correlations "
        f"range from -1 to +1 and are calculated using "
        f"the full available half-hourly dataset."
    )

    # ============================================================
    # GRAPH 3 - CARBON INTENSITY VS AGILE PRICE
    # ============================================================

    fig = scatter_preview(

        market,

        x=CI,

        y=A,

        opacity=0.35,

        title=(

            "Carbon intensity vs Agile electricity price "

            f"| Pearson r = {carbon_r:+.2f}"

        ),

        labels={

            CI: "Grid carbon intensity (gCO₂/kWh)",

            A: "Agile electricity price (p/kWh)"

        },

        color_discrete_sequence=[BLUE]

    )

    raw_plot(fig)

    if np.isfinite(carbon_r) and carbon_r > 0:

        _v13_render(st.caption, 'caption')(
            f"The positive correlation of {carbon_r:.2f} "
            f"indicates that more carbon-intensive electricity "
            f"tends to coincide with higher Agile prices. "
            f"Households pursuing lower electricity costs "
            f"may therefore also reduce their associated "
            f"emissions, although the relationship is "
            f"not perfect."
        )

    elif np.isfinite(carbon_r) and carbon_r < 0:

        _v13_render(st.caption, 'caption')(
            f"The negative correlation of {carbon_r:.2f} "
            f"suggests that cheaper electricity can coincide "
            f"with higher carbon intensity, creating a "
            f"potential conflict between financial and "
            f"environmental objectives."
        )

    else:

        _v13_render(st.caption, 'caption')(
            "No clear linear relationship can be "
            "established from the calculated correlation."
        )

    _v13_render(st.caption, 'caption')(
        "The scatter plot displays a sample of observations "
        "for readability. The reported correlation uses "
        "the full available dataset. Correlation does not "
        "establish a causal effect."
    )


    # ============================================================
    # GRAPH 4 - ANNUAL ASSOCIATED EMISSIONS
    # ============================================================

    st.subheader("Environmental impact of household flexibility")

    levels = sorted(

        set(RESEARCH_LEVELS + [flexibility])

    )

    environment_results = {

        level: scenario_result(

            level,

            receiving_slots,

            strategy

        )

        for level in levels

    }

    scenarios = scenario_summary(
        environment_results
    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=scenarios["Flexibility (%)"],

            y=scenarios["Mean emissions (kg)"],

            mode="lines+markers",

            name="Mean annual household emissions",

            line=dict(
                color=BLUE,
                width=3
            ),

            marker=dict(size=9)

        )

    )

    fig.add_hline(

        y=baseline_emissions,

        line_dash="dash",

        line_color=GREY,

        annotation_text="Baseline annual emissions"

    )

    fig.update_layout(

        title="Mean annual associated emissions by flexibility scenario",

        xaxis_title="Peak electricity consumption shifted (%)",

        yaxis_title="Mean annual emissions (kg CO₂/household/year)"

    )

    fig.update_xaxes(

        tickvals=levels,

        ticktext=[f"{level}%" for level in levels]

    )

    raw_plot(fig)

    zero_emissions = environment_results[0][
        "emissions_after"
    ].mean()

    forty_emissions = environment_results[40][
        "emissions_after"
    ].mean()

    forty_change = (
        zero_emissions - forty_emissions
    )

    forty_relative = (

        forty_change / zero_emissions * 100

        if zero_emissions > 0

        else float("nan")

    )

    _v13_render(st.caption, 'caption')(
        f"With the selected optimisation strategy, moving "
        f"from 0% to 40% flexibility changes average annual "
        f"consumption-associated emissions from "
        f"{zero_emissions:.1f} to {forty_emissions:.1f} "
        f"kg CO₂ per household/year. The estimated reduction "
        f"is {forty_change:.1f} kg CO₂/year "
        f"({forty_relative:.2f}%). "
        f"Total electricity consumption remains unchanged."
    )

    _v13_render(st.caption, 'caption')(
        "Each point represents a mean across all selected "
        "households, calculated after simulating each "
        "individual household's full year of electricity "
        "consumption. The graph does not represent "
        "daily mean emissions or national emissions."
    )


    # ============================================================
    # GRAPH 5 - ECONOMIC AND ENVIRONMENTAL TRADE-OFFS
    # ============================================================

    tradeoffs = []

    for option in STRATEGIES:

        for level in levels:

            result = scenario_result(

                level,

                receiving_slots,

                option

            )

            tradeoffs.append({

                "Strategy": option,

                "Flexibility (%)": level,

                "Mean annual additional savings (£)":
                    result["extra_savings"].mean(),

                "Mean annual emissions reduction (kg CO₂)":
                    result["emissions_reduction"].mean()

            })

    tradeoffs = pd.DataFrame(
        tradeoffs
    )

    fig = px.line(

        tradeoffs,

        x="Mean annual additional savings (£)",

        y="Mean annual emissions reduction (kg CO₂)",

        color="Strategy",

        markers=True,

        hover_data=["Flexibility (%)"],

        title="Economic and environmental trade-offs",

        labels={

            "Mean annual additional savings (£)":
                "Mean additional savings (£/household/year)",

            "Mean annual emissions reduction (kg CO₂)":
                "Mean emissions reduction (kg CO₂/household/year)"

        },

        color_discrete_map={

            STRATEGIES[0]: "#153B79",

            STRATEGIES[1]: "#14A5CA",

            STRATEGIES[2]: "#7198D0"

        }

    )

    fig.update_traces(

        line=dict(width=3),

        marker=dict(size=8)

    )

    raw_plot(fig)

    # Compare all three strategies at the selected
    # flexibility percentage.

    strategy_comparison = tradeoffs.loc[

        tradeoffs["Flexibility (%)"] == flexibility

    ].copy()

    if flexibility > 0:

        best_cost = strategy_comparison.loc[

            strategy_comparison[
                "Mean annual additional savings (£)"
            ].idxmax()

        ]

        best_carbon = strategy_comparison.loc[

            strategy_comparison[
                "Mean annual emissions reduction (kg CO₂)"
            ].idxmax()

        ]

        _v13_render(st.caption, 'caption')(
            f"At {flexibility}% flexibility, the highest "
            f"mean additional annual saving among the "
            f"three strategies is "
            f"£{best_cost['Mean annual additional savings (£)']:.2f} "
            f"per household. The highest estimated annual "
            f"emissions reduction is "
            f"{best_carbon['Mean annual emissions reduction (kg CO₂)']:.2f} "
            f"kg CO₂ per household. These outcomes illustrate "
            f"whether prioritising costs and prioritising "
            f"emissions lead to different results."
        )

    else:

        _v13_render(st.caption, 'caption')(
            "Without consumption redistribution, all three "
            "strategies produce identical outcomes. Select "
            "a positive flexibility percentage to compare "
            "their economic and environmental consequences."
        )

    _v13_render(st.caption, 'caption')(
        "Cost optimisation selects the cheapest eligible "
        "intervals. Emissions optimisation selects the "
        "least carbon-intensive intervals. Balanced "
        "optimisation gives equal weight to daily "
        "normalised prices and carbon intensity."
    )

    # ------------------------------------------------------------
    # FINAL ENVIRONMENTAL INTERPRETATION
    # ------------------------------------------------------------

    st.divider()

    st.subheader("What the environmental analysis tells us")

    _v13_render(st.markdown, 'markdown')(
        """
        Changing electricity tariffs without changing
        electricity consumption does not alter
        consumption-associated emissions.

        Environmental differences arise when households
        redistribute their electricity demand across
        intervals with different grid carbon intensities.

        The economic and environmental objectives can
        therefore be complementary, but selecting the
        cheapest electricity intervals does not necessarily
        maximise the reduction in associated emissions.
        """
    )

    _v13_render(st.caption, 'caption')(
        "The emissions estimates use observed average "
        "grid carbon intensity, not marginal emissions "
        "factors. They are consumption-accounting estimates "
        "and should not be interpreted as measured or "
        "causally estimated national emissions reductions."
    )

    _v13_render(st.caption, 'caption')(
        "Embedded wind is not separately added to the "
        "generation chart because of the unresolved "
        "reconciliation discrepancy. The generation "
        "timestamp convention remains a methodological "
        "limitation."
    )

elif page.startswith("05"):
    import hashlib

    from plotly.subplots import make_subplots

    section(

        "05 — Distributive justice",

        "Fair distribution of the benefits and burdens "
        "of dynamic electricity pricing."

    )

    _v13_render(st.markdown, 'markdown')("**Methodological note**")
    _v13_render(st.caption, 'caption')('Following Wang and Lo (2021), the analysis distinguishes the distribution of economic benefits, the distribution of financial burdens and the recognition of different household circumstances. Potential flexibility benefits are estimated by comparing identical consumption profiles under the fixed 0%, 20% and 40% scenarios. Monthly financial exposure is defined relative to the conventional monthly bill, including daily standing charges. Ofgem subgroup evidence comes from a separate household study and is not linked to our historical consumption profiles. Household income, vulnerability and actual flexibility capabilities cannot be inferred for individual households in our dataset. The results identify potential inequalities in economic opportunities and financial exposure, rather than directly demonstrating distributive injustice.')
    st.divider()


    # ------------------------------------------------------------
    # THEORETICAL FRAMEWORK
    # ------------------------------------------------------------

    _v13_render(st.markdown, 'markdown')(
        """
        ### From energy flexibility to energy justice

        Dynamic electricity pricing creates economic and
        environmental opportunities. However, households
        may have different resources and capabilities
        to exploit these opportunities.

        Drawing on Wang and Lo (2021), this analysis
        distinguishes three interconnected issues:

        **Distribution of benefits:** who can access
        the additional financial value of flexibility?

        **Distribution of burdens:** who may be exposed
        to unfavourable electricity prices and higher bills?

        **Recognition of household circumstances:** what
        differences in technology, resources and needs
        may influence these opportunities?
        """
    )

    research_scenario_caption()

    _v13_render(st.caption, 'caption')(
        "The 0%, 20% and 40% scenarios describe hypothetical "
        "shares of peak electricity consumption shifted. "
        "They are not observed socioeconomic groups or "
        "measured household flexibility capabilities."
    )

    fixed = {

        level: scenario_result(level)

        for level in RESEARCH_LEVELS

    }

    baseline = fixed[0]
    moderate = fixed[20]
    high = fixed[40]

    gap = (

        baseline["agile_after"]

        - high["agile_after"]

    )

    relative_gap = pct_of_bill(
        gap,
        baseline["conventional_bill"]
    )

    check(
        np.allclose(
            gap,
            high["extra_savings"],
            rtol=1e-9,
            atol=1e-7
        ),
        "Flexibility benefit gap reconciliation failed."
    )

    # ============================================================
    # LOAD MONTHLY GOLD
    # ============================================================

    @st.cache_resource(
        show_spinner=False,
        max_entries=1
    )
    def load_monthly_justice(
        path,
        manifest_digest,
        annual_digest
    ):

        with np.load(
            path,
            allow_pickle=False
        ) as archive:

            result = {
                key: archive[key]
                for key in archive.files
            }

        if (
            str(result["manifest_digest"].item())
            != manifest_digest
        ):

            raise ValueError(
                "Monthly Gold and Silver manifest mismatch."
            )

        if (
            str(result["annual_gold_digest"].item())
            != annual_digest
        ):

            raise ValueError(
                "Monthly Gold and annual Gold mismatch."
            )

        for value in result.values():

            value.setflags(write=False)

        return result


    monthly_path = (

        ROOT / "data" / "gold"

        / "step_3_justice_v6"

        / "monthly_household_outcomes.npz"

    )

    annual_path = (

        ROOT / "data" / "gold"

        / "step_3_dashboard_v3"

        / "dashboard_analytics.npz"

    )

    monthly = load_monthly_justice(

        str(monthly_path),

        hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),

        hashlib.sha256(
            annual_path.read_bytes()
        ).hexdigest()

    )

    check(
        np.array_equal(
            base["household_id"].astype(str).to_numpy(),
            monthly["household_ids"]
        ),
        "Monthly household mapping mismatch."
    )

    months = monthly["months"].tolist()

    conventional_monthly = (
        monthly["conventional"][mask]
    )

    agile_monthly = (
        monthly["agile"][:, mask, :]
    )

    check(
        agile_monthly.shape
        == (3, len(households), 12),
        "Monthly household dimensions mismatch."
    )

    for i, level in enumerate(RESEARCH_LEVELS):

        check(
            np.allclose(

                agile_monthly[i].sum(axis=1),

                fixed[level]["agile_after"].to_numpy(),

                rtol=1e-9,

                atol=1e-6

            ),
            f"Monthly and annual scenario mismatch: {level}%"
        )

    check(
        np.allclose(
            conventional_monthly.sum(axis=1),
            baseline["conventional_bill"].to_numpy(),
            rtol=1e-9,
            atol=1e-6
        ),
        "Monthly conventional bill mismatch."
    )

    # ============================================================
    # PRIMARY DISTRIBUTIVE JUSTICE INDICATORS
    # ============================================================

    monthly_penalty = (
        agile_monthly - conventional_monthly[None, :, :]
    )

    monthly_losers = monthly_penalty > 1e-8

    exposed_baseline = (
        monthly_losers[0].any(axis=1).mean() * 100
    )

    exposed_high = (
        monthly_losers[2].any(axis=1).mean() * 100
    )

    metrics([

        (

            "Mean annual flexibility benefit gap",

            money(gap.mean())

        ),

        (

            "Median annual flexibility benefit gap",

            money(gap.median())

        ),

        (

            "Median relative benefit gap",

            f"{relative_gap.median():.2f}%"

        ),

        (

            "Monthly financial exposure — 0%",

            f"{exposed_baseline:.2f}%"

        )

    ])

    _v13_render(st.caption, 'caption')(
        "The benefit gap compares 0% with 40% flexibility "
        "for the same household. Relative benefits are "
        "expressed as a percentage of the conventional "
        "annual electricity bill. Monthly financial exposure "
        "is the percentage of households with at least one "
        "month in which Agile is more expensive."
    )

    # ============================================================
    # GRAPH 1 - DISTRIBUTION OF BENEFITS
    # ============================================================

    st.divider()

    _v10_justice_start()

    st.subheader(
        "1. Unequal access to flexibility benefits"
    )

    _v13_render(st.markdown, 'markdown')(
        """
        The same tariff can generate different economic
        opportunities depending on how much electricity
        households are able to redistribute.

        This comparison measures the additional savings
        that could be missed if a household cannot move
        from 0% to 40% flexibility.
        """
    )

    unit = st.radio(

        "Measure flexibility benefits in",

        [
            "Absolute annual savings (£)",
            "Relative annual savings (%)"
        ],

        horizontal=True,

        key="justice_v6_unit"

    )

    if unit.startswith("Absolute"):

        values = gap.dropna()

        axis_title = (
            "Additional annual savings (£/household/year)"
        )

        unit_name = "£"

    else:

        values = relative_gap.dropna()

        axis_title = (
            "Additional savings (% of conventional annual bill)"
        )

        unit_name = "%"

    distribution = pd.DataFrame({
        "Benefit gap": values
    })

    fig = px.histogram(

        distribution,

        x="Benefit gap",

        nbins=45,

        title="Distribution of the 0%–40% flexibility benefit gap",

        labels={
            "Benefit gap": axis_title
        },

        color_discrete_sequence=[BLUE]

    )

    fig.update_yaxes(
        title="Number of households"
    )

    fig.add_vline(

        x=values.median(),

        line_color=NAVY,

        line_dash="dash",

        annotation_text="Median"

    )

    raw_plot(fig)

    def format_benefit(value):

        if unit_name == "£":

            return f"£{value:,.2f}"

        return f"{value:.2f}%"

    p10 = values.quantile(0.10)
    p90 = values.quantile(0.90)

    _v13_render(st.caption, 'caption')(

        f"The median potential flexibility benefit is "
        f"{format_benefit(values.median())}. "
        f"The central 80% of household outcomes lies "
        f"between {format_benefit(p10)} "
        f"and {format_benefit(p90)}. "
        f"This quantifies how differently households "
        f"could benefit from the same increase "
        f"in hypothetical flexibility."

    )

    _v13_render(st.caption, 'caption')(
        "A benefit gap represents a simulated opportunity "
        "difference, not an observed financial loss or "
        "proof that a household is unable to adapt."
    )

    # ============================================================
    # GRAPH 2 - DISTRIBUTION OF FINANCIAL BURDENS
    # ============================================================

    st.divider()

    st.subheader(
        "2. Monthly financial exposure"
    )

    _v13_render(st.markdown, 'markdown')(
        """
        Annual savings can conceal temporary financial
        disadvantages. A household may benefit from
        Agile over the entire year while facing a higher
        electricity bill during particular months.

        This is especially relevant when considering
        households that may have limited financial
        capacity to absorb temporary increases in costs.
        """
    )

    exposure_rows = []

    for i, level in enumerate(RESEARCH_LEVELS):

        for j, month in enumerate(months):

            affected = monthly_losers[i, :, j]

            count = int(affected.sum())

            penalty = monthly_penalty[i, :, j]

            mean_penalty = (

                float(penalty[affected].mean())

                if count > 0

                else 0.0

            )

            exposure_rows.append({

                "Month": month,

                "Scenario": f"{level}% flexibility",

                "Flexibility": level,

                "Households exposed": count,

                "Exposed households (%)":
                    count / len(households) * 100,

                "Mean excess monthly bill (£)":
                    mean_penalty

            })

    exposure = pd.DataFrame(exposure_rows)

    fig = px.bar(

        exposure,

        x="Month",

        y="Exposed households (%)",

        color="Scenario",

        barmode="group",

        title="Households paying more under Agile by month",

        color_discrete_map={

            "0% flexibility": "#8295AD",

            "20% flexibility": BLUE,

            "40% flexibility": NAVY

        },

        hover_data={

            "Households exposed": True,

            "Mean excess monthly bill (£)": ":.2f",

            "Flexibility": False

        }

    )

    fig.update_xaxes(

        tickvals=months,

        ticktext=[

            pd.Timestamp(m + "-01").strftime("%b")

            for m in months

        ],

        title="Month of 2025"

    )

    fig.update_yaxes(

        title="Households with a higher monthly bill (%)",

        range=[0, 100]

    )

    raw_plot(fig)

    baseline_exposure = exposure.loc[
        exposure["Flexibility"] == 0
    ]

    worst_month = baseline_exposure.loc[

        baseline_exposure[
            "Exposed households (%)"
        ].idxmax()

    ]

    annual_winners = (
        baseline["total_savings"].to_numpy() > 1e-8
    )

    annual_winners_with_monthly_penalty = int(

        (

            annual_winners

            & monthly_losers[0].any(axis=1)

        ).sum()

    )

    _v13_render(st.caption, 'caption')(

        f"Without flexibility, {exposed_baseline:.2f}% "
        f"of households experience at least one month "
        f"in which Agile is more expensive than the "
        f"conventional tariff. Under 40% flexibility, "
        f"this share becomes {exposed_high:.2f}%."

    )

    if worst_month["Households exposed"] > 0:

        month_name = pd.Timestamp(

            worst_month["Month"] + "-01"

        ).strftime("%B")

        _v13_render(st.caption, 'caption')(

            f"At 0% flexibility, {month_name} records "
            f"the highest share of households facing "
            f"a higher monthly Agile bill "
            f"({worst_month['Exposed households (%)']:.2f}%). "
            f"Among affected households in that month, "
            f"the average additional expenditure is "
            f"£{worst_month['Mean excess monthly bill (£)']:.2f}."

        )

    else:

        _v13_render(st.caption, 'caption')(
            "Under the baseline assumptions, no selected "
            "household faces a higher monthly Agile bill "
            "in the analysed year."
        )

    _v13_render(st.markdown, 'markdown')(

        f"""
        **Annual benefits versus monthly burdens**

        {annual_winners_with_monthly_penalty:,} households
        benefit from Agile annually without flexibility
        but still experience at least one disadvantageous
        month.
        """

    )

    # Compact supporting table, not another graph.

    risk_summary = []

    for i, level in enumerate(RESEARCH_LEVELS):

        months_exposed = monthly_losers[i].sum(axis=1)

        positive_excess = np.maximum(
            monthly_penalty[i],
            0
        )

        risk_summary.append({

            "Flexibility": f"{level}%",

            "Households with ≥1 adverse month (%)":
                (months_exposed > 0).mean() * 100,

            "Mean number of adverse months":
                months_exposed.mean(),

            "Mean cumulative adverse-month excess (£/year)":
                positive_excess.sum(axis=1).mean()

        })

    st.dataframe(

        pd.DataFrame(risk_summary).round(2),

        hide_index=True,

        width="stretch"

    )

    _v13_render(st.caption, 'caption')(
        "Monthly bills include the actual standing charges "
        "applicable on each calendar day. The cumulative "
        "adverse-month excess sums positive monthly bill "
        "differences and does not subtract savings in "
        "favourable months."
    )

    _v13_render(st.caption, 'caption')(
        "These are simulated monthly electricity charges, "
        "not observed customer payments or direct-debit "
        "amounts. The analysis does not measure household "
        "income, liquidity or actual financial hardship."
    )

    # ============================================================
    # GRAPH 3 - RECOGNITION OF HOUSEHOLD CIRCUMSTANCES
    # ============================================================

    st.divider()

    import html
    import textwrap

    st.subheader(
        "3. Household circumstances and flexibility opportunities"
    )

    _v13_render(st.markdown, 'markdown')(
        """
        ### Independent Ofgem evidence

        The graph compares the average proportion of
        electricity consumed during peak hours by households
        with different technological and socioeconomic
        characteristics.

        Each percentage represents the share of a household's
        total daily electricity consumption occurring
        between **16:00 and 19:00**.

        For example, a peak consumption share of 18% means
        that approximately 18% of total daily electricity
        consumption occurs during these three hours.
        """
    )

    _v13_render(st.caption, 'caption')(
        "The household categories are different for each "
        "comparison and can overlap. The percentages do "
        "not represent the percentage of households, "
        "financial savings or directly observed flexibility "
        "capabilities."
    )

    required_columns = [

        "dimension",

        "group_a",

        "group_b",

        "group_a_peak_share_pct",

        "group_b_peak_share_pct"

    ]

    missing = [

        column

        for column in required_columns

        if column not in social.columns

    ]

    if not missing:

        evidence = social[
            required_columns
        ].copy()

        for column in [

            "group_a_peak_share_pct",

            "group_b_peak_share_pct"

        ]:

            evidence[column] = pd.to_numeric(

                evidence[column],

                errors="coerce"

            )

        valid = (

            evidence[required_columns].notna().all(axis=1)

            & evidence["group_a_peak_share_pct"].between(
                0, 100
            )

            & evidence["group_b_peak_share_pct"].between(
                0, 100
            )

        )

        if not valid.all():

            _v13_render(st.warning, 'warning')(
                f"{int((~valid).sum())} invalid subgroup "
                "comparisons cannot be displayed."
            )

        evidence = evidence.loc[valid].copy()

        evidence["Characteristic"] = (

            evidence["dimension"]

            .astype(str)

            .str.replace("_", " ", regex=False)

            .str.title()

        )

        evidence["Gap (pp)"] = (

            evidence["group_a_peak_share_pct"]

            - evidence["group_b_peak_share_pct"]

        ).abs()

        evidence = evidence.sort_values(

            "Gap (pp)",

            ascending=False

        ).reset_index(drop=True)

        if not evidence.empty:

            # ----------------------------------------------------
            # Create two explicitly labelled bars per characteristic
            # ----------------------------------------------------

            positions = []

            tick_labels = []

            values = []

            colours = []

            hover_data = []

            table_rows = []

            def wrap_label(value, width=43):

                return "<br>".join(

                    html.escape(part)

                    for part in textwrap.wrap(
                        str(value),
                        width=width
                    )

                )

            for i, row in evidence.iterrows():

                dimension = str(
                    row["Characteristic"]
                )

                group_a = str(
                    row["group_a"]
                )

                group_b = str(
                    row["group_b"]
                )

                value_a = float(
                    row["group_a_peak_share_pct"]
                )

                value_b = float(
                    row["group_b_peak_share_pct"]
                )

                difference = float(
                    row["Gap (pp)"]
                )

                # Leave one empty space between each pair.

                positions.extend([
                    i * 3,
                    i * 3 + 1
                ])

                tick_labels.extend([

                    f"<b>{html.escape(dimension)}</b>"
                    f"<br>{wrap_label(group_a)}",

                    wrap_label(group_b)

                ])

                values.extend([
                    value_a,
                    value_b
                ])

                colours.extend([
                    "#153B79",
                    "#84B6F4"
                ])

                hover_data.extend([

                    [
                        dimension,
                        group_a,
                        difference
                    ],

                    [
                        dimension,
                        group_b,
                        difference
                    ]

                ])

                table_rows.extend([

                    {

                        "Household characteristic":
                            dimension,

                        "Household category":
                            group_a,

                        "Peak consumption share (%)":
                            value_a,

                        "Difference within comparison (pp)":
                            difference

                    },

                    {

                        "Household characteristic":
                            dimension,

                        "Household category":
                            group_b,

                        "Peak consumption share (%)":
                            value_b,

                        "Difference within comparison (pp)":
                            difference

                    }

                ])

            # ----------------------------------------------------
            # Plot using actual household category names
            # ----------------------------------------------------

            fig = go.Figure()

            fig.add_trace(

                go.Bar(

                    x=values,

                    y=positions,

                    orientation="h",

                    width=0.68,

                    marker_color=colours,

                    text=[

                        f"{value:.1f}%"

                        for value in values

                    ],

                    textposition="outside",

                    cliponaxis=False,

                    customdata=hover_data,

                    hovertemplate=(

                        "<b>%{customdata[0]}</b><br>"

                        "%{customdata[1]}<br>"

                        "Peak consumption share: %{x:.1f}%<br>"

                        "Difference within comparison: "
                        "%{customdata[2]:.1f} pp"

                        "<extra></extra>"

                    ),

                    showlegend=False

                )

            )

            fig.update_layout(

                title=(

                    "Average share of daily electricity "
                    "consumed during peak hours"

                ),

                height=max(
                    550,
                    95 * len(evidence) + 100
                ),

                showlegend=False,

                xaxis_title=(

                    "Electricity consumed between "
                    "16:00 and 19:00 "
                    "(% of total daily consumption)"

                ),

                yaxis_title=None

            )

            fig.update_xaxes(

                range=[

                    0,

                    max(
                        1,
                        max(values) * 1.20
                    )

                ],

                ticksuffix="%",

                automargin=True

            )

            fig.update_yaxes(

                tickmode="array",

                tickvals=positions,

                ticktext=tick_labels,

                autorange="reversed",

                automargin=True,

                tickfont=dict(size=12),

                showgrid=False

            )

            raw_plot(fig)

            _v13_render(st.caption, 'caption')(
                "The two shades of green distinguish the two "
                "categories within each characteristic. "
                "They do not represent two fixed groups of "
                "households across the entire graph. "
                "pp = percentage points."
            )

            # ----------------------------------------------------
            # Data-driven interpretation
            # ----------------------------------------------------

            largest = evidence.iloc[0]

            _v13_render(st.markdown, 'markdown')(
                "**What does this graph tell us?**"
            )

            _v13_render(st.write, 'write')(

                f"The largest observed difference concerns "
                f"{largest['Characteristic'].lower()}. "
                f"Households classified as "
                f"'{largest['group_a']}' consume "
                f"{largest['group_a_peak_share_pct']:.1f}% "
                f"of their daily electricity during peak hours, "
                f"compared with "
                f"{largest['group_b_peak_share_pct']:.1f}% "
                f"for '{largest['group_b']}'. "
                f"The difference is "
                f"{largest['Gap (pp)']:.1f} percentage points."

            )

            _v13_render(st.write, 'write')(

                "Since Agile electricity prices are generally "
                "highest between 16:00 and 19:00, households "
                "consuming a greater proportion of electricity "
                "during these hours may face greater proportional "
                "exposure to expensive electricity periods."

            )

            _v13_render(st.caption, 'caption')(

                "A smaller peak consumption share does not "
                "necessarily indicate lower absolute electricity "
                "consumption or a greater ability to redistribute "
                "future demand. These comparisons describe observed "
                "associations, not causal effects."

            )

            # ----------------------------------------------------
            # Supporting table with actual household categories
            # ----------------------------------------------------

            with st.expander(
                "View exact household subgroup comparisons"
            ):

                st.dataframe(

                    pd.DataFrame(
                        table_rows
                    ).round(2),

                    hide_index=True,

                    width="stretch"

                )

        else:

            _v13_render(st.warning, 'warning')(
                "No valid Ofgem subgroup comparisons are available."
            )

    else:

        _v13_render(st.warning, 'warning')(

            "The Ofgem comparison schema requires review. "
            f"Missing columns: {', '.join(missing)}"

        )

        st.dataframe(

            social,

            hide_index=True,

            width="stretch"

        )


    # ------------------------------------------------------------
    # RETAIN ADDITIONAL OFGEM CONSUMER EVIDENCE
    # ------------------------------------------------------------

    with st.expander(
        "Additional Ofgem consumer behaviour evidence"
    ):

        st.dataframe(

            consumer,

            hide_index=True,

            width="stretch"

        )

        _v13_render(st.caption, 'caption')(

            "Consumer survey evidence is independent of "
            "the historical household consumption profiles. "
            "Reported behaviour or willingness to shift "
            "particular activities should not be interpreted "
            "as the percentage of total household electricity "
            "consumption that can be shifted."

        )

    _v13_render(st.markdown, 'markdown')(
        """
        **Connection with distributive justice**

        Household consumption patterns may reflect differences
        in technology ownership, financial resources, essential
        electricity needs and daily routines.

        If these circumstances restrict the ability of some
        households to avoid expensive electricity periods,
        the opportunities created by dynamic pricing may
        not be equally accessible.

        However, these comparisons alone do not establish
        that a particular household is unable to adapt,
        nor do they demonstrate distributive injustice.
        """
    )

    st.divider()

    _v10_justice_transition()

    st.subheader(
        "Managing the distributive justice tension"
    )

    _v13_render(st.markdown, 'markdown')(

        f"""
        The simulations show that creating economic value
        and distributing that value fairly are different
        challenges.

        Increasing flexibility from 0% to 40% generates
        a median additional annual benefit of
        **{money(gap.median())} per household**.

        Meanwhile, **{exposed_baseline:.2f}%** of households
        experience at least one simulated adverse month
        without flexibility.

        These results motivate an assessment of both
        access to benefits and exposure to burdens.
        """

    )

    left, right = st.columns(2)

    with left:

        _v13_render(st.markdown, 'markdown')(
            "**1. More accessible flexibility opportunities**"
        )

        _v13_render(st.write, 'write')(
            "Provide affordable scheduling assistance, "
            "automation and guidance to help households "
            "capture additional savings without requiring "
            "substantial technological investment."
        )

        _v13_render(st.markdown, 'markdown')(
            "**2. Recognition of household constraints**"
        )

        _v13_render(st.write, 'write')(
            "Identify customers whose essential electricity "
            "needs, financial resources or daily routines "
            "limit their ability to respond to dynamic prices."
        )

    with right:

        _v13_render(st.markdown, 'markdown')(
            "**3. Financial risk protection**"
        )

        _v13_render(st.write, 'write')(
            "Provide transparent comparisons of monthly "
            "and annual tariff outcomes, price alerts and "
            "suitable tariff alternatives for households "
            "with limited capacity to absorb volatility."
        )

        _v13_render(st.markdown, 'markdown')(
            "**4. Distributional monitoring**"
        )

        _v13_render(st.write, 'write')(
            "Monitor both proportional and absolute "
            "benefits, exposure to adverse months and "
            "access to flexibility-enabling resources "
            "rather than relying only on average savings."
        )

    _v13_render(st.caption, 'caption')(
        "These managerial responses are proposed applications "
        "of the energy justice framework, not interventions "
        "whose effectiveness has been empirically evaluated."
    )

    # ============================================================
    # CONCLUDING INTERPRETATION
    # ============================================================

    st.subheader("What does this mean for a just energy transition?")

    _v13_render(st.markdown, 'markdown')(
        """
        Dynamic pricing can support the energy transition
        by creating incentives for more flexible electricity
        consumption.

        However, the ability to capture its full economic
        value may depend on resources and opportunities
        that are not equally accessible to every household.

        A just transition therefore requires attention
        not only to aggregate economic and environmental
        performance, but also to the distribution of
        benefits, financial risks and the circumstances
        that influence household participation.

        The managerial challenge is to preserve the
        incentives created by dynamic pricing while
        improving equitable access to their benefits.
        """
    )

    _v13_render(st.caption, 'caption')(
        "The simulations quantify potential differences "
        "in financial outcomes. They do not establish "
        "observed socioeconomic injustice or causal "
        "relationships between household characteristics "
        "and flexibility capabilities."
    )

    with st.expander(
        "Academic framework and methodological references"
    ):

        _v13_render(st.markdown, 'markdown')(
            """
            **Wang, X., & Lo, K. (2021).**
            Just transition: A conceptual review.
            Energy Research & Social Science, 82, 102291.

            https://doi.org/10.1016/j.erss.2021.102291

            **Relevant sections**

            - Introduction, pp. 1–2: sustainability transitions
              may generate new inequalities and vulnerabilities.

            - Section 3.1, p. 3: recognition of different
              circumstances as a foundation for justice.

            - Section 3.3, pp. 4–5: distribution of energy
              benefits and burdens and the principles
              of energy justice.

            - Section 7, pp. 7–8: vulnerable groups and
              structural causes of injustice.

            **Application to Agile**

            The analytical indicators and managerial
            responses presented here are our application
            of the conceptual framework to dynamic
            electricity pricing. They are not findings
            originally reported by Wang and Lo.
            """
        )

# METHODOLOGY AND LIMITATIONS
# ============================================================

st.divider()

with st.expander(
    "Research methodology, data quality and limitations"
):

    _v13_render(st.markdown, 'markdown')(
        """
        ### Research question

        How does Octopus Energy's Agile tariff create
        tensions between energy flexibility and distributive
        justice, and how can these tensions be managed?

        ### Household consumption

        Historical household consumption from 2013 is
        aligned with electricity market data from 2025
        using the approved calendar mapping.

        The analysis includes 3,955 household profiles,
        of which 881 are fully observed. Only 0.0337%
        of consumption measurements are imputed.

        ### Electricity bills

        Annual electricity bills are calculated by
        multiplying half-hourly electricity consumption
        by the applicable VAT-inclusive unit price,
        then adding annual standing charges.

        
    ### Reference research scenarios

    The fixed research scenarios are 0%, 20% and
    40% of peak electricity consumption shifted.

    All three reference scenarios use cost optimisation
    and eight receiving half-hours per day.

    Every scenario is applied individually to each
    selected household before calculating mean and
    median outcomes.

    Exploratory controls inside Environmental
    Sustainability do not change
    these reference research assumptions.

    ### Flexibility simulation
    

        The selected percentage of consumption between
        16:00 and 19:00 is removed proportionally from
        peak intervals and redistributed equally across
        selected off-peak intervals within the same day.

        Total daily and annual energy consumption
        remain unchanged.

        Cost optimisation uses Agile electricity prices.

        Environmental optimisation uses observed
        average grid carbon intensity.

        Balanced optimisation assigns equal weights to
        daily min-max normalised prices and carbon intensity.

        ### Limitations

        - Historical 2013 consumption proxy.
        - Calendar alignment assumptions.
        - Imputed consumption measurements.
        - Atypical household profiles.
        - Embedded wind reconciliation discrepancy.
        - Unconfirmed generation timestamp convention.
        - No individual socioeconomic information.
        - Perfect retrospective knowledge of prices
          and carbon intensity.
        - No appliance-specific or power-capacity constraints.
        - No feedback from simulated demand shifts
          into market prices or grid carbon intensity.
        - No nationally representative household weighting.
        - Associated emissions are not marginal or
          causal system-wide emissions reductions.
        - Flexibility levels are hypothetical, not
          empirically observed household capabilities.

        ### Interpretation

        The results are descriptive and counterfactual,
        rather than causal estimates or guaranteed
        real-world financial outcomes.
        """
    )

    st.json({
        "silver_run": manifest.get("run_id"),
        "market_intervals": len(market),
        "household_profiles": len(registry),
        "fully_observed_profiles":
            int(registry["complete"].sum()),
        "approved_manifest": str(MANIFEST)
    })

_v13_render(st.caption, 'caption')(
    "Agile Energy Analytics | Academic research project | "
    "Frozen Silver | Precomputed Gold | Dashboard V13"
)
