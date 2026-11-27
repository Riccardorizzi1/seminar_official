from octopus_v13_render import renderer as _v13_render, STYLE as _v13_style

from pathlib import Path

import colorsys
import inspect
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

GOLD = (
    Path.home()
    / "Octopus Energy Project"
    / "data/gold/step_3_dashboard_v3"
    / "dashboard_analytics.npz"
)

LEVELS = (0, 20, 40)


def money(x):
    return f"£{float(x):,.2f}"


# ============================================================
# EXISTING BASELINE VERIFICATION
# ============================================================

@st.cache_data(show_spinner=False)
def baseline():

    root = Path.home() / "Octopus Energy Project"

    manifest = json.loads(
        (
            root / "data/silver/approved_silver_manifest.json"
        ).read_text(encoding="utf-8")
    )

    if (
        manifest.get("status") != "PASS"
        or not manifest.get("silver_frozen")
    ):
        raise ValueError("Silver layer not approved.")

    files = manifest["data_files"]

    registry = pd.read_csv(
        files["household_registry"]
    ).sort_values("matrix_row")

    ids_silver = registry.household_id.astype(str).to_numpy()

    if (
        len(registry) != 3955
        or len(set(ids_silver)) != len(registry)
        or not np.array_equal(
            registry.matrix_row.to_numpy(),
            np.arange(len(registry))
        )
    ):
        raise ValueError("Invalid Silver household registry.")

    market = pd.read_csv(
        files["integrated_market"],
        usecols=[
            "timestamp_utc",
            "source_interval_index",
            "agile_unit_rate_gbp_kwh",
            "conventional_unit_rate_gbp_kwh"
        ]
    )

    market = market.sort_values(
        "timestamp_utc"
    ).reset_index(drop=True)

    mapping = market.source_interval_index.to_numpy(dtype=int)

    if not np.array_equal(
        np.sort(mapping), np.arange(17520)
    ):
        raise ValueError("Invalid Silver calendar mapping.")

    standing = pd.read_csv(files["standing_charges"])

    if len(standing) != 365:
        raise ValueError("Invalid standing charges.")

    ap = market.agile_unit_rate_gbp_kwh.to_numpy(float)
    cp = market.conventional_unit_rate_gbp_kwh.to_numpy(float)

    sa = float(standing.agile_standing_gbp_day.sum())
    sc = float(standing.conventional_standing_gbp_day.sum())

    matrix = np.load(
        files["household_matrix"],
        mmap_mode="r"
    )

    if matrix.shape != (3955, 17520):
        raise ValueError("Invalid Silver matrix.")

    required = [
        "annual_kwh",
        "agile_bill",
        "conventional_bill"
    ]

    try:

        with np.load(GOLD, allow_pickle=False) as z:

            ids = z["household_ids"].astype(str)
            cols = z["baseline_columns"].astype(str).tolist()
            values = z["baseline_values"]

            if values.shape == (len(cols), len(ids)):
                values = values.T

            if values.shape != (len(ids), len(cols)):
                raise ValueError("Gold baseline shape differs.")

            b = pd.DataFrame(
                values,
                columns=cols,
                index=ids
            )[required].astype(float)

        if (
            not b.index.is_unique
            or set(b.index) != set(ids_silver)
            or not np.isfinite(b.to_numpy()).all()
        ):
            raise ValueError("Gold baseline does not reconcile.")

        # Validate a small sample without loading the full matrix.

        for i in [0, len(registry)//2, len(registry)-1]:

            q = matrix[i, mapping]

            expected = [
                q.sum(),
                np.sum(q * ap) + sa,
                np.sum(q * cp) + sc
            ]

            if not np.allclose(
                b.loc[ids_silver[i], required].to_numpy(float),
                expected,
                atol=0.05,
                rtol=0
            ):
                raise ValueError(
                    "Gold bills differ from Silver references."
                )

        b.attrs["source"] = (
            "Existing Gold baseline, reconciled against Silver"
        )

    except (KeyError, ValueError, TypeError, OSError) as exc:

        if str(exc) in (
            "Gold bills differ from Silver references.",
            "Gold baseline does not reconcile."
        ):
            raise

        # Fallback only if the Gold archive schema differs.
        # Reconstruct the baseline, not the flexibility scenarios.

        values = []

        for i in range(0, len(registry), 32):

            q = matrix[i:i+32][:, mapping]

            values.append(
                np.column_stack([
                    q.sum(axis=1),
                    (q * ap).sum(axis=1) + sa,
                    (q * cp).sum(axis=1) + sc
                ])
            )

        b = pd.DataFrame(
            np.vstack(values),
            columns=required,
            index=ids_silver
        )

        if not np.isfinite(b.to_numpy()).all():
            raise ValueError("Invalid reconstructed baseline.")

        b.attrs["source"] = (
            f"Silver baseline verification fallback: {exc}"
        )

    del matrix

    return b


# ============================================================
# ALIGN EXISTING V7 REFERENCE SCENARIOS
# ============================================================

def align_fixed(fixed, b):

    frames = {}

    for level in LEVELS:

        f = fixed[level].copy()

        needed = [
            "agile_after",
            "extra_savings",
            "shifted_kwh"
        ]

        if (
            not isinstance(f, pd.DataFrame)
            or not set(needed).issubset(f.columns)
        ):
            raise ValueError(
                f"Unexpected reference scenario {level}: "
                f"{getattr(f, 'columns', None)}"
            )

        f[needed] = f[needed].astype(float)

        if (
            f.empty
            or not np.isfinite(f[needed].to_numpy()).all()
        ):
            raise ValueError("Invalid scenario values.")

        observed_base = (
            f["agile_after"] + f["extra_savings"]
        ).to_numpy(float)

        id_col = next(
            (
                c for c in f
                if str(c).lower() == "household_id"
            ),
            None
        )

        if id_col is not None:

            ids = f[id_col].astype(str).to_numpy()

        elif set(f.index.astype(str)).issubset(b.index):

            ids = f.index.astype(str).to_numpy()

        else:

            ids = None

            # Validate original numerical row indices.

            if (
                pd.api.types.is_integer_dtype(f.index.dtype)
                and f.index.min() >= 0
                and f.index.max() < len(b)
            ):

                candidate = b.index.to_numpy()[
                    f.index.to_numpy(dtype=int)
                ]

                if np.allclose(
                    b.loc[candidate, "agile_bill"],
                    observed_base,
                    atol=0.00001,
                    rtol=0
                ):
                    ids = candidate

            # Handle filtered frames whose indices were reset.
            # Accept only unambiguous exact bill matches.

            if ids is None:

                ordered = b.sort_values("agile_bill")
                bills = ordered.agile_bill.to_numpy()

                lo = np.searchsorted(
                    bills,
                    observed_base - 0.00001,
                    "left"
                )

                hi = np.searchsorted(
                    bills,
                    observed_base + 0.00001,
                    "right"
                )

                if not np.all(hi - lo == 1):
                    raise ValueError(
                        "No unambiguous household identity "
                        "in reference scenarios."
                    )

                ids = ordered.index.to_numpy()[lo]

        f.index = ids

        if (
            not f.index.is_unique
            or not set(ids).issubset(b.index)
        ):
            raise ValueError("Invalid scenario household IDs.")

        if not np.allclose(
            observed_base,
            b.loc[ids, "agile_bill"],
            atol=0.03,
            rtol=0
        ):
            raise ValueError(
                "Scenario IDs or bills do not reconcile "
                "with the baseline."
            )

        if "annual_kwh" in f:

            if not np.allclose(
                f["annual_kwh"].astype(float),
                b.loc[ids, "annual_kwh"],
                atol=0.03,
                rtol=0
            ):
                raise ValueError(
                    "Scenario annual consumption does not reconcile."
                )

        if level == 0:

            if (
                not np.allclose(
                    f["extra_savings"], 0, atol=0.001
                )
                or not np.allclose(
                    f["shifted_kwh"], 0, atol=0.001
                )
            ):
                raise ValueError(
                    "Invalid zero-flexibility scenario."
                )

        frames[level] = f

    ids = frames[0].index

    if any(
        set(f.index) != set(ids)
        for f in frames.values()
    ):
        raise ValueError(
            "Reference scenarios have different household samples."
        )

    return {
        k: v.loc[ids]
        for k, v in frames.items()
    }


# ============================================================
# WINDOW 03
# SAME HOUSEHOLD - THREE DIFFERENT SCENARIOS
# ============================================================

def individual(fixed):

    st.divider()

    st.subheader(
        "Individual household — Flexibility scenario comparison"
    )

    _v13_render(st.caption, 'caption')(
        "Same household and consumption; only the share "
        "of peak electricity shifted changes."
    )

    try:

        b = baseline()

        frames = align_fixed(fixed, b)

    except (ValueError, KeyError, TypeError, OSError) as exc:

        _v13_render(st.error, 'error')(f"V10 scenario verification: {exc}")

        return

    ids = frames[0].index.tolist()

    # Start with a representative household close to
    # the median consumption of the selected sample.

    default = int(
        np.argmin(
            abs(
                b.loc[ids, "annual_kwh"]
                - b.loc[ids, "annual_kwh"].median()
            )
        )
    )

    hid = st.selectbox(
        "Select household",
        ids,
        index=default,
        key="v10_household"
    )

    base = b.loc[hid]

    conv = float(base.conventional_bill)

    a0 = float(
        frames[0].loc[hid, "agile_after"]
    )

    adoption = conv - a0

    _v13_render(st.markdown, 'markdown')("#### Selected household")

    info = st.columns(3)

    info[0].metric(
        "Annual electricity consumption",
        f"{base.annual_kwh:,.1f} kWh"
    )

    info[1].metric(
        "Conventional annual bill",
        money(conv)
    )

    info[2].metric(
        "Adoption Benefit",
        money(adoption)
    )

    _v13_render(st.caption, 'caption')(
        b.attrs.get("source", "Validated baseline")
    )

    _v13_render(st.markdown, 'markdown')("#### Flexibility scenarios")

    rows = []

    for col, level, title in zip(
        st.columns(3),
        LEVELS,
        [
            "No flexibility",
            "Moderate flexibility",
            "Higher flexibility"
        ]
    ):

        r = frames[level].loc[hid]

        bill = float(r.agile_after)

        extra = a0 - bill

        total = conv - bill

        shifted = float(r.shifted_kwh)

        with col:

            with st.container(border=True):

                _v13_render(st.markdown, 'markdown')(f"**{title}**")

                _v13_render(st.markdown, 'markdown')(f"## {level}%")

                st.divider()

                st.metric(
                    "Annual Agile bill",
                    money(bill)
                )

                st.metric(
                    "Flexibility Premium",
                    money(extra)
                )

                st.metric(
                    "Total Benefit",
                    money(total)
                )

                _v13_render(st.caption, 'caption')(
                    f"Annual electricity shifted: "
                    f"{shifted:,.1f} kWh"
                )

        rows.append({
            "Scenario": f"{level}%",
            "Conventional bill (£)": conv,
            "Agile bill (£)": bill,
            "Adoption Benefit (£)": adoption,
            "Flexibility Premium (£)": extra,
            "Total Benefit (£)": total,
            "Shifted energy (kWh)": shifted
        })

    table = pd.DataFrame(rows)

    assert np.allclose(
        table["Adoption Benefit (£)"]
        + table["Flexibility Premium (£)"],
        table["Total Benefit (£)"]
    )

    # Annual electricity expenditure.

    fig = go.Figure()

    fig.add_bar(
        x=table["Scenario"],
        y=table["Agile bill (£)"],
        marker_color=[
            "#427B60",
            "#63A981",
            "#A0D4B0"
        ],
        name="Agile bill",
        text=[
            money(v)
            for v in table["Agile bill (£)"]
        ],
        textposition="auto"
    )

    fig.add_scatter(
        x=table["Scenario"],
        y=[conv] * 3,
        name="Conventional bill",
        mode="lines+markers",
        line=dict(
            color="#34735A",
            dash="dash",
            width=3
        )
    )

    fig.update_layout(
        title="Annual electricity expenditure",
        yaxis_title="Annual electricity bill (£)",
        xaxis_title="Share of peak electricity shifted",
        height=390
    )

    plot(
        fig,
        width="stretch",
        key="v10_individual_bills"
    )

    # Decomposition of economic benefits.

    fig2 = go.Figure()

    fig2.add_bar(
        x=table["Scenario"],
        y=table["Adoption Benefit (£)"],
        name="Adoption Benefit",
        marker_color="#318A5D"
    )

    fig2.add_bar(
        x=table["Scenario"],
        y=table["Flexibility Premium (£)"],
        name="Flexibility Premium",
        marker_color="#A0D4B0"
    )

    fig2.update_layout(
        barmode="relative",
        title="Initial and additional economic benefits",
        yaxis_title="Annual savings (£)",
        xaxis_title="Flexibility scenario",
        height=380
    )

    plot(
        fig2,
        width="stretch",
        key="v10_individual_benefits"
    )

    st.dataframe(
        table.round(2),
        hide_index=True,
        width="stretch"
    )

    _v13_render(st.markdown, 'markdown')("#### Interpretation")

    _v13_render(st.write, 'write')(
        f"The selected household obtains an initial Adoption "
        f"Benefit of {money(adoption)}. "
        f"Under the 20% flexibility scenario, the additional "
        f"economic benefit is "
        f"{money(table.iloc[1]['Flexibility Premium (£)'])}. "
        f"Under the 40% scenario, it is "
        f"{money(table.iloc[2]['Flexibility Premium (£)'])}."
    )

    _v13_render(st.caption, 'caption')(
        "Reference scenarios already used in V7: "
        "16:00–19:00 consumption shifted equally to the "
        "eight cheapest eligible off-peak half-hours within "
        "the same day. Adoption Benefit = conventional bill "
        "minus Agile bill at 0%. Flexibility Premium = Agile "
        "bill at 0% minus scenario bill. Positive values mean "
        "savings. These are hypothetical capabilities, not "
        "observed behaviour."
    )

    st.divider()


# ============================================================
# WINDOW 05
# ECONOMIC STARTING POINT
# ============================================================

def justice_start():

    st.subheader(
        "The starting point: the economic benefit of adopting Agile"
    )

    try:

        b = baseline()

        saving = (
            b.conventional_bill - b.agile_bill
        )

        n = len(b)

        winners = int(
            (saving > 0).sum()
        )

        cols = st.columns(4)

        cols[0].metric(
            "Households benefiting",
            f"{winners/n:.2%}"
        )

        cols[1].metric(
            "Mean conventional bill",
            money(b.conventional_bill.mean())
        )

        cols[2].metric(
            "Mean Agile bill (0%)",
            money(b.agile_bill.mean())
        )

        cols[3].metric(
            "Mean Adoption Benefit",
            money(saving.mean())
        )

        _v13_render(st.caption, 'caption')(
            f"Full analytical sample: {winners:,} "
            f"of {n:,} households obtain positive simulated "
            "savings without behavioural adaptation. "
            "Historical 2013 profiles under 2025 London tariffs; "
            "not a guarantee of future savings."
        )

        _v13_render(st.caption, 'caption')(
            b.attrs.get("source", "Validated baseline")
        )

    except (ValueError, KeyError, OSError) as exc:

        _v13_render(st.error, 'error')(f"V10 baseline verification: {exc}")

    _v13_render(st.write, 'write')(
        "Our analysis starts from tariff adoption without "
        "any behavioural adaptation. The distributive justice "
        "question then concerns both financial exposure and "
        "unequal opportunities to obtain further benefits "
        "through flexibility."
    )

    st.divider()


# ============================================================
# WINDOW 05
# REPLACEMENT OF THE OLD HOUSEHOLD COMPARISON
# ============================================================

def justice_transition():

    st.subheader(
        "4. From individual choice to distributive justice"
    )

    _v13_render(st.markdown, 'markdown')(
        """
        The baseline **Adoption Benefit** is distinct from
        the **Flexibility Premium**, the additional economic
        benefit achieved by adapting the timing of electricity
        consumption.

        Window 3 illustrates that difference for the same
        household under 0%, 20% and 40% flexibility.

        ### Individual choice and unequal opportunities

        Individual choice matters, but the opportunity to act
        is not necessarily equal.

        Financial resources, technology access, housing
        conditions, work schedules and essential electricity
        needs can constrain behavioural adaptation.

        The independent Ofgem evidence contextualises these
        differences; it does not identify the flexibility
        capability of individual households in our simulations.

        ### Managing the tension

        The managerial objective is therefore to widen access
        to additional flexibility benefits while preserving
        economic incentives and protecting households exposed
        to financial risks.

        This follows the distributive justice perspective of
        Wang and Lo (2021).
        """
    )

    st.divider()


# ============================================================
# FOREST GREEN CHART THEME
# ONLY VISUAL PROPERTIES ARE MODIFIED
# ============================================================

def green_colour(value):

    if not isinstance(value, str):
        return value

    named = {
        "blue": "#318A5D",
        "royalblue": "#318A5D",
        "lightblue": "#91D3A8",
        "navy": "#347354",
        "purple": "#537F69",
        "cyan": "#72BC93",
        "Blues": "Greens",
        "blues": "Greens",
        "Purples": "Greens"
    }

    if value in named:
        return named[value]

    alpha = None

    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):

        r, g, b = [
            int(value[i:i+2], 16) / 255
            for i in (1, 3, 5)
        ]

    else:

        m = re.fullmatch(
            r"rgba?\(([^)]+)\)",
            value
        )

        if not m:
            return value

        try:

            vals = [
                float(x.strip())
                for x in m.group(1).split(",")
            ]

            if (
                len(vals) not in (3, 4)
                or not all(
                    0 <= x <= 255
                    for x in vals[:3]
                )
            ):
                return value

            r, g, b = [
                x / 255 for x in vals[:3]
            ]

            if len(vals) == 4:
                alpha = vals[3]

        except ValueError:

            return value

    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Convert the previous blue/cyan/purple visual
    # identity into coordinated green shades.

    # Preserve red and orange where they carry
    # warning or financial-risk meanings.

    if 0.45 <= h <= 0.85 and s >= 0.20:

        rgb = [
            round(x * 255)
            for x in colorsys.hls_to_rgb(
                0.40,
                min(0.85, max(0.36, l)),
                min(0.56, max(0.25, s))
            )
        ]

        if alpha is not None:

            return (
                f"rgba({rgb[0]},{rgb[1]},"
                f"{rgb[2]},{alpha})"
            )

        return "#" + "".join(
            f"{x:02x}" for x in rgb
        )

    return value


def visual(obj):

    if isinstance(obj, dict):

        for k, v in list(obj.items()):

            if k in (
                "color",
                "colors",
                "colorway",
                "fillcolor",
                "linecolor",
                "markercolor",
                "marker_color",
                "colorscale",
                "bordercolor"
            ):

                if isinstance(v, str):

                    obj[k] = green_colour(v)

                elif isinstance(v, (list, tuple, np.ndarray)):

                    obj[k] = [

                        green_colour(x)

                        if isinstance(x, str)

                        else (

                            [x[0], green_colour(x[1])]

                            if (
                                isinstance(x, (tuple, list))
                                and len(x) == 2
                                and isinstance(x[1], str)
                            )

                            else x
                        )

                        for x in v
                    ]

            elif isinstance(v, (dict, list)):

                visual(v)

    elif isinstance(obj, list):

        for v in obj:
            visual(v)

    return obj


def inherit_fonts(obj):

    if isinstance(obj, dict):

        for k, v in list(obj.items()):

            if (
                k in (
                    "font",
                    "tickfont",
                    "textfont",
                    "insidetextfont",
                    "outsidetextfont"
                )
                and isinstance(v, dict)
            ):

                v.pop("color", None)

            elif k in (
                "paper_bgcolor",
                "plot_bgcolor",
                "gridcolor",
                "zerolinecolor"
            ):

                obj.pop(k, None)

            elif isinstance(v, (dict, list)):

                inherit_fonts(v)

    elif isinstance(obj, list):

        for v in obj:
            inherit_fonts(v)


def plot(figure, *args, **kwargs):

    fig = go.Figure(figure)

    obj = fig.to_plotly_json()

    layout = obj.setdefault("layout", {})

    layout.pop("template", None)

    visual(obj)

    inherit_fonts(layout)

    for trace in obj.get("data", []):
        inherit_fonts(trace)

    # Use the native active Streamlit theme for
    # backgrounds, axis labels, grids and fonts.

    kwargs["theme"] = "streamlit"

    return st.plotly_chart(
        go.Figure(obj),
        *args,
        **kwargs
    )


# ============================================================
# LIGHT / DARK SIDEBAR SWITCH
# ============================================================

SWITCH = """

<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:8px 0;
    color:#FFFFFF;
    font:600 14px sans-serif;
">

    <span>Dark mode</span>

    <button
        id="v10-mode"
        role="switch"
        aria-label="Dark mode"
        aria-checked="false"
        style="
            background:#527966;
            color:white;
            border:1px solid #8bc6a3;
            border-radius:20px;
            padding:4px 10px;
            cursor:pointer;
            min-width:52px;
            font-weight:600;
        "
    >OFF</button>

</div>

<div
    id="v10-mode-msg"
    style="font:11px sans-serif;color:#C2EDB5"
></div>

<script>

(()=>{

    const btn = document.getElementById("v10-mode");
    const msg = document.getElementById("v10-mode-msg");

    const d = window.parent.document;

    const sleep = ms =>
        new Promise(resolve => setTimeout(resolve, ms));

    const app = () =>
        d.querySelector('[data-testid="stApp"]')
        || d.querySelector(".stApp")
        || d.body;

    const dark = () => {

        const elements = [
            app(),
            d.querySelector('[data-testid="stAppViewContainer"]'),
            d.body,
            d.documentElement
        ];

        for (const e of elements) {

            if (!e) continue;

            const c = d.defaultView
                .getComputedStyle(e)
                .backgroundColor.match(/[\\d.]+/g);

            if (
                c && c.length >= 3
                && (c.length < 4 || +c[3] > 0.5)
            ) {

                return (
                    +c[0] * .299
                    +c[1] * .587
                    +c[2] * .114
                ) < 128;

            }

        }

        return d.defaultView.matchMedia(
            "(prefers-color-scheme: dark)"
        ).matches;

    };

    const refresh = () => {

        const active = !!dark();

        btn.setAttribute(
            "aria-checked",
            String(active)
        );

        btn.textContent = active ? "ON" : "OFF";

        btn.style.background =
            active ? "#26734E" : "#527966";

    };

    const exact = (selector, text) =>

        Array.from(
            d.querySelectorAll(selector)
        ).find(e =>

            e.getBoundingClientRect().width > 0

            && (
                e.innerText
                || e.textContent
                || ""
            ).trim() === text

        );

    btn.onclick = async () => {

        btn.disabled = true;
        msg.textContent = "";

        const target = dark() ? "Light" : "Dark";

        try {

            const menu =

                d.querySelector(
                    '[data-testid="stMainMenuButton"]'
                )

                || d.querySelector("#MainMenu button")

                || d.querySelector(
                    '[data-testid="stMainMenu"] button'
                );

            if (!menu)
                throw Error("Menu unavailable");

            if (
                !d.querySelector(
                    '[data-testid="stMainMenuList"]'
                )
            ) {

                menu.click();

                await sleep(250);

            }

            let item =

                d.querySelector(
                    '[data-testid="stMainMenuItem-theme-'
                    + target + '"]'
                )

                || exact(
                    '[role="menuitemradio"],'
                    + '[role="option"],[role="menuitem"]',
                    target
                );

            if (!item) {

                const settings = exact(
                    '[role="option"],[role="menuitem"]',
                    "Settings"
                );

                if (settings) {

                    settings.click();

                    await sleep(350);

                }

                const dialog = d.querySelector(
                    '[role="dialog"]'
                );

                if (dialog) {

                    item = exact(
                        '[role="dialog"] label,'
                        + '[role="dialog"] [role="radio"]',
                        target
                    );

                    if (!item) {

                        const select = dialog.querySelector(
                            '[data-baseweb="select"]'
                        );

                        if (select) {

                            select.click();

                            await sleep(250);

                            item = exact(
                                '[role="option"]',
                                target
                            );

                        }

                    }

                }

            }

            if (!item)
                throw Error("Theme selector unavailable");

            item.click();

            await sleep(400);

            if ((!!dark()) !== (target === "Dark"))
                throw Error("Theme not confirmed");

            const close = d.querySelector(
                '[role="dialog"] button[aria-label="Close"]'
            );

            if (close)
                close.click();

            if (
                menu.getAttribute("aria-expanded") === "true"
            ) {
                menu.click();
            }

        } catch(e) {

            msg.textContent =
                "Use the native menu ⋮ → Theme / Settings.";

        } finally {

            refresh();

            btn.disabled = false;

        }

    };

    refresh();

    const timer = setInterval(() => {

        if (!btn.isConnected) {

            clearInterval(timer);

            return;

        }

        refresh();

    }, 600);

})();

</script>

"""


def theme_control():

    with st.sidebar:

        if (
            "unsafe_allow_javascript"
            in inspect.signature(st.html).parameters
        ):

            _v13_render(st.html, 'html')(
                SWITCH,
                unsafe_allow_javascript=True
            )

        else:

            import streamlit.components.v1 as components

            _v13_render(components.html, 'html')(
                SWITCH,
                height=64
            )

        _v13_render(st.caption, 'caption')(
            "Appearance: Light / Dark · native menu ⋮ also available."
        )

