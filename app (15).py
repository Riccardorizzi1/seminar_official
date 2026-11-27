"""Octopus Energy V13 — Streamlit Cloud entry point for seminar_official.

Uses the original V13 dashboard and the flat repository export. No changes
are made to the checked-in Python modules, datasets or scientific checks.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import logging
import os
import re
import runpy
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

REPO = Path(__file__).resolve().parent
PROJECT = "Octopus Energy Project"
APP = "octopus_dashboard_v13.py"
HELPER = "octopus_v13_components.py"
RENDER = "octopus_v13_render.py"
MANIFEST = "approved_silver_manifest.json"
CLOUD_MANIFEST = "approved_silver_manifest.cloud.json"
ANNUAL = "dashboard_analytics.npz"
MONTHLY = "monthly_household_outcomes.npz"
MATRIX = "household_consumption_2013.npy"
MAP = "file_path_map.json"
PARTS = "household_parts.json"
N, T = 3955, 17520
ORIGINAL_SIZE = 554332928
PART_COUNT = 25
BLOCK = 1024 * 1024
LOGGER = logging.getLogger("octopus.cloud")

# Original V11 approved chart styling and UI refinements.
V11_PRESENTATION = '\n# V11 presentation only: existing calculations remain untouched.\n_old_theme_control = theme_control\n\ndef theme_control():\n    _old_theme_control()\n    st.markdown(\'\\n<style>\\n[data-testid="stHeader"] {background: transparent;}\\n[data-testid="stMetric"] {padding: 14px 16px; border: 1px solid color-mix(in srgb,currentColor 14%,transparent); border-radius: 12px; background: color-mix(in srgb,currentColor 4%,transparent);}\\n[data-testid="stMetricLabel"] {white-space: normal;}\\n[data-testid="stMetricValue"] {font-weight: 650; font-variant-numeric: tabular-nums;}\\n[data-testid="stMarkdownContainer"] p {line-height: 1.65; overflow-wrap: break-word;}\\n[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {line-height: 1.45;}\\n[data-testid="stPlotlyChart"] {border: 1px solid color-mix(in srgb,currentColor 13%,transparent); border-radius: 12px; padding: 8px;}\\n[data-testid="stExpander"] details {border-radius: 10px;}\\n</style>\\n\', unsafe_allow_html=True)\n\n_old_green_colour = green_colour\n\ndef green_colour(value):\n    value = _old_green_colour(value)\n    if not isinstance(value, str):\n        return value\n    # Keep numerical scales, semantic warning hues and opacity.\n    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):\n        rgb = [int(value[i:i+2],16)/255 for i in (1,3,5)]\n        h,l,s = colorsys.rgb_to_hls(*rgb)\n        if .24 <= h <= .48 and s >= .18:\n            l = .43 + .19 * l\n            rgb = colorsys.hls_to_rgb(.39,l,min(.50,max(.24,s)))\n            return \'#\' + \'\'.join(f\'{round(x*255):02x}\' for x in rgb)\n    return value\n\n\ndef _neutral_layout(obj):\n    if isinstance(obj, dict):\n        for k,v in list(obj.items()):\n            if k in (\'bgcolor\',\'bordercolor\',\'gridcolor\',\'zerolinecolor\',\'linecolor\',\'tickcolor\'):\n                obj.pop(k,None)\n            elif isinstance(v,(dict,list)):\n                _neutral_layout(v)\n    elif isinstance(obj,list):\n        for v in obj:\n            _neutral_layout(v)\n\n\ndef plot(figure,*args,**kwargs):\n    obj = go.Figure(figure).to_plotly_json()\n    layout = obj.setdefault(\'layout\',{})\n    layout.pop(\'template\',None)\n    visual(obj)\n    inherit_fonts(layout)\n    _neutral_layout(layout)\n    for trace in obj.get(\'data\',[]):\n        inherit_fonts(trace)\n    # Transparent plot surfaces + native active theme for fonts and axes.\n    layout[\'paper_bgcolor\'] = \'rgba(0,0,0,0)\'\n    layout[\'plot_bgcolor\'] = \'rgba(0,0,0,0)\'\n    kwargs[\'theme\'] = \'streamlit\'\n    return st.plotly_chart(go.Figure(obj),*args,**kwargs)\n'

# Exact names and Silver manifest keys from the actual exported path map.
SILVER = {
    "complete_households": "complete_household_sample.csv",
    "consumer_behaviour": "consumer_behaviour_evidence.csv",
    "household_matrix": MATRIX,
    "household_registry": "household_registry.csv",
    "integrated_market": "electricity_market_2025.csv",
    "peak_consumption": "peak_consumption_evidence.csv",
    "social_comparisons": "social_group_comparisons.csv",
    "source_interval_index": "source_interval_index.npy",
    "standing_charges": "tariff_standing_charges_2025.csv",
}
REQUIRED = {APP, HELPER, RENDER, MANIFEST, ANNUAL, MONTHLY, *SILVER.values()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def log(message):
    print(f"[Octopus V13 Cloud] {message}", flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def relative_path(value):
    p = PurePosixPath(str(value).replace("\\", "/"))
    require(bool(p.parts) and not p.is_absolute() and ".." not in p.parts
            and ":" not in str(p), "Invalid path in file_path_map.json.")
    return Path(*p.parts)


def existing_file(repo, name):
    require(Path(name).name == name and not any(c in name for c in "/\\:"),
            f"Invalid repository filename: {name}")
    path = repo / name
    if name == "config.toml" and (repo / ".streamlit/config.toml").is_file():
        path = repo / ".streamlit/config.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing repository file: {name}")
    with path.open("rb") as f:
        require(not f.read(50).startswith(b"version https://git-lfs.github.com/spec/v1"),
                f"Unresolved Git LFS pointer: {name}")
    return path


def discover(repo):
    map_path = existing_file(repo, MAP)
    meta_path = existing_file(repo, PARTS)
    mapping = read_json(map_path)
    meta = read_json(meta_path)
    require(mapping.get("dashboard_version") == "V13", "Expected V13 export.")
    entries = mapping["files"]
    require(REQUIRED <= set(entries),
            "Missing path-map entries: " + ", ".join(sorted(REQUIRED - set(entries))))
    require(all(mapping["silver_manifest_keys"].get(k) == v for k, v in SILVER.items()),
            "Silver file mapping differs from the V13 export.")
    require(meta.get("original_filename") == MATRIX and meta.get("compression") == "gzip",
            "household_parts.json does not describe the original household matrix.")
    require(meta.get("parts") == [f"household_consumption_part_{i:03d}.gz" for i in range(1, PART_COUNT + 1)],
            "The expected 25 household parts are missing or out of order.")
    require(meta.get("original_size") == ORIGINAL_SIZE,
            "Unexpected original household matrix byte size.")
    require(isinstance(meta.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", meta["sha256"]),
            "Invalid household checksum.")
    # The uncompressed matrix must never be required in the repository.
    assets = {name: existing_file(repo, name) for name in sorted(REQUIRED - {MATRIX})}
    parts = [existing_file(repo, name) for name in meta["parts"]]
    paths = {name: relative_path(entries[name]) for name in REQUIRED}
    require(len(set(paths.values())) == len(paths), "Duplicate destinations in path map.")
    require(paths[APP] == Path("dashboard") / APP and
            paths[HELPER] == Path("dashboard") / HELPER and
            paths[RENDER] == Path("dashboard") / RENDER and
            paths[MANIFEST] == Path("data/silver") / MANIFEST and
            paths[ANNUAL] == Path("data/gold/step_3_dashboard_v3") / ANNUAL and
            paths[MONTHLY] == Path("data/gold/step_3_justice_v6") / MONTHLY,
            "Unexpected V13 source or Gold destination in path map.")
    return paths, meta, assets, parts, [map_path, meta_path, *assets.values(), *parts]


def original_manifest(assets):
    import numpy as np
    with np.load(assets[ANNUAL], allow_pickle=False) as a, np.load(assets[MONTHLY], allow_pickle=False) as m:
        expected = str(a["manifest_digest"].item())
        require(expected == str(m["manifest_digest"].item()),
                "Annual and monthly Gold use different Silver manifests.")
        require(sha_file(assets[ANNUAL]) == str(m["annual_gold_digest"].item()),
                "Annual Gold checksum differs from the one recorded in monthly Gold.")
    # Only accept a byte representation with the exact archived SHA256.
    # Git may normalize CRLF to LF. BOM and final newline variants are tested
    # without changing any JSON data, and only the cryptographic match is used.
    raw = assets[MANIFEST].read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    candidates = [raw]
    for base in (lf, lf.removeprefix(b"\xef\xbb\xbf")):
        for end in (base, base.rstrip(b"\n"), base.rstrip(b"\n") + b"\n"):
            for bom in (end, b"\xef\xbb\xbf" + end.removeprefix(b"\xef\xbb\xbf")):
                candidates.extend((bom, bom.replace(b"\n", b"\r\n")))
    for b in dict.fromkeys(candidates):
        if hashlib.sha256(b).hexdigest() == expected:
            manifest = json.loads(b.decode("utf-8-sig"))
            require(manifest.get("status") == "PASS" and manifest.get("silver_frozen") is True
                    and manifest.get("ready_for_step_3") is True,
                    "Silver manifest is not approved.")
            log("Original Silver SHA256 and annual/monthly Gold SHA256: PASS")
            return b, manifest
    raise ValueError("Silver SHA256 differs from both Gold archives. Restore the original approved_silver_manifest.json without editing its contents.")


def restore_matrix(meta, parts, target):
    import numpy as np
    require(shutil.disk_usage(target.parent).free > meta["original_size"] + 32 * BLOCK,
            "Not enough temporary disk space for the household matrix.")
    h = hashlib.sha256()
    written = 0
    temp = target.with_suffix(".npy.tmp")
    try:
        with temp.open("wb") as out:
            for i, part in enumerate(parts, 1):
                with gzip.open(part, "rb") as src:
                    for block in iter(lambda: src.read(BLOCK), b""):
                        written += len(block)
                        require(written <= meta["original_size"], "Household data exceeds original size.")
                        h.update(block)
                        out.write(block)
                if i % 5 == 0:
                    log(f"Household decompression: {i}/{PART_COUNT} parts")
        require(written == meta["original_size"] and h.hexdigest() == meta["sha256"],
                "Household matrix SHA256 or byte size mismatch.")
        os.replace(temp, target)
        matrix = np.load(target, mmap_mode="r", allow_pickle=False)
        require(matrix.shape == (N, T) and matrix.dtype.kind == "f" and matrix.dtype.itemsize == 8,
                "Unexpected household matrix shape or dtype.")
        del matrix
        log("Household matrix: SHA256, shape and dtype PASS (memory mapped)")
    finally:
        temp.unlink(missing_ok=True)


def relocate(value, root):
    if isinstance(value, dict):
        return {k: relocate(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, root) for v in value]
    if isinstance(value, str):
        normal = value.replace("\\", "/")
        marker = PROJECT + "/"
        pos = normal.lower().find(marker.lower())
        if pos >= 0:
            tail = normal[pos + len(marker):]
            return str(root / relative_path(tail)) if tail else str(root)
    return value


def portable_source(source, home, name):
    """Keep the ORIGINAL manifest at MANIFEST. Redirect only operational reads.

    This leaves both original SHA256 calculations in V13 untouched.
    The helper's explicit manifest path must point to the operational copy too.
    """
    tree = ast.parse(source)
    raw = source.encode("utf-8")
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    counts = {"home": 0, "read": 0, "helper": 0, "digest": 0}
    for n in ast.walk(tree):
        replacement = None
        if isinstance(n, ast.Call):
            fn = ast.unparse(n.func)
            if fn in ("Path.home", "pathlib.Path.home") and not n.args and not n.keywords:
                replacement = f"{fn.rsplit('.', 1)[0]}({str(home)!r})"
                counts["home"] += 1
            elif name == APP and fn == "MANIFEST.read_text":
                # Preserve the read_text arguments, e.g. encoding='utf-8-sig'.
                n = n.func
                replacement = f"MANIFEST.with_name({CLOUD_MANIFEST!r}).read_text"
                counts["read"] += 1
            elif name == APP and fn == "MANIFEST.read_bytes":
                counts["digest"] += 1
        elif name == HELPER and isinstance(n, ast.Constant) and n.value == "data/silver/approved_silver_manifest.json":
            replacement = repr("data/silver/" + CLOUD_MANIFEST)
            counts["helper"] += 1
        if replacement is not None:
            start = offsets[n.lineno - 1] + n.col_offset
            end = offsets[n.end_lineno - 1] + n.end_col_offset
            edits.append((start, end, replacement.encode("utf-8")))
    if name == APP:
        require(counts["read"] == 1 and counts["digest"] == 2,
                f"Unexpected V13 manifest access pattern: {counts}")
    if name == HELPER:
        require(counts["helper"] == 1 and counts["home"] >= 1,
                f"Unexpected V13 helper paths: {counts}")
    for start, end, replacement in sorted(edits, reverse=True):
        raw = raw[:start] + replacement + raw[end:]
    out = raw.decode("utf-8")
    if name == HELPER:
        # Restore the approved V11 presentation layer on the V13 helper.
        # No economic or environmental functions are replaced.
        functions = {n.name for n in ast.parse(out).body if isinstance(n, ast.FunctionDef)}
        require({"theme_control", "green_colour", "visual", "inherit_fonts", "plot"} <= functions,
                "V13 visual helper functions are missing.")
        out += "\n" + V11_PRESENTATION + "\n"
    compile(out, name, "exec")
    return out


def preflight(assets, matrix_path):
    """Validate Gold arrays and reconcile monthly results before rendering."""
    import numpy as np
    import pandas as pd
    registry = pd.read_csv(assets[SILVER["household_registry"]],
                           usecols=["household_id", "matrix_row"]).sort_values("matrix_row")
    ids = registry["household_id"].astype(str).to_numpy()
    require(len(ids) == N and len(set(ids)) == N and
            np.array_equal(registry["matrix_row"].to_numpy(), np.arange(N)),
            "Invalid household registry mapping.")
    market = pd.read_csv(assets[SILVER["integrated_market"]], usecols=[
        "target_interval_index", "source_interval_index",
        "agile_unit_rate_p_kwh_inc_vat", "conventional_unit_rate_p_kwh_inc_vat"
    ]).sort_values("target_interval_index")
    idx = market["source_interval_index"].to_numpy(dtype=int)
    require(len(market) == T and np.array_equal(market["target_interval_index"], np.arange(T))
            and np.array_equal(np.sort(idx), np.arange(T)), "Invalid market interval mapping.")
    source_index = np.load(assets[SILVER["source_interval_index"]], allow_pickle=False)
    require(np.array_equal(source_index, idx), "Silver source index and market mapping differ.")
    standing = pd.read_csv(assets[SILVER["standing_charges"]])
    require(len(standing) == 365, "Unexpected number of standing charge days.")
    pa = market["agile_unit_rate_p_kwh_inc_vat"].to_numpy(float) / 100
    pc = market["conventional_unit_rate_p_kwh_inc_vat"].to_numpy(float) / 100
    sa = float(standing["agile_standing_p_day_inc_vat"].sum()) / 100
    sc = float(standing["conventional_standing_p_day_inc_vat"].sum()) / 100
    with np.load(assets[ANNUAL], allow_pickle=False) as a, np.load(assets[MONTHLY], allow_pickle=False) as m:
        require(np.array_equal(a["household_ids"], ids) and np.array_equal(m["household_ids"], ids),
                "Silver, annual Gold and monthly Gold household ordering differs.")
        require(int(a["market_intervals"].item()) == T, "Gold market interval count differs.")
        cols = a["baseline_columns"].tolist()
        base = a["baseline_values"]
        require(base.shape == (N, len(cols)) and len(set(cols)) == len(cols), "Invalid Gold baseline dimensions.")
        for col in ("annual_kwh", "agile_bill", "conventional_bill", "emissions_kg"):
            require(col in cols, f"Missing baseline column: {col}")
        agile = base[:, cols.index("agile_bill")]
        conv = base[:, cols.index("conventional_bill")]
        cost = a["cost_effects"]
        carbon = a["carbon_effects"]
        energy = a["shiftable_energy"]
        require(cost.shape == carbon.shape == (3, 3, N) and energy.shape == (N,),
                "Invalid Gold flexibility dimensions.")
        for arr in (agile, conv, cost, carbon, energy):
            require(np.isfinite(arr).all(), "Non-finite Gold outcomes.")
        ma, mc = m["agile"], m["conventional"]
        require(ma.shape == (3, N, 12) and mc.shape == (N, 12) and len(m["months"]) == 12,
                "Invalid monthly Gold dimensions.")
        require(np.allclose(mc.sum(axis=1), conv, rtol=1e-9, atol=1e-6),
                "Monthly conventional bills do not reconcile with annual Gold.")
        for i, fraction in enumerate((0.0, 0.2, 0.4)):
            require(np.allclose(ma[i].sum(axis=1), agile + fraction * cost[1, 0],
                                rtol=1e-9, atol=1e-6),
                    f"Monthly Agile scenario {int(fraction*100)}% does not reconcile with annual Gold.")
        # Same tariff calculation as the V13's bill verification; read only 3 rows.
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        for row in (0, N // 2, N - 1):
            x = matrix[row, idx]
            require(np.allclose([np.sum(x*pa)+sa, np.sum(x*pc)+sc],
                                [agile[row], conv[row]], rtol=1e-9, atol=1e-7),
                    f"Household bill verification failed for matrix row {row}.")
        del matrix
    log("Silver/Gold household IDs, interval mapping, bill samples and monthly 0/20/40 reconciliation: PASS")


def prepare(repo_str, snapshot):
    # snapshot participates in Streamlit's cache key.
    repo = Path(repo_str)
    paths, meta, assets, parts, _ = discover(repo)
    raw, manifest = original_manifest(assets)
    holder = tempfile.TemporaryDirectory(prefix="octopus_v13_cloud_")
    home = Path(holder.name)
    root = home / PROJECT
    try:
        targets = {n: root / p for n, p in paths.items()}
        for name, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == MATRIX:
                restore_matrix(meta, parts, target)
            elif name in (APP, HELPER, RENDER):
                source = assets[name].read_text(encoding="utf-8-sig")
                target.write_text(portable_source(source, home, name), encoding="utf-8")
            elif name == MANIFEST:
                # Immutable canonical bytes for every original hash check.
                target.write_bytes(raw)
                operational = relocate(manifest, root)
                for key, filename in SILVER.items():
                    require(key in manifest["data_files"], f"Missing Silver key: {key}")
                    require(Path(str(manifest["data_files"][key]).replace("\\", "/")).name == filename,
                            f"Silver original path disagrees with exported filename: {key}")
                    operational["data_files"][key] = str(targets[filename])
                target.with_name(CLOUD_MANIFEST).write_text(
                    json.dumps(operational, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                # No copies of the large repository assets on Linux.
                target.symlink_to(assets[name].resolve())
        preflight(assets, targets[MATRIX])
        log("Runtime verified. Launching the original V13.")
        return holder, targets[APP]
    except BaseException:
        holder.cleanup()
        raise


def main():
    import streamlit as st
    try:
        _, _, _, _, sources = discover(REPO)
        snapshot = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                         for p in sorted(set(sources)))
        cached = st.cache_resource(show_spinner=False, max_entries=1)(prepare)
        holder, entry = cached(str(REPO), snapshot)
    except Exception as exc:
        LOGGER.exception("V13 Cloud preflight failed")
        st.error("V13 Cloud data validation failed. See Manage app logs for the exact failed check.")
        st.stop()
        return
    folder = str(entry.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    for name in (HELPER, RENDER):
        module = Path(name).stem
        old = sys.modules.get(module)
        if old is not None and Path(getattr(old, "__file__", "")).resolve() != (entry.parent/name).resolve():
            sys.modules.pop(module, None)
    # Streamlit still owns widget execution and reruns. No global chdir,
    # monkeypatch of Path.home, change to source files, or skipped checks.
    runpy.run_path(str(entry), run_name="__main__")


    # ========================================================
    # V13 — FIX LIGHT / DARK ON-OFF SWITCH
    # ========================================================

    import importlib
    import re

    components = importlib.import_module(
        "octopus_v13_components"
    )

    SWITCH_SCRIPT = r"""
<script>
(() => {

    const btn = document.getElementById("v10-mode");
    const msg = document.getElementById("v10-mode-msg");

    if (!btn || !msg) return;

    const sleep = ms =>
        new Promise(resolve => setTimeout(resolve, ms));

    // Locate the actual Streamlit document.
    // Supports local execution and Cloud embedding.

    function findDocument() {

        let w = window;

        for (let i = 0; i < 5; i++) {

            try {

                const d = w.document;

                if (
                    d.querySelector(
                        '[data-testid="stMainMenuButton"]'
                    ) ||
                    d.querySelector("#MainMenu")
                ) {
                    return d;
                }

                if (w.parent === w) break;

                w = w.parent;

            } catch (_) {
                break;
            }

        }

        return document;

    }

    function getMenu(d) {

        return (
            d.querySelector(
                '[data-testid="stMainMenuButton"]'
            ) ||
            d.querySelector(
                'button[aria-label="Main menu"]'
            ) ||
            d.querySelector("#MainMenu button")
        );

    }

    function isDark(d) {

        const selectors = [
            '[data-testid="stAppViewContainer"]',
            '[data-testid="stApp"]',
            '.stApp'
        ];

        for (const selector of selectors) {

            const element = d.querySelector(selector);

            if (!element) continue;

            const background = d.defaultView
                .getComputedStyle(element)
                .backgroundColor;

            const values = background
                .match(/[0-9.]+/g);

            if (!values || values.length < 3)
                continue;

            if (
                values.length === 4 &&
                Number(values[3]) < 0.5
            ) continue;

            const brightness =
                Number(values[0]) * 0.299 +
                Number(values[1]) * 0.587 +
                Number(values[2]) * 0.114;

            return brightness < 128;

        }

        return d.defaultView.matchMedia(
            "(prefers-color-scheme: dark)"
        ).matches;

    }

    function refresh() {

        const active = isDark(findDocument());

        btn.textContent = active ? "ON" : "OFF";

        btn.setAttribute(
            "aria-checked",
            String(active)
        );

        btn.style.background = active
            ? "#26734E"
            : "#527966";

    }

    async function waitFor(fn, timeout = 5000) {

        const start = Date.now();

        while (Date.now() - start < timeout) {

            const result = fn();

            if (result) return result;

            await sleep(100);

        }

        return null;

    }

    btn.onclick = async () => {

        btn.disabled = true;
        msg.textContent = "";

        const d = findDocument();

        const target = isDark(d)
            ? "Light"
            : "Dark";

        let openedHere = false;

        try {

            const menu = await waitFor(
                () => getMenu(d)
            );

            if (!menu) {
                throw new Error(
                    "Streamlit menu not found"
                );
            }

            // Open the native Streamlit theme menu.

            if (
                menu.getAttribute("aria-expanded")
                !== "true"
            ) {

                menu.click();
                openedHere = true;

            }

            // Select the real native theme.
            // Streamlit 1.60 theme selector.

            const item = await waitFor(() =>

                d.querySelector(
                    '[data-testid="stMainMenuItem-theme-'
                    + target + '"]'
                )

            );

            if (!item) {
                throw new Error(
                    "Native theme option unavailable"
                );
            }

            item.click();

            // Confirm the actual rendered appearance.
            // Do not display ON/OFF without confirmation.

            const confirmed = await waitFor(
                () => isDark(d) === (target === "Dark")
            );

            if (!confirmed) {
                throw new Error(
                    "Theme change not confirmed"
                );
            }

        } catch (error) {

            msg.textContent = error.message;

            console.error(
                "Octopus V13 theme switch:",
                error
            );

        } finally {

            const menu = getMenu(d);

            if (
                openedHere &&
                menu &&
                menu.getAttribute("aria-expanded")
                === "true"
            ) {

                menu.click();

            }

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

    }, 500);

})();
</script>
"""

    # Replace only the defective switch logic.
    # Preserve the original button and its appearance.

    components.SWITCH, replacements = re.subn(

        r"<script\b[^>]*>.*?</script>",

        lambda _: SWITCH_SCRIPT,

        components.SWITCH,

        count=1,

        flags=re.DOTALL | re.IGNORECASE

    )

    if replacements != 1:

        raise RuntimeError(
            "Original V13 theme switch not found."
        )

    # Execute the original dashboard unchanged.

    runpy.run_path(
        str(entry),
        run_name="__main__"
    )
