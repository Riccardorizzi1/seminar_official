"""Octopus Energy V13 — Streamlit Community Cloud entry point."""
import ast
import json
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent
MAP = REPO / "file_path_map.json"
PROJECT = "Octopus Energy Project"
PY_FILES = (
    "octopus_dashboard_v13.py",
    "octopus_v13_components.py",
    "octopus_v13_render.py",
)
MANIFEST = "approved_silver_manifest.json"


def relative_path(value):
    p = PurePosixPath(str(value).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or ":" in str(p) or not p.parts:
        raise ValueError(f"Invalid relative path: {value}")
    return Path(*p.parts)


def source_file(repo, name):
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid flat filename: {name}")
    if name == "config.toml" and (repo / ".streamlit/config.toml").is_file():
        return repo / ".streamlit/config.toml"
    return repo / name


def portable_source(source, home):
    # Change only actual Path.home() expressions, never displayed text.
    raw = source.encode("utf-8")
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or node.args or node.keywords:
            continue
        fn = ast.unparse(node.func)
        if fn not in ("Path.home", "pathlib.Path.home"):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        expr = f"{fn.rsplit('.', 1)[0]}({str(home)!r})"
        edits.append((start, end, expr.encode("utf-8")))
    for start, end, replacement in sorted(edits, reverse=True):
        raw = raw[:start] + replacement + raw[end:]
    result = raw.decode("utf-8")
    compile(result, "<portable V13>", "exec")
    return result


def rewrite_paths(value, root):
    if isinstance(value, dict):
        return {k: rewrite_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(v, root) for v in value]
    if isinstance(value, str):
        normal = value.replace("\\", "/")
        marker = PROJECT + "/"
        pos = normal.lower().find(marker.lower())
        if pos >= 0:
            tail = normal[pos + len(marker):]
            return str(root / relative_path(tail)) if tail else str(root)
    return value


def link_data(source, target):
    # On Linux/Streamlit Cloud, this does not duplicate large data.
    try:
        target.symlink_to(source)
    except OSError:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def prepare(repo_str, snapshot):
    repo = Path(repo_str)
    mapping = json.loads((repo / MAP.name).read_text(encoding="utf-8-sig"))
    entries = mapping["files"]
    if mapping.get("dashboard_version") != "V13":
        raise ValueError("The export must contain dashboard V13.")
    for name in (*PY_FILES, MANIFEST):
        if name not in entries:
            raise ValueError(f"Missing entry in path map: {name}")

    missing = [n for n in entries if not source_file(repo, n).is_file()]
    if missing:
        raise FileNotFoundError("Missing repository files: " + ", ".join(missing))

    holder = tempfile.TemporaryDirectory(prefix="octopus_v13_")
    home = Path(holder.name)
    root = home / PROJECT
    targets = {n: root / relative_path(p) for n, p in entries.items()}
    if len(set(targets.values())) != len(targets):
        raise ValueError("Duplicate original file paths in path map.")

    for name, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        source = source_file(repo, name)
        if source.suffix.lower() in (".npy", ".npz"):
            with source.open("rb") as f:
                if f.read(50).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(f"Git LFS content not downloaded: {name}")
        if name in PY_FILES:
            target.write_text(portable_source(source.read_text(encoding="utf-8-sig"), home), encoding="utf-8")
        elif name == MANIFEST:
            original = json.loads(source.read_text(encoding="utf-8-sig"))
            manifest = rewrite_paths(original, root)
            for key, filename in mapping.get("silver_manifest_keys", {}).items():
                if filename not in targets:
                    raise ValueError(f"Missing exported Silver asset: {filename}")
                manifest["data_files"][key] = str(targets[filename])
            target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            link_data(source, target)
    return holder, targets[PY_FILES[0]]


def main():
    import streamlit as st
    try:
        mapping = json.loads(MAP.read_text(encoding="utf-8-sig"))
        sources = [MAP] + [source_file(REPO, n) for n in mapping["files"]]
        snapshot = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources if p.is_file())
        cached_prepare = st.cache_resource(show_spinner=False)(prepare)
        holder, entry = cached_prepare(str(REPO), snapshot)
    except Exception as exc:
        st.error("Unable to initialise the V13 dashboard.")
        st.exception(exc)
        st.stop()
        return

    folder = str(entry.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    for filename in PY_FILES[1:]:
        module = Path(filename).stem
        loaded = sys.modules.get(module)
        if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != (entry.parent / filename).resolve():
            sys.modules.pop(module, None)
    # Execute the existing dashboard, including its original page config.
    # No new calculations, charts or navigation are introduced here.
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
