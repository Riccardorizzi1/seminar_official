"""
OCTOPUS ENERGY - AGILE ENERGY ANALYTICS
Dashboard V13 - Streamlit Community Cloud

Supports compressed household consumption data
and nested GitHub repository folders.
"""

import ast
import gzip
import hashlib
import json
import os
import re
import runpy
import shutil
import sys
import tempfile

from pathlib import Path, PurePosixPath

# ============================================================
# 1. NUMERICAL ENVIRONMENT
# ============================================================

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS"
):
    os.environ.setdefault(key, "1")

# ============================================================
# 2. REPOSITORY CONFIGURATION
# ============================================================

REPO = Path(__file__).resolve().parent

PROJECT = "Octopus Energy Project"

PY_FILES = (
    "octopus_dashboard_v13.py",
    "octopus_v13_components.py",
    "octopus_v13_render.py"
)

MANIFEST = "approved_silver_manifest.json"

# ============================================================
# 3. PORTABLE PATH MANAGEMENT
# ============================================================

def flat_name(name):

    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or any(c in name for c in ("/", "\\", ":"))
    ):

        raise ValueError(
            f"Invalid filename: {name!r}"
        )

    return name


def relative_path(value):

    p = PurePosixPath(
        str(value).replace("\\", "/")
    )

    if (
        p.is_absolute()
        or ".." in p.parts
        or ":" in str(p)
        or not p.parts
    ):

        raise ValueError(
            f"Invalid relative path: {value}"
        )

    return Path(*p.parts)


def locate(repo, name, preferred=None):

    """
    Locate files automatically, including files
    stored inside repository subfolders.
    """

    flat_name(name)

    choices = (
        [repo / ".streamlit" / name]
        if name == "config.toml"
        else []
    )

    if preferred is not None:

        choices.append(
            preferred / name
        )

    choices.append(repo / name)

    for path in choices:

        if path.is_file():

            return path

    matches = sorted(

        p for p in repo.rglob(name)

        if p.is_file()

        and ".git" not in p.relative_to(repo).parts

    )

    if len(matches) != 1:

        raise RuntimeError(

            f"Expected one {name}; "
            f"found {len(matches)}: "

            + ", ".join(
                str(p.relative_to(repo))
                for p in matches
            )

        )

    return matches[0]


def read_json(path):

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


# ============================================================
# 4. ADAPT ORIGINAL V13 PYTHON PATHS
# ============================================================

def portable_source(source, home):

    """
    Modify local Path.home() expressions without
    changing the analytical calculations.
    """

    raw = source.encode("utf-8")

    offsets = [0]

    for line in raw.splitlines(keepends=True):

        offsets.append(
            offsets[-1] + len(line)
        )

    edits = []

    for node in ast.walk(ast.parse(source)):

        if (
            not isinstance(node, ast.Call)
            or node.args
            or node.keywords
        ):

            continue

        fn = ast.unparse(node.func)

        if fn not in (
            "Path.home",
            "pathlib.Path.home"
        ):

            continue

        start = (
            offsets[node.lineno - 1]
            + node.col_offset
        )

        end = (
            offsets[node.end_lineno - 1]
            + node.end_col_offset
        )

        replacement = (

            f"{fn.rsplit('.', 1)[0]}"
            f"({str(home)!r})"

        ).encode("utf-8")

        edits.append(
            (start, end, replacement)
        )

    for start, end, replacement in sorted(
        edits,
        reverse=True
    ):

        raw = (
            raw[:start]
            + replacement
            + raw[end:]
        )

    result = raw.decode("utf-8")

    compile(
        result,
        "<portable V13>",
        "exec"
    )

    return result


# ============================================================
# 5. CONVERT WINDOWS MANIFEST PATHS
# ============================================================

def rewrite_paths(value, root):

    if isinstance(value, dict):

        return {

            k: rewrite_paths(v, root)

            for k, v in value.items()

        }

    if isinstance(value, list):

        return [

            rewrite_paths(v, root)

            for v in value

        ]

    if isinstance(value, str):

        normal = value.replace("\\", "/")

        marker = PROJECT + "/"

        pos = normal.lower().find(
            marker.lower()
        )

        if pos >= 0:

            tail = normal[
                pos + len(marker):
            ]

            return (

                str(root / relative_path(tail))

                if tail

                else str(root)

            )

    return value


# ============================================================
# 6. MEMORY-EFFICIENT DATA ACCESS
# ============================================================

def check_lfs(path):

    with path.open("rb") as f:

        if f.read(50).startswith(

            b"version https://git-lfs.github.com/spec/v1"

        ):

            raise ValueError(

                f"Git LFS content not downloaded: "
                f"{path.name}"

            )


def link_data(source, target):

    check_lfs(source)

    try:

        target.symlink_to(source)

    except OSError:

        try:

            os.link(source, target)

        except OSError:

            shutil.copy2(source, target)


# ============================================================
# 7. DISCOVER ALL REQUIRED REPOSITORY FILES
# ============================================================

def discover(repo):

    map_path = locate(
        repo,
        "file_path_map.json"
    )

    mapping = read_json(map_path)

    if mapping.get("dashboard_version") != "V13":

        raise ValueError(
            "The export must contain dashboard V13."
        )

    entries = mapping["files"]

    matrix = mapping[
        "silver_manifest_keys"
    ]["household_matrix"]

    flat_name(matrix)

    for name in (*PY_FILES, MANIFEST, matrix):

        if name not in entries:

            raise ValueError(
                f"Missing entry in path map: {name}"
            )

    # --------------------------------------------------------
    # The original household NPY file is no longer
    # required inside the GitHub repository.
    # --------------------------------------------------------

    meta_path = locate(
        repo,
        "household_parts.json"
    )

    meta = read_json(meta_path)

    if (

        meta.get("original_filename") != matrix

        or meta.get("compression") != "gzip"

    ):

        raise ValueError(

            "Household compression manifest does "
            "not match the V13 matrix."

        )

    parts = meta["parts"]

    if (

        not isinstance(parts, list)

        or not parts

        or len(set(parts)) != len(parts)

    ):

        raise ValueError(
            "Invalid household parts list."
        )

    if any(

        not re.fullmatch(

            r"household_consumption_part_\d+\.gz",

            flat_name(n)

        )

        for n in parts

    ):

        raise ValueError(
            "Unexpected household part filename."
        )

    if (

        not isinstance(
            meta.get("original_size"),
            int
        )

        or meta["original_size"] <= 0

    ):

        raise ValueError(
            "Invalid household original size."
        )

    if not re.fullmatch(

        r"[0-9a-f]{64}",

        meta.get("sha256", "")

    ):

        raise ValueError(
            "Invalid household SHA256 checksum."
        )

    # Automatically locate all compressed files.

    part_paths = [

        locate(
            repo,
            name,
            meta_path.parent
        )

        for name in parts

    ]

    # Locate all other dashboard assets.
    # Explicitly exclude the original household matrix.

    assets = {

        name: locate(
            repo,
            name,
            map_path.parent
        )

        for name in entries

        if name != matrix

    }

    return (

        mapping,
        matrix,
        meta,
        part_paths,
        assets,
        map_path,
        meta_path

    )


# ============================================================
# 8. RECONSTRUCT HOUSEHOLD CONSUMPTION
# ============================================================

def restore_household(meta, parts, target):

    """
    Reconstruct the original NPY file by streaming
    the compressed parts directly to disk.

    The complete matrix is never assembled in RAM
    during decompression.
    """

    digest = hashlib.sha256()

    written = 0

    temporary = target.with_name(

        target.name + ".tmp"

    )

    try:

        with temporary.open("wb") as out:

            # The correct sequence is specified
            # in household_parts.json.

            for path in parts:

                check_lfs(path)

                with gzip.open(path, "rb") as src:

                    while True:

                        block = src.read(
                            1024 * 1024
                        )

                        if not block:

                            break

                        written += len(block)

                        if written > meta["original_size"]:

                            raise ValueError(

                                "Household data exceeds "
                                "expected size."

                            )

                        digest.update(block)

                        out.write(block)

        # Verify the exact original file size.

        if written != meta["original_size"]:

            raise ValueError(

                "Household reconstruction "
                "size mismatch."

            )

        # Verify the original SHA256 checksum.

        if digest.hexdigest() != meta["sha256"]:

            raise ValueError(

                "Household reconstruction "
                "SHA256 mismatch."

            )

        # Only accept a completely verified file.

        os.replace(
            temporary,
            target
        )

    finally:

        temporary.unlink(
            missing_ok=True
        )


# ============================================================
# 9. PREPARE STREAMLIT CLOUD ENVIRONMENT
# ============================================================

def prepare(repo_str, snapshot):

    repo = Path(repo_str)

    (
        mapping,
        matrix,
        meta,
        parts,
        assets,
        _,
        _

    ) = discover(repo)

    # Create an isolated temporary environment
    # on the Streamlit Cloud server.

    holder = tempfile.TemporaryDirectory(

        prefix="octopus_v13_"

    )

    home = Path(holder.name)

    root = home / PROJECT

    try:

        targets = {

            n: root / relative_path(p)

            for n, p in mapping["files"].items()

        }

        if len(set(targets.values())) != len(targets):

            raise ValueError(

                "Duplicate original file paths "
                "in path map."

            )

        for name, target in targets.items():

            target.parent.mkdir(

                parents=True,

                exist_ok=True

            )

            # ------------------------------------------------
            # HOUSEHOLD MATRIX
            # Reconstruct from compressed GitHub files.
            # ------------------------------------------------

            if name == matrix:

                restore_household(

                    meta,

                    parts,

                    target

                )

                continue

            source = assets[name]

            # ------------------------------------------------
            # PYTHON DASHBOARD FILES
            # ------------------------------------------------

            if name in PY_FILES:

                target.write_text(

                    portable_source(

                        source.read_text(
                            encoding="utf-8-sig"
                        ),

                        home

                    ),

                    encoding="utf-8"

                )

            # ------------------------------------------------
            # SILVER MANIFEST
            # ------------------------------------------------

            elif name == MANIFEST:

                manifest = rewrite_paths(

                    read_json(source),

                    root

                )

                for key, filename in mapping[
                    "silver_manifest_keys"
                ].items():

                    if filename not in targets:

                        raise ValueError(

                            f"Missing exported "
                            f"Silver asset: {filename}"

                        )

                    manifest["data_files"][key] = str(

                        targets[filename]

                    )

                target.write_text(

                    json.dumps(

                        manifest,

                        indent=2,

                        ensure_ascii=False

                    ),

                    encoding="utf-8"

                )

            # ------------------------------------------------
            # REMAINING DATASETS AND CONFIGURATION
            # ------------------------------------------------

            else:

                link_data(

                    source,

                    target

                )

        return (

            holder,

            targets[PY_FILES[0]]

        )

    except BaseException:

        holder.cleanup()

        raise


# ============================================================
# 10. STREAMLIT APPLICATION
# ============================================================

def main():

    import streamlit as st

    try:

        (
            _,
            _,
            _,
            parts,
            assets,
            map_path,
            meta_path

        ) = discover(REPO)

        sources = [

            map_path,

            meta_path,

            *parts,

            *assets.values()

        ]

        snapshot = tuple(

            (

                str(p),

                p.stat().st_size,

                p.stat().st_mtime_ns

            )

            for p in sorted(set(sources))

        )

        # Cache the complete reconstructed environment.
        # Household data is restored once per cached
        # environment, not every time filters change.

        cached_prepare = st.cache_resource(

            show_spinner=False

        )(prepare)

        holder, entry = cached_prepare(

            str(REPO),

            snapshot

        )

    except Exception as exc:

        st.error(

            "Unable to initialise the V13 dashboard."

        )

        st.exception(exc)

        st.stop()

        return

    # Make the original V13 components importable.

    folder = str(entry.parent)

    if folder not in sys.path:

        sys.path.insert(

            0,

            folder

        )

    for filename in PY_FILES[1:]:

        module = Path(filename).stem

        loaded = sys.modules.get(module)

        if (

            loaded is not None

            and Path(

                getattr(loaded, "__file__", "")

            ).resolve()

            != (entry.parent / filename).resolve()

        ):

            sys.modules.pop(

                module,

                None

            )

    # ========================================================
    # EXECUTE THE ORIGINAL V13 DASHBOARD
    # ========================================================

    runpy.run_path(

        str(entry),

        run_name="__main__"

    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
