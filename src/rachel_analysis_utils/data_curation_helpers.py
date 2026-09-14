"""Load fiber curation CSVs and apply them to FIP data."""

import warnings
from pathlib import Path

import pandas as pd
from aind_bwnm_fiber_data_curation_utils import data as _curation_data

# the data/ subpackage ships an __init__.py, so this follows the install location
CURATION_DATA_DIR = Path(_curation_data.__file__).parent
CURATABLE_PATCH_CORD = r"(G|R|Iso)_\d+"
NO_FIBER = "no_fiber"


def _to_ses_idx(session_id):
    """'behavior_808054_2025-09-02_10-38-37' -> '808054_2025-09-02'."""
    ses = str(session_id)
    if ses.startswith("behavior_"):
        ses = ses[len("behavior_"):]
    return ses.rsplit("_", 1)[0]


def load_curation(csv_path):
    """
    Read a fiber curation results CSV, keyed to ses_idx.

    csv_path: path under CURATION_DATA_DIR, without the .csv extension, e.g.
        'bilateral_4_channels/curation_results/bilateral_4_channels_results'
        (the _G variant covers only G fibers)

    Returns columns ses_idx, patch_cord, target, keep. A fiber with no intended
    measurement -- NaN or the literal 'no_fiber', both appear -- gets target NA.

    Sessions recorded twice in a day share one ses_idx, so their rows merge:
    targets must agree, and keep is ANDed so an ambiguous fiber is excluded.
    """
    raw = pd.read_csv((CURATION_DATA_DIR / csv_path).with_suffix(".csv"))

    df = pd.DataFrame({
        "ses_idx": raw["session_id"].map(_to_ses_idx),
        "patch_cord": raw["fiber"],
        "target": raw["target"].where(raw["target"].notna() & (raw["target"] != NO_FIBER)),
        "keep": raw["keep"].astype(bool),
    })

    conflicting = df.groupby(["ses_idx", "patch_cord"])["target"].nunique(dropna=False) > 1
    if conflicting.any():
        bad = sorted({ses for ses, _ in conflicting[conflicting].index})
        warnings.warn(f"Conflicting targets for {bad} once the session timestamp is dropped; excluding them.")
        df = df[~df["ses_idx"].isin(bad)]

    return df.groupby(["ses_idx", "patch_cord"], as_index=False).agg(
        target=("target", "first"), keep=("keep", "all")
    )


def apply_curation_df_fip(df_fip, curation):
    """
    Set df_fip['event'] to the curated intended measurement and drop fibers that
    did not pass curation.

    Expects the 'patch_cord' column that nwb_utils.split_fiber adds. Call this
    BEFORE enrich_fip_in_df_trials so the df_trials columns are built from the
    target names and need no second rename.

    Raises if a loaded fiber has no curation row -- curation is expected to cover
    everything requested, so absence is a mistake rather than a silent drop.
    """
    patch_cord = df_fip["patch_cord"]
    curatable = patch_cord.str.fullmatch(CURATABLE_PATCH_CORD, na=False)

    lookup = curation.set_index(["ses_idx", "patch_cord"])
    keys = pd.MultiIndex.from_arrays([df_fip["ses_idx"], patch_cord.where(curatable)])

    absent = curatable & ~pd.Series(keys.isin(lookup.index), index=df_fip.index)
    if absent.any():
        pairs = sorted(set(zip(df_fip.loc[absent, "ses_idx"], patch_cord[absent])))
        raise ValueError(
            f"Curation is missing {len(pairs)} loaded (ses_idx, patch_cord) pair(s), e.g. {pairs[:5]}. "
            "Curation must cover every channel requested in parameters['channels']."
        )

    target = pd.Series(lookup["target"].reindex(keys).to_numpy(), index=df_fip.index)
    keep = pd.Series(lookup["keep"].reindex(keys).to_numpy(), index=df_fip.index).eq(True)

    selected = ~curatable | (keep & target.notna())

    out = df_fip.loc[selected].copy()
    out["event"] = target[selected].fillna(patch_cord[selected])
    return out.reset_index(drop=True)
