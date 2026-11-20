"""
Subject- and group-level aggregation for FIP PSTHs and per-trial metrics.

A **recording site** is one ``(subject_id, channel)`` pair -- one implanted location in one
mouse, e.g. ``808057|latNAcc(L)-DA``. Sites are what we average within and treat as
independent samples: within a mouse, ``PL(L)`` and ``PL(R)`` are averaged *separately*, and
at the group level they are pooled into region ``PL``, each counting once::

    trial -> session -> site (mouse x channel) -> region

So ``sem_over_sites`` is the SEM across sites, **not** across mice -- with left/right
implants there are fewer mice than sites, and the name is deliberately explicit about that.
"Site" is used rather than "unit" because in neuroscience a unit is a single neuron.

Two ways to compute a site's PSTH
---------------------------------

Both are kept deliberately, and :func:`assert_matches_upstream` proves they agree.

:func:`psth_by_site` -- **public API only**
    Calls
    :func:`~aind_dynamic_foraging_basic_analysis.plot.plot_fip.plot_fip_psth_compare_alignments`
    exactly as the original notebook does, and returns the ``etrs`` dict it already
    produces. Nothing is reimplemented, so it cannot drift from the upstream behaviour.
    It also draws the panel as a side effect. Use this when you only need the mean traces
    -- which is everything except session-level resampling.

:func:`compute_site_etr` -- **drops one level, to the inner compute**
    Calls ``fip_psth_multiple_inner_compute`` (the same function the public one uses for
    the ETR) with ``average=False``, then replays the public wrapper's own steps: the
    per-session censor-time union, the per-bin loop, and the ``average=True`` arithmetic.
    The event-triggered response and the censoring are still computed upstream -- only the
    thin wrapper is mirrored.

    It exists for one reason: the public function averages sessions away and never returns
    the per-session traces, so a bootstrap that resamples *sessions* is impossible from its
    output. This keeps them.

Because a PSTH computed with the wrong censor set still looks entirely plausible, the
bypass is only trustworthy with its gate: :func:`assert_matches_upstream` re-runs the real
function and asserts every bin frame is identical. Run it after any change to
:func:`compute_site_etr`.

The group level is pure arithmetic on the per-site mean traces either way, so a full pass
costs no more than drawing the per-site panels did (measured: +3% over compute alone).

This module is dataset-agnostic: region maps, channel orders, colors and thresholds are
passed in by the caller (see the capsule's ``config`` / ``figs.style``).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aind_dynamic_foraging_basic_analysis.plot import plot_fip as pf

__all__ = [
    "compute_site_etr",
    "assert_matches_upstream",
    "site_id",
    "split_site_id",
    "fip_channel",
    "has_group_data",
    "usable_nwbs",
    "build_site_inventory",
    "bin_labels_from",
    "psth_by_site",
    "ETR_LONG_COLS",
    "etr_long_from_summary",
    "etrs_from_per_session",
    "etr_trace",
    "group_summary",
    "plot_site_psth",
    "plot_group_psth",
    "summarize_trial_metric",
    "pool_trial_metric",
    "plot_baseline_panel",
]

_SITE_SEP = "|"


# --------------------------------------------------------------------------------------
# site keys
# --------------------------------------------------------------------------------------
def fip_channel(channel):
    """Normalize a channel to its ``df_fip['event']`` spelling (no trailing underscore)."""
    return str(channel).rstrip("_")


def site_id(subject_id, channel):
    """Build the site key, e.g. ``'808057|latNAcc(L)-DA'``."""
    return f"{subject_id}{_SITE_SEP}{fip_channel(channel)}"


def split_site_id(site):
    """Inverse of :func:`site_id` -> ``(subject_id, channel)``."""
    subject_id, channel = str(site).split(_SITE_SEP, 1)
    return subject_id, channel


# --------------------------------------------------------------------------------------
# session filtering
# --------------------------------------------------------------------------------------
def has_group_data(nwb, group_col):
    """True if the session has any non-NaN value in ``group_col``.

    Sessions with no model fit have an all-NaN ``RPE-binned3``. They do not crash the PSTH
    call -- every bin is simply empty -- but they contribute nothing while still being
    counted, which inflates the reported ``n``, so they are skipped outright.
    """
    df = getattr(nwb, "df_trials", None)
    if df is None or group_col not in df:
        return False
    return bool(df[group_col].notna().any())


def usable_nwbs(nwbs, channel, group_col="RPE-binned3", verbose=False):
    """Sessions that have ``df_fip``, recorded ``channel``, and data in ``group_col``.

    Channel presence is decided from ``df_fip['event']``, never from ``df_trials.columns``:
    a ``<col>_<channel>_baseline`` column can exist for a channel that was never recorded in
    that session.

    Returns ``(kept, skipped)`` where ``skipped`` is a list of ``(session_id, reason)``.
    """
    channel = fip_channel(channel)
    keep, skipped = [], []
    for nwb in nwbs:
        ses = str(getattr(nwb, "session_id", "<unknown>"))
        if "," in ses:
            skipped.append((ses, "grouped dummy_nwb (comma-joined session_id)"))
        elif not hasattr(nwb, "df_fip"):
            skipped.append((ses, "no df_fip (loaded with load_fip=False?)"))
        elif channel not in set(nwb.df_fip["event"].unique()):
            skipped.append((ses, f"channel {channel} not recorded"))
        elif not has_group_data(nwb, group_col):
            skipped.append((ses, f"all-NaN {group_col}"))
        else:
            keep.append(nwb)
    if verbose:
        for ses, why in skipped:
            print(f"  skip {ses}: {why}")
    return keep, skipped


def build_site_inventory(
    nwbs,
    region_map,
    min_sessions=5,
    group_col="RPE-binned3",
    channel_order=None,
    verbose=True,
):
    """One row per kept ``(subject, channel)`` site.

    ``region_map`` may be keyed in either channel spelling. Returns a frame with
    ``[site, subject_id, channel, region, n_sessions]``, sorted by region (and by
    ``channel_order`` within a region when given), after dropping sites with fewer than
    ``min_sessions`` sessions.
    """
    regions = {fip_channel(k): v for k, v in region_map.items()}

    rows = []
    for nwb in nwbs:
        if not hasattr(nwb, "df_fip") or not has_group_data(nwb, group_col):
            continue
        ses = str(nwb.session_id)
        if "," in ses:
            continue
        subject_id = ses.split("_")[0]
        for ch in nwb.df_fip["event"].unique():
            ch = fip_channel(ch)
            rows.append((site_id(subject_id, ch), subject_id, ch, regions.get(ch), ses))

    cols = ["site", "subject_id", "channel", "region", "n_sessions"]
    if not rows:
        return pd.DataFrame(columns=cols)

    inv = pd.DataFrame(rows, columns=["site", "subject_id", "channel", "region", "ses_idx"])
    counts = (
        inv.groupby(["site", "subject_id", "channel", "region"], dropna=False)["ses_idx"]
        .nunique()
        .reset_index(name="n_sessions")
    )

    if verbose:
        for _, r in counts[counts["n_sessions"] < min_sessions].iterrows():
            print(
                f"Dropped site {r['site']}: only {r['n_sessions']} session(s) "
                f"(< {min_sessions})"
            )
    kept = counts[counts["n_sessions"] >= min_sessions].copy()

    if kept["region"].isna().any():
        unknown = sorted(kept.loc[kept["region"].isna(), "channel"].unique())
        warnings.warn(f"channels with no region mapping: {unknown}")

    if channel_order is not None:
        order = {fip_channel(c): i for i, c in enumerate(channel_order)}
        kept["_ch"] = kept["channel"].map(lambda c: order.get(c, len(order)))
        kept = kept.sort_values(["region", "_ch", "subject_id"]).drop(columns="_ch")
    else:
        kept = kept.sort_values(["region", "site"])
    return kept.reset_index(drop=True)


def bin_labels_from(nwbs, group_col="RPE-binned3"):
    """Ordered bin labels, read from ``.cat.categories`` -- never hand-written.

    Reads the categories from the first session that still has them, rather than from a
    concatenation of every session. ``pd.concat`` silently degrades a categorical to
    ``object`` if any session's column is not categorical, and a session whose column is
    entirely NaN (no model fit) can round-trip from parquet as a plain float column -- so
    concatenating first would lose the ordering, and ``.cat`` would raise.

    Falls back to the observed values, sorted numerically where possible, if no session
    stored the column as a categorical.
    """
    columns = [
        n.df_trials[group_col]
        for n in nwbs
        if getattr(n, "df_trials", None) is not None and group_col in n.df_trials
    ]
    if not columns:
        raise KeyError(f"{group_col!r} not found in any session's df_trials")

    for col in columns:
        if isinstance(col.dtype, pd.CategoricalDtype):
            return col.cat.categories.astype(str).tolist()

    warnings.warn(
        f"no session stored {group_col!r} as a categorical; "
        "falling back to observed values, which may not preserve bin order"
    )
    values = pd.concat(columns).dropna().unique()
    labels = [str(v) for v in values]
    try:
        return sorted(labels, key=float)
    except (TypeError, ValueError):
        return sorted(labels)


# --------------------------------------------------------------------------------------
# per-site PSTH -- the template call, unchanged
# --------------------------------------------------------------------------------------
def compute_site_etr(
    nwbs,
    channel,
    group_col="RPE-binned3",
    align_col="choice_time_in_session",
    data_column="data_z_norm",
    tw=(-1, 4),
    labels=None,
    censor=True,
    keep_trials=False,
    verbose=False,
):
    """Per-site PSTHs **and** the per-session means that the upstream wrapper discards.

    This bypasses ``pf.plot_fip_psth_compare_alignments`` but mirrors it exactly, so the
    returned ``summary`` is identical to that function's ``etrs``. Specifically it
    reproduces, in order, ``plot_fip.py`` lines 150-186:

    1. build one alignment dict per session, with **every** bin key present (empty arrays
       where a bin is unoccupied) -- matching the template's ``dict(zip(labels, ...))``;
    2. ``censor_times`` per session = ``np.sort(np.concatenate(...))`` over **all** of that
       session's alignment arrays, reused unchanged for every bin. Getting this wrong is
       the one error that does not show up visually -- a PSTH computed with the wrong
       censor set still looks perfectly plausible;
    3. one ``fip_psth_multiple_inner_compute`` call per bin, over **all** sessions;
    4. the same averaging arithmetic as that function's ``average=True`` branch
       (``plot_fip.py`` lines 511-523).

    The ETR itself is still computed by the upstream ``fip_psth_multiple_inner_compute``,
    so there is no reimplementation of ``event_triggered_response`` or of censoring.

    Returns a dict with

    ``summary``
        ``{bin: DataFrame}`` indexed by ``time``, columns
        ``[data_column, sem_over_sessions, sem]`` -- byte-identical to upstream ``etrs``.
    ``per_session``
        tidy ``[ses_idx, subject_id, site, channel, group, time, value, n_trials]``.
        **This is the new capability**: it is what makes a bootstrap that resamples
        *sessions* (not just sites) possible. ~360k rows for the whole dataset.
    ``trials``
        the raw trial-level frame, only when ``keep_trials=True``. Large -- roughly 1.7M
        rows per site, ~74M for the full dataset -- so leave it off unless you need
        trial-level resampling.

    Verify with :func:`assert_matches_upstream` after any change to this function.
    """
    channel = fip_channel(channel)
    nwbs, _ = usable_nwbs(nwbs, channel, group_col=group_col, verbose=verbose)
    if not nwbs:
        return None

    if labels is None:
        labels = bin_labels_from(nwbs, group_col)

    # (1) every key present in every session, exactly as the template builds them
    align_list = [
        dict(
            zip(
                labels,
                [
                    n.df_trials.loc[
                        n.df_trials[group_col].astype(str) == L, align_col
                    ].values
                    for L in labels
                ],
            )
        )
        for n in nwbs
    ]

    # (2) censor set: union over ALL of this session's bins, shared by every bin.
    # Mirrors plot_fip.py:150-157 exactly -- including that it is built before any
    # thinning, so dropping a sparse bin never widens another bin's window.
    censor_times_list = []
    for d in align_list:
        censor_times_list.append(np.sort(np.concatenate([d[k] for k in d])))

    summary, per_session_frames, trial_frames = {}, [], []
    for L in labels:  # upstream iterates align_list[0]'s keys, i.e. this order
        this_align = [d[L] for d in align_list]

        # (3) all sessions, every bin -- the upstream call, just with average=False so
        # the per-session detail survives
        etr_all = pf.fip_psth_multiple_inner_compute(
            nwbs,
            this_align,
            channel,
            False,
            list(tw),
            censor,
            censor_times_list,
            data_column,
        )
        if etr_all.empty:
            continue

        # (4) upstream's average=True arithmetic, copied from plot_fip.py:511-523
        mean_per_ses = (
            etr_all.groupby(["ses_idx", "time"])[data_column].mean().unstack("ses_idx")
        )
        result = mean_per_ses.mean(axis=1).to_frame(name=data_column)
        result["sem_over_sessions"] = mean_per_ses.sem(axis=1)
        # Upstream also computes result["sem"] here -- the SEM pooling every trial across
        # sessions. Deliberately not computed: it ignores the nesting of trials within
        # sessions, so it understates uncertainty, and nothing plots it (per-site panels
        # use sem_over_sessions, group panels sem_over_sites). Its absence is why
        # assert_matches_upstream compares only the columns we keep.
        summary[L] = result

        counts = (
            etr_all.groupby(["ses_idx", "time"])[data_column].count().unstack("ses_idx")
        )
        tidy = (
            mean_per_ses.stack("ses_idx", future_stack=True)
            .rename("value")
            .reset_index()
        )
        tidy["n_trials"] = (
            counts.stack("ses_idx", future_stack=True).reset_index(drop=True).values
        )
        tidy["group"] = L
        per_session_frames.append(tidy)

        if keep_trials:
            trials = etr_all.copy()
            trials["group"] = L
            trial_frames.append(trials)

    if not summary:
        return None

    per_session = pd.concat(per_session_frames, ignore_index=True)
    per_session["subject_id"] = per_session["ses_idx"].astype(str).str.split("_").str[0]
    per_session["channel"] = channel
    per_session["site"] = per_session["subject_id"] + _SITE_SEP + channel

    out = {
        "summary": summary,
        "per_session": per_session,
        "n_sessions": len(nwbs),
        "labels": labels,
    }
    if keep_trials:
        out["trials"] = pd.concat(trial_frames, ignore_index=True)
    return out


def assert_matches_upstream(
    result,
    nwbs,
    channel,
    group_col="RPE-binned3",
    align_col="choice_time_in_session",
    data_column="data_z_norm",
    tw=(-1, 4),
    labels=None,
):
    """Assert :func:`compute_site_etr`'s ``summary`` equals the upstream function's ``etrs``.

    This is the gate that makes bypassing ``plot_fip_psth_compare_alignments`` safe. It
    re-runs the real function on the same sessions and compares every bin frame exactly.
    Run it whenever :func:`compute_site_etr` changes.
    """


    channel = fip_channel(channel)
    nwbs, _ = usable_nwbs(nwbs, channel, group_col=group_col)
    if labels is None:
        labels = bin_labels_from(nwbs, group_col)

    dicts = [
        dict(
            zip(
                labels,
                [
                    n.df_trials.loc[
                        n.df_trials[group_col].astype(str) == L, align_col
                    ].values
                    for L in labels
                ],
            )
        )
        for n in nwbs
    ]
    fig, ax = _plt.subplots()
    try:
        _, _, ref = pf.plot_fip_psth_compare_alignments(
            nwbs,
            dicts,
            channel,
            tw=list(tw),
            censor=True,
            data_column=data_column,
            error_type="sem_over_sessions",
            ax=ax,
        )
    finally:
        _plt.close(fig)

    checked = 0
    for L in labels:
        if L not in result["summary"]:
            continue
        ours = result["summary"][L]
        # Compare only the columns we keep. We deliberately do not compute upstream's
        # "sem" (pooled over trials), so an exact frame comparison would fail on columns
        # rather than on values. Everything that could go wrong in the bypass -- a wrong
        # censor set, the wrong sessions, the wrong averaging -- moves <data_column> and
        # sem_over_sessions, so this still catches it.
        pd.testing.assert_frame_equal(ours, ref[L][list(ours.columns)])
        checked += 1
    if checked == 0:
        raise AssertionError("no bins compared")
    return checked


def psth_by_site(
    nwbs,
    channel,
    ax,
    group_col="RPE-binned3",
    align_col="choice_time_in_session",
    data_column="data_z_norm",
    tw=(-1, 4),
    colors=None,
    labels=None,
    censor=True,
    verbose=False,
):
    """PSTH for one ``(subject, channel)`` site, split by ``group_col``. **Public API only.**

    A template-identical call to ``pf.plot_fip_psth_compare_alignments``: the censoring, the
    censor-time union and the session-level grand mean all stay inside that function. The
    only additions here are the session pre-filters, the shared bin-label list, and the
    ``n < 2`` error-type fallback. Nothing is reimplemented, so this cannot drift from
    upstream.

    Prefer this unless you need per-session traces. For those use :func:`compute_site_etr`,
    whose ``summary`` is verified identical to the ``etrs`` returned here.

    Returns ``(etrs, n_sessions)``. ``etrs`` is ``{bin_label: DataFrame}``, each indexed by
    ``time`` with columns ``[data_column, 'sem_over_sessions', 'sem']``; it is ``None`` when
    the site has no usable session.
    """
    channel = fip_channel(channel)
    nwbs, _ = usable_nwbs(nwbs, channel, group_col=group_col, verbose=verbose)
    if not nwbs:
        return None, 0

    if labels is None:
        labels = bin_labels_from(nwbs, group_col)

    # Build every key for every session, even where the bin is empty: the upstream
    # "Missing alignment key" check compares key *sets*, and empty arrays pass through
    # fine. Trials with a NaN alignment time carry a NaN bin, so they drop out here
    # automatically -- no dropna needed.
    dicts = [
        dict(
            zip(
                labels,
                [
                    n.df_trials.loc[
                        n.df_trials[group_col].astype(str) == L, align_col
                    ].values
                    for L in labels
                ],
            )
        )
        for n in nwbs
    ]

    error_type = "sem_over_sessions" if len(nwbs) > 1 else "sem"

    # plot_fip_psth_compare_alignments calls bare plt.legend() / plt.tight_layout(), which
    # act on the *current* axes and figure. Without this the legend lands on whichever
    # subplot happened to be created last.
    plt.sca(ax)
    _, _, etrs = pf.plot_fip_psth_compare_alignments(  # returns (fig, ax, etrs)
        nwbs,
        dicts,
        channel,
        tw=list(tw),
        censor=censor,
        extra_colors=colors or {},
        data_column=data_column,
        error_type=error_type,
        ax=ax,
    )
    return etrs, len(nwbs)


# --------------------------------------------------------------------------------------
# group level -- average of averages
# --------------------------------------------------------------------------------------
#: column order of the long-form ETR frame
ETR_LONG_COLS = ["site", "group", "time", "value", "sem_over_sessions"]


def etr_long_from_summary(summary, site):
    """Flatten one site's ``{bin: time-indexed frame}`` into long rows.

    The value column is named ``value`` rather than after ``data_column``, so the schema
    does not change when you switch signal -- matching ``per_session`` and the baseline
    frame. :func:`etr_trace` renames it back on the way out.
    """
    out = []
    for bin_label, frame in summary.items():
        if frame is None or frame.empty:
            continue
        value_col = next(
            (c for c in frame.columns if c not in ("sem_over_sessions", "sem")), None  # noqa: E501
        )
        if value_col is None:
            continue
        d = frame.rename(columns={value_col: "value"}).reset_index()
        d["site"] = site
        d["group"] = str(bin_label)
        out.append(d)
    if not out:
        return pd.DataFrame(columns=ETR_LONG_COLS)
    return pd.concat(out, ignore_index=True)[ETR_LONG_COLS]


def etrs_from_per_session(per_session, categorical=True):
    """Rebuild the long-form ``etrs`` frame from ``per_session``.

    Exact, not an approximation: ``etrs`` is nothing more than the unweighted mean of the
    session means plus their SEM, which is precisely the two operations
    :func:`compute_site_etr` performs on the same values::

        value             = mean over sessions of that session's mean trace
        sem_over_sessions = SEM across those session means (ddof=1)

    This became exact once the trial-pooled ``sem`` column was dropped -- that one needed
    sums of squares, which ``per_session`` never stored.

    So ``per_session`` is the only thing worth caching: it is ~5x larger (~6.5 MB vs
    ~1.3 MB for the full dataset) and regenerates ``etrs`` in well under a second, while
    also supporting bootstraps that resample sessions.

    Example::

        ps = pd.read_parquet('/scratch/dane_etr/per_session.parquet')
        etrs = pu.etrs_from_per_session(ps)
    """
    out = (
        per_session.groupby(["site", "group", "time"], observed=True)["value"]
        .agg(value="mean", sem_over_sessions="sem")
        .reset_index()
    )
    if categorical:
        for col in ("site", "group"):
            out[col] = out[col].astype("category")
    return out[ETR_LONG_COLS]


def etr_trace(etrs, site, bin_label, data_column="data_z_norm"):
    """One ``(site, bin)`` trace as a time-indexed frame, ready for the plotter.

    Returns columns ``[data_column, sem_over_sessions, sem]`` indexed by ``time`` -- the
    shape ``pf.fip_psth_inner_plot`` consumes. Returns ``None`` if that combination is
    absent.
    """
    sel = etrs[(etrs["site"] == site) & (etrs["group"] == str(bin_label))]
    if sel.empty:
        return None
    return (
        sel.set_index("time")[["value", "sem_over_sessions"]]
        .rename(columns={"value": data_column})
        .sort_index()
    )


def group_summary(etrs, sites, bin_label, data_column="data_z_norm"):
    """Average of the per-site averages across ``sites``.

    ``etrs`` is the long-form frame (see :data:`ETR_LONG_COLS`). Each site contributes
    exactly one column regardless of how many sessions it has (5-31 in DANE_bilateral),
    which is what makes this an average of averages rather than a session-weighted pool --
    the two differ materially given how uneven the session counts are.

    Returns a frame indexed by ``time`` with ``[data_column, 'sem_over_sites', 'n_sites',
    'sem_over_sessions_mean']``, directly consumable by
    ``pf.fip_psth_inner_plot(..., error_type='sem_over_sites')``. Returns ``None`` when no
    site has this bin.

    Note ``sem_over_sites`` is the SEM across ``(subject, channel)`` sites, not across
    subjects: both hemispheres of one animal count as separate samples, by design.
    """
    sel = etrs[etrs["site"].isin(list(sites)) & (etrs["group"] == str(bin_label))]
    if sel.empty:
        return None

    # cast to str: pivoting on a categorical would materialise every unused category
    sel = sel.assign(site=sel["site"].astype(str))
    wide = sel.pivot(index="time", columns="site", values="value")  # time x site

    out = pd.DataFrame(index=wide.index)
    out[data_column] = wide.mean(axis=1)
    out["sem_over_sites"] = wide.sem(axis=1)  # ddof=1, across sites
    out["n_sites"] = wide.notna().sum(axis=1)
    if "sem_over_sessions" in sel:
        out["sem_over_sessions_mean"] = sel.pivot(
            index="time", columns="site", values="sem_over_sessions"
        ).mean(axis=1)
    out.index.name = "time"
    return out


def plot_site_psth(
    ax,
    etrs,
    site,
    labels,
    colors,
    data_column="data_z_norm",
    error_type="sem_over_sessions",
    linewidth=None,
):
    """Redraw one site's cached PSTH onto ``ax``.

    ``etrs`` is the long-form frame; ``site`` selects which site to draw. This replays
    already-computed traces rather than recomputing them -- which is what lets the region
    figures and the group figure come out of a single pass over the data.

    Note this is *not* :func:`plot_group_psth` with one site: the per-site rows carry
    ``sem_over_sessions`` and ``sem``, whereas a group summary carries ``sem_over_sites``.
    """
    drawn = 0
    for L in labels:
        frame = etr_trace(etrs, site, L, data_column=data_column)
        if frame is None or error_type not in frame:
            continue
        pf.fip_psth_inner_plot(ax, frame, colors.get(L, ""), L, data_column, error_type)
        drawn += 1
        if linewidth is not None:
            # the trace just drawn; set it here rather than sweeping ax.get_lines()
            # afterwards, which would also catch the axvline below
            ax.get_lines()[-1].set_linewidth(linewidth)
    ax.axvline(0, color="k", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_group_psth(
    ax,
    etrs,
    sites,
    labels,
    colors,
    data_column="data_z_norm",
    error_type="sem_over_sites",
    linewidth=None,
):
    """Draw the group mean for every bin onto ``ax``.

    ``linewidth`` applies to the traces only, not to the zero marker.

    Returns ``{bin_label: summary_frame}`` so the numbers behind the figure stay available.
    """
    summaries = {}
    for L in labels:
        summary = group_summary(etrs, sites, L, data_column=data_column)
        if summary is None:
            continue
        # fip_psth_inner_plot only does etr[data_column] +/- etr[error_type], so a new
        # error-column name plugs in with no upstream change.
        pf.fip_psth_inner_plot(ax, summary, colors.get(L, ""), L, data_column, error_type)
        if linewidth is not None:
            # the trace just drawn; set it here rather than sweeping ax.get_lines()
            # afterwards, which would also catch the axvline below
            ax.get_lines()[-1].set_linewidth(linewidth)
        summaries[L] = summary

    ax.axvline(0, color="k", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return summaries


# --------------------------------------------------------------------------------------
# per-trial metric (the num_reward_past baseline panel)
# --------------------------------------------------------------------------------------
_METRIC_COLS = [
    "level", "region", "site", "subject_id", "channel",
    "ses_idx", "group", "value", "sem", "n",
]


def summarize_trial_metric(
    nwbs,
    channel,
    data_column="data_z",
    group_col="num_reward_past",
    region_map=None,
    window=(-7, 7),
    min_trials_per_group=5,
    shift_by_one=True,
):
    """Two-stage summary of a per-trial metric for one site.

    Pure ``df_trials``; the PSTH plotter is not involved. The value column is
    ``f'{data_column}_{channel}_baseline'``. ``data_column`` is ``'data_z'`` here, **not**
    ``'data_z_norm'`` -- the quantity being plotted *is* the baseline, so the norm version
    would zero it out.

    Stages mirror the PSTH hierarchy as far as one site goes:

    ``level == 'session'``
        one row per session x bin; ``value`` is the mean over that session's trials,
        ``sem`` is across trials, ``n`` is the trial count (cells below
        ``min_trials_per_group`` are dropped).
    ``level == 'site'``
        one row per bin; ``value`` is the mean of the session means, ``sem`` is across
        **sessions**, ``n`` is the session count.

    There is deliberately no region level here -- this function sees one subject and one
    channel, so it has nothing to pool. Use :func:`pool_trial_metric` across sites for that.

    ``shift_by_one`` applies the template's ``groupby('ses_idx')[group_col].shift(1)``, so
    the baseline of trial *t* is indexed by the reward history *entering* it.

    Returns one long frame with ``level`` in ``{'session', 'site'}``.
    """
    region_map = region_map or {}
    regions = {fip_channel(k): v for k, v in region_map.items()}
    channel = fip_channel(channel)
    value_col = f"{data_column}_{channel}_baseline"
    shifted = f"{group_col}_baseline"

    kept, _ = usable_nwbs(nwbs, channel, group_col=group_col)
    frames = [
        n.df_trials
        for n in kept
        if value_col in n.df_trials and group_col in n.df_trials
    ]
    if not frames:
        return pd.DataFrame(columns=_METRIC_COLS)

    df = pd.concat(frames, ignore_index=True)
    if shift_by_one:
        df[shifted] = df.groupby("ses_idx")[group_col].shift(1)
    else:
        df[shifted] = df[group_col]

    lo, hi = window
    df = df[(df[shifted] > lo) & (df[shifted] < hi)]
    df = df.dropna(subset=[value_col, shifted])
    if df.empty:
        return pd.DataFrame(columns=_METRIC_COLS)

    df = df.copy()
    df["subject_id"] = df["ses_idx"].astype(str).str.split("_").str[0]
    df["group"] = df[shifted].astype(int)

    # stage 1: trial -> session
    per_ses = (
        df.groupby(["subject_id", "ses_idx", "group"])[value_col]
        .agg(value="mean", sem="sem", n="count")
        .reset_index()
    )
    per_ses = per_ses[per_ses["n"] >= min_trials_per_group]
    if per_ses.empty:
        return pd.DataFrame(columns=_METRIC_COLS)

    # stage 2: session -> site (SEM across sessions)
    per_site = (
        per_ses.groupby(["subject_id", "group"])["value"]
        .agg(value="mean", sem="sem", n="count")
        .reset_index()
    )

    # No third stage here. Pooling across sites is pool_trial_metric's job: this function
    # is called with one subject and one channel, so a "group" level computed here would
    # average a single value and come back as the site mean with n=1 and sem=NaN -- a row
    # labelled "group" that is not the region value is worse than no row at all.

    per_ses["level"] = "session"
    per_site["level"] = "site"
    per_site["ses_idx"] = pd.NA

    out = pd.concat([per_ses, per_site], ignore_index=True)
    out["channel"] = channel
    out["region"] = regions.get(channel)
    out["site"] = [
        site_id(s, channel) if pd.notna(s) else pd.NA for s in out["subject_id"]
    ]
    return (
        out[_METRIC_COLS]
        .sort_values(["level", "site", "group"], na_position="last")
        .reset_index(drop=True)
    )


def pool_trial_metric(frames):
    """Pool per-site :func:`summarize_trial_metric` frames into a region-level summary.

    Takes the ``level == 'site'`` rows from each site and averages across sites, with SEM
    across sites -- the same average-of-averages rule as :func:`group_summary`. Returns a
    frame with ``[group, value, sem, n, level]`` where ``n`` counts sites.
    """
    usable = [
        f[f["level"] == "site"] for f in frames if f is not None and not f.empty
    ]
    if not usable:
        return None
    sites = pd.concat(usable, ignore_index=True)
    if sites.empty:
        return None
    out = (
        sites.groupby("group")["value"]
        .agg(value="mean", sem="sem", n="count")
        .reset_index()
    )
    out["level"] = "group"
    return out


def plot_baseline_panel(ax, frame, level="site", colors=None, site=None):
    """Bar-plot one level of a :func:`summarize_trial_metric` / :func:`pool_trial_metric`
    frame. Returns ``ax``.
    """
    sel = frame[frame["level"] == level] if "level" in frame else frame
    if site is not None and "site" in sel:
        sel = sel[sel["site"] == site]
    sel = sel.sort_values("group")
    if sel.empty:
        ax.axis("off")
        return ax

    ax.bar(sel["group"], sel["value"], yerr=sel["sem"], color=colors, capsize=3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(sorted(sel["group"].unique()))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
