def generate_inter_energy_features(df, index_col='patient', cols_to_exclude=None,
                                   recon_col='reconstruction_type', feature_cols=None):

    cols_to_exclude = cols_to_exclude or []
    feature_cols = feature_cols or [
        c for c in df.columns if c not in [index_col, recon_col] + cols_to_exclude
    ]

    df = _clean_input(df, index_col, recon_col, feature_cols)
    df_wide, recon_types = _pivot_data(df, index_col, cols_to_exclude, recon_col, feature_cols)

    out = pd.DataFrame(index=df_wide.index)

    _add_ratio_diff_features(out, df_wide, feature_cols, recon_types)
    mono_types = _extract_sorted_mono_types(recon_types)

    _add_slope_features(out, df_wide, feature_cols, mono_types)
    _add_mono_statistics(out, df_wide, feature_cols, mono_types)
    _add_slope_variability(out, feature_cols)

    return out.fillna(0).reset_index()

def _clean_input(df, index_col, recon_col, feature_cols):
    return df.dropna(subset=[index_col, recon_col] + feature_cols)


def _pivot_data(df, index_col, cols_to_exclude, recon_col, feature_cols):
    meta_cols = [index_col] + cols_to_exclude
    df_wide = (
        df.pivot(index=meta_cols, columns=recon_col, values=feature_cols)
          .fillna(0)
    )
    recon_types = df[recon_col].dropna().unique()
    return df_wide, recon_types


def _add_ratio_diff_features(out, df_wide, feature_cols, recon_types):
    eps = 1e-6
    pairs = list(combinations(recon_types, 2))

    for f in feature_cols:
        for e1, e2 in pairs:
            if (f, e1) in df_wide.columns and (f, e2) in df_wide.columns:
                a = df_wide[(f, e1)].fillna(0)
                b = df_wide[(f, e2)].fillna(0)
                out[f"{f}_ratio_{e1}_vs_{e2}"] = a / (b.replace(0, np.nan).fillna(eps))
                out[f"{f}_diff_{e1}_vs_{e2}"] = a - b


def _extract_keV(label):
    try:
        return int(label.split('_')[1].replace('keV', ''))
    except Exception:
        return None


def _extract_sorted_mono_types(recon_types):
    mono = [r for r in recon_types if 'Mono' in str(r)]
    return sorted(mono, key=_extract_keV)


def _add_slope_features(out, df_wide, feature_cols, mono_types):
    if len(mono_types) < 2:
        return

    mono_pairs = list(zip(mono_types[:-1], mono_types[1:]))
    long_range = [(mono_types[0], mono_types[-1])]

    for e1, e2 in mono_pairs + long_range:
        k1, k2 = _extract_keV(e1), _extract_keV(e2)
        if k1 is None or k2 is None or k1 == k2:
            continue

        diff_keV = float(k2 - k1)

        for f in feature_cols:
            if (f, e1) in df_wide.columns and (f, e2) in df_wide.columns:
                v1 = df_wide[(f, e1)].fillna(0)
                v2 = df_wide[(f, e2)].fillna(0)
                out[f"{f}_slope_{e1}_to_{e2}"] = (v1 - v2) / diff_keV


def _add_mono_statistics(out, df_wide, feature_cols, mono_types):
    for f in feature_cols:
        mono_cols = [(f, m) for m in mono_types if (f, m) in df_wide.columns]
        if len(mono_cols) > 1:
            vals = df_wide[mono_cols].fillna(0)
            out[f"{f}_std_across_mono"] = vals.std(axis=1)
            out[f"{f}_mono_mean"] = vals.mean(axis=1)
            out[f"{f}_mono_range"] = vals.max(axis=1) - vals.min(axis=1)


def _add_slope_variability(out, feature_cols):
    for f in feature_cols:
        cols = [c for c in out.columns if isinstance(c, str) and c.startswith(f"{f}_slope_")]
        if cols:
            out[f"{f}_slope_std_across_keV"] = out[cols].fillna(0).std(axis=1)
