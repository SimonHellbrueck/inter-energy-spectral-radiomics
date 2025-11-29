def generate_inter_energy_features(df, index_col='patient', cols_to_exclude=None, recon_col='reconstruction_type', feature_cols=None):
    # sensible defaults
    cols_to_exclude = cols_to_exclude or []
    feature_cols = feature_cols or [c for c in df.columns if c not in [index_col, recon_col] + cols_to_exclude]

    # 0) Drop rows missing critical fields (prevents NaNs from entering via pivot)
    df = df.dropna(subset=[index_col, recon_col] + feature_cols)

    # Columns to keep in the index: patient + other excluded cols
    meta_cols = [index_col] + cols_to_exclude

    # 1) Pivot and fill missing values (prevents NaNs if some recon/features are absent)
    df_wide = (
        df.pivot(index=meta_cols, columns=recon_col, values=feature_cols)
          .fillna(0)  # fill missing measurements with 0 (adjust if another sentinel is better)
    )

    relative_features = pd.DataFrame(index=df_wide.index)

    recon_types = df[recon_col].dropna().unique()
    comparisons = list(combinations(recon_types, 2))
    print("Reconstruction combinations:", comparisons)

    # 2) Ratio & difference features (zero-safe denominator)
    eps = 1e-6
    for feature in feature_cols:
        for (e1, e2) in comparisons:
            if (feature, e1) in df_wide.columns and (feature, e2) in df_wide.columns:
                a = df_wide[(feature, e1)].fillna(0)
                b = df_wide[(feature, e2)].fillna(0)
                ratio_name = f"{feature}_ratio_{e1}_vs_{e2}"
                diff_name  = f"{feature}_diff_{e1}_vs_{e2}"
                relative_features[ratio_name] = a / (b.replace(0, np.nan).fillna(eps))
                relative_features[diff_name]  = a - b

    # Helper to extract keV
    def extract_keV(label):
        try:
            return int(label.split('_')[1].replace('keV', ''))
        except Exception:
            return None

    # 3) Mono energies sorted
    mono_types = sorted([r for r in recon_types if 'Mono' in str(r)], key=lambda x: extract_keV(x))

    # 4) Slope features (consecutive + long-range), zero-safe delta
    mono_keV_pairs = list(zip(mono_types[:-1], mono_types[1:]))
    long_range_pairs = [(mono_types[0], mono_types[-1])] if len(mono_types) >= 2 else []

    for (e1, e2) in mono_keV_pairs + long_range_pairs:
        k1, k2 = extract_keV(e1), extract_keV(e2)
        if k1 is None or k2 is None or (k2 - k1) == 0:
            continue
        delta_keV = float(k2 - k1)
        for feature in feature_cols:
            if (feature, e1) in df_wide.columns and (feature, e2) in df_wide.columns:
                v1 = df_wide[(feature, e1)].fillna(0)
                v2 = df_wide[(feature, e2)].fillna(0)
                slope_name = f"{feature}_slope_{e1}_to_{e2}"
                relative_features[slope_name] = (v1 - v2) / delta_keV

    # 5) Mono energy stats (operate on filled data)
    for feature in feature_cols:
        mono_cols = [(feature, r) for r in mono_types if (feature, r) in df_wide.columns]
        if len(mono_cols) > 1:
            mono_values = df_wide[mono_cols].fillna(0)
            relative_features[f"{feature}_std_across_mono"]  = mono_values.std(axis=1)
            relative_features[f"{feature}_mono_mean"]        = mono_values.mean(axis=1)
            relative_features[f"{feature}_mono_range"]       = mono_values.max(axis=1) - mono_values.min(axis=1)

    # 6) Slope std across keV (fill first to avoid NaNs in std)
    for feature in feature_cols:
        slope_cols = [col for col in relative_features.columns if isinstance(col, str) and col.startswith(f"{feature}_slope_")]
        if slope_cols:
            relative_features[f"{feature}_slope_std_across_keV"] = relative_features[slope_cols].fillna(0).std(axis=1)

    # Final: ensure no NaNs leave the function
    return relative_features.fillna(0).reset_index()