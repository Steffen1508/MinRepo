
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})

OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)


#  WORKSHOP 1 - Horse Survival (manglende data)
print("=" * 60)
print("WORKSHOP 1: Manglende data (Horse Survival)")
print("=" * 60)

horse = pd.read_csv(os.path.join(OUT, "horse.csv"))
print(f"Datasaet: {horse.shape[0]} raekker, {horse.shape[1]} kolonner")

#Kolonner med >50% manglende droppes (for sparsomme til imputering)
high_miss = horse.columns[horse.isna().mean() > 0.5].tolist()
print(f"Droppede kolonner (>50% manglende): {high_miss}")
horse = horse.drop(columns=high_miss)

#Lesion- og hospital_number-kolonner indeholder ID-lignende info, droppes
horse = horse.drop(columns=["lesion_2", "lesion_3"], errors="ignore")

#Identificer manglende data 
miss_pct = (horse.drop(columns=["hospital_number", "outcome"], errors="ignore")
            .isna().mean().sort_values(ascending=False) * 100)
print("\nManglende data per kolonne (%):")
print(miss_pct[miss_pct > 0].round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
miss_pct[miss_pct > 0].plot.barh(ax=axes[0], color="#c0504d")
axes[0].set_xlabel("Manglende data (%)")
axes[0].set_title("Andel manglende per feature")
axes[0].invert_yaxis()

# Moenster for foerste 80 raekker
display_cols = miss_pct[miss_pct > 0].index.tolist()
axes[1].imshow(horse[display_cols].iloc[:80].isna().values,
               aspect="auto", cmap="Greys", interpolation="nearest")
axes[1].set_xticks(range(len(display_cols)))
axes[1].set_xticklabels(display_cols, rotation=80, fontsize=7)
axes[1].set_ylabel("Patient (raekke)")
axes[1].set_title("Moenster af manglende data (sort = mangler)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "missing_overview.pdf"), bbox_inches="tight")
plt.close()

#(4) Imputering + classifier
# Encode kategoriske kolonner med label encoding (nan -> nan bevares)
horse_enc = horse.copy()
cat_cols = horse_enc.select_dtypes(include=["object", "string"]).columns
encoders = {}
for c in cat_cols:
    if c == "outcome":
        continue
    # Konvertér foerst til object-dtype saa vi kan have blandede typer
    col = horse_enc[c].astype("object")
    le = LabelEncoder()
    mask = col.notna()
    encoded = pd.Series(np.nan, index=horse_enc.index, dtype="float64")
    encoded.loc[mask] = le.fit_transform(col.loc[mask].astype(str))
    horse_enc[c] = encoded
    encoders[c] = le

# Outcome -> 1=lived, 2=died, 3=euthanized
horse_enc["outcome"] = horse_enc["outcome"].map(
    {"lived": 1, "died": 2, "euthanized": 3}
)

# Features: alle numeriske kolonner undtagen ID og outcome
feature_cols = [c for c in horse_enc.columns
                if c not in {"hospital_number", "outcome"}
                and pd.api.types.is_numeric_dtype(horse_enc[c])]
X_raw = horse_enc[feature_cols].copy()
y = horse_enc["outcome"].values
print(f"\nFeatures til klassifikation ({len(feature_cols)}): {feature_cols}")

methods = {
    "Drop NaN rows": "drop",
    "Mean imputation": SimpleImputer(strategy="mean"),       # var: "Mean imputering"
    "Median imputation": SimpleImputer(strategy="median"),   # var: "Median imputering"
    "KNN imputation (k=5)": KNNImputer(n_neighbors=5),       # var: "KNN imputering (k=5)"
    "Iterative (MICE)": IterativeImputer(max_iter=10, random_state=42),  # var: "Iterativ (MICE)"
}
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for name, imp in methods.items():
    if imp == "drop":
        mask = X_raw.notna().all(axis=1)
        Xc, yc = X_raw[mask].values, y[mask]
        if len(np.unique(yc)) < 2 or len(yc) < 30:
            results[name] = (np.nan, np.nan, int(mask.sum()))
            print(f"  {name:<25s} - for faa data efter drop ({mask.sum()} raekker)")
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        scores = cross_val_score(clf, Xc, yc, cv=cv, scoring="accuracy")
        results[name] = (scores.mean(), scores.std(), int(mask.sum()))
    else:
        Xi = imp.fit_transform(X_raw.values)
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        scores = cross_val_score(clf, Xi, y, cv=cv, scoring="accuracy")
        results[name] = (scores.mean(), scores.std(), len(y))

print("\nClassifier-sammenligning (5-fold CV accuracy):")
for k, (m, s, n) in results.items():
    if np.isnan(m):
        print(f"  {k:<25s} - ikke nok data (N={n})")
    else:
        print(f"  {k:<25s} acc={m:.3f} +/- {s:.3f}   N={n}")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
names = list(results.keys())
means = [results[k][0] for k in names]
stds = [results[k][1] for k in names]
colors = ["#7f7f7f", "#5b9bd5", "#70ad47", "#ed7d31", "#9e480e"]
bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("Accuracy (5-fold CV)")
ax.set_ylim(0, 1.0)
ax.set_title("Random Forest accuracy under forskellige imputeringsstrategier")
for b, m in zip(bars, means):
    if not np.isnan(m):
        ax.text(b.get_x() + b.get_width()/2, m + 0.02, f"{m:.2f}",
                ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "imputation_acc.pdf"), bbox_inches="tight")
plt.close()

# Pulse-fordeling foer/efter
if "pulse" in feature_cols:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    ref = horse_enc["pulse"].dropna()
    axes[0].hist(ref, bins=25, color="#5b9bd5", edgecolor="white")
    axes[0].set_title(f"Original (n={len(ref)} obs.)")
    imp_med = SimpleImputer(strategy="median").fit_transform(X_raw.values)
    imp_knn = KNNImputer(n_neighbors=5).fit_transform(X_raw.values)
    pulse_idx = feature_cols.index("pulse")
    axes[1].hist(imp_med[:, pulse_idx], bins=25, color="#70ad47", edgecolor="white")
    axes[1].set_title("Efter median-imputering")
    axes[2].hist(imp_knn[:, pulse_idx], bins=25, color="#ed7d31", edgecolor="white")
    axes[2].set_title("Efter KNN-imputering")
    for a in axes:
        a.set_xlabel("Pulse (BPM)")
    axes[0].set_ylabel("Antal heste")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "pulse_dist.pdf"), bbox_inches="tight")
    plt.close()

#(5) Relationelle databaser
def cols_present(*names):
    return [c for c in names if c in horse.columns]

demographics = horse[cols_present("hospital_number", "age", "surgery",
                                  "surgical_lesion", "cp_data")].copy()
vitals = horse[cols_present("hospital_number", "rectal_temp", "pulse",
                            "respiratory_rate", "temp_of_extremities",
                            "peripheral_pulse",
                            "capillary_refill_time")].copy()
clinical = horse[cols_present("hospital_number", "mucous_membrane", "pain",
                              "peristalsis", "abdominal_distention",
                              "nasogastric_tube", "nasogastric_reflux",
                              "rectal_exam_feces", "abdomen")].copy()
labs = horse[cols_present("hospital_number", "packed_cell_volume",
                          "total_protein")].copy()
outcomes = horse[cols_present("hospital_number", "outcome",
                              "lesion_1")].copy()

db_path = os.path.join(OUT, "horse.db")
conn = sqlite3.connect(db_path)
demographics.to_sql("demographics", conn, if_exists="replace", index=False)
vitals.to_sql("vitals", conn, if_exists="replace", index=False)
clinical.to_sql("clinical", conn, if_exists="replace", index=False)
labs.to_sql("labs", conn, if_exists="replace", index=False)
outcomes.to_sql("outcomes", conn, if_exists="replace", index=False)

example_sql = """
SELECT d.hospital_number, d.age, v.pulse, l.packed_cell_volume, o.outcome
FROM demographics d
JOIN vitals    v ON d.hospital_number = v.hospital_number
JOIN labs      l ON d.hospital_number = l.hospital_number
JOIN outcomes  o ON d.hospital_number = o.hospital_number
WHERE v.pulse > 100
ORDER BY l.packed_cell_volume DESC
LIMIT 5;
"""
join_result = pd.read_sql(example_sql, conn)
print("\nEksempel JOIN-resultat:")
print(join_result.to_string(index=False))
conn.close()


#  WORKSHOP 2 - Daily Climate (stoejfuld data
print("\n" + "=" * 60)
print("WORKSHOP 2: Stoejfuld data (Daily Climate)")
print("=" * 60)

climate = pd.read_csv(os.path.join(OUT, "DailyDelhiClimateTrain.csv"),
                      parse_dates=["date"])
print(f"Datasaet: {climate.shape[0]} dage, {climate.shape[1]} kolonner")
print(climate.describe().round(1).to_string())

# meanpressure har tydelige outliers (max=7679 hPa er sensorfejl)
# Vi bevarer dem til den foerste plot for at vise stoej, men markerer dem
n_outliers = int((climate["meanpressure"] > 1100).sum() +
                 (climate["meanpressure"] < 900).sum())
print(f"\nmeanpressure: {n_outliers} ekstreme outliers (uden for [900, 1100] hPa)")

num_climate = ["meantemp", "humidity", "wind_speed", "meanpressure"]


def noise_metrics(s, w=15):
    smoothed = s.rolling(w, center=True, min_periods=1).mean()
    residual = s - smoothed
    snr = s.var() / residual.var() if residual.var() > 0 else np.inf
    return {"std": float(s.std()),
            "noise_std": float(residual.std()),
            "SNR": float(snr)}

stats = {col: noise_metrics(climate[col]) for col in num_climate}
stats_df = pd.DataFrame(stats).T
print("\nStoej-metrikker:")
print(stats_df.round(2).to_string())

labels_map = {"meantemp": "Temperatur (C)", "humidity": "Fugtighed (%)",
              "wind_speed": "Vindhastighed (km/h)",
              "meanpressure": "Tryk (hPa)"}
colors2 = ["#c0504d", "#4f81bd", "#9bbb59", "#8064a2"]

# Plot raa data - klip y-akse for tryk for at undgaa outliers daekker over moenstret
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
for ax, c, col in zip(axes, num_climate, colors2):
    ax.plot(climate["date"], climate[c], color=col, lw=0.6)
    ax.set_ylabel(labels_map[c])
    ax.grid(alpha=0.3)
    if c == "meanpressure":
        # Vis outliers, men indstil y-akse til meningsfuldt omraade
        ax.set_ylim(990, 1030)
        ax.text(0.99, 0.95, f"({n_outliers} outliers > 1030 hPa)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#666")
axes[-1].set_xlabel("Dato")
axes[0].set_title("Raa klimatidsserier (Delhi 2013-2017) - synlig stoej og outliers")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "climate_raw.pdf"), bbox_inches="tight")
plt.close()


def rolling_smooth(s, w, kind="mean"):
    if kind == "mean":
        return s.rolling(w, center=True, min_periods=1).mean()
    return s.rolling(w, center=True, min_periods=1).median()


# Vinduessammenligning - bruger meantemp som hovedeksempel
windows = [3, 7, 15, 31]
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(climate["date"], climate["meantemp"],
             color="lightgray", lw=0.6, label="Raa")
for w, col in zip(windows, ["#7f7f7f", "#5b9bd5", "#ed7d31", "#c0504d"]):
    axes[0].plot(climate["date"], rolling_smooth(climate["meantemp"], w),
                 lw=1.2, color=col, label=f"vindue={w}d")
axes[0].set_ylabel("Temperatur (C)")
axes[0].set_title("Effekt af vinduesstoerrelse (rolling mean) paa meantemp")
axes[0].legend(ncol=5, fontsize=8)
axes[0].grid(alpha=0.3)

# Demonstrer median-filterets robusthed paa meanpressure (med outliers)
press_clip = climate["meanpressure"].clip(lower=900, upper=1100)
# Brug raa data uden klip - sa ser man tydeligt at mean bliver paavirket
axes[1].plot(climate["date"], climate["meanpressure"],
             color="lightgray", lw=0.6, label="Raa")
axes[1].plot(climate["date"], rolling_smooth(climate["meanpressure"], 15, "mean"),
             color="#5b9bd5", lw=1.3, label="Rolling mean (w=15)")
axes[1].plot(climate["date"], rolling_smooth(climate["meanpressure"], 15, "median"),
             color="#c0504d", lw=1.3, label="Rolling median (w=15)")
axes[1].set_ylabel("Tryk (hPa)")
axes[1].set_xlabel("Dato")
axes[1].set_ylim(990, 1030)
axes[1].set_title("Mean vs. median-filter paa meanpressure (median er robust over for outliers)")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "window_compare.pdf"), bbox_inches="tight")
plt.close()

# Foer/efter for alle 4 variable
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for ax, c, col in zip(axes, num_climate, colors2):
    ax.plot(climate["date"], climate[c], color="lightgray", lw=0.5, label="Raa")
    # Brug median for tryk pga outliers, mean for resten
    method = "median" if c == "meanpressure" else "mean"
    ax.plot(climate["date"], rolling_smooth(climate[c], 15, method),
            color=col, lw=1.3, label=f"Glattet (w=15, {method})")
    ax.set_ylabel(labels_map[c])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    if c == "meanpressure":
        ax.set_ylim(990, 1030)
axes[-1].set_xlabel("Dato")
axes[0].set_title("Foer (graa) og efter (farve) praeprocessering med 15-dages vindue")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "before_after.pdf"), bbox_inches="tight")
plt.close()

# Reduktion i residual-std
red = []
for c in num_climate:
    row = {"feature": c, "raw_std": climate[c].std()}
    for w in [3, 7, 15, 31]:
        row[f"std_w{w}"] = (climate[c] - rolling_smooth(climate[c], w)).std()
    red.append(row)
red_df = pd.DataFrame(red).set_index("feature")
print("\nResidualstandardafvigelse efter glatning:")
print(red_df.round(2).to_string())

print("\n" + "=" * 60)
print("Faerdig - alle figurer er gemt i:", FIGS)
print("=" * 60)
