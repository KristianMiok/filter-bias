"""
make_figures_v2.py
==================
Main figures for the MEE resubmission, drawn ONLY from committed reports/*.csv.

  Fig 1  Mechanism & regimes: the practitioner's propensity is corrupted by the
         very error it corrects; three alignment regimes. (exp0b)
  Fig 2  The scale result: everything enters through varrho(D) — simulation
         closed form + the CHELSA layer curve with the three real systems.
         (exp2, survey_rho_curve, survey_battery)
  Fig 3  Observable signature space (AUC x ESS): simulation regimes as anchors,
         34 survey units + 7 crayfish species. (survey_battery, emp2_envelope)
  Fig 4  Ground truth on real data: strategy recovery vs the seed and MCAR
         ceilings — bias beyond sample loss; positivity binds the oracle. (emp3)
  Fig 5  Envelope & gating: positivity governs envelope width across both
         systems; the continuous threshold sweep. (emp2, emp4)

Outputs figures/fig1_mechanism_regimes.pdf ... fig5_envelope_gating.pdf (+.png)
Run from the repo root:   python scripts/make_figures_v2.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
FIG = os.path.join(ROOT, "figures")

C = {"iso": "#4477AA", "toward_HQ": "#EE6677", "away_HQ": "#228833",
     "geo_fixed": "#4477AA", "ALL": "#66CCEE", "FILTER": "#CCBB44",
     "IPW": "#EE6677", "ORACLE": "#AA3377", "SMITH": "#228833", "CALIB": "#4477AA"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  {name}")


def fig1():
    s = pd.read_csv(os.path.join(REP, "exp0b_alignment_summary.csv"))
    s1 = s[s.rho == 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    ax = axes[0]
    for g, lab in [("toward_HQ", "toward precise"), ("geo_fixed", "neutral"), ("away_HQ", "away from precise")]:
        d = s1[s1.geom == g].sort_values("beta")
        ax.plot(d.beta, d.diag_prop_coef_E1, marker="o", ms=3.5, color=C[g],
                ls="--" if g == "geo_fixed" else "-", label=lab)
    b = np.array(sorted(s1.beta.unique()))
    ax.plot(b, s[(s.rho == 0.0) & (s.geom == "toward_HQ")].sort_values("beta").diag_prop_coef_E1,
            ":", color="gray", label="no displacement")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"true coupling $\beta$"); ax.set_ylabel("fitted propensity coefficient")
    ax.set_title("(a) corruption of the propensity", loc="left")
    ax.legend(fontsize=7, frameon=False)
    for ax, g, ttl in [(axes[1], "iso", "(b) isotropic"), (axes[2], "toward_HQ", "(c) toward precise"),
                       (axes[3], "away_HQ", "(d) away from precise")]:
        src = s[(s.rho == 0.0) & (s.geom == "toward_HQ")] if g == "iso" else s1[s1.geom == g]
        d = src.sort_values("beta")
        for col, lab, key in [("rho_sp_ALL", "ALL", "ALL"), ("rho_sp_FILTER", "FILTER", "FILTER"),
                              ("rho_sp_IPW_est", "IPW", "IPW"), ("rho_sp_IPW_oracle", "IPW oracle", "ORACLE")]:
            ax.plot(d.beta, d[col], marker="o", ms=3, color=C[key], ls=":" if key == "ORACLE" else "-", label=lab)
        ax.set_ylim(0.78, 1.005); ax.set_xlabel(r"$\beta$"); ax.set_title(ttl, loc="left")
        if g == "iso":
            ax.set_ylabel("recovery (Spearman vs truth)"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); save(fig, "fig1_mechanism_regimes")


def fig2():
    s = pd.read_csv(os.path.join(REP, "exp2_summary.csv"))
    rc = pd.read_csv(os.path.join(REP, "survey_rho_curve.csv"))
    bat = pd.read_csv(os.path.join(REP, "survey_battery.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    ax = axes[0]
    a = s[(s.part == "A") & (s.regime == "iso")]
    b = s[s.part == "B"]
    ax.scatter(a.diag_rho_D, a.rel_real, s=22, color="#4477AA", label="vary $D$, corr. length (grain 1)")
    ax.scatter(b.diag_rho_D, b.rel_real, s=26, marker="s", facecolors="none",
               edgecolors="#EE6677", label="vary analysis grain")
    pr = a.sort_values("diag_rho_D")
    ax.plot(pr.diag_rho_D, pr.pred_reliability, "k--", lw=1, label="closed form")
    ax.set_xlabel(r"field autocorrelation at displacement, $\varrho(D)$")
    ax.set_ylabel("reliability of the coupling axis")
    ax.set_title("(a) simulation: one curve", loc="left"); ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ok = rc[np.isfinite(rc.rho)]
    ax.plot(ok.lag_km, np.clip(ok.rho, None, 1.0), "-o", ms=4, color="#4477AA")
    ax.set_xscale("log"); ax.set_xlabel("displacement / uncertainty scale (km)")
    ax.set_ylabel(r"CHELSA layer autocorrelation $\varrho(h)$")
    ax.set_title("(b) real layers, real systems", loc="left")
    marks = {"odonata": ("Odonata\n(median low-quality unc.)", "#228833"),
             "orchidaceae": ("Orchidaceae", "#EE6677")}
    for nm, (lab, col) in marks.items():
        row = bat[(bat.name == nm) & (bat.unit == "group")]
        if len(row) and np.isfinite(row.med_unc_low_m.iloc[0]):
            x = max(row.med_unc_low_m.iloc[0] / 1000.0, ok.lag_km.min())
            ax.axvline(x, color=col, lw=1.2, ls="--")
            ax.text(x, 0.74, lab, rotation=90, fontsize=6.5, color=col, ha="right", va="bottom")
    ax.axvspan(2, 25, color="gray", alpha=0.12)
    ax.text(7, 0.735, "crayfish\ndonor-swap test", fontsize=6.5, ha="center", color="gray")
    ax.set_ylim(0.68, 1.01)
    fig.tight_layout(); save(fig, "fig2_scale_rhoD")


def fig3():
    bat = pd.read_csv(os.path.join(REP, "survey_battery.csv")).dropna(subset=["auc", "ess_p99"])
    cray = pd.read_csv(os.path.join(REP, "emp2_envelope.csv"))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    sims = [("away-from-precise (sim)", 0.933, 0.225, C["away_HQ"]),
            ("isotropic (sim)", 0.778, 0.540, C["iso"]),
            ("toward-precise (sim)", 0.695, 0.706, C["toward_HQ"])]
    for lab, a, e, col in sims:
        ax.scatter(a, e, marker="*", s=260, color=col, edgecolors="k", zorder=5, label=lab)
    g = bat[bat.unit == "group"]; sp = bat[bat.unit == "species"]
    ax.scatter(g.auc, g.ess_p99, s=55, color="#66CCEE", edgecolors="k", lw=.5, label="GBIF survey: groups")
    ax.scatter(sp.auc, sp.ess_p99, s=18, color="#66CCEE", alpha=.7, label="GBIF survey: species")
    gated = cray[cray.ipw_gated.astype(bool)]; okc = cray[~cray.ipw_gated.astype(bool)]
    ax.scatter(okc.env_auc, okc.ess_p99, marker="s", s=55, color="#CCBB44", edgecolors="k", lw=.5,
               label="crayfish (IPW reportable)")
    ax.scatter(gated.env_auc, gated.ess_p99, marker="s", s=55, color="#EE6677", edgecolors="k", lw=.5,
               label="crayfish (gated)")
    ax.axhline(0.30, color="k", lw=0.8, ls=":")
    ax.text(0.985, 0.315, "ESS gate 0.30", fontsize=7, ha="right")
    for _, r in bat[bat.ess_p99 < 0.5].iterrows():
        ax.annotate(r["name"], (r.auc, r.ess_p99), fontsize=6, xytext=(3, -7), textcoords="offset points",
                    style="italic")
    for _, r in cray[cray.ess_p99 < 0.06].iterrows():
        ax.annotate(r.species, (r.env_auc, r.ess_p99), fontsize=6, xytext=(3, 4), textcoords="offset points",
                    style="italic")
    ax.set_xlabel("propensity AUC (coupling strength)")
    ax.set_ylabel("ESS fraction of IPW weights (positivity)")
    ax.set_title("Where real datasets sit in the observable signature space", loc="left", fontsize=9.5)
    ax.legend(fontsize=6.6, frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "fig3_signature_space")


def fig4():
    s = pd.read_csv(os.path.join(REP, "emp3_semisynth_summary.csv"))
    s = s[s.D_km == 25.0]
    strategies = [("sp_ALL", "ALL", "ALL"), ("sp_SMITH_env", "relocate", "SMITH"),
                  ("sp_ALL_calib", "calibrate", "CALIB"), ("sp_FILTER", "FILTER", "FILTER"),
                  ("sp_IPW_est", "IPW", "IPW"), ("sp_IPW_oracle", "IPW oracle", "ORACLE")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), sharey=True)
    for ax, axis_name, ttl in [(axes[0], "long_range", "(a) long-range coupling axis (climate)"),
                               (axes[1], "short_range", "(b) short-range axis (topography)")]:
        d = s[s.axis == axis_name].groupby("regime").mean(numeric_only=True)
        x = np.arange(3); w = 0.13
        for i, (col, lab, key) in enumerate(strategies):
            vals = [d.loc[r, col] for r in ["iso", "toward_HQ", "away_HQ"]]
            ax.bar(x + (i - 2.5) * w, vals, w, color=C[key], label=lab if axis_name == "long_range" else None,
                   edgecolor="k", lw=0.3, hatch="//" if key == "ORACLE" else None)
        ax.axhline(d.ceil_seed.iloc[0], color="k", lw=1)
        ax.axhline(d.ceil_mcar.iloc[0], color="k", lw=1, ls="--")
        ax.text(2.42, d.ceil_seed.iloc[0] + .004, "refit ceiling", fontsize=6.5, ha="right")
        ax.text(2.42, d.ceil_mcar.iloc[0] + .004, "random-deletion ceiling", fontsize=6.5, ha="right")
        ax.set_xticks(x); ax.set_xticklabels(["isotropic", "toward", "away"])
        ax.set_title(ttl, loc="left"); ax.set_ylim(0.6, 0.92)
    axes[0].set_ylabel("recovery of truth (Spearman)")
    axes[0].legend(fontsize=6.6, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(); save(fig, "fig4_groundtruth_ceilings")


def fig5():
    e2 = pd.read_csv(os.path.join(REP, "emp2_envelope.csv"))
    e4 = pd.read_csv(os.path.join(REP, "emp4_species_table.csv"))
    sw = pd.read_csv(os.path.join(REP, "emp4_threshold_sweep.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    ax = axes[0]
    ax.scatter(e2.ess_p99, e2.env_width_D_L2, marker="s", s=48,
               c=["#EE6677" if g else "#CCBB44" for g in e2.ipw_gated.astype(bool)],
               edgecolors="k", lw=.5, label=None)
    ax.scatter(e4.ess_p99, e4.env_width_L2, marker="o", s=40, color="#66CCEE", edgecolors="k", lw=.5)
    ax.axvline(0.30, color="k", lw=0.8, ls=":")
    hndl = [plt.Line2D([], [], marker="s", ls="", color="#EE6677", mec="k", label="crayfish, gated"),
            plt.Line2D([], [], marker="s", ls="", color="#CCBB44", mec="k", label="crayfish, reportable"),
            plt.Line2D([], [], marker="o", ls="", color="#66CCEE", mec="k", label="Odonata")]
    ax.legend(handles=hndl, fontsize=6.8, frameon=False, loc="lower right")
    ax.set_xlabel("ESS fraction (positivity)"); ax.set_ylabel(r"$\Lambda=2$ envelope overlap (Schoener $D$)")
    ax.set_title("(a) positivity governs the envelope", loc="left")
    ax = axes[1]
    p = sw[sw.dataset == "POOLED"].sort_values("threshold_m")
    ax.plot(p.threshold_m, p.auc, "-o", ms=4, color="#EE6677", label="propensity AUC")
    ax.plot(p.threshold_m, p.ess_p99, "-s", ms=4, color="#4477AA", label="ESS fraction")
    ax.plot(p.threshold_m, p.retain, "-^", ms=4, color="#228833", label="retained fraction")
    ax.set_xscale("log"); ax.set_xlabel("accuracy threshold (m)")
    ax.set_title("(b) the threshold is not innocent (Odonata)", loc="left")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); save(fig, "fig5_envelope_gating")


def main():
    os.makedirs(FIG, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("all figures written to figures/")


if __name__ == "__main__":
    main()
