import pandas as pd
import numpy as np
import pickle
import warnings
import xlwings as xw
import os

from scipy.stats import norm
from scipy.optimize import minimize

warnings.simplefilter("ignore")

# =====================================================================
# PATHS  [ORIGINAL - INCHANGÉ]
# =====================================================================
cache_path = r"C:\Users\M121040\OneDrive - GROUP DIGITAL WORKPLACE\Bureau\ProjetVol\Moi\market_cache.pkl"
excel_path = r"C:\Users\M121040\OneDrive - GROUP DIGITAL WORKPLACE\Bureau\ProjetVol\Moi\Moi1.xlsx"

# =====================================================================
# LOAD CACHE  [ORIGINAL - INCHANGÉ]
# =====================================================================
with open(cache_path, "rb") as f:
    market_cache = pickle.load(f)

t0 = market_cache["meta"]["t0"]
t1 = market_cache["meta"]["t1"]
date_t0 = market_cache["meta"]["date_t0"]
date_t1 = market_cache["meta"]["date_t1"]
AssetList = market_cache["meta"]["assets"]

# =====================================================================
# PARAMÈTRES  [ORIGINAL - INCHANGÉ]
# =====================================================================
step_coarse = 0.5
step_fine = 0.01
fine_half_width = 0.25
round_level = 2
r = 0.0
q = 0.0
ATM_MIN = 0.8
ATM_MAX = 1.2

# =====================================================================
# [BERGOMI] NOUVEAU PARAMÈTRE : seuil minimum de mouvement du spot
# ─────────────────────────────────────────────────────────────────────
# Source : Bergomi "Smile Dynamics IV" (2009), section 2.2
#
# Le SSR (Skew Stickiness Ratio) est défini comme :
#   R_T = E[dσ_ATM · dlnS] / (ψ · E[(dlnS)²])
#
# Quand dS/S ≈ 0, les trois facteurs sticky (dStickyDelta, dStickySkew)
# sont tous ≈ 0 (ils sont proportionnels à dS). La régression devient
# dégénérée : on explique un signal ≈ 0 par des régresseurs ≈ 0.
# Les poids résultants sont instables et sans signification économique.
#
# Solution : flaguer ces jours. Le SSR réalisé sera NaN pour ces dates,
# ce qui signale au trader que la décomposition n'est pas fiable.
# =====================================================================
MIN_DS_PCT = 0.003  # 0.3% — seuil minimum |dS/S| pour SSR valide


# =====================================================================
# [BERGOMI] NOUVELLE FONCTION : calcul du skew ATM ψ
# ─────────────────────────────────────────────────────────────────────
# Source : Bergomi IV, section 2 — ψ = dσ/dK|_{K=S} est la pente du
# smile au strike ATM. C'est la quantité centrale du SSR :
#   SSR_implied = a·1 + b·0 + c·2 = a + 2c
#   SSR_realized = dσ_ATM / (ψ · dS/S)
#
# On utilise une différence finie centrée sur le spot avec un pas de 1%
# pour rester dans la zone liquide du smile.
# =====================================================================
def compute_atm_skew(x_vol, y_vol, S, h_pct=0.01):
    """
    Calcule la pente du skew ATM ψ = dσ/dK|_{K=S}
    par différences finies centrées.

    Paramètres :
        x_vol  : array strikes (absolus)
        y_vol  : array vols implicites correspondantes
        S      : spot (strike ATM)
        h_pct  : pas en % du spot pour la différenciation (défaut 1%)

    Retourne :
        psi    : pente du skew en vol/strike (ex: -0.002 par point de strike)
    """
    h = S * h_pct
    vol_up = interp_xy(x_vol, y_vol, np.array([S + h]))[0]
    vol_dn = interp_xy(x_vol, y_vol, np.array([S - h]))[0]
    return (vol_up - vol_dn) / (2.0 * h)


# =====================================================================
# LOOP CALCULS  [ORIGINAL - STRUCTURE INCHANGÉE]
# Seules des sections clairement marquées [BERGOMI] ont été ajoutées
# =====================================================================
all_rows = []

for asset in AssetList:
    try:
        print(f"Processing {asset} ...")

        vols = market_cache["data"][asset]["vols"]
        spots = market_cache["data"][asset]["spots"]

        if vols is None or spots is None or len(vols) == 0 or len(spots) == 0:
            print(f"Pas de données pour {asset}")
            continue

        spots = spots.copy()
        spots.index = spots["DATE"]
        spots = spots["MID"].astype(float)

        if date_t0 not in spots.index or date_t1 not in spots.index:
            print(f"Spot manquant pour {asset}")
            continue

        S0 = float(spots.loc[date_t0])
        S1 = float(spots.loc[date_t1])
        dS = S1 - S0

        # [BERGOMI] Calcul du mouvement spot en % — utilisé pour le SSR
        # et pour le flag LowSpotMove.
        # Référence : Bergomi IV, dénominateur du SSR : E[(dlnS)²]
        dS_pct = abs(dS / S0) if S0 > 0 else 0.0
        low_spot_move = (dS_pct < MIN_DS_PCT)

        vols = vols.copy()
        vols["MATURITY"] = pd.to_datetime(vols["MATURITY"])
        vols = vols[vols["DATE"].isin([date_t0, date_t1])].copy()

        if vols.empty:
            print(f"Pas de vols pour {asset}")
            continue

        vols["T_from_t0"] = vols["MATURITY"].apply(lambda x: business_days_between_dates(x, t0))

        spot_map = {date_t0: S0, date_t1: S1}
        vols["K"] = (
            vols["STRIKE"].astype(float)
            * vols["DATE"].map(spot_map).astype(float)
            / 100.0
        )

        vols["vol"] = vols["VOLATILITY"].astype(float) / 100.0

        vol_t0_all = vols[vols["DATE"] == date_t0].copy()
        vol_t1_all = vols[vols["DATE"] == date_t1].copy()

        common_maturities = sorted(
            set(vol_t0_all["MATURITY"]).intersection(set(vol_t1_all["MATURITY"]))
        )

        for maturity in common_maturities:
            try:
                slice_t0 = vol_t0_all[vol_t0_all["MATURITY"] == maturity].copy()
                slice_t1 = vol_t1_all[vol_t1_all["MATURITY"] == maturity].copy()
                market_strikes_t0 = np.round(slice_t0["K"].unique(), 2)
                market_strikes_t1 = np.round(slice_t1["K"].unique(), 2)

                market_strikes_union = np.unique(
                    np.round(np.r_[market_strikes_t0, market_strikes_t1], 2)
                )

                if slice_t0.empty or slice_t1.empty:
                    continue

                T0_days = business_days_between_dates(maturity, t0)
                T1_days = business_days_between_dates(maturity, t1)

                if T0_days <= 0 or T1_days <= 0:
                    continue

                T0_years = T0_days / 252.0
                T1_years = T1_days / 252.0

                x0 = slice_t0["K"].to_numpy(dtype=float)
                y0 = slice_t0["vol"].to_numpy(dtype=float)

                x1 = slice_t1["K"].to_numpy(dtype=float)
                y1 = slice_t1["vol"].to_numpy(dtype=float)

                if len(x0) < 3 or len(x1) < 3:
                    continue

                k_min = max(np.min(x0), np.min(x1))
                k_max = min(np.max(x0), np.max(x1))

                if k_min >= k_max:
                    continue

                K_grid = build_adaptive_grid(
                    k_min=k_min,
                    k_max=k_max,
                    market_strikes=market_strikes_union,
                    S0=S0,
                    S1=S1,
                    step_coarse=step_coarse,
                    step_fine=step_fine,
                    fine_half_width=fine_half_width
                )

                if len(K_grid) < 5:
                    continue

                # ── courbes marché  [ORIGINAL - INCHANGÉ] ────────────
                vol0 = interp_xy(x0, y0, K_grid)
                vol1 = interp_xy(x1, y1, K_grid)

                sticky_strike = vol0.copy()

                vol_t0_full = build_interpolated_slice_on_grid(slice_t0, K_grid)

                sticky_delta = sticky_delta_curve(
                    vol_t0_full=vol_t0_full,
                    K_grid=K_grid,
                    S0=S0,
                    S1=S1,
                    T0_years=T0_years,
                    T1_years=T1_years,
                    r=r,
                    q=q
                )

                sigma_t0_S0 = interp_xy(x0, y0, np.array([S0]))[0]
                sigma_t0_S1 = interp_xy(x0, y0, np.array([S1]))[0]
                sticky_skew = interp_xy(x0, y0, K_grid - dS) - sigma_t0_S0 + sigma_t0_S1

                # ── [BERGOMI] Calcul du skew ATM ψ ───────────────────
                # ψ = dσ/dK|_{K=S0} : pente du smile au strike ATM en t0
                # C'est le dénominateur du SSR (Bergomi IV, sec. 2.2).
                # Un ψ négatif (normal en equity) signifie que les puts
                # downside sont plus chers que les calls upside.
                psi_t0 = compute_atm_skew(x0, y0, S0, h_pct=0.01)

                # ── [BERGOMI] Vol ATM à t0 et t1 ─────────────────────
                # Ces deux valeurs permettent de calculer le mouvement
                # de vol ATM réalisé, numérateur du SSR réalisé.
                sigma_atm_t0 = sigma_t0_S0  # déjà calculé plus haut
                sigma_atm_t1 = interp_xy(x1, y1, np.array([S1]))[0]
                d_sigma_atm  = sigma_atm_t1 - sigma_atm_t0

                # ── dataframe  [ORIGINAL - INCHANGÉ] ─────────────────
                df_reg = pd.DataFrame({
                    "Strike": K_grid,
                    "%Strike": K_grid / S1,
                    "VolT0": vol0,
                    "VolT1": vol1,
                    "dVol": vol1 - vol0,
                    "Sticky Strike Vol": sticky_strike,
                    "Sticky Delta Vol": sticky_delta,
                    "Sticky Skew Vol": sticky_skew,
                    "dStickyDelta": sticky_delta - vol0,
                    "dStickySkew": sticky_skew - vol0
                }).dropna().copy()

                df_reg = df_reg[(df_reg["%Strike"] > ATM_MIN) & (df_reg["%Strike"] < ATM_MAX)]
                df_reg["IsMarket_t0"] = df_reg["Strike"].round(2).isin(market_strikes_t0)
                df_reg["IsMarket_t1"] = df_reg["Strike"].round(2).isin(market_strikes_t1)

                if len(df_reg) < 5:
                    continue

                # ── vega  [ORIGINAL - INCHANGÉ] ──────────────────────
                df_reg["Vega"] = bs_vega(
                    S=S0,
                    K=df_reg["Strike"].to_numpy(dtype=float),
                    T=T0_years,
                    sigma=df_reg["VolT0"].to_numpy(dtype=float),
                    r=r,
                    q=q
                )

                y  = df_reg["dVol"].to_numpy(dtype=float)
                sd = df_reg["dStickyDelta"].to_numpy(dtype=float)
                sk = df_reg["dStickySkew"].to_numpy(dtype=float)

                # ── pondération sqrt(vega)  [ORIGINAL - INCHANGÉ] ────
                w = np.sqrt(np.clip(df_reg["Vega"].to_numpy(dtype=float), 1e-10, None))
                w = w / np.mean(w)

                ss_tot_test = np.sum((y - y.mean()) ** 2)
                if ss_tot_test < 1e-10:
                    continue

                # ── REGRESSION CONTRAINTE  [ORIGINAL - INCHANGÉ] ─────
                # dVol ~ b*dStickyDelta + c*dStickySkew + d
                # b,c >= 0 ; b+c <= 1 ; a = 1-b-c
                def objective(params):
                    b, c, d = params
                    y_pred = b * sd + c * sk + d
                    return np.sum(w * (y - y_pred) ** 2)

                constraints = [
                    {"type": "ineq", "fun": lambda p: 1 - p[0] - p[1]}
                ]

                bounds = [
                    (0, 1),
                    (0, 1),
                    (None, None)
                ]

                starting_points = [
                    np.array([0.20, 0.20, 0.0]),
                    np.array([0.60, 0.20, 0.0]),
                    np.array([0.20, 0.60, 0.0]),
                    np.array([0.45, 0.45, 0.0]),
                    np.array([0.05, 0.80, 0.0]),
                    np.array([0.80, 0.05, 0.0]),
                ]

                best_result = None
                best_obj = np.inf

                for x0_init in starting_points:
                    result = minimize(
                        objective,
                        x0_init,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                        options={"maxiter": 1000, "ftol": 1e-12}
                    )

                    if result.success:
                        obj = objective(result.x)
                        if obj < best_obj:
                            best_obj = obj
                            best_result = result

                if best_result is None:
                    continue

                b, c, d = best_result.x
                a = 1 - b - c

                df_reg["dVol_pred"] = b * sd + c * sk + d

                # ── R² classique  [ORIGINAL - INCHANGÉ] ──────────────
                ss_res = np.sum((df_reg["dVol"] - df_reg["dVol_pred"]) ** 2)
                ss_tot = np.sum((df_reg["dVol"] - df_reg["dVol"].mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                # ── R² pondéré  [ORIGINAL - INCHANGÉ] ────────────────
                y_bar_w   = np.sum(w * y) / np.sum(w)
                ss_res_w  = np.sum(w * (y - df_reg["dVol_pred"].to_numpy(dtype=float)) ** 2)
                ss_tot_w  = np.sum(w * (y - y_bar_w) ** 2)
                wr2 = 1 - ss_res_w / ss_tot_w if ss_tot_w > 0 else np.nan

                # ── [BERGOMI] SSR IMPLICITE ───────────────────────────
                # Source : Bergomi "Smile Dynamics IV" (2009), section 2.2
                #
                # Chaque régime sticky implique un SSR différent :
                #   • Sticky Delta  → SSR = 0  (vol ATM ne bouge pas)
                #   • Sticky Strike → SSR = 1  (vol ATM bouge exactement comme le skew)
                #   • Sticky Skew   → SSR = 2  (local vol, vol ATM sur-réagit)
                #
                # Bergomi démontre que SSR ∈ [1, 2] pour tout modèle de vol
                # stochastique à diffusion (Bergomi IV, section 3.1).
                # Des valeurs hors de cet intervalle signalent soit des chocs
                # autonomes (vol-of-vol), soit des jours à faible spot move.
                #
                # Formule : SSR_implied = a×1 + b×0 + c×2 = a + 2c
                SSR_implied = a + 2.0 * c

                # ── [BERGOMI] SSR RÉALISÉ ─────────────────────────────
                # Source : Bergomi IV, section 2.2, définition de R_T
                # en version point-à-point (une seule observation t0→t1) :
                #
                #   SSR_realized = Δσ_ATM / (ψ × ΔlnS)
                #
                # Interprétation pour le trader :
                #   SSR_realized > SSR_implied → vol a bougé PLUS que le
                #     modèle sticky ne le prédisait → P&L cross-gamma positif
                #     si short vanna (Bergomi IV, eq. 4.8)
                #   SSR_realized < SSR_implied → sous-réaction de la vol →
                #     P&L cross-gamma négatif
                #
                # Valeur NaN si le spot n'a pas assez bougé (low_spot_move)
                # car le dénominateur ψ×ΔlnS devient trop petit pour être
                # fiable numériquement.
                d_ln_S = np.log(S1 / S0) if S0 > 0 else np.nan
                denom_ssr = psi_t0 * d_ln_S if (not low_spot_move and abs(psi_t0) > 1e-8) else np.nan

                if denom_ssr is not np.nan and abs(denom_ssr) > 1e-8:
                    SSR_realized = d_sigma_atm / denom_ssr
                else:
                    SSR_realized = np.nan

                # ── [BERGOMI] ÉCART SSR (diagnostic hedge) ───────────
                # Source : Bergomi IV, section 4, eq. 4.8
                #
                # L'écart SSR_realized - SSR_implied est directement lié
                # au P&L cross-gamma/theta d'une position vanilla delta-hedgée
                # et vega-hedgée. Un écart persistant peut être arbitré
                # (Bergomi IV, section 4.3 : backtest sur EuroStoxx50).
                #
                # Pour les traders : si SSR_error > 0 systématiquement
                # sur une maturité, le skew de marché est trop "steep"
                # par rapport à la dynamique réalisée.
                SSR_error = (SSR_realized - SSR_implied
                             if not np.isnan(SSR_realized) else np.nan)

                # ── [BERGOMI] DÉCOMPOSITION DU MOUVEMENT DE VOL ATM ──
                # Source : Bergomi IV, interprétation des composantes du P&L
                #
                # Le mouvement de vol ATM se décompose en :
                #   1) Partie spot-driven  = SSR_implied × ψ × dS/S
                #      → ce que le modèle sticky prédit pour le mouvement ATM
                #   2) Partie autonome     = d_sigma_atm - partie spot-driven
                #      → choc non expliqué par le spot (vol-of-vol, mean
                #        reversion, choc de marché externe)
                #      → correspond à l'intercept d de la régression
                #
                # Cette décomposition aide les traders à distinguer :
                #   • un mouvement de vol "normal" (spot-driven, coverable
                #     par le vega hedge sticky)
                #   • un choc de vol autonome (nécessite un ajustement de
                #     la position vega indépendamment du spot)
                d_sigma_atm_spot_driven = SSR_implied * psi_t0 * (dS / S0) if S0 > 0 else np.nan
                d_sigma_atm_autonomous  = (d_sigma_atm - d_sigma_atm_spot_driven
                                           if d_sigma_atm_spot_driven is not np.nan else np.nan)

                # ── colonnes finales  [ORIGINAL - INCHANGÉ sauf ajouts] ──
                df_reg["UDL"]          = asset
                df_reg["Maturity"]     = maturity
                df_reg["Date t0"]      = date_t0
                df_reg["Date t1"]      = date_t1
                df_reg["Spot t0"]      = S0
                df_reg["Spot t1"]      = S1
                df_reg["T"]            = T1_days
                df_reg["%StickyStrike"] = 100 * a
                df_reg["%StickyDelta"]  = 100 * b
                df_reg["%StickySkew"]   = 100 * c
                df_reg["R2"]           = r2
                df_reg["Weighted_R2"]  = wr2

                # ── [BERGOMI] NOUVELLES COLONNES OUTPUT ──────────────
                # Toutes les colonnes ci-dessous sont ajoutées par rapport
                # au code original. Elles implémentent les métriques SSR
                # de Bergomi "Smile Dynamics IV" (2009).
                df_reg["Psi_ATM"]       = psi_t0       # skew ATM ψ = dσ/dK|_{K=S}
                df_reg["dSigma_ATM"]    = d_sigma_atm  # mouvement vol ATM réalisé

                df_reg["SSR_implied"]   = SSR_implied
                # SSR théorique déduit des poids : a + 2c
                # ∈ [1,2] pour modèles stochastiques, <1 si sticky delta domine

                df_reg["SSR_realized"]  = SSR_realized
                # SSR empirique t0→t1 : dσ_ATM / (ψ × dlnS)
                # NaN si |dS/S| < MIN_DS_PCT (jour plat, non interprétable)

                df_reg["SSR_error"]     = SSR_error
                # Écart = SSR_realized - SSR_implied
                # > 0 : vol sur-réagit vs modèle → P&L cross-gamma positif
                # < 0 : vol sous-réagit → P&L cross-gamma négatif

                df_reg["dSigma_ATM_pred"]   = d_sigma_atm_spot_driven
                # Partie spot-driven du mouvement ATM prédit par le modèle

                df_reg["dSigma_ATM_auto"]   = d_sigma_atm_autonomous
                # Partie autonome = choc non expliqué par le spot
                # ≈ intercept d de la régression évalué à ATM

                df_reg["LowSpotMove"]   = low_spot_move
                # True si |dS/S| < MIN_DS_PCT : SSR_realized non fiable,
                # décomposition sticky potentiellement dégénérée

                market_strikes = np.round(slice_t0["K"].unique(), 2)

                df_out = df_reg[df_reg["Strike"].round(2).isin(market_strikes_union)][[
                    # ── colonnes originales ──────────────────────────
                    "UDL",
                    "Maturity",
                    "Date t0",
                    "Date t1",
                    "Spot t0",
                    "Spot t1",
                    "VolT0",
                    "VolT1",
                    "dVol",
                    "Sticky Strike Vol",
                    "Sticky Delta Vol",
                    "Sticky Skew Vol",
                    "T",
                    "%StickyStrike",
                    "%StickyDelta",
                    "%StickySkew",
                    "R2",
                    "Vega",
                    "Weighted_R2",
                    "Strike",
                    "%Strike",
                    "IsMarket_t0",
                    "IsMarket_t1",
                    # ── colonnes Bergomi (nouvelles) ─────────────────
                    "Psi_ATM",
                    "dSigma_ATM",
                    "SSR_implied",
                    "SSR_realized",
                    "SSR_error",
                    "dSigma_ATM_pred",
                    "dSigma_ATM_auto",
                    "LowSpotMove",
                ]].copy()

                all_rows.append(df_out)

            except Exception:
                continue

        print(f"OK {asset}")

    except Exception as e:
        print(f"Erreur sur {asset}: {e}")
        continue

# =====================================================================
# CONCAT + EXPORT EXCEL VIA XLWINGS  [ORIGINAL - INCHANGÉ sauf colonnes]
# =====================================================================
if len(all_rows) == 0:
    print("Aucun résultat à exporter.")
else:
    df_final = pd.concat(all_rows, ignore_index=True)
    df_final = df_final.sort_values(["UDL", "Maturity", "Strike"]).reset_index(drop=True)

    # ── arrondis légers  [ORIGINAL - INCHANGÉ] ────────────────────────
    for col in [
        "Spot t0", "Spot t1", "VolT0", "VolT1", "dVol",
        "Sticky Strike Vol", "Sticky Delta Vol", "Sticky Skew Vol",
        "%StickyStrike", "%StickyDelta", "%StickySkew", "R2", "Vega",
        "Weighted_R2", "Strike", "%Strike",
        # ── [BERGOMI] nouvelles colonnes numériques à convertir ────────
        "Psi_ATM", "dSigma_ATM",
        "SSR_implied", "SSR_realized", "SSR_error",
        "dSigma_ATM_pred", "dSigma_ATM_auto",
    ]:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    # ── [BERGOMI] Résumé console des métriques SSR ────────────────────
    # Affiche un bilan rapide pour diagnostiquer la session t0→t1
    print("\n" + "=" * 60)
    print(f"[BERGOMI SSR] Résumé session {date_t0} → {date_t1}")
    print("=" * 60)

    summary = (
        df_final[df_final["LowSpotMove"] == False]
        .groupby("UDL")[["SSR_implied", "SSR_realized", "SSR_error"]]
        .mean()
        .round(3)
    )
    print(summary.to_string())
    print()
    print(f"Jours à faible spot move (SSR_realized=NaN) : "
          f"{df_final['LowSpotMove'].sum()} lignes sur {len(df_final)}")
    print("=" * 60 + "\n")

    app = xw.App(visible=False)
    try:
        if os.path.exists(excel_path):
            wb = app.books.open(excel_path)
        else:
            wb = app.books.add()
            wb.save(excel_path)

        try:
            sht = wb.sheets["DATA"]
        except:
            sht = wb.sheets.add("DATA", after=wb.sheets[-1])

        sht.clear_contents()
        sht["A1"].value = df_final

        wb.save()
        wb.close()

    finally:
        app.quit()

    print(f"Fichier exporté : {excel_path}")
    display(df_final.head(20))
