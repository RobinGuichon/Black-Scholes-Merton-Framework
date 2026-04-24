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
step_coarse    = 0.5
step_fine      = 0.01
fine_half_width = 0.25
round_level    = 2
r = 0.0
q = 0.0
ATM_MIN = 0.8
ATM_MAX = 1.2

# =====================================================================
# [BERGOMI + BLEND] NOUVEAUX PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────
# MIN_DS_PCT   : en-dessous, le SSR est numériquement instable (dS≈0)
# MIN_DVOL_STD : en-dessous, la régression est dégénérée (dVol≈0 partout)
# NO_SIGNAL_WEIGHTS : poids neutres quand les DEUX méthodes échouent
#   → on peut mettre (1/3, 1/3, 1/3) ou la conviction a priori du desk
# =====================================================================
MIN_DS_PCT      = 0.003          # 0.3% de mouvement spot minimum pour SSR
MIN_DVOL_STD    = 1e-5           # std(dVol) minimum pour régression valide
NO_SIGNAL_WEIGHTS = (1/3, 1/3, 1/3)   # poids neutres si aucun signal


# =====================================================================
# [BERGOMI] FONCTIONS UTILITAIRES NOUVELLES
# =====================================================================

def compute_atm_skew(x_vol, y_vol, S, h_pct=0.01):
    """
    ψ = dσ/dK|_{K=S} — pente du smile ATM.
    Quantité fondamentale du SSR (Bergomi IV, sec. 2.2).
    Différences finies centrées avec un pas de h_pct * S.
    """
    h = S * h_pct
    vol_up = interp_xy(x_vol, y_vol, np.array([S + h]))[0]
    vol_dn = interp_xy(x_vol, y_vol, np.array([S - h]))[0]
    return (vol_up - vol_dn) / (2.0 * h)


def compute_local_skew(x_vol, y_vol, K_grid, h_pct=0.005):
    """
    ψ(K) = dσ/dK pour chaque strike du grid.
    Version pointwise du skew ATM, nécessaire pour le SSR par strike.
    h_pct plus petit qu'ATM car on reste plus local sur la surface.
    """
    psi = np.zeros(len(K_grid))
    for i, K in enumerate(K_grid):
        h = K * h_pct
        vol_up = interp_xy(x_vol, y_vol, np.array([K + h]))[0]
        vol_dn = interp_xy(x_vol, y_vol, np.array([K - h]))[0]
        psi[i] = (vol_up - vol_dn) / (2.0 * h)
    return psi


def ssr_to_weights(ssr_value):
    """
    Convertit un SSR scalaire en triplet de poids (a, b, c).

    Source : Bergomi IV, sec. 2.2 — le SSR paramétrise un axe :
        SSR=0 → pur Sticky Delta   (b=1, a=0, c=0)
        SSR=1 → pur Sticky Strike  (a=1, b=0, c=0)
        SSR=2 → pur Local Vol/Skew (c=1, a=0, b=0)

    Interpolation linéaire par morceaux sur le simplexe :
        SSR ∈ [0,1] : entre SD et SS  → a=SSR,   b=1-SSR, c=0
        SSR ∈ [1,2] : entre SS et SK  → a=2-SSR, b=0,     c=SSR-1

    Vérification : a + 2c = SSR dans les deux cas ✓
    Cas hors-bornes clippé sur [0,2] avant conversion.
    """
    ssr_clipped = np.clip(ssr_value, 0.0, 2.0)
    if ssr_clipped <= 1.0:
        a = ssr_clipped
        b = 1.0 - ssr_clipped
        c = 0.0
    else:
        a = 2.0 - ssr_clipped
        b = 0.0
        c = ssr_clipped - 1.0
    return a, b, c


def blend_weights(a_reg, b_reg, c_reg,
                  a_ssr, b_ssr, c_ssr,
                  alpha):
    """
    Fusion des deux estimateurs.
    alpha = R²_weighted ∈ [0,1] — poids de la régression.
    (1-alpha) = poids du SSR.

    Interprétation :
        alpha→1 (R² élevé, dVol grand)  : régression domine
        alpha→0 (R² faible, dVol petit) : SSR domine

    Les poids blendés sommant à 1 par construction car
    chaque triplet est dans le simplexe.
    """
    a = alpha * a_reg + (1.0 - alpha) * a_ssr
    b = alpha * b_reg + (1.0 - alpha) * b_ssr
    c = alpha * c_reg + (1.0 - alpha) * c_ssr
    # normalisation de sécurité (float arithmetic)
    total = a + b + c
    if total > 0:
        a, b, c = a / total, b / total, c / total
    return a, b, c


# =====================================================================
# LOOP CALCULS  [ORIGINAL - STRUCTURE INCHANGÉE]
# Les seules modifications sont balisées [BERGOMI] ou [BLEND]
# =====================================================================
all_rows = []

for asset in AssetList:
    try:
        print(f"Processing {asset} ...")

        vols  = market_cache["data"][asset]["vols"]
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

        # [BERGOMI] Flags de qualité du signal spot
        dS_pct       = abs(dS / S0) if S0 > 0 else 0.0
        d_ln_S       = np.log(S1 / S0) if S0 > 0 and S1 > 0 else np.nan
        low_spot_move = (dS_pct < MIN_DS_PCT)

        vols = vols.copy()
        vols["MATURITY"] = pd.to_datetime(vols["MATURITY"])
        vols = vols[vols["DATE"].isin([date_t0, date_t1])].copy()

        if vols.empty:
            print(f"Pas de vols pour {asset}")
            continue

        vols["T_from_t0"] = vols["MATURITY"].apply(
            lambda x: business_days_between_dates(x, t0)
        )

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
                    k_min=k_min, k_max=k_max,
                    market_strikes=market_strikes_union,
                    S0=S0, S1=S1,
                    step_coarse=step_coarse, step_fine=step_fine,
                    fine_half_width=fine_half_width
                )

                if len(K_grid) < 5:
                    continue

                # ── courbes marché [ORIGINAL - INCHANGÉ] ─────────────
                vol0 = interp_xy(x0, y0, K_grid)
                vol1 = interp_xy(x1, y1, K_grid)

                sticky_strike = vol0.copy()

                vol_t0_full = build_interpolated_slice_on_grid(slice_t0, K_grid)

                sticky_delta = sticky_delta_curve(
                    vol_t0_full=vol_t0_full, K_grid=K_grid,
                    S0=S0, S1=S1,
                    T0_years=T0_years, T1_years=T1_years,
                    r=r, q=q
                )

                sigma_t0_S0 = interp_xy(x0, y0, np.array([S0]))[0]
                sigma_t0_S1 = interp_xy(x0, y0, np.array([S1]))[0]
                sticky_skew = interp_xy(x0, y0, K_grid - dS) - sigma_t0_S0 + sigma_t0_S1

                # ── [BERGOMI] Skew ATM ψ et vol ATM ──────────────────
                psi_t0       = compute_atm_skew(x0, y0, S0)
                sigma_atm_t0 = sigma_t0_S0
                sigma_atm_t1 = interp_xy(x1, y1, np.array([S1]))[0]
                d_sigma_atm  = sigma_atm_t1 - sigma_atm_t0

                # ── [BERGOMI] SSR pointwise sur le grid ───────────────
                # Pour chaque strike K, on calcule :
                #   SSR(K) = dVol(K) / (ψ(K) · dlnS)
                # Cela généralise le SSR ATM à toute la surface.
                # On utilisera la moyenne pondérée vega comme SSR global.
                psi_grid = compute_local_skew(x0, y0, K_grid)

                # ── dataframe [ORIGINAL - INCHANGÉ] ──────────────────
                df_reg = pd.DataFrame({
                    "Strike":          K_grid,
                    "%Strike":         K_grid / S1,
                    "VolT0":           vol0,
                    "VolT1":           vol1,
                    "dVol":            vol1 - vol0,
                    "Sticky Strike Vol": sticky_strike,
                    "Sticky Delta Vol":  sticky_delta,
                    "Sticky Skew Vol":   sticky_skew,
                    "dStickyDelta":    sticky_delta - vol0,
                    "dStickySkew":     sticky_skew - vol0,
                    # [BERGOMI] skew local stocké pour SSR pointwise
                    "Psi_K":           psi_grid,
                }).dropna().copy()

                df_reg = df_reg[
                    (df_reg["%Strike"] > ATM_MIN) &
                    (df_reg["%Strike"] < ATM_MAX)
                ]
                df_reg["IsMarket_t0"] = df_reg["Strike"].round(2).isin(market_strikes_t0)
                df_reg["IsMarket_t1"] = df_reg["Strike"].round(2).isin(market_strikes_t1)

                if len(df_reg) < 5:
                    continue

                # ── vega [ORIGINAL - INCHANGÉ] ────────────────────────
                df_reg["Vega"] = bs_vega(
                    S=S0,
                    K=df_reg["Strike"].to_numpy(dtype=float),
                    T=T0_years,
                    sigma=df_reg["VolT0"].to_numpy(dtype=float),
                    r=r, q=q
                )

                y_arr  = df_reg["dVol"].to_numpy(dtype=float)
                sd_arr = df_reg["dStickyDelta"].to_numpy(dtype=float)
                sk_arr = df_reg["dStickySkew"].to_numpy(dtype=float)

                # ── pondération sqrt(vega) [ORIGINAL - INCHANGÉ] ─────
                w = np.sqrt(np.clip(df_reg["Vega"].to_numpy(dtype=float), 1e-10, None))
                w = w / np.mean(w)

                # ── [BLEND] Flag signal régression ───────────────────
                # La régression est valide seulement si dVol a une
                # dispersion cross-sectionelle suffisante.
                # Quand std(dVol)≈0, tous les strikes ont le même dVol≈0
                # → la régression est dégénérée (R²=0/0 ou numériquement
                # instable). On flag ce cas séparément du low_spot_move.
                dvol_std = np.std(y_arr)
                low_vol_signal = (dvol_std < MIN_DVOL_STD)

                # ── [BLEND] Flag global "aucun signal" ───────────────
                # Les deux méthodes échouent simultanément quand :
                #   • dVol≈0 partout (régression dégénérée)
                #   • dS/S≈0 (SSR numérique instable)
                # Dans ce cas, on utilise un prior neutre.
                no_signal = low_spot_move and low_vol_signal

                # ── RÉGRESSION CONTRAINTE [ORIGINAL - INCHANGÉ] ──────
                # dVol ~ b·dSD + c·dSK + d
                # b,c ≥ 0 ; b+c ≤ 1 ; a = 1-b-c
                def objective(params):
                    b, c, d = params
                    y_pred = b * sd_arr + c * sk_arr + d
                    return np.sum(w * (y_arr - y_pred) ** 2)

                constraints = [{"type": "ineq", "fun": lambda p: 1 - p[0] - p[1]}]
                bounds = [(0, 1), (0, 1), (None, None)]

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
                        objective, x0_init, method="SLSQP",
                        bounds=bounds, constraints=constraints,
                        options={"maxiter": 1000, "ftol": 1e-12}
                    )
                    if result.success:
                        obj = objective(result.x)
                        if obj < best_obj:
                            best_obj = obj
                            best_result = result

                if best_result is None:
                    continue

                b_reg, c_reg, d_reg = best_result.x
                a_reg = 1.0 - b_reg - c_reg

                df_reg["dVol_pred"] = b_reg * sd_arr + c_reg * sk_arr + d_reg

                # ── R² classique [ORIGINAL - INCHANGÉ] ───────────────
                ss_res = np.sum((df_reg["dVol"] - df_reg["dVol_pred"]) ** 2)
                ss_tot = np.sum((df_reg["dVol"] - df_reg["dVol"].mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                # ── R² pondéré [ORIGINAL - INCHANGÉ] ─────────────────
                y_bar_w  = np.sum(w * y_arr) / np.sum(w)
                ss_res_w = np.sum(w * (y_arr - df_reg["dVol_pred"].to_numpy()) ** 2)
                ss_tot_w = np.sum(w * (y_arr - y_bar_w) ** 2)
                wr2 = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else np.nan

                # ── [BERGOMI] SSR vega-pondéré sur le grid ───────────
                # Calcul du SSR(K) pour chaque strike, puis agrégation
                # par la vega comme poids (les strikes ATM comptent plus).
                #
                # SSR(K) = dVol(K) / (ψ(K) · dlnS)
                #
                # Si ψ(K)·dlnS ≈ 0 pour un strike (pente du smile ≈ 0
                # à ce strike), ce strike n'est pas informatif pour le SSR
                # et est exclu de la moyenne (mask).
                if not low_spot_move and not np.isnan(d_ln_S):
                    psi_K_arr = df_reg["Psi_K"].to_numpy(dtype=float)
                    denom_K   = psi_K_arr * d_ln_S

                    # masque : exclud strikes avec skew local trop faible
                    mask_valid = np.abs(denom_K) > 1e-7
                    if mask_valid.sum() >= 3:
                        ssr_K       = np.full(len(df_reg), np.nan)
                        ssr_K[mask_valid] = (
                            df_reg["dVol"].to_numpy()[mask_valid] /
                            denom_K[mask_valid]
                        )
                        # moyenne vega-pondérée (ignore NaN)
                        vega_arr = df_reg["Vega"].to_numpy(dtype=float)
                        valid    = mask_valid & np.isfinite(ssr_K)
                        if valid.sum() >= 3:
                            SSR_realized = (
                                np.sum(vega_arr[valid] * ssr_K[valid]) /
                                np.sum(vega_arr[valid])
                            )
                        else:
                            SSR_realized = np.nan
                    else:
                        SSR_realized = np.nan
                else:
                    SSR_realized = np.nan

                # SSR ATM (scalaire, pour référence)
                denom_atm = psi_t0 * d_ln_S if not low_spot_move else np.nan
                SSR_atm = (d_sigma_atm / denom_atm
                           if (denom_atm is not np.nan and abs(denom_atm) > 1e-8)
                           else np.nan)

                # ── [BLEND] FUSION RÉGRESSION + SSR ──────────────────
                # Source : intuition du trader confirmée par le fait que
                # R²_weighted mesure exactement la qualité du signal
                # cross-sectionnel de la régression.
                #
                # alpha = R²_weighted clippé dans [0,1] comme poids.
                # Si R²<0 (régression pire que la moyenne), alpha=0
                # → on utilise uniquement le SSR.
                #
                # Étapes :
                #   1. Convertir SSR_realized → (a_ssr, b_ssr, c_ssr)
                #   2. alpha = clip(R²_weighted, 0, 1)
                #   3. Blender les deux triplets

                if no_signal:
                    # Aucun signal nulle part : prior neutre
                    a_ssr, b_ssr, c_ssr = NO_SIGNAL_WEIGHTS
                    SSR_for_weights = np.nan
                    alpha = 0.0
                    blend_regime = "NoSignal"

                elif np.isnan(SSR_realized):
                    # SSR non calculable (low_spot_move) mais régression OK
                    # → on utilise uniquement la régression
                    a_ssr, b_ssr, c_ssr = a_reg, b_reg, c_reg
                    SSR_for_weights = np.nan
                    alpha = 1.0
                    blend_regime = "RegressionOnly"

                elif low_vol_signal:
                    # dVol≈0 (régression dégénérée) mais SSR calculable
                    # → on utilise uniquement le SSR
                    a_ssr, b_ssr, c_ssr = ssr_to_weights(SSR_realized)
                    SSR_for_weights = SSR_realized
                    alpha = 0.0
                    blend_regime = "SSROnly"

                else:
                    # Cas général : les deux méthodes sont valides
                    # alpha = R²_weighted, clippé dans [0,1]
                    a_ssr, b_ssr, c_ssr = ssr_to_weights(SSR_realized)
                    SSR_for_weights = SSR_realized
                    alpha = float(np.clip(wr2 if not np.isnan(wr2) else 0.0, 0.0, 1.0))
                    blend_regime = "Blended"

                a_final, b_final, c_final = blend_weights(
                    a_reg, b_reg, c_reg,
                    a_ssr, b_ssr, c_ssr,
                    alpha
                )

                # ── [BERGOMI] SSR implicite des poids finaux ──────────
                SSR_implied_final = a_final + 2.0 * c_final
                SSR_implied_reg   = a_reg   + 2.0 * c_reg
                SSR_error = (SSR_realized - SSR_implied_final
                             if not np.isnan(SSR_realized) else np.nan)

                # ── colonnes finales [ORIGINAL - INCHANGÉ] ───────────
                df_reg["UDL"]           = asset
                df_reg["Maturity"]      = maturity
                df_reg["Date t0"]       = date_t0
                df_reg["Date t1"]       = date_t1
                df_reg["Spot t0"]       = S0
                df_reg["Spot t1"]       = S1
                df_reg["T"]             = T1_days

                # Poids originaux de la régression (inchangés)
                df_reg["%StickyStrike_Reg"] = 100 * a_reg
                df_reg["%StickyDelta_Reg"]  = 100 * b_reg
                df_reg["%StickySkew_Reg"]   = 100 * c_reg
                df_reg["R2"]            = r2
                df_reg["Weighted_R2"]   = wr2

                # [BLEND] Poids finaux blendés — c'est CE QUE LE TRADER VOIT
                df_reg["%StickyStrike"] = 100 * a_final
                df_reg["%StickyDelta"]  = 100 * b_final
                df_reg["%StickySkew"]   = 100 * c_final

                # [BERGOMI] Métriques SSR
                df_reg["Psi_ATM"]           = psi_t0
                df_reg["dSigma_ATM"]        = d_sigma_atm
                df_reg["SSR_implied_reg"]   = SSR_implied_reg
                df_reg["SSR_realized"]      = SSR_realized
                df_reg["SSR_atm"]           = SSR_atm
                df_reg["SSR_implied_final"] = SSR_implied_final
                df_reg["SSR_error"]         = SSR_error

                # [BLEND] Méta-données du blend
                df_reg["Blend_Alpha"]   = alpha   # poids de la régression
                df_reg["Blend_Regime"]  = blend_regime
                df_reg["LowSpotMove"]   = low_spot_move
                df_reg["LowVolSignal"]  = low_vol_signal
                df_reg["NoSignal"]      = no_signal

                market_strikes = np.round(slice_t0["K"].unique(), 2)

                df_out = df_reg[df_reg["Strike"].round(2).isin(market_strikes_union)][[
                    # ── colonnes originales ──────────────────────────
                    "UDL", "Maturity", "Date t0", "Date t1",
                    "Spot t0", "Spot t1",
                    "VolT0", "VolT1", "dVol",
                    "Sticky Strike Vol", "Sticky Delta Vol", "Sticky Skew Vol",
                    "T",
                    # Poids régression seule (pour diagnostics)
                    "%StickyStrike_Reg", "%StickyDelta_Reg", "%StickySkew_Reg",
                    "R2", "Weighted_R2",
                    # Poids blendés → output principal trader
                    "%StickyStrike", "%StickyDelta", "%StickySkew",
                    "Vega", "Strike", "%Strike",
                    "IsMarket_t0", "IsMarket_t1",
                    # ── colonnes Bergomi ─────────────────────────────
                    "Psi_ATM", "dSigma_ATM",
                    "SSR_implied_reg", "SSR_realized", "SSR_atm",
                    "SSR_implied_final", "SSR_error",
                    # ── colonnes blend ───────────────────────────────
                    "Blend_Alpha", "Blend_Regime",
                    "LowSpotMove", "LowVolSignal", "NoSignal",
                ]].copy()

                all_rows.append(df_out)

            except Exception:
                continue

        print(f"OK {asset}")

    except Exception as e:
        print(f"Erreur sur {asset}: {e}")
        continue

# =====================================================================
# CONCAT + EXPORT  [ORIGINAL - INCHANGÉ sauf colonnes additionnelles]
# =====================================================================
if len(all_rows) == 0:
    print("Aucun résultat à exporter.")
else:
    df_final = pd.concat(all_rows, ignore_index=True)
    df_final = df_final.sort_values(["UDL", "Maturity", "Strike"]).reset_index(drop=True)

    numeric_cols = [
        "Spot t0", "Spot t1", "VolT0", "VolT1", "dVol",
        "Sticky Strike Vol", "Sticky Delta Vol", "Sticky Skew Vol",
        "%StickyStrike_Reg", "%StickyDelta_Reg", "%StickySkew_Reg",
        "%StickyStrike", "%StickyDelta", "%StickySkew",
        "R2", "Weighted_R2", "Vega", "Strike", "%Strike",
        "Psi_ATM", "dSigma_ATM",
        "SSR_implied_reg", "SSR_realized", "SSR_atm",
        "SSR_implied_final", "SSR_error", "Blend_Alpha",
    ]
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    # ── [BLEND] Résumé console ────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"[BLEND] Résumé session {date_t0} → {date_t1}")
    print("=" * 65)
    regime_counts = df_final.drop_duplicates(["UDL","Maturity"])["Blend_Regime"].value_counts()
    print("Régimes de blend par (UDL, Maturity) :")
    print(regime_counts.to_string())
    print()

    # Résumé des poids blendés vs régression pure par UDL
    summary = (
        df_final.drop_duplicates(["UDL","Maturity"])
        .groupby("UDL")[[
            "%StickyStrike", "%StickyDelta", "%StickySkew",
            "%StickyStrike_Reg", "%StickyDelta_Reg", "%StickySkew_Reg",
            "SSR_realized", "Blend_Alpha"
        ]]
        .mean()
        .round(1)
    )
    print("Poids moyens blendés vs régression (par UDL) :")
    print(summary.to_string())
    print("=" * 65 + "\n")

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
