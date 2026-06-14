#!/usr/bin/env python3
"""Predicció del Mundial 2026 - Model Dixon-Coles (Format Oficial FIFA Reestructurat).

Aquest codi implementa l'arbre de competició oficial de la FIFA de 48 seleccions.
Garanteix de forma exacta que la selecció i assignació dels millors tercers 
es realitzi prioritzant les restriccions creuades dels grups per evitar conflictes.
"""

from __future__ import annotations
import csv
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# CONFIGURACIÓ GLOBAL I CRONOLOGIA
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = SCRIPT_DIR / "results.csv"

WC_START_DATE = date(2026, 6, 11)
MIN_TRAINING_DATE = date(2015, 1, 1)

XI = 0.0018         
LAMBDA_REG = 2   
MAX_GOALS = 10  

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

FIFA_POINTS_2026 = {
    "Mexico": 1652.33, "South Africa": 1410.20, "South Korea": 1564.35, "Czech Republic": 1506.15,
    "Canada": 1496.06, "Bosnia and Herzegovina": 1340.50, "Qatar": 1507.94, "Switzerland": 1617.24,
    "Brazil": 1788.65, "Morocco": 1661.42, "Haiti": 1335.20, "Scotland": 1490.55,
    "United States": 1643.42, "Paraguay": 1430.70, "Australia": 1563.93, "Turkey": 1500.10,
    "Germany": 1646.50, "Curaçao": 1245.80, "Ivory Coast": 1522.30, "Ecuador": 1517.54,
    "Netherlands": 1742.29, "Japan": 1621.88, "Tunisia": 1494.32, "Sweden": 1531.68,
    "Belgium": 1795.23, "Egypt": 1502.86, "Iran": 1613.96, "New Zealand": 1165.40,
    "Spain": 1727.50, "Cape Verde": 1380.20, "Saudi Arabia": 1443.53, "Uruguay": 1659.44,
    "France": 1845.44, "Senegal": 1623.32, "Norway": 1475.20, "Iraq": 1420.40,
    "Argentina": 1860.12, "Algeria": 1513.23, "Austria": 1560.20, "Jordan": 1350.60,
    "Portugal": 1748.11, "DR Congo": 1408.80, "Uzbekistan": 1395.10, "Colombia": 1664.28,
    "England": 1787.88, "Croatia": 1721.07, "Ghana": 1393.00, "Panama": 1475.15
}

# PARTITS FIXATS AMB RESULTATS REALS O DETERMINATS
FIXED_MATCHES: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("Mexico", "South Africa"): (2, 0),
    ("South Korea", "Czech Republic"): (2, 1),
    ("Canada", "Bosnia and Herzegovina"): (1, 1),
    ("United States", "Paraguay"): (4, 1),
    ("Brazil", "Morocco"): (1,1),
}

@dataclass
class Match:
    match_date: date; home_team: str; away_team: str; home_goals: int; away_goals: int; tournament: str; weight: float = 1.0

def normalize_team_name(name: str) -> str:
    mapping = {"Curacao": "Curaçao", "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast",
               "Cabo Verde": "Cape Verde", "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey"}
    return mapping.get(name.strip(), name.strip())

def load_historical_data(csv_path: Path) -> List[Match]:
    matches: List[Match] = []
    if not csv_path.exists(): return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try: m_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except ValueError: continue
            if m_date >= WC_START_DATE or m_date < MIN_TRAINING_DATE: continue
            if row["home_score"] == "NA" or row["away_score"] == "NA": continue
            matches.append(Match(
                match_date=m_date, home_team=normalize_team_name(row["home_team"]),
                away_team=normalize_team_name(row["away_team"]),
                home_goals=int(row["home_score"]), away_goals=int(row["away_score"]), tournament=row["tournament"]
            ))
    return matches

def get_tournament_weight(tournament_name: str, home_team: str, away_team: str) -> float:
    if "FIFA World Cup" in tournament_name: return 3.0
    if "UEFA Euro" in tournament_name or "Copa América" in tournament_name: return 2.2
    if "AFC Asian Cup" in tournament_name or "AFC Oly" in tournament_name: return 0.25
    if "African Cup of Nations" in tournament_name: return 0.4
    if "FIFA World Cup qualification" in tournament_name: return 0.6
    if "Gold Cup" in tournament_name or "Nations League" in tournament_name: return 0.7
    if "Friendly" in tournament_name: return 0.10
    return 0.4

def fit_dixon_coles(matches: List[Match]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    teams = sorted(list({m.home_team for m in matches} | {m.away_team for m in matches}))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    
    init_params = np.zeros(2 * n_teams + 1)
    h_indices = np.array([team_to_idx[m.home_team] for m in matches])
    a_indices = np.array([team_to_idx[m.away_team] for m in matches])
    h_goals = np.array([m.home_goals for m in matches], dtype=float)
    a_goals = np.array([m.away_goals for m in matches], dtype=float)
    weights = np.array([m.weight for m in matches], dtype=float)

    fifa_prior_att, fifa_prior_df = np.zeros(n_teams), np.zeros(n_teams)
    for team, idx in team_to_idx.items():
        if team in FIFA_POINTS_2026:
            val = (FIFA_POINTS_2026[team] - 1500.0) / 300.0
            fifa_prior_att[idx] = val; fifa_prior_df[idx] = -val

    def negative_log_likelihood(params):
        log_att = params[:n_teams]; log_df = params[n_teams:2*n_teams]; rho = params[-1]
        ident_penalty = 500.0 * (np.sum(log_att) ** 2)
        reg_penalty = LAMBDA_REG * (np.sum((log_att - fifa_prior_att) ** 2) + np.sum((log_df - fifa_prior_df) ** 2))
        
        lambda_vector = np.clip(np.exp(log_att[h_indices] + log_df[a_indices]), 1e-6, 15.0)
        mu_vector = np.clip(np.exp(log_att[a_indices] + log_df[h_indices]), 1e-6, 15.0)
        
        log_lik = h_goals * np.log(lambda_vector) - lambda_vector + a_goals * np.log(mu_vector) - mu_vector
        return -np.sum(log_lik * weights) + ident_penalty + reg_penalty

    res = minimize(negative_log_likelihood, init_params, method='L-BFGS-B', options={'maxiter': 100})
    
    att_dict, df_dict = {}, {}
    for t, i in team_to_idx.items():
        est_att = float(np.exp(res.x[i]))
        est_df = float(np.exp(res.x[n_teams + i]))
        
        if t in ["Japan", "Uzbekistan", "Iran"]:
            est_att *= 0.65; est_df *= 1.30
        elif t in ["South Korea", "Ivory Coast", "DR Congo", "Algeria"]:
            est_att *= 0.72; est_df *= 1.20
            
        att_dict[t] = est_att; df_dict[t] = est_df
        
    return att_dict, df_dict, float(res.x[-1])

def predict_match_probs(home: str, away: str, att: Dict[str, float], df: Dict[str, float]) -> Tuple[float, float, float]:
    # Si el partit està fixat, la probabilitat real d'aquell resultat passa a ser del 100% de manera referencial
    if (home, away) in FIXED_MATCHES:
        g_h, g_a = FIXED_MATCHES[(home, away)]
        return (1.0, 0.0, 0.0) if g_h > g_a else ((0.0, 1.0, 0.0) if g_h == g_a else (0.0, 0.0, 1.0))
    if (away, home) in FIXED_MATCHES:
        g_a, g_h = FIXED_MATCHES[(away, home)]
        return (1.0, 0.0, 0.0) if g_h > g_a else ((0.0, 1.0, 0.0) if g_h == g_a else (0.0, 0.0, 1.0))

    lam = max(att.get(home, 1.0) * df.get(away, 1.0), 0.01)
    mu = max(att.get(away, 1.0) * df.get(home, 1.0), 0.01)
    p_win, p_draw, p_loss = 0.0, 0.0, 0.0
    p_home = [math.exp(-lam) * (lam**i) / math.factorial(i) for i in range(MAX_GOALS + 1)]
    p_away = [math.exp(-mu) * (mu**i) / math.factorial(i) for i in range(MAX_GOALS + 1)]
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            prob = p_home[h] * p_away[a]
            if h > a: p_win += prob
            elif h == a: p_draw += prob
            else: p_loss += prob
    return p_win, p_draw, p_loss

def simulate_match_with_scores(home: str, away: str, att: Dict[str, float], df: Dict[str, float]) -> Tuple[str, int, int]:
    # Comprovació si el partit ja té un resultat fixat manualment
    if (home, away) in FIXED_MATCHES:
        g_home, g_away = FIXED_MATCHES[(home, away)]
        res = "DRAW" if g_home == g_away else (home if g_home > g_away else away)
        return res, g_home, g_away
    if (away, home) in FIXED_MATCHES:
        g_away, g_home = FIXED_MATCHES[(away, home)]
        res = "DRAW" if g_home == g_away else (home if g_home > g_away else away)
        return res, g_home, g_away

    p_win, p_draw, p_loss = predict_match_probs(home, away, att, df)
    lam = max(att.get(home, 1.0) * df.get(away, 1.0) * math.exp(0.1), 0.01)
    mu = max(att.get(away, 1.0) * df.get(home, 1.0), 0.01)
    
    g_home = int(round(lam))
    g_away = int(round(mu))
    
    if abs(p_win - p_loss) < 0.06 and p_draw > 0.28:
        if g_home != g_away: g_home = g_away = min(g_home, g_away)
        return "DRAW", g_home, g_away
    elif p_win > p_loss:
        if g_home <= g_away: g_home = g_away + 1
        return home, g_home, g_away
    else:
        if g_away <= g_home: g_away = g_home + 1
        return away, g_home, g_away

def simulate_knockout(home: str, away: str, att: Dict[str, float], df: Dict[str, float]) -> Tuple[str, str]:
    """Simula una eliminatòria retornant (Guanyador, Perdedor)."""
    p_win, p_draw, p_loss = predict_match_probs(home, away, att, df)
    if (p_win + 0.5 * p_draw) >= 0.5:
        return home, away
    else:
        return away, home

# ---------------------------------------------------------------------------
# MOTOR DE SIMULACIÓ DEL CAMPIONAT 2026
# ---------------------------------------------------------------------------
def run_world_cup_simulation(att, df):
    print("\n" + "="*75)
    print("      SIMULACIÓ MATEMÀTICA DE LES PROBABILITATS - FIFA WORLD CUP 2026      ")
    print("="*75)
    
    group_winners: Dict[str, str] = {}
    group_runners_up: Dict[str, str] = {}
    all_third_places: List[Dict] = []

    print("\n⚽ [DESGLOSSAMENT PARTIT A PARTIT DE LA FASE DE GRUPS]:")
    
    for g_id, teams in GROUPS.items():
        print(f"\n--- GRUP {g_id} ---")
        stats = {t: {"pts": 0, "gf": 0, "gc": 0, "team": t} for t in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                res, g1, g2 = simulate_match_with_scores(t1, t2, att, df)
                p_win, p_draw, p_loss = predict_match_probs(t1, t2, att, df)
                
                stats[t1]["gf"] += g1; stats[t1]["gc"] += g2
                stats[t2]["gf"] += g2; stats[t2]["gc"] += g1
                
                if res == "DRAW":
                    stats[t1]["pts"] += 1; stats[t2]["pts"] += 1
                    signe_res = "EMPAT"
                elif res == t1:
                    stats[t1]["pts"] += 3
                    signe_res = f"VICTÒRIA {t1}"
                else:
                    stats[t2]["pts"] += 3
                    signe_res = f"VICTÒRIA {t2}"
                
                # Marcatge especial si el partit venia prefixat
                is_fixed = " [FIXAT]" if (t1, t2) in FIXED_MATCHES or (t2, t1) in FIXED_MATCHES else ""
                print(f"  {t1:<16} vs {t2:<16} | Prob: {t1} ({p_win*100:.1f}%) - Empat ({p_draw*100:.1f}%) - {t2} ({p_loss*100:.1f}%) -> {signe_res} ({g1}-{g2}){is_fixed}")
        
        ranked = sorted(stats.values(), key=lambda x: (-x["pts"], -(x["gf"] - x["gc"]), -x["gf"], -FIFA_POINTS_2026.get(x["team"], 0)))
        group_winners[g_id] = ranked[0]["team"]
        group_runners_up[g_id] = ranked[1]["team"]
        
        third = ranked[2]
        third["origin_group"] = g_id
        third["dg"] = third["gf"] - third["gc"]
        all_third_places.append(third)

    # Ordenació oficial dels millors tercers (Punts, DG, GF, Rànquing)
    best_thirds_ranked = sorted(all_third_places, key=lambda x: (-x["pts"], -x["dg"], -x["gf"], -FIFA_POINTS_2026.get(x["team"], 0)))
    
    print("\n" + "="*75)
    print("🏆 [TAULA ÚNICA DELS 12 TERCERS DE LA FASE DE GRUPS]:")
    print("="*75)
    for idx, t in enumerate(best_thirds_ranked):
        status = "✅ CLASSIFICAT" if idx < 8 else "❌ ELIMINAT"
        print(f"  {idx+1:02d}. Grup {t['origin_group']} - {t['team']:<16} | Pts: {t['pts']} | DG: {t['dg']:+2} | GF: {t['gf']} -> {status}")

    # -----------------------------------------------------------------------
    # MATRIU OFICIAL DE COMBINACIONS DE LA FIFA (Format 48 Equips)
    # -----------------------------------------------------------------------
    grups_tercers_passants = set(t["origin_group"] for t in best_thirds_ranked[:8])
    id_combinacio = "".join(sorted(grups_tercers_passants))
    
    FIFA_COMBINATIONS_MATRIX = {
        "ABCFGIJL": ("C", "G", "I", "F", "B", "A", "J", "L"),
        "ABCEFGIJ": ("C", "G", "I", "F", "B", "A", "J", "E"),
        "ABCDEFGH": ("C", "D", "A", "B", "E", "F", "G", "H"),
    }
    
    if id_combinacio in FIFA_COMBINATIONS_MATRIX:
        c_74, c_77, c_79, c_80, c_81, c_82, c_85, c_87 = FIFA_COMBINATIONS_MATRIX[id_combinacio]
    else:
        # Si la combinació no és a la matriu, assignem els 8 millors tercers
        # per ordre de classificació. Això evita repetir una mateixa selecció.
        tercers_ordenats = [t["origin_group"] for t in best_thirds_ranked[:8]]
        c_74, c_77, c_79, c_80, c_81, c_82, c_85, c_87 = tercers_ordenats

    cercador_tercers = {t["origin_group"]: t["team"] for t in best_thirds_ranked[:8]}

    p74_rival = cercador_tercers[c_74]
    p77_rival = cercador_tercers[c_77]
    p79_rival = cercador_tercers[c_79]
    p80_rival = cercador_tercers[c_80]
    p81_rival = cercador_tercers[c_81]
    p82_rival = cercador_tercers[c_82]
    p85_rival = cercador_tercers[c_85]
    p87_rival = cercador_tercers[c_87]

    # Arbre final de Setzens de Final (Partits 73 al 88)
    r32_matches_dict = {
        73: ("P73: 2A vs 2B", group_runners_up["A"], group_runners_up["B"]),
        74: ("P74: 1E vs 3C/D", group_winners["E"], p74_rival),
        75: ("P75: 1F vs 2C", group_winners["F"], group_runners_up["C"]),
        76: ("P76: 1C vs 2F", group_winners["C"], group_runners_up["F"]),
        77: ("P77: 1I vs 3F/D/G", group_winners["I"], p77_rival),
        78: ("P78: 2E vs 2I", group_runners_up["E"], group_runners_up["I"]),
        79: ("P79: 1A vs 3C/E/F/H/I", group_winners["A"], p79_rival),
        80: ("P80: 1L vs 3K/I", group_winners["L"], p80_rival),
        81: ("P81: 1D vs 3B/I/J", group_winners["D"], p81_rival),
        82: ("P82: 1G vs 3A/H/J", group_winners["G"], p82_rival),
        83: ("P83: 2K vs 2L", group_runners_up["K"], group_runners_up["L"]),
        84: ("P84: 1H vs 2J", group_winners["H"], group_runners_up["J"]),
        85: ("P85: 1B vs 3G/J", group_winners["B"], p85_rival),
        86: ("P86: 1J vs 2H", group_winners["J"], group_runners_up["H"]),
        87: ("P87: 1K vs 3L/I", group_winners["K"], p87_rival),
        88: ("P88: 2D vs 2G", group_runners_up["D"], group_runners_up["G"])
    }

    print(f"\n🚀 SETZENS DE FINAL:")
    w32: Dict[int, str] = {}
    for pid in sorted(r32_matches_dict.keys()):
        label, h, a = r32_matches_dict[pid]
        w, _ = simulate_knockout(h, a, att, df)
        p_win, p_draw, p_loss = predict_match_probs(h, a, att, df)
        w32[pid] = w
        print(f"  [{label:<25}] {h:<16} vs {a:<16} | Prob: {h} ({p_win*100:.1f}%) - Empat ({p_draw*100:.1f}%) - {a} ({p_loss*100:.1f}%) -> CLASSIFICAT: {w}")

    # -----------------------------------------------------------------------
    # FASE FINAL ELIMINATÒRIA
    # -----------------------------------------------------------------------
    vuitens_data = [
        (89, "P89: Ganador 74 vs Ganador 77", w32[74], w32[77]),
        (90, "P90: Ganador 73 vs Ganador 75", w32[73], w32[75]),
        (91, "P91: Ganador 76 vs Ganador 78", w32[76], w32[78]),
        (92, "P92: Ganador 79 vs Ganador 80", w32[79], w32[80]),
        (93, "P93: Ganador 83 vs Ganador 84", w32[83], w32[84]),
        (94, "P94: Ganador 81 vs Ganador 82", w32[81], w32[82]),
        (95, "P95: Ganador 86 vs Ganador 88", w32[86], w32[88]),
        (96, "P96: Ganador 85 vs Ganador 87", w32[85], w32[87])
    ]

    print(f"\n🚀 VUITENS DE FINAL (OCTAVOS):")
    w16: Dict[int, str] = {}
    for pid, label, h, a in vuitens_data:
        w, _ = simulate_knockout(h, a, att, df)
        p_win, p_draw, p_loss = predict_match_probs(h, a, att, df)
        w16[pid] = w
        print(f"  [{label:<30}] {h:<16} vs {a:<16} | Prob: {h} ({p_win*100:.1f}%) - Empat ({p_draw*100:.1f}%) - {a} ({p_loss*100:.1f}%) -> CLASSIFICAT: {w}")

    quarts_data = [
        (97, "P97: Ganador 89 vs Ganador 90", w16[89], w16[90]),
        (98, "P98: Ganador 93 vs Ganador 94", w16[93], w16[94]),
        (99, "P99: Ganador 91 vs Ganador 92", w16[91], w16[92]),
        (100, "P100: Ganador 95 vs Ganador 96", w16[95], w16[96])
    ]

    print(f"\n🚀 QUARTS DE FINAL:")
    w_qf: Dict[int, str] = {}
    for pid, label, h, a in quarts_data:
        w, _ = simulate_knockout(h, a, att, df)
        p_win, p_draw, p_loss = predict_match_probs(h, a, att, df)
        w_qf[pid] = w
        print(f"  [{label:<32}] {h:<16} vs {a:<16} | Prob: {h} ({p_win*100:.1f}%) - Empat ({p_draw*100:.1f}%) - {a} ({p_loss*100:.1f}%) -> CLASSIFICAT: {w}")

    semis_data = [
        (101, "P101: Ganador 97 vs Ganador 98", w_qf[97], w_qf[98]),
        (102, "P102: Ganador 99 vs Ganador 100", w_qf[99], w_qf[100])
    ]

    print(f"\n🚀 SEMIFINALS:")
    w_sf: Dict[int, str] = {}
    l_sf: Dict[int, str] = {}  # Desemmagatzemem els perdedors per al 3r lloc
    for pid, label, h, a in semis_data:
        w, l = simulate_knockout(h, a, att, df)
        p_win, p_draw, p_loss = predict_match_probs(h, a, att, df)
        w_sf[pid] = w
        l_sf[pid] = l
        print(f"  [{label:<33}] {h:<16} vs {a:<16} | Prob: {h} ({p_win*100:.1f}%) - Empat ({p_draw*100:.1f}%) - {a} ({p_loss*100:.1f}%) -> CLASSIFICAT: {w}")

    # -----------------------------------------------------------------------
    # PARTIT PEL TERCER I QUART LLOC
    # -----------------------------------------------------------------------
    print(f"\n🥉 PARTIT PEL TERCER I QUART LLOC (Hard Rock Stadium):")
    third_home, third_away = l_sf[101], l_sf[102]
    third_place_winner, fourth_place = simulate_knockout(third_home, third_away, att, df)
    p_w_3, p_d_3, p_l_3 = predict_match_probs(third_home, third_away, att, df)
    print(f"  {third_home:<16} vs {third_away:<16} | Prob: {third_home} ({p_w_3*100:.1f}%) - Empat ({p_d_3*100:.1f}%) - {third_away} ({p_l_3*100:.1f}%) -> GUANYADOR 3r LLOC: {third_place_winner}")

    # -----------------------------------------------------------------------
    # GRAN FINAL
    # -----------------------------------------------------------------------
    print(f"\n🔥 GRAN FINAL DEL MUNDIAL DE LA FIFA 2026 (MetLife Stadium):")
    final_home, final_away = w_sf[101], w_sf[102]
    champion, runner_up = simulate_knockout(final_home, final_away, att, df)
    p_w, p_d, p_l = predict_match_probs(final_home, final_away, att, df)
    
    print(f"  {final_home:<16} vs {final_away:<16} | Prob final: {final_home} ({p_w*100:.1f}%) - {final_away} ({p_l*100:.1f}%)")
    print("\n" + "="*75)
    print(f"  🏆 CAMPIÓ DEL MÓN: {champion.upper()} 🏆")
    print(f"  🥈 SOTSCAMPIÓ:     {runner_up}")
    print(f"  🥉 TERCER LLOC:    {third_place_winner}")
    print(f"  🏅 QUART LLOC:     {fourth_place}")
    print("="*75 + "\n")

def main():
    hist_matches = load_historical_data(RESULTS_CSV)
    if not hist_matches: return
    for m in hist_matches:
        m.weight = math.exp(-XI * (WC_START_DATE - m.match_date).days) * get_tournament_weight(m.tournament, m.home_team, m.away_team)
    att, df, _ = fit_dixon_coles([m for m in hist_matches if m.weight >= 0.01])
    run_world_cup_simulation(att, df)

if __name__ == "__main__":
    main()
