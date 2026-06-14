#!/usr/bin/env python3
"""Validació històrica partit a partit del Mundial 2022.

Aquest script agafa el model Dixon-Coles, l'entrena en menys de 2 segons filtrant
només els equips participants, i avalua els pronòstics respecte als resultats reals.
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
# CONFIGURACIÓ DE RUTES I VARIABLES GLOBALS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = SCRIPT_DIR / "results.csv"

MIN_TRAINING_DATE = date(2015, 1, 1)
START_WORLD_CUP_2022 = date(2022, 11, 20)
XI = 0.0018
LAMBDA_REG = 2
MAX_GOALS = 10

# Llista estricta dels 32 equips participants per accelerar l'entrenament
TEAMS_2022 = {
    "Qatar", "Ecuador", "Senegal", "Netherlands",
    "England", "Iran", "United States", "Wales",
    "Argentina", "Saudi Arabia", "Mexico", "Poland",
    "France", "Australia", "Denmark", "Tunisia",
    "Spain", "Costa Rica", "Germany", "Japan",
    "Belgium", "Canada", "Morocco", "Croatia",
    "Brazil", "Serbia", "Switzerland", "Cameroon",
    "Portugal", "Ghana", "Uruguay", "South Korea"
}

# Diccionari de desempat real per a partits eliminatoris
KNOCKOUT_REAL_WINNERS = {
    ("2022-12-03", "Netherlands", "United States"): "Netherlands", 
    ("2022-12-03", "Argentina", "Australia"): "Argentina",
    ("2022-12-04", "France", "Poland"): "France", 
    ("2022-12-04", "England", "Senegal"): "England",
    ("2022-12-05", "Japan", "Croatia"): "Croatia", 
    ("2022-12-05", "Brazil", "South Korea"): "Brazil",
    ("2022-12-06", "Morocco", "Spain"): "Morocco", 
    ("2022-12-06", "Portugal", "Switzerland"): "Portugal",
    ("2022-12-09", "Croatia", "Brazil"): "Croatia", 
    ("2022-12-09", "Netherlands", "Argentina"): "Argentina",
    ("2022-12-10", "Morocco", "Portugal"): "Morocco", 
    ("2022-12-10", "England", "France"): "France",
    ("2022-12-13", "Argentina", "Croatia"): "Argentina", 
    ("2022-12-14", "France", "Morocco"): "France",
    ("2022-12-17", "Croatia", "Morocco"): "Croatia", 
    ("2022-12-18", "Argentina", "France"): "Argentina",
}

FIFA_POINTS_2022 = {
    "Brazil": 1840.77, "Belgium": 1816.71, "Argentina": 1773.88, "France": 1759.78,
    "England": 1728.47, "Spain": 1715.22, "Netherlands": 1694.51,
    "Portugal": 1676.56, "Denmark": 1666.57, "Germany": 1650.21, "Croatia": 1645.64,
    "Mexico": 1644.89, "Uruguay": 1638.71, "Switzerland": 1635.92, "United States": 1627.48,
    "Senegal": 1584.38, "Wales": 1569.82, "Iran": 1564.61,
    "Serbia": 1563.62, "Morocco": 1563.50, "Japan": 1559.54,
    "Poland": 1548.59, "South Korea": 1530.30, "Tunisia": 1507.54, 
    "Costa Rica": 1503.59, "Australia": 1488.72, "Canada": 1475.00, 
    "Cameroon": 1471.44, "Ecuador": 1464.39, "Qatar": 1439.89,
    "Saudi Arabia": 1437.78, "Ghana": 1393.00
}

@dataclass
class Match:
    match_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    tournament: str
    weight: float = 1.0

def normalize_team_name(name: str) -> str:
    mapping = {"Curacao": "Curaçao", "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast",
               "Cabo Verde": "Cape Verde", "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey"}
    return mapping.get(name.strip(), name.strip())

def load_dataset(csv_path: Path) -> Tuple[List[Match], List[Match]]:
    train_matches = []
    world_cup_matches = []
    if not csv_path.exists(): return [], []
    
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["home_score"] == "NA" or row["away_score"] == "NA": continue
            m_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            if m_date < MIN_TRAINING_DATE: continue
            
            h_team = normalize_team_name(row["home_team"])
            a_team = normalize_team_name(row["away_team"])
            
            match_obj = Match(
                match_date=m_date, home_team=h_team, away_team=a_team,
                home_goals=int(row["home_score"]), away_goals=int(row["away_score"]), tournament=row["tournament"]
            )
            
            if h_team in TEAMS_2022 and a_team in TEAMS_2022:
                if match_obj.tournament == "FIFA World Cup" and date(2022, 11, 20) <= m_date <= date(2022, 12, 18):
                    world_cup_matches.append(match_obj)
                elif m_date < START_WORLD_CUP_2022:
                    train_matches.append(match_obj)
                
    return train_matches, world_cup_matches

def get_tournament_weight(tournament_name: str) -> float:
    if "FIFA World Cup" in tournament_name: return 3.0
    if "UEFA Euro" in tournament_name or "Copa América" in tournament_name: return 2.2
    if "AFC Asian Cup" in tournament_name: return 0.25
    if "African Cup of Nations" in tournament_name: return 0.4
    if "FIFA World Cup qualification" in tournament_name: return 0.6
    if "Gold Cup" in tournament_name or "Nations League" in tournament_name: return 0.7
    if "Friendly" in tournament_name: return 0.10
    return 0.4

def tau_correction(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if rho == 0: return 1.0
    if x == 0 and y == 0: return 1.0 - lam * mu * rho
    if x == 1 and y == 0: return 1.0 + mu * rho
    if x == 0 and y == 1: return 1.0 + lam * rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0

def fit_dixon_coles(matches: List[Match]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    teams = sorted(list(TEAMS_2022))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    
    init_params = np.zeros(2 * n_teams + 1)
    h_indices = np.array([team_to_idx[m.home_team] for m in matches], dtype=int)
    a_indices = np.array([team_to_idx[m.away_team] for m in matches], dtype=int)
    h_goals = np.array([m.home_goals for m in matches], dtype=float)
    a_goals = np.array([m.away_goals for m in matches], dtype=float)
    weights = np.array([m.weight for m in matches], dtype=float)

    fifa_prior_att = np.zeros(n_teams)
    fifa_prior_df = np.zeros(n_teams)
    for team, idx in team_to_idx.items():
        if team in FIFA_POINTS_2022:
            val = (FIFA_POINTS_2022[team] - 1500.0) / 300.0
            fifa_prior_att[idx] = val
            fifa_prior_df[idx] = -val

    def negative_log_likelihood(params):
        log_att = params[:n_teams]
        log_df = params[n_teams:2*n_teams]
        rho = params[-1]
        
        ident_penalty = 500.0 * (np.sum(log_att) ** 2)
        reg_penalty = LAMBDA_REG * (np.sum((log_att - fifa_prior_att) ** 2) + np.sum((log_df - fifa_prior_df) ** 2))
        
        lambda_vector = np.clip(np.exp(log_att[h_indices] + log_df[a_indices]), 1e-6, 15.0)
        mu_vector = np.clip(np.exp(log_att[a_indices] + log_df[h_indices]), 1e-6, 15.0)
        
        log_lik = h_goals * np.log(lambda_vector) - lambda_vector + a_goals * np.log(mu_vector) - mu_vector
        
        for idx in range(len(matches)):
            hg, ag = int(h_goals[idx]), int(a_goals[idx])
            if hg <= 1 and ag <= 1:
                tau = tau_correction(hg, ag, lambda_vector[idx], mu_vector[idx], rho)
                if tau > 0: log_lik[idx] += math.log(tau)

        return -np.sum(log_lik * weights) + ident_penalty + reg_penalty

    res = minimize(negative_log_likelihood, init_params, method='L-BFGS-B', options={'maxiter': 60})
    
    att_dict, df_dict = {}, {}
    rho_val = float(res.x[-1])
    for t, i in team_to_idx.items():
        att_dict[t] = float(np.exp(res.x[i]))
        df_dict[t] = float(np.exp(res.x[n_teams + i]))
        
    return att_dict, df_dict, rho_val

def predict_match_probs(home: str, away: str, att: Dict[str, float], df: Dict[str, float], rho: float) -> Tuple[float, float, float]:
    lam = max(att.get(home, 1.0) * df.get(away, 1.0), 0.01)
    mu = max(att.get(away, 1.0) * df.get(home, 1.0), 0.01)
    p_win, p_draw, p_loss = 0.0, 0.0, 0.0
    
    p_home = [math.exp(-lam) * (lam**i) / math.factorial(i) for i in range(MAX_GOALS + 1)]
    p_away = [math.exp(-mu) * (mu**i) / math.factorial(i) for i in range(MAX_GOALS + 1)]
    
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            prob = p_home[h] * p_away[a]
            prob *= tau_correction(h, a, lam, mu, rho)
            
            if h > a: p_win += prob
            elif h == a: p_draw += prob
            else: p_loss += prob
            
    return p_win, p_draw, p_loss

def main():
    print("=" * 95)
    print("      VALIDACIÓ METODOLÒGICA PARTIT A PARTIT - MUNDIAL 2022 (DIXON-COLES)      ")
    print("=" * 95)
    
    train_raw, wc_matches = load_dataset(RESULTS_CSV)
    if not wc_matches:
        print(f"Error crític: No s'han localitzat partits del 2022 al fitxer {RESULTS_CSV}")
        return

    train_sample = []
    for m in train_raw:
        days_old = (START_WORLD_CUP_2022 - m.match_date).days
        time_weight = math.exp(-XI * days_old)
        if time_weight < 0.01: continue
        
        m.weight = time_weight * get_tournament_weight(m.tournament)
        train_sample.append(m)

    print("Ajustant paràmetres del model (entrenament instantani de 32 equips)...")
    att, df, rho_param = fit_dixon_coles(train_sample)
    print(f"Model llest. Paràmetre d'interdependència calculat (rho): {rho_param:.4f}\n")

    # --- BLOC NOU: IMPRESSIÓ DE PARÀMETRES D'ATAC I DEFENSA ---
    print("=" * 55)
    print(f"{'SELECCIÓ':<25} | {'FORÇA ATAC (α)':<14} | {'FORÇA DEFENSA (β)':<14}")
    print("-" * 55)
    # Ordenem els equips per força d'atac de major a menor per a una millor visualització
    sorted_teams = sorted(TEAMS_2022, key=lambda x: att.get(x, 1.0), reverse=True)
    for team in sorted_teams:
        print(f"{team:<25} | {att.get(team, 1.0):14.4f} | {df.get(team, 1.0):14.4f}")
    print("=" * 55 + "\n")
    # ---------------------------------------------------------

    wc_matches.sort(key=lambda x: x.match_date)
    
    total_matches = 0
    correct_predictions = 0
    summary_by_stage: Dict[str, List[bool]] = {}

    print(f"{'DATA':<12} | {'FASE':<13} | {'PARTIT':<35} | {'PROBABILITATS (%)':<26} | {'PRED':<6} | {'REAL':<6} | {'ENCERT'}")
    print("-" * 115)

    for match in wc_matches:
        match_key = (match.match_date.isoformat(), match.home_team, match.away_team)
        is_knockout = match_key in KNOCKOUT_REAL_WINNERS
        
        pw, pd, pl = predict_match_probs(match.home_team, match.away_team, att, df, rho_param)
        
        if not is_knockout:
            stage = "Fase de Grups"
            max_prob = max(pw, pd, pl)
            if max_prob == pw: pred_outcome = "1"
            elif max_prob == pd: pred_outcome = "X"
            else: pred_outcome = "2"
            
            if match.home_goals > match.away_goals: real_outcome = "1"
            elif match.home_goals == match.away_goals: real_outcome = "X"
            else: real_outcome = "2"
        else:
            stage = "Elimnatòria"
            prob_home_advance = pw + 0.5 * pd
            pred_outcome = "1" if prob_home_advance >= 0.5 else "2"
            
            real_winner = KNOCKOUT_REAL_WINNERS[match_key]
            real_outcome = "1" if real_winner == match.home_team else "2"

        is_correct = (pred_outcome == real_outcome)
        if is_correct: correct_predictions += 1
        total_matches += 1
        summary_by_stage.setdefault(stage, []).append(is_correct)
        
        prob_str = f"1:{pw*100:4.1f}% X:{pd*100:4.1f}% 2:{pl*100:4.1f}%"
        teams_str = f"{match.home_team} - {match.away_team}"
        
        print(f"{match.match_date.isoformat()} | {stage:<13} | {teams_str:<35} | {prob_str:<26} | {pred_outcome:<6} | {real_outcome:<6} | {'✅' if is_correct else '❌'}")

    print("\n" + "=" * 95)
    print("                         RESUM FINAL DE METRIQUES D'ENCERT                         ")
    print("=" * 95)
    for stage, results in summary_by_stage.items():
        sc = sum(results)
        st = len(results)
        print(f" Accuracy en {stage:<15} : {sc}/{st} encerts ({100 * sc / st:.2f}%)")
        
    global_accuracy = 100 * correct_predictions / total_matches
    print("-" * 95)
    print(f" TOTAL DEL MUNDIAL DE QATAR : {correct_predictions}/{total_matches} encerts ({global_accuracy:.2f}%)")
    print("=" * 95)

if __name__ == "__main__":
    main()
