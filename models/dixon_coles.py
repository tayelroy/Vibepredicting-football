import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import gammaln

class DixonColesModel:
    def __init__(self):
        self.teams = []
        self.team_to_idx = {}
        self.idx_to_team = {}
        self.alpha = None
        self.beta = None
        self.gamma = 1.0
        self.rho = 0.0
        
    def _rho_correction_single(self, x, y, lambda_, mu, rho):
        if x == 0 and y == 0:
            val = 1.0 - lambda_ * mu * rho
        elif x == 0 and y == 1:
            val = 1.0 + lambda_ * rho
        elif x == 1 and y == 0:
            val = 1.0 + mu * rho
        elif x == 1 and y == 1:
            val = 1.0 - rho
        else:
            val = 1.0
        return max(val, 1e-10)

    def fit(self, matches_df, xi=0.2):
        """
        Fits the Dixon-Coles model to the international matches.
        matches_df columns: [date, home_team, away_team, home_score, away_score, neutral]
        xi: time decay factor
        """
        print("\n--- Fitting Dixon-Coles Model ---")
        df = matches_df.copy()
        
        # Calculate time decay weights
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()
        df["days_ago"] = (max_date - df["date"]).dt.days
        df["weight"] = np.exp(-xi * (df["days_ago"] / 365.25))
        
        # Build team mappings
        self.teams = sorted(list(set(df["home_team"].unique()) | set(df["away_team"].unique())))
        self.team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        self.idx_to_team = {idx: team for idx, team in enumerate(self.teams)}
        N = len(self.teams)
        print(f"Fitting model for {N} unique national teams across {len(df)} matches...")
        
        # Precompute NumPy arrays for vectorized likelihood calculation
        home_indices = np.array([self.team_to_idx[r["home_team"]] for _, r in df.iterrows()])
        away_indices = np.array([self.team_to_idx[r["away_team"]] for _, r in df.iterrows()])
        home_goals = np.array(df["home_score"].values, dtype=float)
        away_goals = np.array(df["away_score"].values, dtype=float)
        is_neutral = np.array(df["neutral"].values, dtype=bool)
        weights = np.array(df["weight"].values, dtype=float)
        
        # Precompute log factorials of goals: log(x!) = gammaln(x + 1)
        log_fact_h = gammaln(home_goals + 1.0)
        log_fact_a = gammaln(away_goals + 1.0)
        
        # Masks for Dixon-Coles low-score correction
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        
        # Optimization variables: a (attack log, N), b (defense log, N), g (home advantage log, 1), rho (1)
        init_a = np.zeros(N)
        init_b = np.zeros(N)
        init_g = 0.1
        init_rho = 0.0
        
        init_params = np.concatenate([init_a, init_b, [init_g, init_rho]])
        
        # Bounds
        bounds = []
        for _ in range(N):
            bounds.append((-5.0, 5.0)) # a bounds
        for _ in range(N):
            bounds.append((-5.0, 5.0)) # b bounds
        bounds.append((-2.0, 2.0))     # g bounds
        bounds.append((-0.3, 0.3))     # rho bounds
        
        # Constraint: mean of exp(a) must be 1.0
        def constraint_mean_attack(params):
            a = params[:N]
            return np.mean(np.exp(a)) - 1.0
            
        constraints = [{"type": "eq", "fun": constraint_mean_attack}]
        
        def obj_func(params):
            a = params[:N]
            b = params[N:2*N]
            g = params[2*N]
            rho = params[2*N+1]
            
            alpha_arr = np.exp(a)
            beta_arr = np.exp(b)
            gamma_val = np.exp(g)
            
            # lambdas & mus for all matches
            lambdas = alpha_arr[home_indices] * beta_arr[away_indices]
            lambdas = np.where(is_neutral, lambdas, lambdas * gamma_val)
            mus = alpha_arr[away_indices] * beta_arr[home_indices]
            
            # Avoid division by zero/log issues
            lambdas = np.clip(lambdas, 1e-10, None)
            mus = np.clip(mus, 1e-10, None)
            
            # Poisson probabilities in log space: x * log(lambda) - lambda - log(x!)
            log_p_h = home_goals * np.log(lambdas) - lambdas - log_fact_h
            log_p_a = away_goals * np.log(mus) - mus - log_fact_a
            
            p_h = np.exp(log_p_h)
            p_a = np.exp(log_p_a)
            
            # Vectorized tau correlation correction
            tau = np.ones_like(lambdas)
            tau[mask_00] = 1.0 - lambdas[mask_00] * mus[mask_00] * rho
            tau[mask_01] = 1.0 + lambdas[mask_01] * rho
            tau[mask_10] = 1.0 + mus[mask_10] * rho
            tau[mask_11] = 1.0 - rho
            
            tau = np.clip(tau, 1e-10, None)
            
            probs = p_h * p_a * tau
            neg_log_lik = -np.sum(weights * np.log(np.clip(probs, 1e-15, None)))
            
            return neg_log_lik
            
        print("Running optimization (SLSQP)...")
        opt_res = opt.minimize(
            obj_func,
            init_params,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "disp": False}
        )
        
        if not opt_res.success:
            print(f"⚠️ Warning: Optimization failed to converge: {opt_res.message}")
            
        # Extract parameters
        self.alpha = np.exp(opt_res.x[:N])
        self.beta = np.exp(opt_res.x[N:2*N])
        self.gamma = np.exp(opt_res.x[2*N])
        self.rho = opt_res.x[2*N+1]
        
        print("Optimization complete.")
        print(f"  Fitted Home Advantage (gamma): {self.gamma:.4f}")
        print(f"  Fitted Low-Score Correlation (rho): {self.rho:.4f}")
        
        # Display team parameters
        team_params = []
        for idx, team in self.idx_to_team.items():
            team_params.append({
                "team": team,
                "attack": self.alpha[idx],
                "defense": self.beta[idx]
            })
        df_params = pd.DataFrame(team_params).sort_values("attack", ascending=False)
        print("\nTop 5 Attacking Teams:")
        print(df_params.head(5).to_string(index=False))
        print("\nTop 5 Defensive Teams (lower is better):")
        print(df_params.sort_values("defense", ascending=True).head(5).to_string(index=False))

    def predict_probability(self, home_team, away_team, neutral=True, home_covariates=None, away_covariates=None, theta_a=0.15, theta_d=0.04):
        """
        Predicts match probabilities using Dixon-Coles formula, including optional aggregated covariates.
        home_covariates: dict with keys {"attack", "defense"}
        away_covariates: dict with keys {"attack", "defense"}
        """
        if home_team not in self.team_to_idx or away_team not in self.team_to_idx:
            raise ValueError(f"One of the teams ({home_team}, {away_team}) was not in the training set.")
            
        h_idx = self.team_to_idx[home_team]
        a_idx = self.team_to_idx[away_team]
        
        alpha_h = self.alpha[h_idx]
        beta_h = self.beta[h_idx]
        alpha_a = self.alpha[a_idx]
        beta_a = self.beta[a_idx]
        
        # Adjust using covariates if present
        # Log-linear scaling: alpha_new = alpha * exp(theta_a * covariate)
        if home_covariates:
            alpha_h *= np.exp(theta_a * home_covariates.get("attack", 0.0))
            beta_h *= np.exp(theta_d * home_covariates.get("defense", 0.0))
        if away_covariates:
            alpha_a *= np.exp(theta_a * away_covariates.get("attack", 0.0))
            beta_a *= np.exp(theta_d * away_covariates.get("defense", 0.0))
            
        # Calculate lambda and mu
        if neutral:
            lambda_ = alpha_h * beta_a
        else:
            lambda_ = alpha_h * beta_a * self.gamma
            
        mu = alpha_a * beta_h
        
        # Create probability matrix (up to 10 goals)
        max_goals = 10
        prob_matrix = np.zeros((max_goals, max_goals))
        
        from scipy.stats import poisson as poisson_dist
        for x in range(max_goals):
            for y in range(max_goals):
                p_h = poisson_dist.pmf(x, lambda_)
                p_a = poisson_dist.pmf(y, mu)
                tau = self._rho_correction_single(x, y, lambda_, mu, self.rho)
                prob_matrix[x, y] = p_h * p_a * tau
                
        # Calculate Win/Draw/Loss probabilities
        prob_home = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away = np.sum(np.triu(prob_matrix, 1))
        
        # Normalize to ensure sum is exactly 1.0
        total_prob = prob_home + prob_draw + prob_away
        prob_home /= total_prob
        prob_draw /= total_prob
        prob_away /= total_prob
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda": lambda_,
            "mu": mu,
            "home_win": prob_home,
            "draw": prob_draw,
            "away_win": prob_away,
            "matrix": prob_matrix
        }
