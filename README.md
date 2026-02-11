# The Black-Scholes-Merton Framework: Stochastic Calculus & Dynamic Hedging

## 📄 Project Overview

This project presents a comprehensive reconstruction of the **Black-Scholes-Merton (BSM) option pricing model**, moving from discrete probabilistic foundations to continuous-time hedging strategies.

Based on the accompanying engineering report (**Centrale Nantes - Applied Mathematics Department**), this repository bridges the gap between fundamental probability theory and advanced financial engineering. It demonstrates that continuous-time financial models are not arbitrary constructions but rigorous limits of discrete processes.

## 📊 Jupyter Notebook & Visualizations

The core of this repository is the **Jupyter Notebook**, which contains the Python implementations of the mathematical concepts discussed in the report.

**The notebook allows you to:**
* **Trace Random Walks:** Visualize the convergence of discrete Bernoulli processes to continuous Brownian Motion as $\Delta t \to 0$.
* **Simulate Asset Paths:** Generate Monte Carlo simulations for Arithmetic and Geometric Brownian Motions (GBM).
* **Plot the "Greeks":** Generate the 3D surfaces (as seen in the report) to visualize option sensitivities (Delta, Gamma, Theta, Vega, Rho) with respect to spot price and time to maturity.
* **Verify Distributions:** Validate the log-normal distribution of asset prices at maturity.

## 🛠️ Key Mathematical Concepts

This project covers the following theoretical and practical aspects:

1.  **Probabilistic Foundations:**
    * The Bernoulli Process and Random Walks.
    * The Central Limit Theorem and Scaling Laws ($\sigma \propto \sqrt{t}$).
    * Derivation of the Wiener Process (Brownian Motion).

2.  **Stochastic Calculus:**
    * The necessity of Itô Calculus over ordinary calculus.
    * Quadratic variation and Itô's Lemma.
    * Geometric Brownian Motion (GBM) dynamics.

3.  **The Black-Scholes Framework:**
    * Dynamic replication and the self-financing portfolio ($\Pi_t$).
    * Derivation of the Partial Differential Equation (PDE).
    * Risk-neutral valuation and the No-Arbitrage principle.

4.  **Sensitivity Analysis (The Greeks):**
    * **Delta ($\Delta$):** Directional sensitivity and hedging ratios.
    * **Gamma ($\Gamma$):** Convexity and hedging stability.
    * **Theta ($\Theta$):** Time decay.
    * **Vega ($\nu$):** Sensitivity to volatility.
    * **Rho ($\rho$):** Sensitivity to interest rates.

## 🚀 Getting Started

### Prerequisites
To run the simulations and generate the plots, you need the following Python libraries:

```bash
pip install numpy matplotlib scipy pandas seaborn
