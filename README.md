# AI-Driven Pairs Trading with Deep Reinforcement Learning

### Overview

This project implements and evaluates a sophisticated pairs trading strategy for the PEP/KO stock pair. It begins with a classical quantitative approach based on cointegration (as detailed in the book *Successful Algorithmic Trading*) and elevates it by training a Deep Reinforcement Learning (DRL) agent to make dynamic, intelligent trading decisions.

The core achievement of this project is demonstrating that an optimized AI agent can learn a profitable policy on out-of-sample data where a traditional, rules-based strategy fails, turning a significant loss into a positive return.

### Final Performance (Out-of-Sample: 2017-2018)

The value of the adaptive AI approach is clearly demonstrated by comparing the final out-of-sample performance metrics:

| Metric             | Baseline Strategy | Optimized RL Agent | Improvement      |
| ------------------ | ----------------- | ------------------ | ---------------- |
| **Total Return** | **-63.71%** | **+1.85%** | **+65.56%** |
| **Sharpe Ratio** | -0.38             | 0.17               | **Turned Positive** |
| **Max Drawdown** | -89.88%           | -33.97%            | **Reduced by 62%** |

### Performance Chart

*To add your chart, save the final plot image as `final_performance.png` in your repository.*

<img width="993" height="578" alt="image" src="https://github.com/user-attachments/assets/eda118a0-f6c2-4665-85dc-77fd477287a9" />


### Key Concepts Implemented
- **Cointegration & Stationarity:** Statistical validation of the pair's relationship using the CADF test.
- **Mean-Reversion Strategy:** A z-score based baseline strategy for benchmarking.
- **Feature Engineering:** Adding volatility and momentum features to enhance the agent's market context.
- **Reinforcement Learning Environment:** A custom `gymnasium` environment simulating the trading process.
- **Hyperparameter Tuning:** Automated optimization of the agent's parameters using `Optuna`.
- **Agent Training:** Training a Proximal Policy Optimization (PPO) agent from `stable-baselines3`.
- **Rigorous Backtesting:** Out-of-sample (walk-forward) validation to ensure a robust evaluation.

### Project Structure

- `AI_Pairs_Trading.ipynb`: The main Jupyter Notebook containing the entire workflow.
- `rl_environment.py`: The custom Gymnasium environment for the trading agent.
- `requirements.txt`: A list of all necessary Python libraries.

### How to Run

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Set up a Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook:**
    Launch Jupyter and run the cells in `AI_Pairs_Trading.ipynb` from top to bottom.
    ```bash
    jupyter notebook
    ```
    **Note:** The hyperparameter optimization cell will take 20-30+ minutes to run.
