# Metric Definitions

This document provides precise definitions for all contextual, opponent-adjusted metrics implemented in this project.

## CxG (Contextual Expected Goals)

### Definition
**CxG** is the probability that a shot results in a goal, conditioned on:
- **Geometric factors**: Shot distance, angle, centrality
- **Contextual factors**: Game state (score differential, minute), possession patterns
- **Pressure factors**: Defensive pressure indicators
- **Opponent quality**: Defensive strength of the opposing team

### Mathematical Formulation
```
CxG = P(Goal | X_geometric, X_context, X_pressure, X_opponent)
```

Where:
- `X_geometric`: {distance, angle, centrality, distance_to_goal_line}
- `X_context`: {score_diff, minute_bucket, possession_length, possession_duration}
- `X_pressure`: {under_pressure, recent_def_actions, pressure_proxy_score}
- `X_opponent`: {global_rating, zone_rating, block_rate}

### Model
- Algorithm: LightGBM gradient boosting classifier
- Calibration: Isotonic regression post-hoc calibration
- Training target: Binary indicator of goal/no-goal

### Neutralization
**Neutral CxG** removes contextual and opponent effects to isolate inherent shot quality:

Reference context:
- Score differential: 0 (tied game)
- Minute: 55 (mid-second half)
- Pressure: False (no defensive pressure)
- Opponent: Average (global_rating=0, zone_rating=0)

```
Neutral CxG = P(Goal | X_geometric, X_context_ref, X_pressure_ref, X_opponent_avg)
```

### Opponent-Adjusted Metrics
- **Opponent-Adjusted Difference**: `CxG - Neutral CxG`
  - Positive: Shot was harder than neutral conditions
  - Negative: Shot was easier than neutral conditions
  
- **Opponent-Adjusted Ratio**: `CxG / Neutral CxG`
  - > 1: Shot difficulty increased by context/opponent
  - < 1: Shot difficulty decreased by context/opponent

### Use Cases
1. **Player evaluation**: Sum of CxG across all shots measures finishing under various conditions
2. **Shot quality isolation**: Neutral CxG isolates technical finishing ability
3. **Context impact**: Opponent-adjusted difference quantifies situational effects
4. **Strength of schedule**: Aggregate opponent-adjusted metrics account for opponent quality

---

## CxA (Contextual Expected Assists)

### Definition
**CxA** is the expected assist value of a pass, accounting for:
- Pass completion probability given context and opponent
- Probability of generating a shot within k actions/seconds
- Expected CxG of that shot

### Mathematical Formulation
```
CxA = P(complete | X_pass, X_opponent) × 
      E[P(shot within k | complete) × E[CxG | shot]]
```

### Components

#### 1. Pass Completion Model
```
P(complete) = f(distance, angle, height, body_part, pressure, opponent_press_rate)
```

#### 2. Shot Generation Hazard
```
P(shot within k | complete) = g(receive_zone, momentum, opponent_compact_rating)
```

#### 3. Future Shot CxG
Sample or model distribution of next-shot location conditioned on pass receive zone.

### Neutralization
Replace pressure, opponent, and momentum with reference values.

---

## CxT (Contextual Expected Threat)

### Definition
**CxT** is the expected change in goal probability resulting from an on-ball action, accounting for state transitions and opponent quality.

### Mathematical Formulation
Discretize pitch into NxM grid. For action transitioning from state s to s':

```
CxT(action) = V(s', context, opponent) - V(s, context, opponent)
```

Where V(s, context, opponent) is the value function:
```
V(s) = P(goal_for | s, context, opponent) - P(goal_against | s, context, opponent)
```

### Value Function Learning
- Method: Fitted Value Iteration or Temporal Difference (TD) learning
- Terminal hazards: P(shot | s), P(goal | shot, s, opponent)
- Opponent encoding: Team embedding or one-hot categorical

### Neutralization
Compute CxT with reference opponent and context:
```
Neutral CxT(action) = V(s', context_ref, opponent_avg) - V(s, context_ref, opponent_avg)
```

---

## C-OBV (Contextual On-Ball Value)

### Definition
**C-OBV** is the comprehensive value of an on-ball action using a Markov Decision Process (MDP) framework with Off-Policy Evaluation (OPE).

### Mathematical Formulation
Value of action a in state s:
```
C-OBV(a, s) = E[Σ_{t=0}^h γ^t R_t | a, s, context, opponent]
```

Where:
- h: Horizon (e.g., 10-15 seconds or 6 actions)
- γ: Discount factor (optional, ~0.95)
- R_t: Reward at time t

### Reward Structure
```
R_t = w1 × P(goal_for)_t 
      - w2 × P(goal_against)_t 
      + w3 × ΔField_Position_t
      + w4 × Retain_Possession_t
```

Weights (w1, w2, w3, w4) are tunable based on desired emphasis.

### Estimation Method
- **Fitted Q Evaluation (FQE)**: Learn Q(s, a) from data
- **Doubly Robust Estimator**: Combine model and importance sampling
- **Hazard Models**: P(goal_for), P(goal_against), P(turnover), P(foul)

### Opponent and Context Integration
Each hazard model conditions on opponent quality and game context:
```
P(goal_for | s, a, opponent, context)
P(turnover | s, a, opponent_press_rating, context)
```

### Neutralization
```
Neutral C-OBV = E[Σ_{t=0}^h γ^t R_t | a, s, context_ref, opponent_avg]
```

---

## Acceptance Criteria

### CxG
- **Calibration**: ECE ≤ 0.06 overall, ≤ 0.05 under pressure
- **Discrimination**: Brier score improvement ≥ 0.002 vs baseline xG
- **Neutralization**: Mean(opponent_adjusted_diff) within ±0.005
- **Slices**: Consistent performance across pressure, opponent strength, game state

### CxA, CxT, C-OBV
- **Stability**: Rank correlation > 0.7 across time periods
- **Predictive Power**: Correlation with future outcomes (goals, points)
- **Face Validity**: Top players/teams align with domain expert expectations

---

## References

1. Eggels et al. (2016). "Expected Threat (xT)." Friends of Tracking.
2. Fernández et al. (2019). "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer."
3. StatsBomb. "On-Ball Value (OBV) methodology." StatsBomb IQ.
4. Decroos et al. (2019). "Actions Speak Louder than Goals: Valuing Player Actions in Soccer."
