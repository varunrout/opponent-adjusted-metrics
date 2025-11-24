# Methodology: Prediction Module and Inference Pipeline

**Version:** 1.0.0  
**Module:** Production & Inference  
**Technical Stack:** Python CLI, Joblib, Matplotlib

---

## 1. Abstract

The Prediction Module is the operational interface of the project. It consumes the artifacts produced by the Modelling module (serialized models, coefficient maps) and applies them to new, unseen match data. This document outlines the software architecture for inference, the handling of "Cold Start" problems for new teams, and the generation of visual and tabular reports.

## 2. Inference Architecture

The inference process is designed to be stateless and idempotent. It is triggered via the Command Line Interface (CLI) script `scripts/run_cxg_analysis.py`.

### 2.1 The Pipeline Steps
1.  **Context Loading:** The system loads the trained Meta-Learner and Submodels from the `models/` directory using `joblib`. It also loads the `coefficients.json` map, which contains the $A_k$ (Attack) and $D_j$ (Defense) values for known teams.
2.  **Data Ingestion (Micro-Batch):** The script accepts a `match_id`. It fetches the event data for that specific match from the raw data store (or API).
3.  **Feature Transformation:** The raw events are passed through the *exact same* `build_shot_features` function used in training. This ensures feature parity (e.g., "Distance" is calculated identically).
4.  **Neutral Prediction:** The features are fed into the Stacked Ensemble to generate $\text{xG}_{neutral}$.
5.  **Adjustment Application:** The system looks up the Home and Away team IDs in the coefficient map and applies the adjustment formula:
    $$ \text{xG}_{final} = \text{xG}_{neutral} + A_{team} + D_{opponent} $$

### 2.2 Handling Unknown Entities (Cold Start)
A critical challenge in production is encountering a team that was not in the training set (e.g., a newly promoted team).
*   **Strategy:** Zero-Imputation.
*   **Logic:** If a `team_id` is missing from the coefficient map, we assume their coefficient is $0.0$. This effectively defaults the prediction to the "Neutral" model.
*   **Logging:** The system logs a warning ("Team ID 1234 not found in coefficients, using neutral prior") to alert analysts that the model needs retraining.

## 3. Command Line Interface (CLI)

The prediction module is exposed via a robust CLI for integration into automated workflows (e.g., Airflow, Jenkins).

### 3.1 Usage
```bash
python scripts/run_cxg_analysis.py \
  --match_id 3788741 \
  --model_version "v2_neutral_priors" \
  --output_format "json" \
  --save_plots
```

### 3.2 Arguments
*   `--match_id`: The StatsBomb ID of the match to analyze.
*   `--model_version`: Selects which set of serialized models to use (enables A/B testing of models).
*   `--output_dir`: Destination for generated reports.
*   `--save_plots`: Boolean flag to trigger the generation of Matplotlib visualizations.

## 4. Output Artifacts

The module produces two types of outputs: structured data for downstream systems and visual reports for human consumption.

### 4.1 Structured Data (JSON/CSV)
For every shot in the match, we export a record containing:
```json
{
  "event_id": "8f3a...",
  "timestamp": "00:12:34",
  "player_name": "Harry Kane",
  "xg_neutral": 0.12,
  "xg_adjusted": 0.15,
  "adjustment_delta": +0.03,
  "factors": {
    "distance": 14.5,
    "angle": 22.1,
    "pressure": "High"
  }
}
```
This allows analysts to audit *why* a prediction was made.

### 4.2 Visualizations
The module uses `matplotlib` to generate a "Shot Map" for the match.
*   **Glyphs:** Shots are plotted as circles.
*   **Size:** Proportional to $\text{xG}_{final}$.
*   **Color:** Green for Goal, Red for Miss, Blue for Blocked.
*   **Annotations:** High-xG chances (>0.3) are annotated with the player name and minute.
*   **Pitch Control:** (Optional) If tracking data is available, we overlay the pitch control surface to show space dominance.

## 5. Future Improvements: Real-Time Inference

Currently, the system operates in batch mode (post-match). To move to real-time (live betting or broadcast):
1.  **Stream Processing:** Replace the file-based ingestion with a Kafka consumer reading live event feeds.
2.  **State Management:** The "Game State" features (Score Differential) must be maintained in a state store (Redis) rather than calculated from a full match history.
3.  **Latency:** The feature engineering pipeline must be optimized to run in <50ms per event.

## 6. Conclusion

The Prediction Module is the bridge between theory and practice. By wrapping the complex statistical models in a robust, fail-safe CLI, we ensure that the insights derived from the "Opponent-Adjusted" methodology are accessible, reproducible, and actionable for end-users.
