# Evaluation Engine

> Financial audit: compare predicted market cap to actual and track prediction accuracy over time.

---

## Responsibilities

Evaluate the thesis: **"Given financial statements, can we predict fair market cap?"**

This module performs a **financial audit** by:

1. **Prediction Accuracy** - Compare predicted market cap to actual market cap at snapshot time
2. **Error Distribution** - Analyze relative error distribution across universe
3. **Temporal Stability** - Track how predictions evolve as new statements arrive
4. **Portfolio Simulation** - Evaluate strategy of buying undervalued / selling overvalued

---

## Evaluation Methodology & Pipeline

To ensure rigorous evaluation, the valuation pipeline **MUST** follow these steps before metrics are calculated:

### 1. Data Preparation
- **Input**: Take all available financial data from the database.
- **Filtering**:
  - Filter for the **latest snapshot** per ticker to avoid data leakage and duplication.
  - Exclude explicit `market_cap` and `price` columns from features (X) to prevent target leakage.
- **Target**: `log_mcap` (Natural log of Market Cap).
- **Features (`X`)**:
  - Ticker ID (for tracking, not feature)
  - Numeric variables (normalized/imputed)
  - Categorical variables (one-hot encoded)

### 2. Cross-Validation Loop
- Perform **Bootstrap Cross-Validation** (or K-Fold) to generate out-of-sample predictions.
- **Process**:
  1. Split data into Train/Test sets.
  2. Train model on **Train** set.
  3. Predict on **Held-out Test** set.
  4. Repeat `n` times.

### 3. Result Aggregation
- **Output**:
  - `predicted_mcap_mean`: Average of held-out predictions (converted back from log scale).
  - `predicted_mcap_std`: Standard deviation of held-out predictions.
  - `actual_mcap`: Actual market cap at snapshot time.
- **Storage**: Save these aggregated results to the `valuation_results` table.

---

## Core Metrics

### Relative Error (Primary)

```python
relative_error = (predicted_mcap - actual_mcap) / actual_mcap
```

- Negative → Predicted < Actual → "Undervalued" by model
- Positive → Predicted > Actual → "Overvalued" by model

### Relative Standard Deviation

```python
relative_std = prediction_std / actual_mcap
```

Measures confidence in the prediction. High std → uncertain prediction.

### Price Evolution Tracking

Compare predictions at `t` to actual prices at `t+1`:

```python
# At snapshot t:
predicted_mcap_t = valuation_model(financials_t)
actual_mcap_t = market_cap at t

# At snapshot t+1:
actual_mcap_t1 = market_cap at t+1

# Did price move toward prediction?
correction = (actual_mcap_t1 - actual_mcap_t) / actual_mcap_t
expected_direction = sign(predicted_mcap_t - actual_mcap_t)
```

---

## Inputs

- **Valuation Results** per ticker/snapshot:
  - `predicted_mcap_mean`, `predicted_mcap_std`
  - `actual_mcap`, `relative_error`, `relative_std`
- **Historical Snapshots** for temporal tracking
- **Evaluation Config**:
  - Universe filters (market cap minimums, sectors)
  - Time horizons (1Q, 2Q, 4Q forward)
  - Precision/recall thresholds

---

## Outputs

- **Ticker-Level Metrics**:
  - Prediction error history
  - Confidence intervals over time
  - Direction consistency (did price move toward prediction?)
- **Universe-Level Metrics**:
  - Error distribution (mean, median, percentiles)
  - Accuracy by sector/size bucket
  - Ranked list: most undervalued → most overvalued
- **Portfolio Simulation** (optional):
  - If buying bottom decile, selling top decile:
    - What would returns have been?
    - Sharpe-like metrics on statement cadence

---

## Dependencies

### External Packages
- `numpy` - Numerical computation
- `scipy` - Statistical tests
- `pandas` - Data manipulation

### Internal Modules
- `src/db/` - Repository access
- `src/valuation/` - Valuation results

---

## Folder Structure

```
src/evaluation/
  __init__.py
  service.py              # Main evaluation orchestrator
  metrics/
    __init__.py
    accuracy.py           # Prediction accuracy metrics
    distribution.py       # Error distribution analysis
    temporal.py           # Temporal tracking
    portfolio.py          # Portfolio simulation
  reports.py              # Report generation
  config.py               # Evaluation configuration
```

---

## Accuracy Metrics

```python
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class AccuracyMetrics:
    # Error statistics
    mean_relative_error: float
    median_relative_error: float
    error_percentiles: dict  # {5: val, 25: val, 75: val, 95: val}
    
    # Confidence-weighted
    mean_confidence_score: float  # Based on relative_std
    
    # Direction accuracy
    direction_accuracy: float  # % predictions where price moved toward prediction
    
    # Coverage
    n_tickers: int
    n_snapshots: int

def compute_accuracy(results: List[ValuationResult]) -> AccuracyMetrics:
    errors = [r.relative_error for r in results]
    
    return AccuracyMetrics(
        mean_relative_error=np.mean(errors),
        median_relative_error=np.median(errors),
        error_percentiles={
            5: np.percentile(errors, 5),
            25: np.percentile(errors, 25),
            75: np.percentile(errors, 75),
            95: np.percentile(errors, 95),
        },
        mean_confidence_score=np.mean([1 / (1 + r.relative_std) for r in results]),
        direction_accuracy=compute_direction_accuracy(results),
        n_tickers=len(set(r.ticker for r in results)),
        n_snapshots=len(results),
    )
```

---

## Error Distribution Analysis

```python
def analyze_error_distribution(results: List[ValuationResult]) -> dict:
    """
    Analyze the distribution of prediction errors.
    """
    errors = [r.relative_error for r in results]
    
    return {
        "distribution": {
            "mean": np.mean(errors),
            "std": np.std(errors),
            "skew": scipy.stats.skew(errors),
            "kurtosis": scipy.stats.kurtosis(errors),
        },
        "by_decile": compute_decile_stats(results),
        "by_sector": compute_sector_stats(results),
        "outliers": identify_outliers(results, threshold=2.0),
    }
```

---

## Temporal Tracking

Track prediction accuracy over statement cycles:

```python
def temporal_analysis(ticker: str, snapshots: List[Snapshot]) -> dict:
    """
    For each snapshot, check if subsequent prices moved toward prediction.
    """
    results = []
    
    for i in range(len(snapshots) - 1):
        current = snapshots[i]
        next_snap = snapshots[i + 1]
        
        # Predicted direction
        if current.predicted_mcap_mean > current.actual_mcap:
            expected_direction = "up"  # Model thinks undervalued
        else:
            expected_direction = "down"  # Model thinks overvalued
        
        # Actual direction
        actual_change = next_snap.actual_mcap - current.actual_mcap
        actual_direction = "up" if actual_change > 0 else "down"
        
        results.append({
            "snapshot_t": current.snapshot_timestamp,
            "snapshot_t1": next_snap.snapshot_timestamp,
            "expected_direction": expected_direction,
            "actual_direction": actual_direction,
            "correct": expected_direction == actual_direction,
            "magnitude": actual_change / current.actual_mcap,
        })
    
    return {
        "ticker": ticker,
        "direction_accuracy": sum(r["correct"] for r in results) / len(results),
        "details": results,
    }
```

---

## Portfolio Simulation

Simulate strategy based on mispricing signals:

```python
def simulate_portfolio(
    universe: List[ValuationResult],
    strategy: str = "long_short",
    decile_size: int = 10,
) -> dict:
    """
    Simulate portfolio returns on statement cadence.
    
    Strategy:
    - Long bottom decile (most undervalued by model)
    - Short top decile (most overvalued by model)
    """
    sorted_universe = sorted(universe, key=lambda r: r.relative_error)
    
    n = len(sorted_universe) // decile_size
    long_positions = sorted_universe[:n]   # Most undervalued
    short_positions = sorted_universe[-n:] # Most overvalued
    
    # Compute returns at next snapshot
    long_return = np.mean([r.next_period_return for r in long_positions])
    short_return = np.mean([r.next_period_return for r in short_positions])
    
    return {
        "long_return": long_return,
        "short_return": short_return,
        "spread_return": long_return - short_return,
        "n_long": len(long_positions),
        "n_short": len(short_positions),
    }
```

---

## Design Decisions

### ✅ RESOLVED: Evaluation Approach

**Decision**: Financial audit perspective - evaluate prediction accuracy, not trading returns as primary metric.

Primary output is **relative error distribution**, with portfolio simulation as secondary validation.

### ⚠️ NEEDS REVIEW: Success Criteria

What constitutes "good" prediction accuracy?

**Options**:
- Median absolute relative error < X%
- Direction accuracy > 60%
- Error std < mean (consistent predictions)

### ⚠️ NEEDS REVIEW: Benchmark Comparison

Compare model predictions to:
- Random predictions
- Simple heuristics (e.g., sector median multiple)
- Previous snapshot's actual

### 📌 TODO: Statistical Significance

How to determine if model predictions are significantly better than baseline?

---

## Constraints

- ⚡ Primary metric is **prediction accuracy**, not returns
- ⚡ Must evaluate on **held-out data** (no look-ahead bias)
- ⚡ Must track temporal evolution across statement cycles
- ⚡ Portfolio simulation is for validation, not primary output

---

## Integration Tests

### Test Scope

Integration tests verify evaluation metrics calculation with real valuation results.

### Test Cases

```python
# tests/integration/test_evaluation.py

class TestEvaluationIntegration:
    """Integration tests for evaluation engine."""
    
    def test_accuracy_metrics(self, sample_valuation_results):
        """
        Test accuracy metrics computation.
        
        Verifies:
        - Mean/median error computed correctly
        - Percentiles are accurate
        - Confidence score is valid (0-1)
        """
        metrics = compute_accuracy(sample_valuation_results)
        
        assert -1.0 <= metrics.mean_relative_error <= 1.0
        assert 5 in metrics.error_percentiles
        assert 0 <= metrics.mean_confidence_score <= 1
    
    def test_error_distribution(self, sample_valuation_results):
        """
        Test error distribution analysis.
        
        Verifies:
        - Distribution stats computed
        - Decile stats available
        - Sector stats grouped correctly
        """
        analysis = analyze_error_distribution(sample_valuation_results)
        
        assert "mean" in analysis["distribution"]
        assert "by_decile" in analysis
        assert "by_sector" in analysis
    
    def test_temporal_analysis(self, test_db, ticker_history):
        """
        Test temporal tracking across snapshots.
        
        Verifies:
        - Direction accuracy computed
        - Correct/incorrect predictions tracked
        - All snapshots analyzed
        """
        result = temporal_analysis("AAPL", ticker_history)
        
        assert "direction_accuracy" in result
        assert len(result["details"]) == len(ticker_history) - 1
    
    def test_portfolio_simulation(self, sample_valuation_results):
        """
        Test portfolio simulation on valuation results.
        
        Verifies:
        - Long/short returns computed
        - Spread return calculated
        - Position counts correct
        """
        sim = simulate_portfolio(sample_valuation_results, decile_size=10)
        
        assert "long_return" in sim
        assert "short_return" in sim
        assert "spread_return" in sim
```

### Running Tests

```bash
# Run evaluation integration tests
pytest tests/integration/test_evaluation.py -v

# Run with real valuation data
pytest tests/integration/test_evaluation.py -v --use-db

# Run with coverage
pytest tests/integration/test_evaluation.py --cov=src/evaluation
```
