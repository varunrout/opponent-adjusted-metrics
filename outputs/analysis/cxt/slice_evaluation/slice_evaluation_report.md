# CxT Post-Model Slice Evaluation Report

Generated: 2026-03-06 12:33:38

## Overview

- **Total samples**: 436,050
- **Slices evaluated**: 26
- **OK slices**: 25
- **Warning slices**: 0

## Slice Metrics

| Slice | N | Success Rate | Completion AUC | xT Gain R² | CxT Mean | Status |
|-------|---|--------------|----------------|------------|----------|--------|
| Overall | 436,050 | 90.6% | 0.889 | 0.620 | 0.0012 | OK |
| Under Pressure | 98,333 | 90.8% | 0.905 | 0.558 | 0.0018 | OK |
| No Pressure | 337,717 | 90.6% | 0.878 | 0.636 | 0.0011 | OK |
| vs Strong Opponent | 0 | N/A | N/A | N/A | N/A | SKIP - Too few samples |
| vs Weak Opponent | 436,050 | 90.6% | 0.889 | 0.620 | 0.0012 | OK |
| Action: pass | 240,105 | 83.0% | 0.780 | 0.659 | 0.0005 | OK |
| Action: carry | 192,225 | 100.0% | N/A | 0.489 | 0.0022 | OK |
| Action: dribble | 3,720 | 100.0% | N/A | 0.000 | -0.0000 | OK |
| Start: MID | 223,661 | 92.9% | 0.878 | 0.681 | 0.0010 | OK |
| Start: DEF | 110,394 | 89.3% | 0.875 | 0.322 | 0.0007 | OK |
| Start: ATT | 101,995 | 87.2% | 0.901 | 0.594 | 0.0024 | OK |
| Zone: 4 | 123,274 | 93.3% | 0.871 | 0.683 | 0.0010 | OK |
| Zone: 1 | 73,250 | 90.3% | 0.880 | 0.319 | 0.0009 | OK |
| Zone: 6 | 49,243 | 92.1% | 0.874 | 0.671 | 0.0009 | OK |
| Zone: 3 | 18,533 | 87.0% | 0.851 | 0.317 | 0.0003 | OK |
| Zone: 9 | 30,605 | 87.3% | 0.902 | 0.636 | 0.0033 | OK |
| Zone: 7 | 39,727 | 85.9% | 0.898 | 0.523 | 0.0006 | OK |
| Zone: 8 | 31,663 | 88.6% | 0.903 | 0.665 | 0.0037 | OK |
| Zone: 5 | 51,144 | 92.8% | 0.874 | 0.672 | 0.0010 | OK |
| Zone: 2 | 18,611 | 87.2% | 0.848 | 0.291 | 0.0004 | OK |
| Late Game (75+) | 90,436 | 89.8% | 0.891 | 0.628 | 0.0015 | OK |
| Early Game (<45) | 77,751 | 90.8% | 0.887 | 0.627 | 0.0009 | OK |
| Progressive Actions | 54,699 | 64.1% | 0.797 | 0.460 | 0.0119 | OK |
| Non-Progressive | 381,351 | 94.5% | 0.843 | 0.056 | -0.0003 | OK |
| Into Final Third | 32,131 | 72.0% | 0.825 | 0.611 | 0.0053 | OK |
| Into Penalty Area | 21,586 | 54.3% | 0.760 | 0.436 | 0.0150 | OK |

## Key Findings

### Completion Model Performance

- **Pressure impact**: Under pressure AUC=0.905 vs No pressure AUC=0.878
- **Action types**:
  - Action: pass: R²=0.659, CxT mean=0.0005
  - Action: carry: R²=0.489, CxT mean=0.0022
  - Action: dribble: R²=0.000, CxT mean=-0.0000

### xT Gain Model Performance

- **By macro zone**:
  - Zone: 4: R²=0.683
  - Zone: 1: R²=0.319
  - Zone: 6: R²=0.671
  - Zone: 3: R²=0.317
  - Zone: 9: R²=0.636
  - Zone: 7: R²=0.523
  - Zone: 8: R²=0.665
  - Zone: 5: R²=0.672
  - Zone: 2: R²=0.291

## Acceptance Criteria

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Overall Completion AUC ≥ 0.55 | 0.889 | ✓ |
| Overall xT Gain R² ≥ 0 | 0.620 | ✓ |
| All slices AUC ≥ 0.50 | - | ✓ |

## Conclusion

✓ **All slices pass acceptance criteria.** Model is ready for final integration.