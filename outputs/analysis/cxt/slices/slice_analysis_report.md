# CxT Pre-Model Slice Analysis Report

Generated: 2026-03-06 12:35:33

## Summary

This analysis validates feature signals before model training by examining 
how success rates vary across different slice dimensions.

## Signal Strength Summary

| Feature | Lift Ratio | Lift Range |
|---------|------------|------------|
| action_type | infx | 117.218 |
| under_pressure | infx | 4.434 |
| macro_zone_start | 6.34x | 2.075 |
| start_zone_name | 6.34x | 2.075 |
| period | 1.49x | 0.438 |
| minute_normalized | 1.38x | 0.316 |
| opponent_global_rating | 1.14x | 0.129 |

## Detailed Slice Analysis

### opponent_global_rating

| Bin | Count | Success Rate | Lift |
|-----|-------|--------------|------|
| Weak | 110,175 | 0.8% | 0.938 |
| Medium-Weak | 115,319 | 0.9% | 1.027 |
| Medium-Strong | 102,712 | 0.9% | 1.067 |
| Strong | 107,844 | 0.8% | 0.971 |

### action_type

| Bin | Count | Success Rate | Lift |
|-----|-------|--------------|------|
| carry | 192,225 | 0.0% | 0.000 |
| dribble | 3,720 | 100.0% | 117.218 |
| pass | 240,105 | 0.0% | 0.000 |

### macro_zone_start

| Bin | Count | Success Rate | Lift |
|-----|-------|--------------|------|
| 1 | 73,250 | 0.3% | 0.389 |
| 2 | 18,611 | 0.7% | 0.857 |
| 3 | 18,533 | 0.8% | 0.936 |
| 4 | 123,274 | 0.5% | 0.610 |
| 5 | 51,144 | 0.8% | 0.947 |
| 6 | 49,243 | 0.8% | 0.900 |
| 7 | 39,727 | 2.1% | 2.464 |
| 8 | 31,663 | 1.6% | 1.855 |
| 9 | 30,605 | 1.4% | 1.628 |

### start_zone_name

| Bin | Count | Success Rate | Lift |
|-----|-------|--------------|------|
| ATT_CENTRAL | 39,727 | 2.1% | 2.464 |
| ATT_WIDE_L | 31,663 | 1.6% | 1.855 |
| ATT_WIDE_R | 30,605 | 1.4% | 1.628 |
| DEF_CENTRAL | 73,250 | 0.3% | 0.389 |
| DEF_WIDE_L | 18,611 | 0.7% | 0.857 |
| DEF_WIDE_R | 18,533 | 0.8% | 0.936 |
| MID_CENTRAL | 123,274 | 0.5% | 0.610 |
| MID_WIDE_L | 51,144 | 0.8% | 0.947 |
| MID_WIDE_R | 49,243 | 0.8% | 0.900 |

### under_pressure

| Bin | Count | Success Rate | Lift |
|-----|-------|--------------|------|
| False | 337,717 | 0.0% | 0.000 |
| True | 98,333 | 3.8% | 4.434 |

### minute_normalized

| Bin | Count | Success Rate | Lift |
|-----|-------|--------------|------|
| Early | 110,504 | 0.8% | 0.924 |
| Mid-Early | 114,801 | 0.7% | 0.836 |
| Mid-Late | 104,113 | 0.9% | 1.106 |
| Late | 106,632 | 1.0% | 1.152 |

### period

| Bin | Count | Success Rate | Lift |
|-----|-------|--------------|------|
| 1 | 218,336 | 0.8% | 0.892 |
| 2 | 204,568 | 0.9% | 1.106 |
| 3 | 6,802 | 0.8% | 0.948 |
| 4 | 6,344 | 1.1% | 1.330 |

## Visualizations

- [Signal Strength Chart](signal_strength.png)
- [Key Slice Lifts](key_slice_lifts.png)
- [Zone Lift Heatmap](zone_lift_heatmap.png)

## Interpretation

**Lift > 1.0**: Higher success rate than overall average

**Lift < 1.0**: Lower success rate than overall average

**Lift Ratio > 1.1**: Meaningful discriminative signal for modeling
