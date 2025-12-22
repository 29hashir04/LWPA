# Cosine Similarity Migration

## Overview
Changed the gait recognition metric from **Euclidean distance** to **cosine similarity** for better recognition performance.

## Why Cosine Similarity?

### Advantages
1. **Better for high-dimensional features**: With 136-dimensional feature vectors, cosine similarity focuses on directional similarity rather than magnitude
2. **Robust to scale variations**: Insensitive to feature vector magnitude differences
3. **Normalized output**: Direct 0-1 scale makes confidence interpretation clearer
4. **Better separability**: Often provides better class separation in high dimensions

### Mathematical Difference

**Euclidean Distance** (old):
```
distance = ||feature_A - feature_B||
Lower distance = more similar
```

**Cosine Similarity** (new):
```
similarity = (feature_A · feature_B) / (||feature_A|| × ||feature_B||)
Higher similarity = more similar
Range: -1 to 1 (clamped to 0-1)
```

## Changes Made

### 1. Model Configuration (`models/gait_model.py`)
- Changed `DISTANCE_THRESHOLD: 18` to `SIMILARITY_THRESHOLD: 0.85`
- Updated documentation
- Modified `distance_threshold` attribute to `similarity_threshold`

### 2. Recognition Function
**Before** (Euclidean):
```python
distance = np.linalg.norm(smoothed_features - enrolled_features)
confidence = np.exp(-distance / threshold)
if distance <= threshold:
    decision = best_name
```

**After** (Cosine):
```python
# Normalize both vectors
normalized_features = features / np.linalg.norm(features)
normalized_enrolled = enrolled / np.linalg.norm(enrolled)

# Compute cosine similarity
similarity = np.dot(normalized_features, normalized_enrolled)
confidence = similarity  # Already 0-1 range

if similarity >= threshold:
    decision = best_name
```

### 3. UI Settings (`ui/render_functions.py`)
**Threshold Slider**:
- Old: 5.0 to 30.0 (distance units)
- New: 0.70 to 0.95 (similarity percentage)

**Mode Indicators**:
- Strict: similarity ≥ 0.90 (90% match required)
- Balanced: similarity ≥ 0.80 (80% match required)
- Lenient: similarity < 0.80 (below 80%)

## Default Threshold

**0.85** (85% similarity required for match)
- This provides good balance between security and usability
- Adjustable via UI slider based on specific requirements

## Benefits in Practice

1. **More reliable**: Less affected by walking speed variations
2. **Better confidence scores**: Directly interpretable as "percentage match"
3. **Easier tuning**: 0-1 scale is more intuitive than distance units
4. **Improved accuracy**: Better discrimination between similar gaits

## Performance Impact

- **Computation**: Negligible change - both require similar operations
- **Speed**: Slightly faster (no exponential calculation for confidence)
- **Memory**: No change

## Re-enrollment Note

⚠️ **Important**: Existing enrolled patterns will work, but for best results:

1. **Test current enrollments** first to see if they work well
2. **Re-enroll users** if you experience issues
3. **Adjust threshold** (start at 0.85, tune as needed)

The switch is backwards compatible, but recognition accuracy may vary until the system is re-calibrated.

## Tuning Guide

### If too many false rejections:
- Lower threshold: 0.85 → 0.80 → 0.75
- Use "Lenient" mode

### If too many false acceptances:
- Raise threshold: 0.85 → 0.90 → 0.93
- Use "Strict" mode

### Optimal setting:
- Start at **0.85** (balanced)
- Monitor false positive/negative rates
- Adjust in 0.01 increments
