# Multi-Person Recognition Diagnostic Guide

## Issue: Second Person Not Recognized

If you have enrolled 2 persons but only one is being recognized, follow these steps:

### Step 1: Check Enhanced Logs

After the update, the logs now show **ALL enrolled persons' similarity scores**:

**Example log output:**
```
MATCH: hashir (similarity=0.822, conf=82%) | All: [hashir=0.822, person2=0.654]
```

This shows:
- **hashir**: 82.2% match (recognized)
- **person2**: 65.4% match (too low)

### Step 2: Common Causes

#### Cause 1: Threshold Too High
If person2's similarity is **below 0.72** (current threshold):
- **Solution**: Lower the threshold via UI slider
- Try: 0.65 or 0.60
- Check if person2 gets recognized

#### Cause 2: Similar Gaits
If both persons have very similar walking patterns:
- **hashir**: 0.822
- **person2**: 0.820

The system correctly picks the **higher** match, but they're too close!

**Solutions:**
- Re-enroll person2 with more varied walking sequences
- Ensure person2 walks at their natural pace
- Capture different walking directions (towards/away/sideways)

#### Cause 3: Poor Enrollment Quality
If person2's enrolled features are bad quality:
- Low confidence during enrollment
- Not enough variation in poses
- Partial body detections

**Solution**: Re-enroll person2:
1. Go to Enroll tab
2. Delete old enrollment (if possible)
3. Re-enroll with:
   - Good lighting
   - Full body visible
   - Natural walking
   - Multiple walking directions

#### Cause 4: Name Capitalization
Check if names match exactly:
- Enrolled as: "Person2"
- System shows: "person2"

**Solution**: Names are case-sensitive. Ensure exact match.

### Step 3: Test with Logs

1. **Restart Streamlit** to get new logs:
   ```bash
   streamlit run main.py
   ```

2. **Have person2 walk in front of camera**

3. **Check logs** for output like:
   ```
   MATCH: hashir (similarity=0.822, conf=82%) | All: [hashir=0.822, person2=0.654]
   ```

4. **Analyze the scores**:
   - If person2's score is **< 0.72**: Lower threshold
   - If person2's score is **close to person1**: Re-enroll with better distinction
   - If person2's score is **not shown**: Enrollment failed (re-enroll)

### Step 4: Verify Enrollment

Check if both persons are actually enrolled:

**Via UI:**
- Go to Recognition tab
- Look at "Status" panel
- Should show: "Monitoring 2 pattern(s)"
- Should list both names

**Via Logs:**
- Look for: `Loaded X patterns for user`
- Should see both names in the list

### Step 5: Optimal Enrollment Tips

For **best multi-person recognition**:

1. **Different walking styles**:
   - Person1: Normal pace
   - Person2: Slower or faster pace
   
2. **Full sequences**:
   - Use 8-10 sequences (default)
   - Walk continuously during capture
   
3. **Good conditions**:
   - Bright lighting
   - Stable camera
   - Full body visible
   - 2-4 meters from camera

### Step 6: Adjust Threshold

The similarity threshold determines **how strict** matching is:

- **0.90**: Very strict (may reject valid users)
- **0.85**: Strict (good security)
- **0.80**: Balanced (recommended)
- **0.75**: Lenient (lower security)
- **0.70**: Very lenient (current setting)
- **0.65**: Too lenient (high false positives)

**For 2 persons with similar gaits**: Use **0.75-0.80**  
**For 2 persons with distinct gaits**: Use **0.70-0.75**

### Example Diagnosis

**Logs show:**
```
MATCH: hashir (similarity=0.822, conf=82%) | All: [hashir=0.822, ali=0.680]
```

**Analysis:**
- hashir: 82.2% (above 0.72 threshold) ✅
- ali: 68.0% (below 0.72 threshold) ❌

**Solutions:**
1. Lower threshold to **0.65** (ali will match at 68%)
2. OR re-enroll ali with better quality data

---

## Quick Fix Checklist

- [ ] Restart Streamlit to see new logs
- [ ] Verify 2 persons enrolled (check Status panel)
- [ ] Have person2 walk in front of camera
- [ ] Check logs for "All: [...]" to see both scores
- [ ] If person2 < 0.72: Lower threshold via UI
- [ ] If person2 not in list: Re-enroll person2
- [ ] If scores too close: Re-enroll with better distinction
