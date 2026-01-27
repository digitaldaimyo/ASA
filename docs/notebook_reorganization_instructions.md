# ASA Mechanistic Probing Notebook Reorganization Instructions

## Project Context

**Architecture:** Addressed State Attention (ASA) - a novel transformer architecture with slot-based memory and second-order control mechanisms

**Purpose:** Transform an exploratory analysis notebook into a publication-ready, narrative-driven mechanistic probing notebook that generates figures for a research paper

**Source:** The current notebook is in a GitHub repository (to be provided by user)

**Output:** A new, clean notebook that preserves all analysis but reorganizes it for clarity and narrative flow

---

## Core Narrative (The Story to Tell)

The notebook should reveal this mechanistic story across depth:

### The Three-Axis Substrate Framework

Attention heads operate in a 3D substrate space with orthogonal axes:

1. **M (Memory Capacity)** - How much temporal state a head maintains
   - Primary: `write_ess`
   - Secondary: `write_tail_half_life`
   - Increases with depth; drives role switching

2. **P (Routing Smoothness)** - How smoothly routing policy evolves
   - Primary: `inertia_mean`
   - Secondary: `inertia_slope`
   - Peaks mid-depth; controls reconfiguration dynamics

3. **C (Routing Commitment)** - How decisively heads select slots
   - Primary: `top4_mass`
   - Secondary: `tok_ent`, `eff_slots`
   - Modulates expression style

### Categorical Flags (Not Axes)

- **L (Routing Lock)** - Whether policy is frozen
  - `inertia_half_lag >= threshold`
  - Distinguishes anchors from adaptive carriers

### Second-Order Control Fields (Layer-Level)

- **E_slot** - Slotspace gate strength (enables substrate coordination)
- **E_gamma** - Content read strength (content-driven attention bias)
- These are *environmental* signals that shape what head roles can exist at each depth

### Developmental Phases Across Depth

1. **Formation (L0-3)** - Local processing, weak memory, interchangeable heads
2. **Remodeling (L8-15)** - Second-order fields peak, M expands, roles diverge
3. **Handoff (L12-14)** - Transition zone between content and slot dominance
4. **Consolidation (L16-20)** - Memory concentrates, locks engage, stability increases

### Head Taxonomy (Substrate Classes)

**Adaptive Classes:**
- EXPLORATORY_ROUTER - Low P, low C
- SEMANTIC_SPECIALIST - High C under content shaping
- HIGH_CAPACITY_INTEGRATOR - High M, unlocked

**Locked Classes:**
- POLICY_ANCHOR - High P, high C, stable
- FROZEN_HIGH_CAPACITY_ANCHOR - High M+P, locked
- BRITTLE_LOCKER - Low M, high P+C

**Transitional:**
- TRANSITIONER - In handoff regime

---

## Target Notebook Structure

```
1. SETUP & MODEL LOADING
   - Clone repo, install dependencies
   - Load HF checkpoint
   - Model summary and parameter count
   - Set random seeds
   
2. DATA GENERATION (SINGLE COMPREHENSIVE PASS)
   - Build/load validation dataset
   - Run inference with hooks capturing ALL tensors
   - Save raw tensors to artifacts/
   - This should be ONE forward pass, not multiple
   
3. CORE METRICS COMPUTATION
   
   3.1 Axis M: Memory Capacity
       - write_ess computation
       - write_tail_half_life computation
       - Z-score normalization
       - Quick validation plot (optional, can be lightweight)
   
   3.2 Axis P: Routing Smoothness
       - inertia_mean computation
       - inertia_slope computation
       - Z-score normalization
       - Quick validation plot
   
   3.3 Axis C: Routing Commitment
       - top4_mass computation
       - tok_ent, eff_slots computation
       - Z-score normalization
       - Quick validation plot
   
   3.4 Flag L: Routing Lock
       - inertia_half_lag computation
       - locked flag (binary) with threshold
       - Distribution plot
   
   3.5 Secondary Metrics
       - Cluster analysis (if used)
       - Regime switching counts
       - Any other derived metrics
   
4. SECOND-ORDER CONTROL FIELDS
   - slotspace_gate extraction/computation
   - content_read_gamma extraction/computation
   - delta_norm computation
   - Build environment table: df_env
     - Columns: layer, E_slot, E_gamma, slot_state, gamma_state
   - Field regime classification per layer
   
5. DRIFT & PHASE ANALYSIS
   - Compute per-layer drift in (M,P,C) space
   - drift_mean per layer
   - Drift regime classification
   - Phase assignment (FORMATION, REMODELING, etc.)
   - Build layer regimes table: df_layer_regimes
   
6. TAXONOMY CLASSIFICATION
   - Threshold computation (quantile-based for M/P/C)
   - Substrate region binning (LOW/MID/HIGH)
   - Substrate class assignment
   - Build canonical head×layer table: df_hl
     - Columns: layer, head, M, P, C, locked, substrate_class, phase, switches, etc.
   - Class distribution summary
   
7. VISUALIZATION (NARRATIVE ORDER)
   
   7.1 Overview: Routing Decomposition
       - Content vs Key routing variance explained
       - Similarity metrics (Argmax agreement, JS divergence)
       - Key-content correlation & confidence shift
       
   7.2 Memory Dynamics
       - Slot memory timescales (tail half-life, ESS)
       - Per-head heatmaps across layers
       - Memory capacity growth across depth
       
   7.3 Substrate Geometry
       - PCA projections per layer (showing head clustering evolution)
       - Slot norm heatmaps (H × K) per layer
       - 3D substrate space (M, P, C) with all heads
       - Depth regime breakdown (EARLY/MID/LATE)
       
   7.4 Trajectory Analysis
       - Per-head trajectories in M-P plane
       - 3D trajectories for key heads (locked vs adaptive)
       - Vector field showing drift colored by control fields
       - Mean drift magnitude across depth
       
   7.5 Control Field Evolution
       - E_slot and E_gamma across layers
       - Second-order strength (gate values)
       - Magnitude of corrections
       
   7.6 Phase & Taxonomy
       - Phase timeline (BEFORE/AFTER if sweep used)
       - Phase portrait (M-P space with arrows)
       - Cluster occupancy across depth
       - Substrate class distributions
       - Predictive features (what correlates with adaptive vs locked)
       
   7.7 Routing Manifold
       - 3D UMAP density map of slot keys
       
8. VALIDATION & EXPORT
   - Consistency checks
   - Export canonical tables (CSV, JSON)
   - Save all plots to artifacts/
   - Summary statistics
   - Print narrative summary
```

---

## Specific Reorganization Tasks

### A. Code Consolidation & Deduplication

**ELIMINATE REDUNDANCY:**

1. **Single data collection pass**
   - Current notebook likely runs inference multiple times
   - Consolidate into ONE forward pass with comprehensive hooks
   - Cache tensors to disk immediately
   - Subsequent sections load from cache

2. **Unified metric computation**
   - Don't recompute M/P/C multiple times
   - Compute once, store in dataframes, reference everywhere
   - Z-score normalization happens once

3. **Single taxonomy classification**
   - Current code may have "BEFORE" and "AFTER" taxonomy
   - Keep the sweep logic but run it ONCE
   - Store best config, apply it, generate final taxonomy

4. **Plotting helper functions**
   - Extract repeated plotting patterns into functions
   - Example: substrate_3d_plot(df, color_by='layer', marker_by='locked')
   - Reduces code duplication in visualization section

### B. Code Quality Improvements

**WITHIN EACH SECTION:**

1. **Clear section headers**
   ```python
   #@title 3.1 Axis M: Memory Capacity
   """
   Compute write_ess and write_tail_half_life for each head×layer.
   These measure how diffuse and long-lived the write distributions are.
   High M → long-range memory capacity.
   """
   ```

2. **Explicit intermediate saves**
   ```python
   # Save to artifacts for inspection
   (artifacts_dir / 'metrics_M.json').write_text(json.dumps(metrics_M, indent=2))
   ```

3. **Progress indicators**
   ```python
   print(f"Computing memory metrics... ", end="")
   # ... computation ...
   print(f"Done. Shape: {df_M.shape}")
   ```

4. **Validation checks**
   ```python
   assert not df_hl['M'].isna().any(), "M contains NaN values"
   assert df_hl['locked'].isin([0,1]).all(), "locked must be binary"
   ```

5. **Clear variable naming**
   - `df_hl` - head×layer table (canonical)
   - `df_layer_regimes` - layer-level table (canonical)
   - `df_env` - environment fields per layer
   - No ambiguous names like `df_pts`, `df_tax`, `df_layer`, `df_layer_reg`

### C. Data Flow

**CANONICAL DATAFRAMES:**

Build these in order and never recompute:

1. **df_metrics_raw** - raw metrics immediately after inference
2. **df_M** - Memory axis metrics (unnormalized)
3. **df_P** - Routing smoothness metrics (unnormalized)
4. **df_C** - Commitment metrics (unnormalized)
5. **df_hl_raw** - Merge M/P/C before normalization
6. **df_hl** - Canonical head×layer table (z-scored M/P/C, locked flag, etc.)
7. **df_env** - Layer-level environment fields
8. **df_layer_regimes** - Layer-level with field regimes, drift, phase

**MERGE ORDER:**
```python
df_hl = (df_M
         .merge(df_P, on=['layer','head'])
         .merge(df_C, on=['layer','head'])
         .merge(df_lock, on=['layer','head']))

# Z-score M, P, C
df_hl['M'] = zscore(df_hl['write_ess'])
df_hl['P'] = zscore(df_hl['inertia_mean'])
df_hl['C'] = zscore(df_hl['top4_mass'])

# Compute drift
df_layer_drift = compute_drift(df_hl)

# Merge layer-level info
df_layer_regimes = (df_env
                    .merge(df_layer_drift, on='layer')
                    .pipe(classify_phases))

# Add phase to df_hl
phase_map = df_layer_regimes.set_index('layer')['phase'].to_dict()
df_hl['phase'] = df_hl['layer'].map(phase_map)

# Classify substrate
df_hl = classify_substrate(df_hl, thresholds)
```

### D. Plotting Organization

**GROUPING PRINCIPLE:** Group by narrative, not by computation

**Current (BAD):** Plots interleaved with metric computation
**Target (GOOD):** All plots in section 7, ordered by story

**PLOT MATRIX:**

| Section | Plot Type | Purpose | Data Source |
|---------|-----------|---------|-------------|
| 7.1 | 4-panel overview | Key vs content decomposition | routing_analysis |
| 7.2 | Timescale plot + heatmaps | Memory dynamics | df_hl |
| 7.3 | PCA + slot norms | Geometry evolution | df_hl |
| 7.3 | 3D substrate space | Full M-P-C structure | df_hl |
| 7.4 | Per-head trajectories | Individual head paths | df_hl |
| 7.4 | Drift vector fields | Flow through substrate | df_hl + df_layer_regimes |
| 7.5 | Field strength plot | Control signal evolution | df_env |
| 7.6 | Phase timeline | Developmental narrative | df_layer_regimes |
| 7.6 | Taxonomy distributions | Class prevalence | df_hl |
| 7.7 | UMAP manifold | Routing topology | slot_keys |

**REUSABLE PLOTTING FUNCTIONS:**

Create these helper functions in section 7.0:

```python
def plot_substrate_3d(df, color_by='layer', marker_by='locked', 
                      size_by=None, title=None, figsize=(10,8)):
    """3D scatter in (M,P,C) space with flexible styling"""
    pass

def plot_trajectory_2d(df, heads=None, color_by='layer', 
                       annotate_locked=True, title=None):
    """2D trajectories in M-P plane"""
    pass

def plot_phase_timeline(df_layer, title=None):
    """Phase scatter plot across layers"""
    pass

def save_plot(name):
    """Save current figure to artifacts/plots/{name}.png"""
    plt.savefig(artifacts_dir / 'plots' / f'{name}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
```

---

## Preservation Requirements

**MUST BE PRESERVED:**

1. ✅ All computed metrics (nothing lost)
2. ✅ All visualizations (can be reorganized but not removed)
3. ✅ Taxonomy sweep logic (if present)
4. ✅ Threshold selection methodology
5. ✅ Statistical summaries and crosstabs
6. ✅ Export functionality (JSON, CSV)
7. ✅ Random seed setting for reproducibility
8. ✅ Model loading and validation

**CAN BE MODIFIED:**

1. ✅ Order of execution (as long as dependencies respected)
2. ✅ Variable names (for clarity)
3. ✅ Code organization within sections
4. ✅ Comments and documentation (improve them!)
5. ✅ Plotting aesthetics (make them publication-quality)

**CAN BE REMOVED:**

1. ✅ Duplicate computations
2. ✅ Debug print statements (unless informative)
3. ✅ Commented-out code
4. ✅ Unused imports
5. ✅ Intermediate dataframes that aren't used downstream

---

## Quality Control Checklist

Before considering the task complete, verify:

### 1. Completeness Checks

- [ ] All metrics from original notebook are computed
- [ ] All plots from original notebook are generated
- [ ] All exports (JSON, CSV) are created
- [ ] Section 1-8 structure is fully implemented
- [ ] No orphaned code blocks

### 2. Data Integrity Checks

- [ ] df_hl has expected shape: (num_layers × num_heads, num_columns)
- [ ] df_layer_regimes has one row per layer
- [ ] No NaN values in critical columns (M, P, C, locked, phase, substrate_class)
- [ ] Z-scored values have mean≈0, std≈1
- [ ] Locked flag is binary (0/1)
- [ ] Phase labels match expected taxonomy

### 3. Reproducibility Checks

- [ ] Random seed set at start
- [ ] All data loading uses cached artifacts when available
- [ ] Forward pass is deterministic
- [ ] Plots are generated from saved dataframes, not recomputed data

### 4. Code Quality Checks

- [ ] No code duplication (DRY principle)
- [ ] Clear section headers with @title and docstrings
- [ ] Helper functions have docstrings
- [ ] Imports are organized (standard lib, third-party, local)
- [ ] No unused imports or variables
- [ ] Consistent naming conventions

### 5. Narrative Checks

- [ ] Sections flow logically (Setup → Data → Metrics → Fields → Taxonomy → Viz)
- [ ] Plots are ordered to tell the developmental story
- [ ] Comments explain WHY, not just WHAT
- [ ] Key findings are summarized after major sections

### 6. Output Checks

- [ ] All artifacts saved to artifacts/ directory
- [ ] Plot filenames are descriptive
- [ ] JSON exports are valid and readable
- [ ] CSV exports have headers and are well-formed

### 7. Execution Checks

- [ ] Notebook runs end-to-end without errors
- [ ] Runtime is reasonable (<10 min on CPU for standard dataset)
- [ ] Memory usage is reasonable (no OOM on typical machine)
- [ ] Progress indicators show what's happening

---

## Specific Code Patterns to Implement

### Pattern 1: Section Template

```python
#@title {Section Number} {Section Name}
"""
Brief description of what this section does and why.
Key outputs: list the main variables/artifacts produced.
"""

print(f"\n{'='*60}")
print(f"  {Section Name}")
print(f"{'='*60}\n")

# Imports specific to this section (if needed)
# ...

# Main computation
# ...

# Validation
assert {condition}, "Error message"

# Save artifacts
# ...

# Summary
print(f"✓ Completed: {summary statistics}")
```

### Pattern 2: Cached Computation

```python
def compute_or_load(cache_path, compute_fn, *args, **kwargs):
    """Load from cache if exists, otherwise compute and save."""
    if cache_path.exists():
        print(f"Loading cached: {cache_path.name}")
        return pd.read_csv(cache_path)  # or pickle.load, etc.
    
    print(f"Computing: {cache_path.name}...")
    result = compute_fn(*args, **kwargs)
    result.to_csv(cache_path, index=False)
    return result

# Usage
df_M = compute_or_load(
    artifacts_dir / 'metrics_M.csv',
    compute_memory_metrics,
    slot_states, write_logits
)
```

### Pattern 3: Taxonomy Classification

```python
def classify_substrate_class(row, thresholds):
    """
    Assign substrate class based on (M,P,C) bins and locked flag.
    Follows priority order: Locked classes first, then adaptive.
    """
    M_bin = bin_axis(row['M'], thresholds['M'])
    P_bin = bin_axis(row['P'], thresholds['P'])
    C_bin = bin_axis(row['C'], thresholds['C'])
    locked = row['locked']
    phase = row['phase']
    
    # Priority order
    if locked and M_bin == 'HIGH' and P_bin == 'HIGH':
        return 'FROZEN_HIGH_CAPACITY_ANCHOR'
    if locked and M_bin == 'LOW' and P_bin == 'HIGH' and C_bin == 'HIGH':
        return 'BRITTLE_LOCKER'
    if locked and P_bin == 'HIGH' and C_bin == 'HIGH':
        return 'POLICY_ANCHOR'
    if phase == 'PHASE_HANDOFF' and not locked:
        return 'TRANSITIONER'
    if not locked and M_bin == 'HIGH':
        return 'HIGH_CAPACITY_INTEGRATOR'
    if not locked and C_bin == 'HIGH' and P_bin != 'HIGH':
        return 'SEMANTIC_SPECIALIST'
    if not locked and P_bin == 'LOW' and C_bin == 'LOW':
        return 'EXPLORATORY_ROUTER'
    
    return 'OTHER'

df_hl['substrate_class'] = df_hl.apply(
    lambda row: classify_substrate_class(row, thresholds),
    axis=1
)
```

### Pattern 4: Plot Saving

```python
def save_figure(name, fig=None):
    """Save current or specified figure to artifacts/plots/"""
    plot_dir = artifacts_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    if fig is None:
        fig = plt.gcf()
    
    path = plot_dir / f'{name}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path.name}")

# Usage
plt.figure(figsize=(12, 6))
# ... plotting code ...
plt.title("Phase Timeline")
save_figure('phase_timeline')
plt.show()
```

---

## Step-by-Step Execution Plan for Claude Code

### Phase 1: Setup & Analysis (Do First)

1. **Read and understand the original notebook**
   - Identify all sections
   - Map all computed variables
   - List all plots generated
   - Note any non-obvious dependencies

2. **Create skeleton of new notebook**
   - Section headers (1-8) with empty cells
   - Imports cell
   - Setup cell with paths and seeds

3. **Verify understanding with user**
   - "I've analyzed the original notebook. Here's what I found..."
   - List key sections, metrics, plots
   - Ask for confirmation before proceeding

### Phase 2: Core Migration (Systematic Transfer)

4. **Section 1: Setup & Model Loading**
   - Copy directly, verify imports
   - Test model loading

5. **Section 2: Data Generation**
   - Consolidate inference code
   - Ensure single forward pass
   - Add caching logic

6. **Section 3: Core Metrics**
   - Extract M computation → 3.1
   - Extract P computation → 3.2
   - Extract C computation → 3.3
   - Extract L computation → 3.4
   - Verify no duplication

7. **Section 4: Control Fields**
   - Extract field computations
   - Build df_env table
   - Verify layer-level aggregations

8. **Section 5: Drift & Phases**
   - Extract drift computation
   - Extract phase classification
   - Build df_layer_regimes

9. **Section 6: Taxonomy**
   - Extract threshold computation
   - Extract classification logic
   - Build df_hl (canonical)
   - Verify class assignments

10. **Section 7: Visualization**
    - Group all plots by narrative order
    - Create plotting helpers
    - Ensure each plot has df source
    - Add save_figure calls

11. **Section 8: Export**
    - Consolidate all export code
    - Add final summary

### Phase 3: Quality Assurance

12. **Run complete notebook**
    - Fix any errors
    - Verify all plots generate
    - Check all artifacts saved

13. **Compare outputs with original**
    - Key metrics match?
    - Plots look correct?
    - Any missing pieces?

14. **Code cleanup**
    - Remove TODOs
    - Add final comments
    - Format consistently

15. **Final verification**
    - Run quality control checklist
    - Test from clean environment
    - Document any known issues

### Phase 4: Documentation

16. **Add introductory markdown cell**
    ```markdown
    # Addressed State Attention: Mechanistic Probing Analysis
    
    This notebook performs comprehensive mechanistic analysis of the ASA architecture,
    revealing how attention heads organize into substrate classes and evolve through
    developmental phases across depth.
    
    **Runtime:** ~5 minutes on CPU
    **Outputs:** Canonical tables (df_hl, df_layer_regimes) and figure suite
    ```

17. **Add section introductions**
    - Each major section gets markdown cell
    - Explains purpose and key outputs

18. **Create README for artifacts**
    - Explain each saved file
    - Document table schemas

---

## Success Criteria

The reorganized notebook is considered complete when:

1. ✅ Runs end-to-end without errors
2. ✅ Produces identical metrics to original (within numerical precision)
3. ✅ Generates all expected plots
4. ✅ Saves all artifacts (JSON, CSV, plots)
5. ✅ Code is DRY (no significant duplication)
6. ✅ Sections follow logical narrative order
7. ✅ Quality control checklist passes
8. ✅ Claude Code has verified completion with user

---

## Notes for Claude Code

### When Stuck

- **Ask before guessing:** If logic is unclear, ask user for clarification
- **Show examples:** When proposing changes, show before/after code
- **Validate incrementally:** Test each section before moving to next

### Communication Style

- **Progress updates:** Every 3-5 sections, summarize progress
- **Flag issues:** If something seems wrong in original, flag it
- **Suggest improvements:** If you see opportunities beyond reorganization

### Priority Order

1. **Preserve correctness** (don't break anything)
2. **Maintain completeness** (don't lose information)
3. **Improve clarity** (make it readable)
4. **Optimize efficiency** (reduce redundancy)

---

## Appendix: Variable Reference

### Canonical Dataframes

| Name | Shape | Key Columns | Purpose |
|------|-------|-------------|---------|
| df_hl | (L×H, many) | layer, head, M, P, C, locked, substrate_class, phase | Canonical head×layer table |
| df_layer_regimes | (L, ~10) | layer, E_slot, E_gamma, field_regime, drift_mean, phase | Layer-level properties |
| df_env | (L, 4-5) | layer, E_slot, E_gamma, slot_state, gamma_state | Environment fields |

### Key Metrics

| Metric | Type | Range | Meaning |
|--------|------|-------|---------|
| write_ess | float | [1, K] | Effective sample size of write distribution |
| write_tail_half_life | float | [1, T] | Tokens until write mass halves |
| inertia_mean | float | [0, 1] | Mean overlap between consecutive read distributions |
| inertia_slope | float | [-1, 1] | Rate of change in inertia |
| inertia_half_lag | float | [1, T] | Lag until inertia drops to 0.5 |
| top4_mass | float | [0, 1] | Probability mass on top 4 slots |
| tok_ent | float | [0, log(K)] | Entropy of per-token routing |
| eff_slots | float | [1, K] | Effective number of slots used |

### Taxonomy Labels

**Phases (layer-level):**
- PHASE_FORMATION
- PHASE_REMODELING
- PHASE_HANDOFF
- PHASE_CONSOLIDATION
- PHASE_TERMINAL_TUNING
- PHASE_MIXED

**Substrate Classes (head-level):**
- EXPLORATORY_ROUTER
- SEMANTIC_SPECIALIST
- HIGH_CAPACITY_INTEGRATOR
- POLICY_ANCHOR
- FROZEN_HIGH_CAPACITY_ANCHOR
- BRITTLE_LOCKER
- TRANSITIONER
- OTHER

---

## End of Instructions

This document should be sufficient for Claude Code to perform the reorganization.
Any questions or ambiguities should be raised with the user before proceeding.

**Next step:** User will provide GitHub repo URL, then Claude Code begins Phase 1.
