# Implementation audit (Burn + Magika)

This document captures a focused audit of the current implementation with emphasis on Burn idioms, reliability, and maintainability.

## Sources consulted

- Burn Book ONNX import guidance: `burn-import`/`burn-onnx` generation flow, load strategies, and compatibility notes.
- Burn crate/docs for current pre-release ecosystem (`0.21.0-pre.3`) used by this repository.
- Existing Magika parity tests and vendored upstream generated metadata.

## Summary of findings

### ✅ Current strengths

1. **Backend-generic API shape is good**  
   `MagikaModel<B>` exposes a generic inference surface over Burn backends, consistent with Burn’s backend abstraction.

2. **Parity-first validation approach is strong**  
   Existing fixture parity test against the upstream Rust `magika` crate is the correct quality anchor for this project.

3. **Minimal runtime dependencies**  
   Current approach avoids requiring an ONNX runtime at execution time (aligned with Burn philosophy).

### ⚠️ Key gaps and improvements

1. **Batch path previously executed N single-item forwards**  
   `infer_batch` iterated and called `forward` once per sample. This caused avoidable overhead and diverged from idiomatic tensor-batch execution in Burn.

   **Status:** fixed in this change by running one batched forward pass and slicing output rows.

2. **ONNX tensor decoding robustness**  
   Parsing assumed all tensor values are in `raw_data`. Some ONNX exports may use `float_data` instead.

   **Status:** fixed in this change by supporting both `raw_data` and `float_data`, with clearer shape/value validation errors.

3. **Hand-implemented ONNX graph execution is brittle long-term**  
   Manual extraction of hard-coded initializer names and hand-written operator graph increases maintenance burden when upstream model assets change.

   **Recommendation:** consider migrating to generated Burn model code via `burn-import`/`burn-onnx` in `build.rs`, especially if model refresh cadence increases.

4. **Unsafe label conversion**
   Current `label_for_index` uses `transmute`. It is range-guarded, but still introduces avoidable unsafe surface.

   **Recommendation:** prefer generated mapping tables or an explicit conversion helper from generated/vendor code.

5. **Limited unit-level invariants around model internals**
   Current tests cover determinism/parity, but there are few targeted checks for tensor loader behavior and edge cases.

   **Recommendation:** add tests for:
   - tensor decode fallback (`raw_data` vs `float_data`)
   - output shape mismatch failures
   - threshold/overwrite map behavior for known scores

## Should this project use `burn-onnx` now?

Short answer: **it can be a good fit**, but only if you want lower maintenance on model graph changes.

- If your goal is **strict reproducibility + easier model upgrades**, generated code via `burn-import` is likely more idiomatic in the Burn ecosystem.
- If your goal is **tight custom control of every operation**, current manual path is valid, but expect higher upkeep.

For this repository specifically, a practical path is:

1. Keep current implementation as baseline.
2. Add an optional generated-model feature branch (same fixtures/parity tests).
3. Compare:
   - parity drift risk
   - code complexity
   - compile/runtime performance
   - binary size
4. Adopt generated path only if it improves at least maintainability without hurting parity.

## Additional medium-term recommendations

1. Add benchmark split by backend + batch size (1, 8, 32, corpus-size).
2. Track model metadata hash/version in tests to fail loudly on silent asset drift.
3. Expose a non-allocating batch API (`&[&[u8]]`) in addition to current owned vector input.
4. Consider structuring model layers as explicit Burn modules if staying manual, to improve readability and future extension.
