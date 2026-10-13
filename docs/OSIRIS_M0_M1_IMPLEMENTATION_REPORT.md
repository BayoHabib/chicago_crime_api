# OSIRIS Predictive Intelligence — M0/M1 implementation report

Date: 2026-10-13. Branch: `work/osiris-predictive-security-m1-20261013`. Draft PR only; no merge.

## Completed
- Removed synthetic `PoissonRegressor` training fallback from `PredictionService`.
- Require a metadata manifest (`validation_status=validated`, `validation_run_id`, `version`, `artifact_sha256`) plus artifact integrity check before loading.
- Absent/unvalidated models return 503 `MODEL_NOT_VALIDATED` at single, batch and grid prediction routes.
- Added negative regression tests for missing, unvalidated and absent models. Assert liveness is independent of model availability.

## Dependency/security audit (static findings, not vulnerability scan)
1. `pyproject.toml` declares unpinned direct Git dependencies for EventFlow and Chicago Crime Downloader. Pin commit hashes or signed releases, lock transitive packages, and enable vulnerability/SBOM scanning.
2. `joblib.load` deserializes executable Python content. A locally stored SHA-256 and a manifest do **not** establish trust if an attacker can replace both. Restrict artifact filesystem permissions, require trusted build provenance or signed manifests, and ideally deploy non-executable model formats.
3. `main.py` enables permissive CORS under debug and includes boilerplate production assertions. Limit allowed origins and correct misleading model status claims.
4. Previously generated intervals were normal approximations with an unsupported nominal coverage statement; grid features/resolution and horizon scaling are not validated. Prediction endpoints should remain disabled for public use until statistical validation and schema migration are complete.
5. Existing `tests/conftest.py` references a `load_model` method that the service does not expose; legacy tests require modernization. The new targeted tests use explicit overrides.
6. OSIRIS SDK ingestion and streaming routes are not appropriate for restricted crime-event data. Introduce an authenticated, bounded, aggregation-only Next.js server-side gateway on a private network, as specified in the integration document.

## Tests and limitations
- Tests added in `tests/test_api.py`; status depends on GitHub Actions CI for draft PR.
- No Chicago data were downloaded, no model trained, and no quality or fairness score claimed.
- No code changes in OSIRIS, EventFlow, or Downloader.
- No branch merges or default-branch updates.

## Follow-up gates
- A01: missing model -> deterministic 503 (test added).
- A02–A06: event-time correctness, future leakage, forecast horizons, interval calibration, provenance (NOT complete).
- A07–A11: gateway auth, privacy, rate limits, UI labels, OSIRIS tests (NOT complete).
- A12: signed build/model provenance and reproducibility (NOT complete).

Next branch milestone: create validated training artifact pipeline with rolling-origin tests and fixed grid specification; remove unsupported risk tiers and confidence interval claims before enabling model serving.
