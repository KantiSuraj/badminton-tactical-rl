# B2 CoachAI+ worker delivery

Copy the files into the matching paths in the `badminton-tactical-rl` repository.
Do not copy this folder as a new project; these are additive adapter files plus
new tests and docs. `pyproject.toml.patch` is optional pytest marker registration.

The worker is designed to run only in `.venv-coachai`; the parent imports only
`protocol.py` and `worker_client.py`.

## Validation performed in the delivery environment

- Adapter-focused suite: **44 passed, 2 skipped**.
- Existing repository tests excluding the two parquet-dependent data test
  modules: **213 passed, 2 skipped**.
- `py_compile` passed for all adapter modules and the smoke runner.
- The 2 skipped tests are opt-in real CoachAI+/RallyNet tests. They require the
  user's WSL checkout and `.venv-coachai`; those paths are not available in the
  delivery environment, so no claim is made that real RallyNet has been loaded
  or a full native rally has run here.
- The full unfiltered suite could not be validated here because the two excluded
  data modules require a Parquet engine (`pyarrow` or `fastparquet`) absent from
  this delivery runtime. The project source was not altered to work around it.

See `docs/B2_WORKER_INTEGRATION.md` for the protocol, invariants, setup,
smoke-test commands, parent-side usage, and known limits.
