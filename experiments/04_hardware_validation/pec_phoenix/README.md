# pec_phoenix — Error mitigation on ibm_phoenix (Supplementary Section S8)

Probabilistic error cancellation (PEC) for the single-step MNIST QRU circuit of Table 10 (OTC group 0, rep 0;
observables Z0, Z4, Z6, X5, X7 at a statevector value of about +0.8), and readout mitigation, gate twirling, and
ZNE applied to the same transpiled circuit. All runs use the built-in options of the Qiskit Runtime Estimator.

PEC was measured on ibm_phoenix (IBM Nighthawk r2) because its sampling overhead for this circuit on
ibm_marrakesh was too large for the available QPU time (Supplementary Section S8).

## Contents

| File | Purpose | Paper |
|---|---|---|
| `PEC_repeats.ipynb` | Transpiles the circuit for ibm_phoenix (optimization level 1, CZ couplers with reported error > 1% excluded, fixed seeds) and saves the ISA circuit; also runs PEC at Qiskit Runtime's default precision (these runs are not reported in the paper) | circuit used in Tables 10, S8, S9 |
| `PEC_fixed_shots.ipynb` | PEC runs with a set number of randomizations and shots per randomization | Table 10 (PEC row), Table S8, Figure 7 |
| `Mitigation_phoenix.ipynb` | Readout mitigation, gate twirling, and ZNE with the settings of Table 10, on the same transpiled circuit | Table S9 |
| `pec_isa_circuit_ibm_phoenix_seed0_excl1pct_20261010_021000.qpy` | Transpiled (ISA) circuit used for all runs in this folder (seed 0; 84 CZ gates, depth 266) | — |
| `pec_observables_ibm_phoenix_seed0_excl1pct_20261010_021000.json` | Observables mapped onto the physical qubits of this circuit | — |
| `pec_repeats_circuits_ibm_phoenix_20261010_021000.json` | Transpilation summary (CZ count, depth, summed reported CZ error, physical qubits). It also lists seeds 1–3, whose circuits are not used in the paper and not included here | — |
| `pec_fixedshots_results_ibm_phoenix.csv` | Results of all `PEC_fixed_shots.ipynb` runs (one row per run) | Table S8 |
| `mitigation_results_ibm_phoenix.csv` | Results of all `Mitigation_phoenix.ipynb` runs (one row per run) | Table S9 |

The notebooks read the OTC parameters from `../error_mitigation_comparison/qru_otc_multi_obs_20260503_205659.csv`.

## Results reported in the paper

**PEC (Table 10, Table S8).** The paper uses the 15 runs of `pec_fixedshots_results_ibm_phoenix.csv` with
`num_randomizations_req >= 400` (number of randomizations set to 400, 800, or 1,600; 128, 256, or 512 shots per
randomization). The other rows of the file are an initial check run and runs with the default 32 randomizations,
which are not reported. Key columns:

- `num_randomizations_req`, `shots_per_rand_req`: values set in the Qiskit Runtime options
- `num_randomizations`, `shots`: values executed, from the job metadata (the executed number of randomizations equals the value set multiplied by the learned sampling overhead, rounded up)
- `noise_overhead`: sampling overhead reported by Qiskit Runtime
- `ev_*`, `std_*`: expectation values and reported statistical uncertainties
- `mean_dev_pct`: mean of the signed relative deviations (ev − SV)/SV over the five observables

**Other methods (Table S9).** `mitigation_results_ibm_phoenix.csv` contains three runs each of `baseline`
(readout mitigation only), `twirling` (300 instances), `zne` (32 twirling instances), and `zne_twirling`
(ZNE with 300 twirling instances). The deviation in Table S9 is computed as in Table 10, from the mean over the
three runs of each observable.

## Running the notebooks

- An IBM Quantum account with access to ibm_phoenix is required. Set `SERVICE_NAME` and `INSTANCE` in Cell 1
  (`<YOUR_INSTANCE_NAME>` is a placeholder).
- Submission and collection are separate cells, so results can be collected after a kernel restart. The notebooks
  write job IDs to local files (`*_pending_*.json`, `pec_job_ids.txt`); these files are not part of this repository.
- Hardware results depend on the device calibration at the time of execution and are not expected to be
  reproduced exactly.
