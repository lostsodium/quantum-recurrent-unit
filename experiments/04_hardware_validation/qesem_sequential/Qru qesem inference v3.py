
#!/usr/bin/env python
# coding: utf-8
"""
QRU MNIST Inference — QESEM with Checkpoint/Resume
====================================================
Runs single-sample MNIST inference across 8 time steps with full
crash recovery. Each step is checkpointed to JSON immediately after
completion so the run can be resumed from any interruption point.

All 10 observables are submitted in a single job per step.
QESEM handles incompatible bases (Z and X on same qubit) internally.

Backend modes (set BACKEND_MODE below)
---------------------------------------
  'statevector'  -- local StatevectorEstimator, no transpile, no noise
  'fake'         -- AerSimulator.from_backend(FakeMarrakesh())
  'noise_model'  -- AerSimulator with NoiseModel from FakeMarrakesh
  'qesem'        -- Qedma QESEM on real IBM backend
"""

import numpy as np
import json, os, time, uuid, pickle
from datetime import datetime

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator
from qiskit_ibm_catalog import QiskitFunctionsCatalog


# In[2]:


# ---------------------------------------------------------------------------
# Configuration — edit this section
# ---------------------------------------------------------------------------
BACKEND_MODE = 'qesem'   # 'statevector' | 'fake' | 'noise_model' | 'real' | 'qesem'
PKL_FILE     = None # 'MNIST 3_5 Noise Injection Sigma005 (J8)_1-1_current.pkl'
SAMPLE_ID    = 0               # which test sample to run
BACKEND_NAME = 'ibm_marrakesh' # used only in qesem mode
INSTANCE     = '<YOUR_INSTANCE_NAME>'
CRN_INSTANCE = '<YOUR_CRN_INSTANCE>'
N_SHOTS      = 1024
PRECISION    = 0.05            # QESEM target precision (absolute, default 0.02)

# Note: CHECKPOINT_FILE is computed inside main() so it updates when SAMPLE_ID changes.

# ---------------------------------------------------------------------------
# Observables
#
# All 10 observables are submitted in a single job per step.
# QESEM handles incompatible bases (Z and X on same qubit) internally.
#
# Index mapping:
#   0: Z0  -- classification output
#   1: Z1  -- classification output
#   2: Z4  -- hidden[0] Z
#   3: X4  -- hidden[0] X
#   4: Z5  -- hidden[1] Z
#   5: X5  -- hidden[1] X
#   6: Z6  -- hidden[2] Z
#   7: X6  -- hidden[2] X
#   8: Z7  -- hidden[3] Z
#   9: X7  -- hidden[3] X
# ---------------------------------------------------------------------------
OBSERVABLES = [
    (0, 'Z', 'Z0'),
    (1, 'Z', 'Z1'),
    (4, 'Z', 'Z4'),
    (4, 'X', 'X4'),
    (5, 'Z', 'Z5'),
    (5, 'X', 'X5'),
    (6, 'Z', 'Z6'),
    (6, 'X', 'X6'),
    (7, 'Z', 'Z7'),
    (7, 'X', 'X7'),
]


# In[3]:


# ---------------------------------------------------------------------------
# Subfunctions
# ---------------------------------------------------------------------------
def make_obs(qubit, pauli, n=10):
    s = ['I'] * n
    s[n-1-qubit] = pauli
    return SparsePauliOp(''.join(s))

def make_obs_list():
    return [make_obs(q, p) for q, p, _ in OBSERVABLES]

def get_obs_names():
    return [name for _, _, name in OBSERVABLES]

def assemble_outputs(evs_dict):
    """
    Assemble classification output and hidden_out from a single job result.

    evs_dict : dict {obs_name: ev}

    Returns
    -------
    output     : np.ndarray, shape (2,)  -- [Z0, Z1]
    hidden_out : np.ndarray, shape (8,)  -- [Z4, X4, Z5, X5, Z6, X6, Z7, X7]
    """
    output = np.array([evs_dict['Z0'], evs_dict['Z1']])
    hidden_out = np.array([
        evs_dict['Z4'], evs_dict['X4'],
        evs_dict['Z5'], evs_dict['X5'],
        evs_dict['Z6'], evs_dict['X6'],
        evs_dict['Z7'], evs_dict['X7'],
    ])
    return output, hidden_out


# In[4]:


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------
def build_qru_circuit(data_input, hidden_input, params):
    data   = QuantumRegister(4, 'data')
    hidden = QuantumRegister(4, 'hidden')
    anc    = QuantumRegister(2, 'anc')
    qc     = QuantumCircuit(data, hidden, anc)

    qc.barrier(label='ENC')
    for i in range(4):
        qc.rx(data_input[2*i]     * params[4*i]     + params[4*i+1],    data[i])
        qc.ry(data_input[2*i+1]   * params[4*i+2]   + params[4*i+3],    data[i])
    for i in range(4):
        qc.rx(hidden_input[2*i]   * params[16+4*i]  + params[16+4*i+1], hidden[i])
        qc.ry(hidden_input[2*i+1] * params[16+4*i+2]+ params[16+4*i+3], hidden[i])
    for i in range(4):
        qc.cx(data[i],   data[(i+1) % 4])
        qc.cx(hidden[i], hidden[(i+1) % 4])

    qc.barrier(label='UPD')
    qc.cx(data[0], hidden[0])
    qc.cswap(hidden[0], hidden[1], anc[0])
    qc.rz(params[32], hidden[0])
    qc.ry(params[33], hidden[0])
    qc.rz(params[34], hidden[0])
    qc.cswap(hidden[0], hidden[3], anc[1])

    idx   = 35
    all_q = [data[i] for i in range(4)] + [hidden[i] for i in range(4)]
    for layer in range(4):
        qc.barrier(label=f'VAR{layer+1}')
        for reg in [data, hidden]:
            for i in range(4):
                qc.rz(params[idx], reg[i]); idx += 1
                qc.ry(params[idx], reg[i]); idx += 1
                qc.rz(params[idx], reg[i]); idx += 1
        for i in range(8):
            qc.cx(all_q[i], all_q[(i+1) % 8])

    return qc
    


# In[5]:


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------
def setup_backend():
    if BACKEND_MODE == 'statevector':
        return None, None, StatevectorEstimator()

    elif BACKEND_MODE == 'fake':
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit_ibm_runtime import EstimatorV2 as Estimator
        backend   = AerSimulator.from_backend(FakeMarrakesh())
        pm        = generate_preset_pass_manager(backend=backend, optimization_level=1)
        estimator = Estimator(mode=backend)
        estimator.options.default_shots = N_SHOTS
        return backend, pm, estimator

    elif BACKEND_MODE == 'noise_model':
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit_ibm_runtime import EstimatorV2 as Estimator
        backend   = AerSimulator(noise_model=NoiseModel.from_backend(FakeMarrakesh()))
        pm        = generate_preset_pass_manager(backend=backend, optimization_level=1)
        estimator = Estimator(mode=backend)
        estimator.options.default_shots = N_SHOTS
        return backend, pm, estimator


    elif BACKEND_MODE == 'real':
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
        from qiskit_ibm_runtime.options import EstimatorOptions
        service   = QiskitRuntimeService(name="NTU")
        backend   = service.backend(BACKEND_NAME, instance=INSTANCE)
        pm        = generate_preset_pass_manager(backend=backend, optimization_level=1)
        options   = EstimatorOptions()
        options.default_shots = N_SHOTS
        estimator = Estimator(mode=backend, options=options)
        return backend, pm, estimator
    elif BACKEND_MODE == 'qesem':
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_catalog import QiskitFunctionsCatalog
        service  = QiskitRuntimeService(name="NTU")
        backend  = service.backend(BACKEND_NAME, instance=INSTANCE)
        pm       = generate_preset_pass_manager(backend=backend, optimization_level=1)
        catalog  = QiskitFunctionsCatalog(channel="ibm_quantum_platform", instance=CRN_INSTANCE)
        qesem_fn = catalog.load('qedma/qesem')
        return backend, pm, qesem_fn

    raise ValueError(f"Unknown BACKEND_MODE: {BACKEND_MODE}")


# In[6]:


# ---------------------------------------------------------------------------
# Single step execution
# ---------------------------------------------------------------------------
def run_step(qc_or_isa, executor, ckpt_step):
    """
    Run all observables for one time step in a single job.
    Handles QESEM job_id recovery on resume.
    Updates ckpt_step in-place.
    Returns dict {obs_name: {'ev': float, 'std': float}}
    """
    names    = get_obs_names()
    obs_list = make_obs_list()

    if BACKEND_MODE == 'statevector':
        evs  = executor.run([(qc_or_isa, obs_list)]).result()[0].data.evs
        stds = [0.0] * len(names)
        ckpt_step.update({
            'job_id': str(uuid.uuid4()), 'backend_mode': BACKEND_MODE,
            'submitted_at': datetime.now().isoformat(), 'qpu_time_s': 0,
        })

    elif BACKEND_MODE in ('fake', 'noise_model', 'real'):
        isa_obs = [o.apply_layout(qc_or_isa.layout) for o in obs_list]
        result  = executor.run([(qc_or_isa, isa_obs)]).result()[0]
        evs     = result.data.evs
        stds    = list(result.data.stds) if hasattr(result.data, 'stds') else [0.0]*len(names)
        ckpt_step.update({
            'job_id': str(uuid.uuid4()), 'backend_mode': BACKEND_MODE,
            'submitted_at': datetime.now().isoformat(), 'qpu_time_s': 0,
        })

    elif BACKEND_MODE == 'qesem':
        from qiskit_ibm_catalog import QiskitFunctionsCatalog
        isa_obs = [o.apply_layout(qc_or_isa.layout) for o in obs_list]

        # Resume: job already submitted but result not yet saved
        if 'job_id' in ckpt_step and 'observables' not in ckpt_step:
            print(f"  Resuming QESEM job {ckpt_step['job_id']} ...")
            catalog = QiskitFunctionsCatalog(channel="ibm_quantum_platform", instance=CRN_INSTANCE)
            job     = catalog.job(ckpt_step['job_id'])
        else:
            job = executor.run(
                pubs         = [(qc_or_isa, isa_obs)],
                backend_name = BACKEND_NAME,
                instance     = CRN_INSTANCE,
                options      = {'default_precision': PRECISION},
            )
            print(f"  job_id: {job.job_id}")
            # Save job_id immediately before calling .result() (crash-safe)
            ckpt_step.update({
                'job_id': job.job_id, 'backend_mode': BACKEND_MODE,
                'submitted_at': datetime.now().isoformat(),
                'precision': PRECISION,
            })

        result = job.result()
        evs    = [result[0].data.evs[i] for i in range(len(isa_obs))]
        stds   = list(result[0].data.stds) if hasattr(result[0].data, 'stds') else [0.0]*len(names)
        meta   = result[0].metadata if hasattr(result[0], 'metadata') else {}
        ckpt_step.update({
            'completed_at': datetime.now().isoformat(),
            'qpu_time_s'  : meta.get('total_qpu_time', None),
        })

    ckpt_step['observables'] = {
        name: {'ev': float(ev), 'std': float(std)}
        for name, ev, std in zip(names, evs, stds)
    }
    return ckpt_step['observables']


# In[7]:


# ---------------------------------------------------------------------------
# QPU time estimation
# ---------------------------------------------------------------------------
def estimate_qpu_time(est_mode, sample_id=0):
    """
    est_mode = 'analytical'  不會用 QPU
    est_mode = 'empirical'   會用到 QPU
    Run QESEM for a single step to estimate QPU time per step.
    Uses current PRECISION and BACKEND_NAME settings.
    Returns dict with qpu_time_s and ev results.
    """
    import time
    assert BACKEND_MODE == 'qesem', "estimate_qpu_time() only works with BACKEND_MODE='qesem'"

    with open(PKL_FILE, 'rb') as f:
        data = pickle.load(f)
    params       = np.array(data[14])
    test_dataset = data[23]
    inp_all, _   = zip(*test_dataset)
    data_input   = np.array(inp_all)[sample_id][0]   # step 0
    hidden_input = np.zeros(8)

    _, pm, qesem_fn = setup_backend()
    qc     = build_qru_circuit(data_input, hidden_input, params)
    isa_qc = pm.run(qc)
    isa_obs = [o.apply_layout(isa_qc.layout) for o in make_obs_list()]

    print(f"Running QESEM time estimation: precision={PRECISION}, backend={BACKEND_NAME}")
    t0  = time.time()
    job = qesem_fn.run(
        pubs         = [(isa_qc, isa_obs)],
        backend_name = BACKEND_NAME,
        instance     = CRN_INSTANCE,
        # options      = {'default_precision': PRECISION},  # 真正跑一個 step
        # options = {'estimate_time_only': 'empirical', 'default_precision': PRECISION},  # 會用到 QPU
        # options      = {'estimate_time_only': 'analytical', 'default_precision': PRECISION},  # 不會用 QPU
        options      = {'estimate_time_only': est_mode, 'default_precision': PRECISION},  # 不會用 QPU
    )
    print(f"  job_id: {job.job_id}")
    result       = job.result()
    wall_s       = time.time() - t0
    meta         = result[0].metadata if hasattr(result[0], 'metadata') else {}
    t_estimation = meta.get('time_estimation_sec', None)
    qpu_s        = meta.get('total_qpu_time', None)
    print(f"  Precision            : {PRECISION}")
    print(f"  QPU consumed         : {qpu_s}s")
    print(f"  Est. time / step     : {t_estimation}s")
    print(f"  Wall time            : {wall_s/60:.1f} min")
    return {'precision': PRECISION, 'qpu_consumed_s': qpu_s, 'estimation_s': t_estimation, 'wall_s': wall_s, 'result': result}


# In[8]:


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None

def save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)


# In[9]:


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global CHECKPOINT_FILE
    safe_name       = PKL_FILE[:-4].replace(' ', '_').replace('(', '').replace(')', '')
    CHECKPOINT_FILE = f'checkpoints/{safe_name}/checkpoint_s{SAMPLE_ID}_{BACKEND_MODE}_p{PRECISION}.json'
    os.makedirs(f'checkpoints/{safe_name}', exist_ok=True)

    # Load parameters and dataset
    with open(PKL_FILE, 'rb') as f:
        data = pickle.load(f)

    params       = np.array(data[14])   # vbest_params
    test_dataset = data[23]             # list of (input, label) tuples
    inp_all, tar_all = zip(*test_dataset)
    inp_all = np.array(inp_all)
    tar_all = np.array(tar_all)

    sample     = inp_all[SAMPLE_ID]     # shape (8, 8)
    true_label = int(tar_all[SAMPLE_ID])

    print(f"Sample ID  : {SAMPLE_ID}")
    print(f"True label : {true_label}")
    print(f"Backend    : {BACKEND_MODE}")
    print(f"Checkpoint : {CHECKPOINT_FILE}")

    # Setup backend
    _, pm, executor = setup_backend()

    # Load or init checkpoint
    ckpt = load_checkpoint()
    if ckpt is None:
        ckpt = {
            'sample_id'  : SAMPLE_ID,
            'true_label' : true_label,
            'backend'    : BACKEND_MODE,
            'params_file': PKL_FILE,
            'created_at' : datetime.now().isoformat(),
            'steps'      : {},
        }
        save_checkpoint(ckpt)
        print("New checkpoint created.")
    else:
        completed = sum(1 for s in ckpt['steps'].values() if 'hidden_out' in s)
        print(f"Resuming — {completed}/8 steps already complete.")

    # Inference loop
    hidden_input = np.zeros(8)
    st = time.time()

    for step in range(8):
        step_key   = str(step)
        data_input = sample[step]

        # Restore hidden_input from previous completed step
        if step > 0:
            prev = ckpt['steps'].get(str(step-1), {})
            if 'hidden_out' not in prev:
                print(f"Step {step-1} incomplete — stopping. Run again to resume.")
                return None
            hidden_input = np.array(prev['hidden_out'])

        elapsed = time.time() - st
        print(f"\n[Step {step+1}/8]  elapsed={elapsed:.0f}s")

        # Skip if already complete
        if step_key in ckpt['steps'] and 'hidden_out' in ckpt['steps'][step_key]:
            print(f"  Already complete, skipping.")
            continue

        # Init step entry
        if step_key not in ckpt['steps']:
            ckpt['steps'][step_key] = {
                'data_input'       : data_input.tolist(),
                'hidden_input_used': hidden_input.tolist(),
            }
            save_checkpoint(ckpt)

        step_ckpt = ckpt['steps'][step_key]

        # Build and transpile circuit
        qc     = build_qru_circuit(data_input, hidden_input, params)
        isa_qc = qc if BACKEND_MODE == 'statevector' else pm.run(qc)

        # Run all observables in a single job
        if 'observables' in step_ckpt:
            print(f"  Observables already done, skipping.")
            obs_result = {k: v['ev'] for k, v in step_ckpt['observables'].items()}
        else:
            print(f"  Running {len(OBSERVABLES)} observables ...")
            obs_result_full = run_step(isa_qc, executor, step_ckpt)
            save_checkpoint(ckpt)
            obs_result = {k: v['ev'] for k, v in obs_result_full.items()}
            print(f"  Done: {obs_result}")

        # Assemble output and hidden state
        output, hidden_out = assemble_outputs(obs_result)

        step_ckpt['output']     = output.tolist()
        step_ckpt['hidden_out'] = hidden_out.tolist()
        save_checkpoint(ckpt)

        print(f"  Output: Z0={output[0]:.4f}  Z1={output[1]:.4f}")
        print(f"  Hidden: {[f'{v:.3f}' for v in hidden_out]}")

        # break; # 先只計算第一步

    # Final result
    last = ckpt['steps'].get('7', {})
    if 'output' in last:
        final_output = np.array(last['output'])
        prediction   = int(np.argmax(final_output))
        correct      = prediction == true_label
        elapsed      = time.time() - st

        ckpt.update({
            'final_output': final_output.tolist(),
            'prediction'  : prediction,
            'correct'     : correct,
            'elapsed_s'   : elapsed,
            'finished_at' : datetime.now().isoformat(),
        })
        save_checkpoint(ckpt)

        print(f"\n{'='*45}")
        print(f"Prediction : {prediction}  (true: {true_label})  {'✓' if correct else '✗'}")
        print(f"Output     : Z0={final_output[0]:.4f}  Z1={final_output[1]:.4f}")
        print(f"Elapsed    : {elapsed/60:.1f} min")
        return correct
    else:
        print("\nInference incomplete — run again to resume.")
        return None


# In[ ]:





# In[ ]:






# In[10]:


PKL_FILE = 'MNIST 3_5 Noise Injection Sigma005 (J8)_1-1_current.pkl'


# In[18]:


if __name__ == '__main__':
    main()


