#!/usr/bin/env python
# coding: utf-8
#
# lit_vqc_common.py
# ---------------------------------------------------------------------------
# Small, explicit ansatz-layer helper for the literature-faithful QGRU/QLSTM
# comparison models (QGRU_lit_j8.py, QLSTM_lit_j8.py, QLSTM_ces_lit_j8.py).
#
# Deliberately NOT built on top of VQC_j8.py's temp_Variation: that class
# hardcodes a fixed rotate-then-entangle order, which happens to match
# Ceschini et al.'s design but not Chen et al.'s (whose circuit diagram
# shows entangle-then-rotate). Reusing a shared QRU utility with an
# implicit fixed order is what caused that mismatch to go unnoticed in an
# earlier version of this code. This module makes the order an explicit,
# required argument instead, so each paper's design has to be checked and
# stated deliberately rather than inherited by accident.
# ---------------------------------------------------------------------------

import pennylane as qml
import jax.numpy as jnp


def circular_cnot_entangle(wires):
    """Circular CNOT entanglement: each qubit -> next qubit, wrapping last -> first.
    Matches Ceschini et al.'s and Chen et al.'s description of circular/ring
    entanglement. No-op for a single-qubit circuit."""
    n = len(wires)
    if n < 2:
        return
    for q1, q2 in zip(wires[:-1], wires[1:]):
        qml.CNOT(wires=[q1, q2])
    qml.CNOT(wires=[wires[-1], wires[0]])


def rotation_layer(weights, wires, gate):
    """Single-qubit rotation on every wire. gate: 'rx' (1 param/qubit) or
    'rot' (3 params/qubit, general Rz-Ry-Rz rotation)."""
    wi = 0
    for w in wires:
        if gate == 'rx':
            qml.RX(weights[wi], wires=w)
            wi += 1
        elif gate == 'rot':
            qml.Rot(weights[wi], weights[wi + 1], weights[wi + 2], wires=w)
            wi += 3
        else:
            raise ValueError(f"unsupported gate '{gate}'")
    return wi


def num_req_params(n_qubits, gate):
    return n_qubits if gate == 'rx' else 3 * n_qubits if gate == 'rot' else None


def ansatz_layer(weights, wires, gate, order):
    """One ansatz layer: rotation + circular CNOT entanglement, in the
    specified order.
    order: 'rotate_then_entangle' (Ceschini et al.) or
           'entangle_then_rotate' (Chen et al.)."""
    if order == 'rotate_then_entangle':
        rotation_layer(weights, wires, gate)
        circular_cnot_entangle(wires)
    elif order == 'entangle_then_rotate':
        circular_cnot_entangle(wires)
        rotation_layer(weights, wires, gate)
    else:
        raise ValueError(f"unsupported order '{order}'")
