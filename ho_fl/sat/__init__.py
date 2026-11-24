"""Satellite scenario for hybrid-order federated fine-tuning (brief v2: ``docs/RHO-FFT-agent-brief-v2.md``).

A Walker constellation of backbone (B), hybrid (H) and forward-only (Z) satellites fine-tunes one model without a
ground server: zeroth-order reports are flooded and replayed from seeds, first-order updates are summed along each
orbital plane and exchanged between the plane roots. Everything specific to that scenario lives in this package;
the modules of ``ho_fl`` it reuses keep their upstream behaviour.

Randomness in this package comes from ``ho_fl.randomness.stable_seed`` only, never from a global generator.
"""
