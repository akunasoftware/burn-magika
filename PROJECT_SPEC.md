# magika-burn — Project Spec

## Goal

Reimplement the inference layer of Google Magika using Burn instead of ONNX Runtime.

Primary objectives:
- Eliminate ONNX Runtime dependency (`ort`)
- Enable portable inference across:
  - CPU (baseline)
  - WebGPU (primary target)
- Maintain compatibility with Magika model outputs and behavior

This is not a fork — it is a clean, runtime-agnostic reimplementation.

## Guiding Principle

Do not change the model.

Only change:
- how it is executed
- where it runs
