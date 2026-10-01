# 10. Streaming / Embedded Relevance

*Legend: **[M]** = measured in this project's Python/PyTorch experiment. **[D]** = derived mathematically
from measured or architectural quantities, no new experiment run. **[H]** = hypothetical/illustrative,
not implemented or tested here. **[F]** = future hardware work; not measured.*

## 10.1 What one window costs today (Model C)

Model C [M] processes one 6x128 window (2.56 s at 50 Hz [M]) as follows:

| Layer | Output shape | Kernel/stride/groups | Local RF (samples) | MACs | Params |
|---|---|---|---|---|---|
| Conv1d | 24x128 | k=9, s=1, g=1 | 9 | 165,888 | 1,320 |
| BatchNorm | 24x128 | - | 9 | 0 | 48 |
| ReLU | 24x128 | - | 9 | 0 | 0 |
| MaxPool1d | 24x64 | k=2, s=2 | 10 | 0 | 0 |
| Conv1d (depthwise) | 24x64 | k=9, s=1, g=24 | 26 | 13,824 | 240 |
| Conv1d (pointwise) | 48x64 | k=1, s=1, g=1 | 26 | 73,728 | 1,200 |
| BatchNorm | 48x64 | - | 26 | 0 | 96 |
| ReLU | 48x64 | - | 26 | 0 | 0 |
| MaxPool1d | 48x32 | k=2, s=2 | 28 | 0 | 0 |
| Conv1d (depthwise) | 48x32 | k=9, s=1, g=48 | 60 | 13,824 | 480 |
| Conv1d (pointwise) | 96x32 | k=1, s=1, g=1 | 60 | 147,456 | 4,704 |
| BatchNorm | 96x32 | - | 60 | 0 | 192 |
| ReLU | 96x32 | - | 60 | 0 | 0 |
| Global Average Pool | 96x1 | - | (see 10.1.1) | 0 | 0 |
| Linear | 6 | 96->6 | - | 576 | 582 |
| **Total** | | | **60** | **415,296** | **8,862** |

All figures **[M]**, recomputed directly from `src/architectures.py`/`src/shape_calc.py` for this report
and reconciled exactly against the frozen Step 3 numbers: 8,862 params, 415,296 MACs/window, local RF 60
samples, peak activation 12.00 KB (the first Conv1d's 24x128 output). No discrepancy found.

**10.1.1 Local RF vs. GAP coverage.** The 60-sample figure is the *local* receptive field: the temporal
context that a single feature at the last conv layer depends on. Global Average Pooling then aggregates
across all 32 remaining temporal positions (stride/jump 4 samples apart), so the classifier's effective
information horizon is the **entire 128-sample window** [D] -- not 60 samples. Both numbers matter for
different purposes: 60 samples is what a streaming, per-feature computation would need buffered; 128
samples is what the current whole-window classifier actually looks at.

**10.1.2 Storage.** float32 weights: 34.62 KB [M]. Peak float32 activation: 12.00 KB [M] (a Python/PyTorch
execution figure -- see 10.6 for why an optimized streaming implementation's memory profile would differ).

## 10.2 Compute rate under the existing 50%-overlap scheme

UCI HAR's native windowing hops by 64 samples (50% overlap) [M], i.e. every 1.28 s [D]:

- Classifications/s = 1 / hop_duration = 1 / 1.28 s = **0.781 Hz** [D] (one classification every 1.28 s)
- MACs/s = MACs/window / hop_duration = 415,296 / 1.28 s = **324,450 MAC/s** (0.324 MMAC/s) [D]
- Raw input rate: 6 channels x 50 Hz = 300 values/s [D]
- float32 input bandwidth: 300 x 4 B = 1,200 B/s = **9.6 kbps** [D]
- int16 input bandwidth (illustrative sensor format, not used in this project): 300 x 2 B = 600 B/s =
  **4.8 kbps** [H]

These are throughput figures for the current recompute-the-whole-window scheme. They are **not** MCU
cycle counts, wall-clock latency, or energy figures -- those require real hardware measurement (10.9).

## 10.3 Why window inference is not yet streaming

The current implementation buffers a full 128-sample window before running any computation, then
classifies it as one shot. Three consequences follow directly from this:

1. **Latency floor.** No classification can be produced before 2.56 s of data has arrived, regardless of
   how fast the compute itself runs.
2. **Redundant computation.** Each new window shares 64 of its 128 samples with the previous window, so
   roughly half of every convolution's input is data the network has already processed once.
3. **Non-incremental.** The whole pipeline re-runs from the first sample of the window every time, rather
   than updating an existing internal state as new samples arrive.

A genuinely *streaming* implementation would instead maintain persistent state (buffers/accumulators) and
update it sample-by-sample or chunk-by-chunk, emitting a decision (or an updated confidence) with lower
and more predictable latency, without repeating the convolution work already done on old samples. The
current network demonstrates that a compact CNN on raw IMU signals reaches strong accuracy -- it is not
yet an optimized streaming implementation of that CNN, and no streaming kernel has been written or tested.

## 10.4 Per-operation streaming feasibility (analysis only -- not implemented)

| Operation | Streaming feasibility | State required |
|---|---|---|
| Conv1d (k=9, stride 1) | Feasible: standard sliding-window/ring-buffer convolution | k-1 = 8 past samples per input channel |
| Depthwise Conv1d (k=9, groups=C) | Feasible, and cheaper: each output channel only needs its own ring buffer, no cross-channel mixing | 8 past samples per channel, independent buffers |
| Pointwise Conv1d (1x1) | Trivially streaming: a per-timestep matrix-vector product, no temporal state at all | None (memoryless in time) |
| BatchNorm (inference) | Can be folded into the preceding conv's weights/bias ahead of time (affine transform using fixed running mean/var) [D, standard technique, not yet applied in this codebase] | None once folded |
| ReLU | Trivially streaming (pointwise) | None |
| MaxPool/downsampling (k=2, s=2) | Feasible: halves the output rate; needs to hold one extra sample to pair before emitting | 1 sample |
| Global Average Pooling | Convertible to a running sum + count, rather than storing the full 32-position feature map | One accumulator (+ count) per channel |
| Linear classifier | Applied once per completed aggregation (window or its streaming equivalent) | None beyond the GAP accumulator |

**This section is analysis, not implementation.** None of these transformations (buffer-based conv,
BN folding, running-sum GAP) exist in the current PyTorch training/evaluation code; they describe what a
future C/streaming port would need to do, and each would need to be verified against the trained
PyTorch model's output before being trusted (see Section 12, item 2).

## 10.5 Reuse across overlapping windows

Because consecutive windows overlap by 64 of 128 samples, a persistent-buffer convolution implementation
would, in principle, avoid recomputing convolutional features for samples it has already convolved in an
earlier window -- only the newly arrived 64 samples' contribution to each conv layer would need fresh
computation, with old ring-buffer contents reused. This is a plausible, standard streaming-CNN argument
[D], **not something we have implemented or measured** here.

Two things complicate a casual "50% of the compute is free" claim, and should not be glossed over:

- **Global Average Pooling is window-specific.** Even if local convolutional features are reused, the
  *classification decision* for a given window depends on GAP-and-classify over that window's own
  32-position feature map. A running-accumulator GAP (10.4) would need careful design to represent
  "the average over the current window" rather than "the average over all time seen so far" -- these are
  different quantities, and only the former matches what the trained model was trained to expect.
- **Boundary correctness.** "Same" padding (10.7) means positions near a window's edges see zero-padding
  in the current implementation; a streaming buffer replacing that edge behavior with real neighbouring
  samples from the adjacent window would change those features' values, not just their timing. Any
  streaming redesign must be validated against the reference PyTorch forward pass, sample range by sample
  range, before its outputs can be trusted to match the trained model (see Section 12, item 2).

## 10.6 Wake-up network interpretation

The research assistantship's core question -- can a small, cheap detector decide whether the *main* classifier is
worth waking up -- is indirectly supported by one measured statistic: **only 6 of 333 test-set errors
(1.8%) crossed the static/dynamic boundary** [M], versus 58.9% within-static and 39.3% within-dynamic
confusions [M]. This suggests the coarse "is something dynamic happening or not" distinction is
substantially easier than fine-grained six-way classification.

This is evidence toward a plausible design, not proof of one. Two important caveats:

- Model C was trained and evaluated as a **six-class** classifier. It has never been trained, evaluated,
  or even run as a binary static/dynamic detector; the 1.8% figure is a *post-hoc property* of a 6-class
  model's confusion pattern, not a measurement of a purpose-built wake-up detector's accuracy.
- No energy or duty-cycle claim can be made without knowing (a) how much of real-world operating time is
  actually static and (b) the real energy cost of the always-on wake-up stage vs. the occasionally-woken
  main classifier -- both require hardware/duty-cycle data we do not have [F].

**What would need to be tested next:** train and evaluate a dedicated binary (static-vs-dynamic, or
event/no-event) objective directly, measure its accuracy and false-wake rate on its own terms, and only
then estimate energy savings against a measured activity duty cycle and per-inference energy cost.

## 10.7 A vs. C: an embedded Pareto question, not a re-selection

C was selected under the pre-registered validation-macro-F1 rule (Phase 1: A 0.9890 +/- 0.0006, B 0.9813
+/- 0.0007, C 0.9898 +/- 0.0005 [M]) and is the model this project's final test result (0.9240 accuracy,
0.9231 macro-F1 [M], 17-epoch checkpoint) refers to. That selection is not being revisited here.

Purely as a forward-looking embedded question, A is worth noting:

| | A | C | Change (A -> C) |
|---|---|---|---|
| Params | 3,766 | 8,862 | **+135.3%** [D] |
| MACs/window | 274,624 | 415,296 | **+51.2%** [D] |
| Local RF | 360 ms | 1,200 ms | +233% |
| Validation macro-F1 | 0.9890 | 0.9898 | +0.0008 |

C costs 135% more parameters and 51% more compute than A [D] for a validation macro-F1 improvement of
0.0008 -- smaller than either model's own Phase-1 seed-to-seed standard deviation (0.0005-0.0006 [M]).
Under the pre-registered rule C is still the correct selection (the rule compares means, and C's mean is
higher), but for a low-power *wake-up* role specifically -- where A's smaller compute and shorter
receptive field are direct advantages -- A deserves its own future investigation as a Pareto-efficient
alternative. This is **not** a proposal to re-select the final classifier: A has not been, and will not
be here, evaluated on the official test set.

## 10.8 Quantization: a storage estimate, nothing more

Naive int8 storage (1 byte/parameter, no scale/zero-point metadata, no BN folding accounted for):

- Model C: 8,862 params -> **8.65 KB** (vs. 34.62 KB float32; a 4x / 75% reduction) [D]
- Model A: 3,766 params -> **3.68 KB** (vs. 14.71 KB float32) [D]

This is a storage arithmetic exercise only. **We make no claim about int8 accuracy, latency, or energy**
-- none of these have been measured, and quantization (post-training or quantization-aware) has not been
performed. Both are natural next experiments (Section 10.10).

## 10.9 Conceptual X-HEEP / DMA / FIFO mapping (future work, not implemented)

```
sensor -> DMA/FIFO -> small sample/chunk buffer -> incremental Conv1D pipeline
        -> activation/downsampling -> incremental aggregation
        -> wake-up/classification decision -> interrupt to main processor
```

Operations that could plausibly overlap DMA reception of the *next* chunk, in principle: convolution on
already-arrived samples, activation, and partial pooling/accumulation on completed portions of a chunk
[D, standard embedded-DSP reasoning]. Whether this overlap is actually achievable, and by how much,
depends entirely on hardware not yet measured: **cycle counts per layer, real DMA-vs-compute overlap,
pipeline stalls, SRAM read/write traffic, energy per operation, and wake-up latency from a physical event
to an interrupt** [F]. None of these numbers exist yet for this project.

## 10.10 Memory: four different things that are not interchangeable

- **Weights**: 34.62 KB float32 [M] (8.65 KB int8, estimate only [D]) -- fixed, read-only at inference.
- **Persistent streaming state** (ring buffers for each conv's k-1 history, GAP accumulator): not
  designed or sized yet; would depend on the specific streaming kernel implementation [F].
- **Intermediate activations**: 12.00 KB peak, measured from the Python/PyTorch whole-window forward pass
  [M]. A streaming C implementation would very likely **not** need to store a full 24x128 (or any full)
  feature map at once -- it would process and discard samples incrementally -- but it would need the
  ring-buffer state above instead. The Python peak-activation figure is therefore an upper bound on
  *whole-window* execution, not a prediction of a streaming implementation's memory footprint.
- **Input buffering / stack / runtime overhead**: not estimated here; genuinely implementation-dependent [F].

## 10.11 Latency: three numbers that are not "the" latency

- **Full window duration**: 2.56 s [M] -- how much data the current implementation waits for.
- **Hop duration**: 1.28 s [D] -- how often a new classification becomes available under the current
  50%-overlap scheme.
- **Local receptive field**: 1.20 s [M] -- the temporal context a single pre-GAP feature depends on.

None of these is a hardware inference latency. That number requires measuring actual compute time on
target hardware [F], which has not been done.

One implementation detail matters for any future causal/streaming redesign, and we checked it directly
in the code rather than assuming: **every convolution in this model uses symmetric ("same") zero-padding**
(`src/architectures.py`: `padding = (kernel - 1) // 2`, confirmed applied in `src/model.py`'s
`nn.Conv1d(..., padding=op.padding, ...)`). This means the trained network is **non-causal** -- each
output position depends on input samples both before *and after* it in time, via padding. A causal
streaming redesign (left-padding only, or no padding with a pure sliding buffer) would shift each layer's
effective alignment in time and is not simply "the same network, computed incrementally" -- it is a
related but distinct computation whose outputs would need to be validated against the trained model
before being trusted (Section 10.5, Section 12 item 3).

## 10.12 Prioritized next steps

1. Implement equivalent NumPy or C streaming/ring-buffer kernels for Model C's operations (Section 10.4).
2. Verify sample-by-sample/chunked streaming outputs against the frozen PyTorch reference, layer by layer,
   before trusting any streaming result.
3. Investigate causal vs. the current "same"-padding behavior and its effect on alignment and RF (10.11).
4. Post-training int8 quantization; separately, quantization-aware training; measure real accuracy impact.
5. Train and evaluate a dedicated binary static-vs-dynamic (or event/no-event) wake-up objective on its
   own terms, rather than inferring one from the 6-class confusion pattern (10.6).
6. Benchmark both A and C (not just the selected model) on target MCU hardware, without reopening or
   retuning the frozen HAR model-selection/test result from Steps 1-5.
7. Measure real cycles, SRAM traffic, and energy per operation on that hardware.
8. Investigate overlapping convolution/activation compute with DMA sensor-chunk reception.
9. Profile to find the actual bottleneck operation before assuming which one it is.
10. Consider a custom accelerator only after (6)-(9) show a specific, measured bottleneck that justifies it.

## 10.13 Limitations of this section

This entire section is desk analysis connecting a completed, frozen ML experiment to a future embedded
implementation. No streaming kernel, quantized model, or hardware benchmark exists yet. Every number
carrying an **[M]** label traces to a specific frozen file from Steps 1-5; every **[D]** label is
arithmetic on those numbers, shown with its formula; every **[H]**/**[F]** label is explicitly not
measured and should not be read as a claim about real hardware behavior.
