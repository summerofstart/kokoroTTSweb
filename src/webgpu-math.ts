/**
 * WebGPU Math Engine
 * ===================
 * Advanced mathematical optimization for Kokoro TTS using WebGPU compute shaders.
 *
 * Provides GPU-accelerated implementations of:
 *   - Pitch estimation (normalized autocorrelation with parallel reduction)
 *   - Audio analysis (RMS energy + zero-crossing rate in single-pass reduction)
 *   - Voice vector blending with optional RMS normalization
 *   - Audio peak/RMS normalization
 *
 * Design principles:
 *   - Shared singleton GPUContext avoids redundant adapter/device creation
 *   - Lazy initialisation with transparent CPU fallback
 *   - Workgroup-optimised WGSL shaders with tree-based parallel reduction
 *   - Zero-copy buffer management where possible
 *
 * Mathematical operations accelerated:
 *   - Autocorrelation:  R(τ) = Σ_{i=0}^{N-τ-1} x[i] · x[i+τ]
 *   - RMS:             x_rms = sqrt( Σ x[i]² / N )
 *   - ZCR:             zcr = count{ sign(x[i]) ≠ sign(x[i-1]) } / N
 *   - Normalised blend: y[i] = ((a[i]·(1-α) + b[i]·α) · g) · (target / max(rms, ε))
 */

// ─── GPU Buffer Usage Constants ───────────────────────────────────────────────
// Numeric values per WebGPU spec (GPUBufferUsage namespace).
// Using raw constants avoids a dependency on @webgpu/types.

const GPU_MAP_READ  = 0x0001;
const GPU_MAP_WRITE = 0x0002;
const GPU_COPY_SRC  = 0x0004;
const GPU_COPY_DST  = 0x0008;
const GPU_UNIFORM   = 0x0040;
const GPU_STORAGE   = 0x0080;
const GPU_INDIRECT  = 0x0100;

// ─── GPU Context (Singleton) ──────────────────────────────────────────────────

/**
 * Manages a shared WebGPU adapter and device instance.
 * The first call to `initialize()` lazily requests the adapter and device;
 * subsequent calls return the cached instance.
 */
export class GPUContext {
  private static _instance: GPUContext | null = null;

  private _adapter: GPUAdapter | null = null;
  private _device: GPUDevice | null = null;
  private _initPromise: Promise<boolean> | null = null;
  private _available = false;

  private constructor() {}

  /** Get the singleton instance */
  static get instance(): GPUContext {
    if (!GPUContext._instance) {
      GPUContext._instance = new GPUContext();
    }
    return GPUContext._instance;
  }

  /** The underlying GPUDevice (null if unavailable / not initialised) */
  get device(): GPUDevice | null {
    return this._device;
  }

  /** The underlying GPUAdapter (null if unavailable / not initialised) */
  get adapter(): GPUAdapter | null {
    return this._adapter;
  }

  /** Whether the GPU context is ready for use */
  get isAvailable(): boolean {
    return this._available;
  }

  /**
   * Initialise the GPU context. Safe to call multiple times — only the first
   * call actually performs work.
   *
   * @returns `true` when WebGPU is available and initialised.
   */
  async initialize(): Promise<boolean> {
    if (this._available) return true;
    if (this._initPromise) return this._initPromise;

    this._initPromise = this._init();
    return this._initPromise;
  }

  private async _init(): Promise<boolean> {
    try {
      if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
        return false;
      }

      const gpu = navigator.gpu as GPU;
      this._adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!this._adapter) return false;

      this._device = await this._adapter.requestDevice();

      if (this._device) {
        this._device.addEventListener('uncapturederror', (evt) => {
          const err = (evt as GPUUncapturedErrorEvent).error;
          console.warn('[WebGPU] Uncaptured error:', err.message);
        });
        this._available = true;
      }
      return this._available;
    } catch (err) {
      console.warn('[WebGPU] Initialisation failed:', err);
      return false;
    }
  }

  // ── Buffer helpers ───────────────────────────────────────────────────────

  /** Upload a Float32Array into a GPU storage buffer. */
  uploadBuffer(data: Float32Array): GPUBuffer | null {
    if (!this._device) return null;
    const byteLen = data.byteLength;
    const buf = this._device.createBuffer({
      size: byteLen,
      usage: GPU_STORAGE | GPU_COPY_DST | GPU_COPY_SRC,
    });
    this._device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer, data.byteOffset, byteLen);
    return buf;
  }

  /** Allocate a GPU storage buffer (uninitialised). */
  allocBuffer(byteLength: number): GPUBuffer | null {
    if (!this._device) return null;
    return this._device.createBuffer({
      size: byteLength,
      usage: GPU_STORAGE | GPU_COPY_DST | GPU_COPY_SRC,
    });
  }

  /** Create a uniform buffer from raw bytes (will be 16-byte aligned). */
  createUniformBuffer(data: ArrayBuffer): GPUBuffer | null {
    if (!this._device) return null;
    const aligned = Math.ceil(data.byteLength / 16) * 16;
    const buf = this._device.createBuffer({
      size: aligned,
      usage: GPU_UNIFORM | GPU_COPY_DST,
    });
    this._device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  /**
   * Dispatch a compute shader and read the result back to CPU.
   *
   * @param pipeline   - compiled compute pipeline
   * @param bindGroup  - bind group with all resources
   * @param wgCount    - number of workgroups to dispatch
   * @param outputBuf  - buffer written to by the shader (GPU_STORAGE | COPY_SRC)
   * @param byteLength - number of bytes to read back
   * @returns Float32Array with the result, or `null` on failure.
   */
  async dispatchAndRead(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    wgCount: number,
    outputBuf: GPUBuffer,
    byteLength: number,
  ): Promise<Float32Array | null> {
    const device = this._device;
    if (!device) return null;

    const readback = device.createBuffer({
      size: byteLength,
      usage: GPU_COPY_DST | GPU_MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuf, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);

    try {
      await readback.mapAsync(GPU_MAP_READ);
      const mapped = readback.getMappedRange();
      const result = new Float32Array(mapped.slice(0));
      readback.unmap();
      return result;
    } catch {
      return null;
    }
  }
}

// ─── WGSL Shader Sources ──────────────────────────────────────────────────────
// Each shader is accompanied by a description of its mathematical operation.

const SHADERS = {

  /* ───────────────────────────────────────────────────────────────────────
   * PITCH – Normalised autocorrelation
   *
   *   score[τ] = Σ_{i=0}^{N-τ-1} x[i] · x[i+τ]
   *
   * Each workgroup computes one lag τ using 256 threads for a parallel
   * dot-product with tree reduction.
   *
   * Dispatch:   ceil((maxLag - minLag + 1) / 256)
   * Workgroup:  256
   * ─────────────────────────────────────────────────────────────────────── */
  PITCH_AUTOCORRELATION: `
struct Params {
  frameOffset: u32,
  frameLength: u32,
  minLag: u32,
  maxLag: u32,
  sampleRate: f32,
};

@group(0) @binding(0) var<storage, read> audio: array<f32>;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let lagIndex = gid.x;
  let numLags = params.maxLag - params.minLag + 1u;
  if (lagIndex >= numLags) { return; }

  let lag = params.minLag + lagIndex;
  let N = params.frameLength;
  let offset = params.frameOffset;
  let total = N - lag;

  // Each thread processes a subset of the frame
  let itemsPerThread = (total + 255u) / 256u;
  let startIdx = lid.x * itemsPerThread;
  let endIdx = min(startIdx + itemsPerThread, total);

  var sum = 0.0;
  for (var i = startIdx; i < endIdx; i = i + 1u) {
    sum = sum + audio[offset + i] * audio[offset + i + lag];
  }

  // Tree reduction in shared memory
  shared[lid.x] = sum;
  workgroupBarrier();

  var reduceSize = 256u;
  while (reduceSize > 1u) {
    reduceSize = (reduceSize + 1u) / 2u;
    if (lid.x < reduceSize) {
      shared[lid.x] = shared[lid.x] + shared[lid.x + reduceSize];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    scores[lagIndex] = shared[0];
  }
}
`,

  /* ───────────────────────────────────────────────────────────────────────
   * AUDIO ANALYSIS – Single-pass RMS + ZCR
   *
   * For each workgroup (256 threads):
   *   sumSq = Σ x[i]²
   *   zcr   = count of sign changes
   *
   * Output: per-workgroup PartialResult, reduced in a second pass.
   *
   * Dispatch:   ceil(N / 256)
   * Workgroup:  256
   * ─────────────────────────────────────────────────────────────────────── */
  AUDIO_ANALYSIS: `
struct Params {
  length: u32,
  step: u32,
  _pad0: u32,
  _pad1: u32,
};

struct PartialResult {
  sumSq: f32,
  zcr: f32,
};

@group(0) @binding(0) var<storage, read> audio: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<PartialResult>;

var<workgroup> sharedSumSq: array<f32, 256>;
var<workgroup> sharedZcr: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let i = gid.x;
  if (i >= params.length) { return; }

  let sample = audio[i];
  var sumSq = sample * sample;
  var zcr = 0.0;

  // Zero-crossing detection
  if (i > 0u) {
    let prev = audio[i - 1u];
    if ((sample >= 0.0 && prev < 0.0) || (sample < 0.0 && prev >= 0.0)) {
      zcr = 1.0;
    }
  }

  sharedSumSq[lid.x] = sumSq;
  sharedZcr[lid.x] = zcr;
  workgroupBarrier();

  // Tree reduction for both quantities
  var reduceSize = 256u;
  while (reduceSize > 1u) {
    reduceSize = (reduceSize + 1u) / 2u;
    if (lid.x < reduceSize) {
      sharedSumSq[lid.x] = sharedSumSq[lid.x] + sharedSumSq[lid.x + reduceSize];
      sharedZcr[lid.x] = sharedZcr[lid.x] + sharedZcr[lid.x + reduceSize];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    let idx = wgid.x;
    partials[idx].sumSq = sharedSumSq[0];
    partials[idx].zcr = sharedZcr[0];
  }
}
`,

  /* ───────────────────────────────────────────────────────────────────────
   * AUDIO ANALYSIS – Final reduction
   *
   * Sums partials from all workgroups into a single result.
   * Dispatch: 1 workgroup
   * ─────────────────────────────────────────────────────────────────────── */
  AUDIO_ANALYSIS_FINAL: `
struct PartialResult {
  sumSq: f32,
  zcr: f32,
};

struct FinalResult {
  sumSq: f32,
  zcr: f32,
  length: f32,
};

@group(0) @binding(0) var<storage, read> partials: array<PartialResult>;
@group(0) @binding(1) var<storage, read_write> result: array<FinalResult>;

var<workgroup> sharedSumSq: array<f32, 256>;
var<workgroup> sharedZcr: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let numPartials = arrayLength(&partials);

  if (lid.x < numPartials) {
    sharedSumSq[lid.x] = partials[lid.x].sumSq;
    sharedZcr[lid.x] = partials[lid.x].zcr;
  } else {
    sharedSumSq[lid.x] = 0.0;
    sharedZcr[lid.x] = 0.0;
  }
  workgroupBarrier();

  var reduceSize = 256u;
  while (reduceSize > 1u) {
    reduceSize = (reduceSize + 1u) / 2u;
    if (lid.x < reduceSize) {
      sharedSumSq[lid.x] = sharedSumSq[lid.x] + sharedSumSq[lid.x + reduceSize];
      sharedZcr[lid.x] = sharedZcr[lid.x] + sharedZcr[lid.x + reduceSize];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    result[0].sumSq = sharedSumSq[0];
    result[0].zcr = sharedZcr[0];
  }
}
`,

  /* ───────────────────────────────────────────────────────────────────────
   * VOICE BLEND + RMS NORMALISATION (single pass)
   *
   * Phase 1:  y[i] = ((a[i]·(1-α)) + (b[i]·α)) · g
   * Phase 2:  compute rms = sqrt( Σ y[i]² / N )
   * Phase 3:  y[i] *= target / max(rms, ε)
   *
   * The normalisation uses a two-phase approach: first accumulate squared
   * values, then apply the scale factor.
   *
   * When normalize == 0, only phase 1 runs.
   *
   * Dispatch:   ceil(length / 256)
   * Workgroup:  256
   * ─────────────────────────────────────────────────────────────────────── */
  VOICE_BLEND_NORM: `
struct Params {
  mix: f32,
  gain: f32,
  length: u32,
  normalize: u32,
  targetRms: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<storage, read> voiceA: array<f32>;
@group(0) @binding(1) var<storage, read> voiceB: array<f32>;
@group(0) @binding(2) var<storage, read_write> outVoice: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = gid.x;
  if (i >= params.length) { return; }

  // Phase 1: blend
  let blended = ((voiceA[i] * (1.0 - params.mix)) + (voiceB[i] * params.mix)) * params.gain;
  outVoice[i] = blended;

  // Phase 2: accumulate squares for RMS (if normalisation enabled)
  if (params.normalize == 1u) {
    shared[lid.x] = blended * blended;
  }
  workgroupBarrier();

  // Phase 3: tree reduction for RMS
  if (params.normalize == 1u) {
    var reduceSize = 256u;
    while (reduceSize > 1u) {
      reduceSize = (reduceSize + 1u) / 2u;
      if (lid.x < reduceSize) {
        shared[lid.x] = shared[lid.x] + shared[lid.x + reduceSize];
      }
      workgroupBarrier();
    }

    // Thread 0 computes scale and broadcasts it via shared[0]
    if (lid.x == 0u) {
      let rms = sqrt(shared[0] / f32(params.length));
      let scale = params.targetRms / max(rms, 1e-8);
      shared[0] = scale;
    }
  }
  workgroupBarrier();

  // Phase 4: apply normalisation
  if (params.normalize == 1u) {
    let scale = shared[0];
    outVoice[i] = outVoice[i] * scale;
  }
}
`,

  /* ───────────────────────────────────────────────────────────────────────
   * AUDIO NORMALISATION (peak or RMS)
   *
   * Peak mode:  y[i] = x[i] · target / max(|x|)
   * RMS  mode:  y[i] = x[i] · target / sqrt( Σ x² / N )
   *
   * Dispatch:   ceil(N / 256)
   * Workgroup:  256
   * ─────────────────────────────────────────────────────────────────────── */
  AUDIO_NORMALIZE: `
struct Params {
  length: u32,
  target: f32,
  mode: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = gid.x;
  if (i >= params.length) { return; }

  let val = input[i];

  // Accumulate
  if (params.mode == 0u) {
    shared[lid.x] = abs(val);      // peak
  } else {
    shared[lid.x] = val * val;     // RMS
  }
  workgroupBarrier();

  // Tree reduction
  var reduceSize = 256u;
  while (reduceSize > 1u) {
    reduceSize = (reduceSize + 1u) / 2u;
    if (lid.x < reduceSize) {
      if (params.mode == 0u) {
        shared[lid.x] = max(shared[lid.x], shared[lid.x + reduceSize]);
      } else {
        shared[lid.x] = shared[lid.x] + shared[lid.x + reduceSize];
      }
    }
    workgroupBarrier();
  }

  // Thread 0 computes global scale
  if (lid.x == 0u) {
    var norm: f32;
    if (params.mode == 0u) {
      norm = max(shared[0], 1e-8);
    } else {
      norm = sqrt(max(shared[0] / f32(params.length), 1e-8));
    }
    shared[0] = params.target / norm;
  }
  workgroupBarrier();

  output[i] = val * shared[0];
}
`,

} as const;

// ─── Result Types ─────────────────────────────────────────────────────────────

export interface PitchResult {
  /** Estimated fundamental frequency in Hz (0 if indeterminate). */
  pitch: number;
  /** Normalised autocorrelation score in [0, 1]. */
  score: number;
}

export interface AudioAnalysisResult {
  pitch: number;
  rms: number;
  zcr: number;
  brightness: number;
}

// ─── WebGPU Math Engine ───────────────────────────────────────────────────────

export class WebGPUMath {
  private ctx: GPUContext;
  private pipelines = new Map<string, GPUComputePipeline>();
  private _ready = false;
  private _initPromise: Promise<boolean> | null = null;

  /**
   * @param ctx  An existing GPUContext, or leave undefined to use the default
   *             singleton.
   */
  constructor(ctx?: GPUContext) {
    this.ctx = ctx ?? GPUContext.instance;
  }

  /** Whether the underlying GPU is initialised and usable. */
  get ready(): boolean {
    return this._ready;
  }

  /** Ensure the GPU context is initialised. Safe to call frequently. */
  async ensureReady(): Promise<boolean> {
    if (this._ready) return true;
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._init();
    return this._initPromise;
  }

  private async _init(): Promise<boolean> {
    const ok = await this.ctx.initialize();
    this._ready = ok;
    return ok;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Estimate pitch via GPU-accelerated normalised autocorrelation.
   *
   * @param samples    Audio buffer (Float32Array).
   * @param sampleRate Sampling rate in Hz.
   * @param minPitch   Lowest detectable pitch (default 70 Hz).
   * @param maxPitch   Highest detectable pitch (default 320 Hz).
   */
  async estimatePitch(
    samples: Float32Array,
    sampleRate: number,
    minPitch = 70,
    maxPitch = 320,
  ): Promise<PitchResult> {
    if (!await this.ensureReady()) {
      return this._pitchCPU(samples, sampleRate, minPitch, maxPitch);
    }

    const device = this.ctx.device!;
    const frameLen = Math.min(4096, samples.length);
    const frameOff = Math.max(0, Math.floor((samples.length - frameLen) / 2));
    const frame = samples.subarray(frameOff, frameOff + frameLen);
    const minLag = Math.floor(sampleRate / maxPitch);
    const maxLag = Math.ceil(sampleRate / minPitch);
    const numLags = maxLag - minLag + 1;

    // Upload frame
    const audioBuf = this.ctx.uploadBuffer(frame);
    if (!audioBuf) return this._pitchCPU(samples, sampleRate, minPitch, maxPitch);

    const scoresBuf = this.ctx.allocBuffer(numLags * 4);
    if (!scoresBuf) return this._pitchCPU(samples, sampleRate, minPitch, maxPitch);

    // Uniforms (pack as u32 array then overlay f32 for the float member)
    const params = new Uint32Array(5);
    const pf = new Float32Array(params.buffer);
    params[0] = 0;           // frameOffset — we uploaded the trimmed frame
    params[1] = frameLen;    // frameLength
    params[2] = minLag;
    params[3] = maxLag;
    pf[4] = sampleRate;
    const paramBuf = this.ctx.createUniformBuffer(params.buffer);
    if (!paramBuf) return this._pitchCPU(samples, sampleRate, minPitch, maxPitch);

    const pipeline = await this._pipeline('pitch', SHADERS.PITCH_AUTOCORRELATION, device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: audioBuf } },
        { binding: 1, resource: { buffer: scoresBuf } },
        { binding: 2, resource: { buffer: paramBuf } },
      ],
    });

    const scores = await this.ctx.dispatchAndRead(
      pipeline, bindGroup, Math.ceil(numLags / 256), scoresBuf, numLags * 4,
    );
    if (!scores) return this._pitchCPU(samples, sampleRate, minPitch, maxPitch);

    // Find best lag
    let bestScore = -Infinity;
    let bestLag = 0;
    for (let i = 0; i < numLags; i++) {
      if (scores[i] > bestScore) {
        bestScore = scores[i];
        bestLag = minLag + i;
      }
    }

    // Normalise score using energy
    let energy = 0;
    for (let i = 0; i < frameLen; i++) energy += frame[i] * frame[i];
    const normScore = energy > 0 ? bestScore / energy : 0;

    return {
      pitch: bestLag > 0 ? sampleRate / bestLag : 0,
      score: Math.max(0, Math.min(1, normScore)),
    };
  }

  /**
   * Analyse audio in a single GPU pass: RMS energy + zero-crossing rate.
   *
   * @param samples    Audio buffer.
   * @param sampleRate Sampling rate in Hz.
   */
  async analyzeAudio(
    samples: Float32Array,
    sampleRate: number,
  ): Promise<AudioAnalysisResult> {
    if (!await this.ensureReady()) {
      return this._analyzeCPU(samples, sampleRate);
    }

    const device = this.ctx.device!;
    const N = samples.length;

    // Upload audio
    const audioBuf = this.ctx.uploadBuffer(samples);
    if (!audioBuf) return this._analyzeCPU(samples, sampleRate);

    // Per-workgroup partials
    const wgCount = Math.ceil(N / 256);
    const partialsBuf = device.createBuffer({
      size: wgCount * 8, // 2 × f32 per workgroup
      usage: GPU_STORAGE | GPU_COPY_DST | GPU_COPY_SRC,
    });

    // Pass 1 params
    const p1 = new Uint32Array(4);
    p1[0] = N;
    p1[1] = 1; // step
    const paramBuf = this.ctx.createUniformBuffer(p1.buffer);
    if (!paramBuf) return this._analyzeCPU(samples, sampleRate);

    const pipe1 = await this._pipeline('ap1', SHADERS.AUDIO_ANALYSIS, device);
    const bg1 = device.createBindGroup({
      layout: pipe1.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: audioBuf } },
        { binding: 1, resource: { buffer: partialsBuf } },
      ],
    });

    // Result buffer: 3 × f32
    const resultBuf = device.createBuffer({
      size: 12,
      usage: GPU_STORAGE | GPU_COPY_DST | GPU_COPY_SRC,
    });

    const pipe2 = await this._pipeline('ap2', SHADERS.AUDIO_ANALYSIS_FINAL, device);
    const bg2 = device.createBindGroup({
      layout: pipe2.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: partialsBuf } },
        { binding: 1, resource: { buffer: resultBuf } },
      ],
    });

    // Record both passes in a single command buffer
    const encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(pipe1);
    pass.setBindGroup(0, bg1);
    pass.dispatchWorkgroups(wgCount);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(pipe2);
    pass.setBindGroup(0, bg2);
    pass.dispatchWorkgroups(1);
    pass.end();

    const readback = device.createBuffer({
      size: 12,
      usage: GPU_COPY_DST | GPU_MAP_READ,
    });
    encoder.copyBufferToBuffer(resultBuf, 0, readback, 0, 12);
    device.queue.submit([encoder.finish()]);

    try {
      await readback.mapAsync(GPU_MAP_READ);
      const mapped = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();

      const sumSq = mapped[0];
      const zcr = mapped[1];
      const rms = Math.sqrt(sumSq / N);
      const zcrRate = zcr / N;

      // Pitch via GPU too
      const { pitch } = await this.estimatePitch(samples, sampleRate);

      return {
        pitch,
        rms,
        zcr: zcrRate,
        brightness: Math.min(1, zcrRate / 0.18),
      };
    } catch {
      return this._analyzeCPU(samples, sampleRate);
    }
  }

  /**
   * Blend two voice vectors on the GPU, with optional RMS normalisation.
   *
   * @returns A new Float32Array with the blended (and optionally normalised)
   *          voice vector.
   */
  async blendVectors(
    voiceA: Float32Array,
    voiceB: Float32Array,
    mix: number,
    gain: number,
    normalize = false,
    targetRms = 0.1,
  ): Promise<Float32Array> {
    if (!await this.ensureReady()) {
      return this._blendCPU(voiceA, voiceB, mix, gain, normalize, targetRms);
    }

    const device = this.ctx.device!;
    const length = Math.min(voiceA.length, voiceB.length);
    const byteLen = length * 4;

    // Upload voices
    const bufA = this.ctx.uploadBuffer(voiceA.subarray(0, length));
    const bufB = this.ctx.uploadBuffer(voiceB.subarray(0, length));
    const outBuf = this.ctx.allocBuffer(byteLen);
    if (!bufA || !bufB || !outBuf) {
      return this._blendCPU(voiceA, voiceB, mix, gain, normalize, targetRms);
    }

    // Uniforms
    const arr = new ArrayBuffer(24);
    const v = new DataView(arr);
    v.setFloat32(0, mix, true);
    v.setFloat32(4, gain, true);
    v.setUint32(8, length, true);
    v.setUint32(12, normalize ? 1 : 0, true);
    v.setFloat32(16, targetRms, true);
    v.setUint32(20, 0, true); // padding
    const paramBuf = this.ctx.createUniformBuffer(arr);
    if (!paramBuf) return this._blendCPU(voiceA, voiceB, mix, gain, normalize, targetRms);

    const pipeline = await this._pipeline('blend', SHADERS.VOICE_BLEND_NORM, device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: paramBuf } },
      ],
    });

    const result = await this.ctx.dispatchAndRead(
      pipeline, bindGroup, Math.ceil(length / 256), outBuf, byteLen,
    );
    return result ?? this._blendCPU(voiceA, voiceB, mix, gain, normalize, targetRms);
  }

  /**
   * Normalise audio to a target level using peak or RMS normalisation.
   */
  async normalizeAudio(
    samples: Float32Array,
    target: number,
    mode: 'peak' | 'rms' = 'rms',
  ): Promise<Float32Array> {
    if (!await this.ensureReady()) {
      return this._normalizeCPU(samples, target, mode);
    }

    const device = this.ctx.device!;
    const N = samples.length;
    const byteLen = N * 4;

    const inputBuf = this.ctx.uploadBuffer(samples);
    const outBuf = this.ctx.allocBuffer(byteLen);
    if (!inputBuf || !outBuf) return this._normalizeCPU(samples, target, mode);

    const arr = new ArrayBuffer(16);
    const v = new DataView(arr);
    v.setUint32(0, N, true);
    v.setFloat32(4, target, true);
    v.setUint32(8, mode === 'peak' ? 0 : 1, true);
    const paramBuf = this.ctx.createUniformBuffer(arr);
    if (!paramBuf) return this._normalizeCPU(samples, target, mode);

    const pipeline = await this._pipeline('norm', SHADERS.AUDIO_NORMALIZE, device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramBuf } },
      ],
    });

    const result = await this.ctx.dispatchAndRead(
      pipeline, bindGroup, Math.ceil(N / 256), outBuf, byteLen,
    );
    return result ?? this._normalizeCPU(samples, target, mode);
  }

  // ── Pipeline cache ─────────────────────────────────────────────────────────

  private async _pipeline(
    key: string,
    code: string,
    device: GPUDevice,
  ): Promise<GPUComputePipeline> {
    const existing = this.pipelines.get(key);
    if (existing) return existing;

    const module = device.createShaderModule({ code });
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.pipelines.set(key, pipeline);
    return pipeline;
  }

  // ── CPU fallbacks ───────────────────────────────────────────────────────────

  private _pitchCPU(
    samples: Float32Array,
    sampleRate: number,
    minPitch: number,
    maxPitch: number,
  ): PitchResult {
    const frameLen = Math.min(4096, samples.length);
    const start = Math.max(0, Math.floor((samples.length - frameLen) / 2));
    const frame = samples.subarray(start, start + frameLen);
    const minLag = Math.floor(sampleRate / maxPitch);
    const maxLag = Math.ceil(sampleRate / minPitch);

    let bestLag = 0;
    let bestScore = -Infinity;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let score = 0;
      for (let i = 0; i < frame.length - lag; i++) {
        score += frame[i] * frame[i + lag];
      }
      if (score > bestScore) {
        bestScore = score;
        bestLag = lag;
      }
    }

    let energy = 0;
    for (let i = 0; i < frameLen; i++) energy += frame[i] * frame[i];
    const normScore = energy > 0 ? bestScore / energy : 0;

    return {
      pitch: bestLag > 0 ? sampleRate / bestLag : 0,
      score: Math.max(0, Math.min(1, normScore)),
    };
  }

  private _analyzeCPU(samples: Float32Array, sampleRate: number): AudioAnalysisResult {
    const N = samples.length;
    let sumSq = 0;
    let zcr = 0;
    let prev = 0;

    for (let i = 0; i < N; i++) {
      const s = samples[i];
      sumSq += s * s;
      if ((s >= 0 && prev < 0) || (s < 0 && prev >= 0)) zcr++;
      prev = s;
    }

    const { pitch } = this._pitchCPU(samples, sampleRate, 70, 320);
    const rms = Math.sqrt(sumSq / N);
    const zcrRate = zcr / N;

    return {
      pitch,
      rms,
      zcr: zcrRate,
      brightness: Math.min(1, zcrRate / 0.18),
    };
  }

  private _blendCPU(
    a: Float32Array,
    b: Float32Array,
    mix: number,
    gain: number,
    normalize: boolean,
    targetRms: number,
  ): Float32Array {
    const len = Math.min(a.length, b.length);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      out[i] = (a[i] * (1 - mix) + b[i] * mix) * gain;
    }
    if (normalize) {
      let sumSq = 0;
      for (let i = 0; i < len; i++) sumSq += out[i] * out[i];
      const rms = Math.sqrt(sumSq / len);
      const scale = targetRms / Math.max(rms, 1e-8);
      for (let i = 0; i < len; i++) out[i] *= scale;
    }
    return out;
  }

  private _normalizeCPU(
    samples: Float32Array,
    target: number,
    mode: 'peak' | 'rms',
  ): Float32Array {
    const out = new Float32Array(samples.length);
    let measure: number;
    if (mode === 'peak') {
      let peak = 0;
      for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
      measure = Math.max(peak, 1e-8);
    } else {
      let sumSq = 0;
      for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
      measure = Math.sqrt(Math.max(sumSq / samples.length, 1e-8));
    }
    const scale = target / measure;
    for (let i = 0; i < samples.length; i++) out[i] = samples[i] * scale;
    return out;
  }
}

// ─── Convenience singleton ────────────────────────────────────────────────────

let _default: WebGPUMath | null = null;

/** Get (or create) the default module-level WebGPUMath singleton. */
export function getWebGPUMath(): WebGPUMath {
  if (!_default) _default = new WebGPUMath();
  return _default;
}
