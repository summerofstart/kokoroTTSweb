import { useMemo, useState } from "react";
import { Download, Loader2, Save, SlidersHorizontal } from "lucide-react";
import { getVoiceUrl, voices, type KokoroVoice } from "./kokoro-types";
import { kokoroApi } from "./kokoro-api";

type Backend = "webgpu" | "cpu";

async function loadVoiceVector(voice: string) {
  const response = await fetch(getVoiceUrl(voice));
  if (!response.ok) throw new Error(`Failed to load ${voice}: ${response.status}`);
  return new Float32Array(await response.arrayBuffer());
}

function blendCpu(a: Float32Array, b: Float32Array, mix: number, gain: number) {
  const length = Math.min(a.length, b.length);
  const output = new Float32Array(length);
  for (let index = 0; index < length; index++) {
    output[index] = (a[index] * (1 - mix) + b[index] * mix) * gain;
  }
  return output;
}

async function blendWebGpu(a: Float32Array, b: Float32Array, mix: number, gain: number) {
  if (!navigator.gpu) throw new Error("WebGPU is not available");

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Failed to get GPU adapter");
  const device = await adapter.requestDevice();
  const length = Math.min(a.length, b.length);
  const byteLength = length * 4;

  const shader = device.createShaderModule({
    code: `
      struct Params {
        mix: f32,
        gain: f32,
        length: u32,
        pad: u32,
      };

      @group(0) @binding(0) var<storage, read> voiceA: array<f32>;
      @group(0) @binding(1) var<storage, read> voiceB: array<f32>;
      @group(0) @binding(2) var<storage, read_write> outVoice: array<f32>;
      @group(0) @binding(3) var<uniform> params: Params;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= params.length) {
          return;
        }
        outVoice[i] = ((voiceA[i] * (1.0 - params.mix)) + (voiceB[i] * params.mix)) * params.gain;
      }
    `
  });

  const makeStorage = (data?: Float32Array) => {
    const buffer = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    if (data) device.queue.writeBuffer(buffer, 0, data.subarray(0, length));
    return buffer;
  };

  const bufferA = makeStorage(a);
  const bufferB = makeStorage(b);
  const output = makeStorage();
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  const paramsBytes = new ArrayBuffer(16);
  const view = new DataView(paramsBytes);
  view.setFloat32(0, mix, true);
  view.setFloat32(4, gain, true);
  view.setUint32(8, length, true);
  device.queue.writeBuffer(params, 0, paramsBytes);

  const readback = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" }
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return result;
}

function toArrayBuffer(vector: Float32Array) {
  return vector.buffer.slice(vector.byteOffset, vector.byteOffset + vector.byteLength);
}

export function VoiceVectorApp() {
  const [voiceA, setVoiceA] = useState<KokoroVoice>("af_heart");
  const [voiceB, setVoiceB] = useState<KokoroVoice>("af_bella");
  const [mix, setMix] = useState(0.5);
  const [gain, setGain] = useState(1);
  const [voiceId, setVoiceId] = useState("custom_blend");
  const [backend, setBackend] = useState<Backend | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [vectorUrl, setVectorUrl] = useState<string | null>(null);
  const [vectorBuffer, setVectorBuffer] = useState<ArrayBuffer | null>(null);

  const canGenerate = useMemo(() => voiceA !== voiceB && voiceId.trim().length > 0, [voiceA, voiceB, voiceId]);

  async function generateVector() {
    if (!canGenerate || busy) return;

    setBusy(true);
    setError(null);
    if (vectorUrl) URL.revokeObjectURL(vectorUrl);
    setVectorUrl(null);
    setVectorBuffer(null);

    try {
      setStatus("Loading source voices");
      const [a, b] = await Promise.all([loadVoiceVector(voiceA), loadVoiceVector(voiceB)]);

      setStatus("Blending vectors");
      let vector: Float32Array;
      try {
        vector = await blendWebGpu(a, b, mix, gain);
        setBackend("webgpu");
      } catch {
        vector = blendCpu(a, b, mix, gain);
        setBackend("cpu");
      }

      const buffer = toArrayBuffer(vector);
      setVectorBuffer(buffer);
      setVectorUrl(URL.createObjectURL(new Blob([buffer], { type: "application/octet-stream" })));
      setStatus("Vector ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Vector generation failed");
      setStatus("Failed");
    } finally {
      setBusy(false);
    }
  }

  async function importGeneratedVoice() {
    if (!vectorBuffer || busy) return;
    setBusy(true);
    setError(null);
    try {
      setStatus("Importing voice");
      await kokoroApi.importVoice(voiceId.trim(), vectorBuffer.slice(0));
      setStatus("Voice imported");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Voice import failed");
      setStatus("Failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Kokoro voice vector generator</p>
            <h1>Voice Vector</h1>
          </div>
          <div className="status" data-busy={busy}>
            {busy ? <Loader2 size={18} className="spin" /> : <SlidersHorizontal size={18} />}
            <span>{status}</span>
          </div>
        </header>

        <div className="controls vectorControls">
          <label>
            Voice A
            <select value={voiceA} onChange={(event) => setVoiceA(event.target.value)}>
              {voices.map(([id, name, region]) => (
                <option key={id} value={id}>
                  {name} - {region}
                </option>
              ))}
            </select>
          </label>

          <label>
            Voice B
            <select value={voiceB} onChange={(event) => setVoiceB(event.target.value)}>
              {voices.map(([id, name, region]) => (
                <option key={id} value={id}>
                  {name} - {region}
                </option>
              ))}
            </select>
          </label>

          <label>
            Output ID
            <input value={voiceId} onChange={(event) => setVoiceId(event.target.value.replace(/[^a-z0-9_]/gi, "_").toLowerCase())} />
          </label>
        </div>

        <div className="controls twoControls">
          <label>
            Mix <span>{Math.round(mix * 100)}%</span>
            <input type="range" min="0" max="1" step="0.01" value={mix} onChange={(event) => setMix(Number(event.target.value))} />
          </label>

          <label>
            Gain <span>{gain.toFixed(2)}x</span>
            <input type="range" min="0.7" max="1.3" step="0.01" value={gain} onChange={(event) => setGain(Number(event.target.value))} />
          </label>
        </div>

        <div className="actions">
          <button onClick={generateVector} disabled={!canGenerate || busy}>
            {busy ? <Loader2 size={18} className="spin" /> : <SlidersHorizontal size={18} />}
            Generate .bin
          </button>
          <button onClick={importGeneratedVoice} disabled={!vectorBuffer || busy}>
            <Save size={18} />
            Import to Kokoro
          </button>
          <a className={!vectorUrl ? "disabled" : ""} href={vectorUrl ?? undefined} download={`${voiceId || "kokoro_voice"}.bin`}>
            <Download size={18} />
            Download
          </a>
        </div>

        {backend && <p className="hint">Backend: {backend === "webgpu" ? "WebGPU compute" : "CPU fallback"}</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <aside className="details">
        <div>
          <p className="eyebrow">Output</p>
          <h2>{voiceId || "custom_voice"}.bin</h2>
          <p>Kokoro-compatible voice vector file generated by blending existing voice packs.</p>
        </div>
        <div className="note">
          This does not clone a WAV recording. It creates a new Kokoro style-vector file by mixing existing voice vectors,
          using WebGPU compute when available and CPU fallback otherwise.
        </div>
      </aside>
    </main>
  );
}
