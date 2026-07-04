import { useMemo, useState } from "react";
import { Download, Loader2, Mic, Save, SlidersHorizontal, Upload } from "lucide-react";
import { getVoiceUrl, voices, type KokoroVoice } from "./kokoro-types";
import { kokoroApi } from "./kokoro-api";
import { getWebGPUMath, type AudioAnalysisResult } from "./webgpu-math";

type Backend = "webgpu" | "cpu";

async function loadVoiceVector(voice: string) {
  const response = await fetch(getVoiceUrl(voice));
  if (!response.ok) throw new Error(`Failed to load ${voice}: ${response.status}`);
  return new Float32Array(await response.arrayBuffer());
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
  const [referenceSummary, setReferenceSummary] = useState<string | null>(null);
  const [vectorUrl, setVectorUrl] = useState<string | null>(null);
  const [vectorBuffer, setVectorBuffer] = useState<ArrayBuffer | null>(null);
  const [normalize, setNormalize] = useState(true);

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

      setStatus("Blending vectors (WebGPU)");
      const math = getWebGPUMath();
      let vector: Float32Array;
      try {
        vector = await math.blendVectors(a, b, mix, gain, normalize, 0.1);
        setBackend("webgpu");
      } catch {
        // WebGPU unavailable or failed — CPU fallback is inside blendVectors
        vector = await math.blendVectors(a, b, mix, gain, normalize, 0.1);
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

  async function matchReferenceFile(file: File | null) {
    if (!file || busy) return;

    setBusy(true);
    setError(null);
    try {
      setStatus("Analyzing reference (WebGPU)");
      const context = new AudioContext({ sampleRate: 24000 });
      let analysis: AudioAnalysisResult;
      try {
        const decoded = await context.decodeAudioData(await file.arrayBuffer());
        const channel = decoded.getChannelData(0);

        // Use the GPU-accelerated audio analysis pipeline
        const math = getWebGPUMath();
        analysis = await math.analyzeAudio(channel, decoded.sampleRate);
        setBackend("webgpu");
      } finally {
        await context.close();
      }

      const higherPitch = analysis.pitch >= 165;
      const bright = analysis.brightness >= 0.5;

      if (higherPitch && bright) {
        setVoiceA("af_heart");
        setVoiceB("af_bella");
      } else if (higherPitch) {
        setVoiceA("af_nicole");
        setVoiceB("af_sarah");
      } else if (bright) {
        setVoiceA("am_puck");
        setVoiceB("am_michael");
      } else {
        setVoiceA("bm_fable");
        setVoiceB("bm_daniel");
      }

      setMix(Math.min(0.85, Math.max(0.15, analysis.brightness)));
      setGain(Math.min(1.2, Math.max(0.8, 0.95 + analysis.rms * 1.8)));
      setVoiceId(`custom_${file.name.replace(/\.[^.]+$/i, "").replace(/[^a-z0-9_]/gi, "_").toLowerCase()}`);
      setReferenceSummary(`Pitch ${Math.round(analysis.pitch)} Hz, brightness ${Math.round(analysis.brightness * 100)}%`);
      setStatus("Reference matched");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reference analysis failed");
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

        <div className="actions">
          <label className="fileButton">
            <Upload size={18} />
            Match My WAV
            <input type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg" onChange={(event) => void matchReferenceFile(event.target.files?.[0] ?? null)} />
          </label>
          <button onClick={generateVector} disabled={!canGenerate || busy}>
            {busy ? <Loader2 size={18} className="spin" /> : <Mic size={18} />}
            Use Match
          </button>
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

        <div className="controls twoControls">
          <label className="checkboxLabel">
            <input
              type="checkbox"
              checked={normalize}
              onChange={(event) => setNormalize(event.target.checked)}
            />
            RMS Normalize
          </label>
          <label>
            Backend
            <select value={backend ?? (navigator.gpu ? "webgpu" : "cpu")} disabled>
              <option value="webgpu">WebGPU compute</option>
              <option value="cpu">CPU fallback</option>
            </select>
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

        {referenceSummary && <p className="hint">Reference match: {referenceSummary}</p>}
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
          <strong>WebGPU Math Engine</strong><br />
          Audio analysis uses GPU-accelerated autocorrelation for pitch detection
          and parallel reduction for RMS/ZCR computation. Vector blending runs
          on the GPU with optional RMS normalisation — all with transparent CPU
          fallback.
        </div>
      </aside>
    </main>
  );
}
