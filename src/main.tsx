import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Download, Gauge, Loader2, Play, Sparkles } from "lucide-react";
import { kokoroApi, voices, type KokoroDType, type KokoroDevice, type KokoroVoice } from "./kokoro-api";
import "./styles.css";

type ApiMessage =
  | { type: "kokoro:preload"; id?: string; config?: { device?: KokoroDevice; dtype?: KokoroDType } }
  | {
      type: "kokoro:synthesize";
      id?: string;
      options: {
        text: string;
        voice?: KokoroVoice;
        speed?: number;
        device?: KokoroDevice;
        dtype?: KokoroDType;
      };
    };

function isVoice(value: string | null): value is KokoroVoice {
  return voices.some(([id]) => id === value);
}

function isDevice(value: string | null): value is KokoroDevice {
  return value === "auto" || value === "wasm" || value === "webgpu";
}

function isDType(value: string | null): value is KokoroDType {
  return value === "q4" || value === "q8" || value === "q4f16" || value === "fp32";
}

function App() {
  const [text, setText] = useState(
    "Kokoro runs fully in your browser. Type a sentence, choose a voice, and generate speech locally."
  );
  const [voice, setVoice] = useState<KokoroVoice>("af_heart");
  const [device, setDevice] = useState<KokoroDevice>("auto");
  const [dtype, setDType] = useState<KokoroDType>("q4");
  const [speed, setSpeed] = useState(1);
  const [status, setStatus] = useState("Ready");
  const [runtime, setRuntime] = useState("auto / q4");
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const queryRunRef = useRef(false);

  const charCount = text.trim().length;
  const selectedVoice = useMemo(() => voices.find(([id]) => id === voice), [voice]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const queryText = params.get("text");
    if (!queryText) return;

    const queryVoice = params.get("voice");
    const queryDevice = params.get("device");
    const queryDType = params.get("dtype");
    const querySpeed = Number(params.get("speed") ?? "1");

    setText(queryText);
    if (isVoice(queryVoice)) setVoice(queryVoice);
    if (isDevice(queryDevice)) setDevice(queryDevice);
    if (isDType(queryDType)) setDType(queryDType);
    if (Number.isFinite(querySpeed)) setSpeed(Math.min(1.35, Math.max(0.7, querySpeed)));
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (queryRunRef.current || params.get("autoplay") !== "1" || !params.get("text")) return;
    queryRunRef.current = true;
    const timer = window.setTimeout(() => {
      void generate();
    }, 400);
    return () => window.clearTimeout(timer);
  }, [text, voice, device, dtype, speed]);

  useEffect(() => {
    async function handleMessage(event: MessageEvent<ApiMessage>) {
      const message = event.data;
      if (!message || typeof message !== "object") return;

      try {
        if (message.type === "kokoro:preload") {
          const runtime = await kokoroApi.preload(message.config);
          if ("postMessage" in (event.source ?? {})) {
            event.source?.postMessage({ type: "kokoro:ready", id: message.id, runtime }, event.origin);
          }
          return;
        }

        if (message.type === "kokoro:synthesize") {
          const result = await kokoroApi.synthesize(message.options);
          if ("postMessage" in (event.source ?? {})) {
            event.source?.postMessage(
              {
                type: "kokoro:result",
                id: message.id,
                result
              },
              event.origin,
              [result.wav]
            );
          }
        }
      } catch (err) {
        if ("postMessage" in (event.source ?? {})) {
          event.source?.postMessage(
            {
              type: "kokoro:error",
              id: message.id,
              message: err instanceof Error ? err.message : "Unknown Kokoro error"
            },
            event.origin
          );
        }
      }
    }

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  useEffect(() => {
    let alive = true;
    setStatus("Preloading model");
    setError(null);

    kokoroApi
      .preload({ device, dtype })
      .then((nextRuntime) => {
        if (!alive) return;
        setRuntime(`${nextRuntime.device} / ${nextRuntime.dtype}`);
        setStatus("Model hot");
      })
      .catch((err) => {
        if (!alive) return;
        setError(err instanceof Error ? err.message : "Preload failed");
        setStatus("Ready");
      });

    return () => {
      alive = false;
    };
  }, [device, dtype]);

  async function generate() {
    if (!text.trim() || busy) return;

    setBusy(true);
    setError(null);
    try {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);

      setStatus("Generating speech");
      const result = await kokoroApi.synthesize({ text, voice, speed, device, dtype });
      const blob = new Blob([result.wav], { type: result.mimeType });
      const url = URL.createObjectURL(blob);

      setAudioUrl(url);
      setRuntime(`${result.runtime.device} / ${result.runtime.dtype}`);
      setElapsed(result.elapsedMs);
      setStatus("Speech ready");

      requestAnimationFrame(() => {
        audioRef.current?.play().catch(() => {
          setStatus("Speech ready");
        });
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown Kokoro error");
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
            <p className="eyebrow">Browser local TTS API</p>
            <h1>Kokoro TTS</h1>
          </div>
          <div className="status" data-busy={busy}>
            {busy ? <Loader2 size={18} className="spin" /> : <Sparkles size={18} />}
            <span>{status}</span>
          </div>
        </header>

        <div className="editor">
          <label htmlFor="script">Text</label>
          <textarea
            id="script"
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength={1200}
            spellCheck
          />
          <div className="counter">{charCount} / 1200</div>
        </div>

        <div className="controls">
          <label>
            Voice
            <select value={voice} onChange={(event) => setVoice(event.target.value as KokoroVoice)}>
              {voices.map(([id, name, region]) => (
                <option key={id} value={id}>
                  {name} - {region}
                </option>
              ))}
            </select>
          </label>

          <label>
            Runtime
            <select value={device} onChange={(event) => setDevice(event.target.value as KokoroDevice)}>
              <option value="auto">Auto fastest</option>
              <option value="webgpu">WebGPU fallback</option>
              <option value="wasm">WASM</option>
            </select>
          </label>

          <label>
            Precision
            <select value={dtype} onChange={(event) => setDType(event.target.value as KokoroDType)}>
              <option value="q4">q4 fastest</option>
              <option value="q4f16">q4f16 GPU</option>
              <option value="q8">q8 balanced</option>
              <option value="fp32">fp32</option>
            </select>
          </label>

          <label>
            Speed <span>{speed.toFixed(2)}x</span>
            <input
              type="range"
              min="0.7"
              max="1.35"
              step="0.05"
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="actions">
          <button onClick={generate} disabled={busy || charCount === 0}>
            {busy ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
            Generate
          </button>
          <a className={!audioUrl ? "disabled" : ""} href={audioUrl ?? undefined} download="kokoro.wav">
            <Download size={18} />
            WAV
          </a>
        </div>

        <audio ref={audioRef} controls src={audioUrl ?? undefined} />

        {error && <p className="error">{error}</p>}
      </section>

      <aside className="details">
        <div>
          <p className="eyebrow">Selected voice</p>
          <h2>{selectedVoice?.[1]}</h2>
          <p>{selectedVoice?.[2]}</p>
        </div>
        <div className="metrics">
          <div>
            <Gauge size={18} />
            <span>{runtime}</span>
          </div>
          <strong>{elapsed === null ? "hot preload" : `${elapsed} ms`}</strong>
        </div>
        <div className="note">
          First run downloads the ONNX model to the browser cache. The public API is exported from{" "}
          <code>src/kokoro-api.ts</code>.
        </div>
      </aside>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
