import { useMemo, useState } from "react";
import { Download, Loader2, Play, Server, Upload } from "lucide-react";

const modelUrl = "https://huggingface.co/jasonzhang76/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX";

const speakers = ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"];
const languages = ["english", "japanese", "chinese", "korean"];

export function QwenApp() {
  const [endpoint, setEndpoint] = useState(localStorage.getItem("qwen3tts.endpoint") ?? "");
  const [text, setText] = useState("Hello from Qwen3 TTS.");
  const [speaker, setSpeaker] = useState("ryan");
  const [language, setLanguage] = useState("english");
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Backend required");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canRun = useMemo(() => endpoint.trim().length > 0 && text.trim().length > 0, [endpoint, text]);

  async function generate() {
    if (!canRun || busy) return;

    setBusy(true);
    setError(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);

    try {
      localStorage.setItem("qwen3tts.endpoint", endpoint.trim());
      setStatus("Calling Qwen3TTS backend");

      const form = new FormData();
      form.set("text", text.trim());
      form.set("speaker", speaker);
      form.set("language", language);
      if (referenceFile) form.set("reference_audio", referenceFile);

      const response = await fetch(endpoint.trim(), {
        method: "POST",
        body: form
      });

      if (!response.ok) {
        throw new Error(`Qwen3TTS backend failed: ${response.status} ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      setStatus("Speech ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Qwen3TTS request failed");
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
            <p className="eyebrow">Qwen3TTS custom voice</p>
            <h1>Qwen3 TTS</h1>
          </div>
          <div className="status" data-busy={busy}>
            {busy ? <Loader2 size={18} className="spin" /> : <Server size={18} />}
            <span>{status}</span>
          </div>
        </header>

        <div className="editor">
          <label htmlFor="qwen-endpoint">Backend API endpoint</label>
          <input
            id="qwen-endpoint"
            value={endpoint}
            onChange={(event) => setEndpoint(event.target.value)}
            placeholder="https://your-server.example.com/api/qwen3tts"
          />
        </div>

        <div className="editor">
          <label htmlFor="qwen-text">Text</label>
          <textarea id="qwen-text" value={text} onChange={(event) => setText(event.target.value)} maxLength={1200} />
          <div className="counter">{text.trim().length} / 1200</div>
        </div>

        <div className="controls qwenControls">
          <label>
            Speaker
            <select value={speaker} onChange={(event) => setSpeaker(event.target.value)}>
              {speakers.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label>
            Language
            <select value={language} onChange={(event) => setLanguage(event.target.value)}>
              {languages.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="actions">
          <button onClick={generate} disabled={!canRun || busy}>
            {busy ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
            Generate
          </button>
          <label className="fileButton">
            <Upload size={18} />
            Reference WAV
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.ogg"
              onChange={(event) => setReferenceFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <a className={!audioUrl ? "disabled" : ""} href={audioUrl ?? undefined} download="qwen3tts.wav">
            <Download size={18} />
            WAV
          </a>
        </div>

        {referenceFile && <p className="hint">Reference: {referenceFile.name}</p>}
        <audio controls src={audioUrl ?? undefined} />
        {error && <p className="error">{error}</p>}
      </section>

      <aside className="details">
        <div>
          <p className="eyebrow">Model</p>
          <h2>Qwen3-TTS ONNX</h2>
          <p>CustomVoice model with predefined speakers and optional reference audio via backend.</p>
        </div>
        <div className="note">
          This ONNX export is multi-gigabyte and targets local ONNX Runtime apps. GitHub Pages cannot host a server API or
          reliably run the full stack in-browser. Use this page as the frontend for a Qwen3TTS backend.
        </div>
        <a className="sourceLink" href={modelUrl} target="_blank" rel="noreferrer">
          Hugging Face model
        </a>
      </aside>
    </main>
  );
}
