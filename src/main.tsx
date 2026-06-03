import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { Mic, Sparkles } from "lucide-react";
import { KokoroApp } from "./KokoroApp";
import { QwenApp } from "./QwenApp";
import "./styles.css";

function App() {
  const [mode, setMode] = useState<"kokoro" | "qwen">("kokoro");

  return (
    <>
      <nav className="modebar">
        <button className={mode === "kokoro" ? "active" : ""} onClick={() => setMode("kokoro")}>
          <Sparkles size={18} />
          Kokoro
        </button>
        <button className={mode === "qwen" ? "active" : ""} onClick={() => setMode("qwen")}>
          <Mic size={18} />
          Qwen3TTS
        </button>
      </nav>
      {mode === "kokoro" ? <KokoroApp /> : <QwenApp />}
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
