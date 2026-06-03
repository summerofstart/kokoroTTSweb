import React, { lazy, Suspense, useState } from "react";
import { createRoot } from "react-dom/client";
import { Mic, Sparkles } from "lucide-react";
import { KokoroApp } from "./KokoroApp";
import "./styles.css";

const F5App = lazy(() => import("./f5/App.jsx"));

function App() {
  const [mode, setMode] = useState<"kokoro" | "f5">("kokoro");

  return (
    <>
      <nav className="modebar">
        <button className={mode === "kokoro" ? "active" : ""} onClick={() => setMode("kokoro")}>
          <Sparkles size={18} />
          Kokoro
        </button>
        <button className={mode === "f5" ? "active" : ""} onClick={() => setMode("f5")}>
          <Mic size={18} />
          F5-TTS WebGPU
        </button>
      </nav>
      {mode === "kokoro" ? (
        <KokoroApp />
      ) : (
        <Suspense fallback={<div className="f5Loading">Loading F5-TTS WebGPU engine...</div>}>
          <F5App />
        </Suspense>
      )}
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
