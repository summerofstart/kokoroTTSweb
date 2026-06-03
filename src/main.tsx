import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { SlidersHorizontal, Sparkles } from "lucide-react";
import { KokoroApp } from "./KokoroApp";
import { VoiceVectorApp } from "./VoiceVectorApp";
import "./styles.css";

function App() {
  const [mode, setMode] = useState<"kokoro" | "vector">("kokoro");

  return (
    <>
      <nav className="modebar">
        <button className={mode === "kokoro" ? "active" : ""} onClick={() => setMode("kokoro")}>
          <Sparkles size={18} />
          Kokoro
        </button>
        <button className={mode === "vector" ? "active" : ""} onClick={() => setMode("vector")}>
          <SlidersHorizontal size={18} />
          Voice Vector
        </button>
      </nav>
      {mode === "kokoro" ? <KokoroApp /> : <VoiceVectorApp />}
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
