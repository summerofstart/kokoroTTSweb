export const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
export const VOICE_CACHE_NAME = "kokoro-voices";
const WEBGPU_DISABLED_KEY = "kokoro.webgpu.disabled";

export const voices = [
  ["af_heart", "Heart", "American English"],
  ["af_bella", "Bella", "American English"],
  ["af_nicole", "Nicole", "American English"],
  ["af_sarah", "Sarah", "American English"],
  ["am_michael", "Michael", "American English"],
  ["am_puck", "Puck", "American English"],
  ["bf_emma", "Emma", "British English"],
  ["bf_alice", "Alice", "British English"],
  ["bm_daniel", "Daniel", "British English"],
  ["bm_fable", "Fable", "British English"],
  ["jf_alpha", "Alpha", "Japanese"],
  ["jf_gongitsune", "Gongitsune", "Japanese"],
  ["jf_nezumi", "Nezumi", "Japanese"],
  ["jf_tebukuro", "Tebukuro", "Japanese"],
  ["jm_kumo", "Kumo", "Japanese"]
] as const;

export type BuiltInKokoroVoice = (typeof voices)[number][0];
export type KokoroVoice = BuiltInKokoroVoice | string;
export type KokoroDevice = "auto" | "wasm" | "webgpu";
export type KokoroRuntimeDevice = "wasm" | "webgpu";
export type KokoroDType = "q4" | "q8" | "q4f16" | "fp32";

export type KokoroGenerateOptions = {
  text: string;
  voice?: KokoroVoice;
  speed?: number;
  device?: KokoroDevice;
  dtype?: KokoroDType;
  batchSize?: number;
};

export type KokoroApiConfig = {
  device?: KokoroDevice;
  dtype?: KokoroDType;
  aggressiveWebGPU?: boolean;
};

export type KokoroSynthesisResult = {
  wav: ArrayBuffer;
  mimeType: "audio/wav";
  runtime: {
    device: KokoroRuntimeDevice;
    dtype: KokoroDType;
  };
  elapsedMs: number;
};

export type KokoroWorkerRequest =
  | { id: number; type: "preload"; config?: KokoroApiConfig }
  | { id: number; type: "generate"; options: KokoroGenerateOptions }
  | { id: number; type: "importVoice"; voiceId: string; data: ArrayBuffer };

export type KokoroWorkerResponse =
  | { id: number; type: "ready"; runtime: KokoroSynthesisResult["runtime"] }
  | { id: number; type: "voiceImported"; voiceId: string }
  | { id: number; type: "result"; result: KokoroSynthesisResult }
  | { id: number; type: "error"; message: string };

export function getVoiceUrl(voiceId: string) {
  return `${MODEL_ID.replace("onnx-community/", "https://huggingface.co/onnx-community/")}/resolve/main/voices/${voiceId}.bin`;
}

export function disableWebGPUFallback() {
  try {
    localStorage.setItem(WEBGPU_DISABLED_KEY, "1");
  } catch {
    // Ignore storage failures in private or restricted contexts.
  }
}

export async function resolveRuntime(
  device: KokoroDevice = "auto",
  aggressiveWebGPU = false
): Promise<KokoroRuntimeDevice> {
  if (device !== "auto") return device;
  if (!aggressiveWebGPU) {
    try {
      if (localStorage.getItem(WEBGPU_DISABLED_KEY) === "1") return "wasm";
    } catch {
      return "wasm";
    }
  }
  if (!("gpu" in navigator)) return "wasm";

  try {
    const adapter = await navigator.gpu?.requestAdapter();
    return adapter ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}
