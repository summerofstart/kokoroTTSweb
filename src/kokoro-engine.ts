import { KokoroTTS } from "kokoro-js";
import {
  MODEL_ID,
  disableWebGPUFallback,
  type KokoroDType,
  type KokoroApiConfig,
  type KokoroGenerateOptions,
  type KokoroRuntimeDevice,
  type KokoroSynthesisResult,
  resolveRuntime
} from "./kokoro-types";

type TTS = Awaited<ReturnType<typeof KokoroTTS.from_pretrained>>;

let loaded: {
  key: string;
  tts: TTS;
  device: KokoroRuntimeDevice;
  dtype: KokoroDType;
} | null = null;
let loading: Promise<NonNullable<typeof loaded>> | null = null;
let loadingKey: string | null = null;

export async function preloadKokoro(
  device: KokoroGenerateOptions["device"] = "auto",
  dtype: KokoroDType = "q4",
  config: Pick<KokoroApiConfig, "aggressiveWebGPU"> = {}
) {
  const runtimeDevice = await resolveRuntime(device, config.aggressiveWebGPU);
  const runtimeDType = runtimeDevice === "webgpu" && dtype === "q4" ? "q4f16" : dtype;
  const key = `${runtimeDevice}:${runtimeDType}`;
  if (loaded?.key === key) return loaded;
  if (loading && loadingKey === key) return loading;
  if (loading) await loading.catch(() => null);
  if (loaded?.key === key) return loaded;

  loadingKey = key;
  loading = (async () => {
    try {
      const tts = await KokoroTTS.from_pretrained(MODEL_ID, {
        device: runtimeDevice,
        dtype: runtimeDType
      });
      loaded = { key, tts, device: runtimeDevice, dtype: runtimeDType };
      return loaded;
    } catch (err) {
      if (runtimeDevice !== "webgpu") throw err;
      disableWebGPUFallback();

      const fallbackDType = dtype === "q4f16" ? "q4" : dtype;
      const fallbackKey = `wasm:${fallbackDType}`;
      if (loaded?.key === fallbackKey) return loaded;

      const tts = await KokoroTTS.from_pretrained(MODEL_ID, {
        device: "wasm",
        dtype: fallbackDType
      });
      loaded = { key: fallbackKey, tts, device: "wasm", dtype: fallbackDType };
      return loaded;
    } finally {
      loading = null;
      loadingKey = null;
    }
  })();

  return loading;
}

export async function synthesizeKokoro(options: KokoroGenerateOptions): Promise<KokoroSynthesisResult> {
  const started = performance.now();
  const model = await preloadKokoro(options.device, options.dtype);
  const audio = await model.tts.generate(options.text.trim(), {
    voice: options.voice ?? "af_heart",
    speed: options.speed ?? 1
  });
  const wav = audio.toWav();
  const buffer =
    wav instanceof ArrayBuffer
      ? wav.slice(0)
      : wav.buffer.slice(wav.byteOffset, wav.byteOffset + wav.byteLength);

  return {
    wav: buffer,
    mimeType: "audio/wav",
    runtime: {
      device: model.device,
      dtype: model.dtype
    },
    elapsedMs: Math.round(performance.now() - started)
  };
}
