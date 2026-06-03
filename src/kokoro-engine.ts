import { KokoroTTS } from "kokoro-js";
import {
  MODEL_ID,
  VOICE_CACHE_NAME,
  disableWebGPUFallback,
  getVoiceUrl,
  type KokoroDType,
  type KokoroApiConfig,
  type KokoroGenerateOptions,
  type KokoroRuntimeDevice,
  type KokoroSynthesisResult,
  resolveRuntime
} from "./kokoro-types";

type TTS = Awaited<ReturnType<typeof KokoroTTS.from_pretrained>>;
type RawAudio = Awaited<ReturnType<TTS["generate"]>>;

let loaded: {
  key: string;
  tts: TTS;
  device: KokoroRuntimeDevice;
  dtype: KokoroDType;
} | null = null;
let loading: Promise<NonNullable<typeof loaded>> | null = null;
let loadingKey: string | null = null;

const englishVoices = new Set([
  "af_heart",
  "af_alloy",
  "af_aoede",
  "af_bella",
  "af_jessica",
  "af_kore",
  "af_nicole",
  "af_nova",
  "af_river",
  "af_sarah",
  "af_sky",
  "am_adam",
  "am_echo",
  "am_eric",
  "am_fenrir",
  "am_liam",
  "am_michael",
  "am_onyx",
  "am_puck",
  "am_santa",
  "bf_emma",
  "bf_isabella",
  "bm_george",
  "bm_lewis",
  "bf_alice",
  "bf_lily",
  "bm_daniel",
  "bm_fable"
]);

export async function importKokoroVoice(voiceId: string, data: ArrayBuffer) {
  if (!voiceId || !/^[a-z][a-z0-9_]{1,48}$/i.test(voiceId)) {
    throw new Error("Voice id must use letters, numbers, and underscores.");
  }
  if (data.byteLength < 256 * 4) {
    throw new Error("Voice file is too small to be a Kokoro voice .bin.");
  }

  const cache = await caches.open(VOICE_CACHE_NAME);
  await cache.put(
    getVoiceUrl(voiceId),
    new Response(data.slice(0), {
      headers: {
        "Content-Type": "application/octet-stream"
      }
    })
  );
  return voiceId;
}

function splitSentences(text: string) {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?。！？])\s+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function chunkSentences(sentences: string[], batchSize: number) {
  const chunks: string[] = [];
  for (let index = 0; index < sentences.length; index += batchSize) {
    chunks.push(sentences.slice(index, index + batchSize).join(" "));
  }
  return chunks;
}

function wavToBytes(wav: ArrayBuffer | ArrayBufferView) {
  return wav instanceof ArrayBuffer
    ? new Uint8Array(wav)
    : new Uint8Array(wav.buffer, wav.byteOffset, wav.byteLength);
}

function readAscii(bytes: Uint8Array, offset: number, length: number) {
  return String.fromCharCode(...bytes.slice(offset, offset + length));
}

function findDataOffset(bytes: Uint8Array) {
  let offset = 12;
  while (offset + 8 <= bytes.byteLength) {
    const id = readAscii(bytes, offset, 4);
    const size = new DataView(bytes.buffer, bytes.byteOffset + offset + 4, 4).getUint32(0, true);
    if (id === "data") return { offset: offset + 8, size, sizeOffset: offset + 4 };
    offset += 8 + size + (size % 2);
  }
  throw new Error("Invalid WAV data chunk");
}

function concatWavs(wavs: ArrayBuffer[]) {
  if (wavs.length === 1) return wavs[0].slice(0);

  const first = wavToBytes(wavs[0]);
  const firstData = findDataOffset(first);
  const header = first.slice(0, firstData.offset);
  const parts = wavs.map((wav) => {
    const bytes = wavToBytes(wav);
    const data = findDataOffset(bytes);
    return bytes.slice(data.offset, data.offset + data.size);
  });
  const dataSize = parts.reduce((total, part) => total + part.byteLength, 0);
  const output = new Uint8Array(header.byteLength + dataSize);
  output.set(header, 0);

  let offset = header.byteLength;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }

  const view = new DataView(output.buffer);
  view.setUint32(4, output.byteLength - 8, true);
  view.setUint32(firstData.sizeOffset, dataSize, true);
  return output.buffer;
}

function normalizeWavBuffer(wav: ReturnType<RawAudio["toWav"]>) {
  const bytes = wavToBytes(wav);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function generateAudio(model: TTS, text: string, options: KokoroGenerateOptions) {
  const voice = options.voice ?? "af_heart";
  const speed = options.speed ?? 1;

  if (!englishVoices.has(voice)) {
    const { input_ids } = model.tokenizer(text, { truncation: true });
    return model.generate_from_ids(input_ids, { voice: voice as never, speed });
  }

  return model.generate(text, { voice: voice as never, speed });
}

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
  const batchSize = Math.min(32, Math.max(1, Math.round(options.batchSize ?? 8)));
  const sentences = splitSentences(options.text.trim());
  const chunks = chunkSentences(sentences.length > 0 ? sentences : [options.text.trim()], batchSize);
  const wavs: ArrayBuffer[] = [];

  for (const chunk of chunks) {
    const audio = await generateAudio(model.tts, chunk, options);
    wavs.push(normalizeWavBuffer(audio.toWav()));
  }

  const buffer = concatWavs(wavs);

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
