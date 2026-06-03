import type {
  KokoroApiConfig,
  KokoroGenerateOptions,
  KokoroSynthesisResult,
  KokoroWorkerRequest,
  KokoroWorkerResponse
} from "./kokoro-types";
import { voices } from "./kokoro-types";
export type {
  KokoroApiConfig,
  KokoroDType,
  KokoroDevice,
  KokoroGenerateOptions,
  KokoroSynthesisResult,
  KokoroVoice
} from "./kokoro-types";
export { voices } from "./kokoro-types";

declare global {
  interface Window {
    KokoroTTSWeb?: {
      preload: KokoroBrowserAPI["preload"];
      importVoice: KokoroBrowserAPI["importVoice"];
      synthesize: KokoroBrowserAPI["synthesize"];
      voices: typeof import("./kokoro-types").voices;
    };
  }
}

type Pending = {
  resolve: (value: KokoroSynthesisResult | KokoroSynthesisResult["runtime"]) => void;
  reject: (reason: Error) => void;
};

export class KokoroBrowserAPI {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();

  constructor() {
    this.worker = new Worker(new URL("./kokoro.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<KokoroWorkerResponse>) => {
      const response = event.data;
      const pending = this.pending.get(response.id);
      if (!pending) return;
      this.pending.delete(response.id);

      if (response.type === "error") {
        pending.reject(new Error(response.message));
      } else if (response.type === "ready") {
        pending.resolve(response.runtime);
      } else if (response.type === "voiceImported") {
        pending.resolve(response.voiceId as never);
      } else {
        pending.resolve(response.result);
      }
    };

    void this.preload({ device: "auto", dtype: "q4" }).catch(() => {
      // UI callers surface preload failures explicitly; the eager warmup should stay quiet.
    });
  }

  preload(config: KokoroApiConfig = {}) {
    return this.call({ type: "preload", config }) as Promise<KokoroSynthesisResult["runtime"]>;
  }

  synthesize(options: KokoroGenerateOptions) {
    if (!options.text.trim()) {
      return Promise.reject(new Error("text is required"));
    }
    return this.call({ type: "generate", options }) as Promise<KokoroSynthesisResult>;
  }

  importVoice(voiceId: string, data: ArrayBuffer) {
    return this.call({ type: "importVoice", voiceId, data }, [data]) as Promise<string>;
  }

  dispose() {
    this.worker.terminate();
    this.pending.clear();
  }

  private call(request: Omit<KokoroWorkerRequest, "id">, transfer: Transferable[] = []) {
    const id = this.nextId++;
    const message = { id, ...request } as KokoroWorkerRequest;

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(message, transfer);
    });
  }
}

export const kokoroApi = new KokoroBrowserAPI();

window.KokoroTTSWeb = {
  preload: kokoroApi.preload.bind(kokoroApi),
  importVoice: kokoroApi.importVoice.bind(kokoroApi),
  synthesize: kokoroApi.synthesize.bind(kokoroApi),
  voices
};
