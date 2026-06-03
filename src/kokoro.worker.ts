import { preloadKokoro, synthesizeKokoro } from "./kokoro-engine";
import type { KokoroWorkerRequest, KokoroWorkerResponse } from "./kokoro-types";

self.onmessage = async (event: MessageEvent<KokoroWorkerRequest>) => {
  const request = event.data;

  try {
    if (request.type === "preload") {
      const model = await preloadKokoro(request.config?.device, request.config?.dtype, {
        aggressiveWebGPU: request.config?.aggressiveWebGPU
      });
      postMessage({
        id: request.id,
        type: "ready",
        runtime: { device: model.device, dtype: model.dtype }
      } satisfies KokoroWorkerResponse);
      return;
    }

    const result = await synthesizeKokoro(request.options);
    postMessage(
      { id: request.id, type: "result", result } satisfies KokoroWorkerResponse,
      [result.wav]
    );
  } catch (err) {
    postMessage({
      id: request.id,
      type: "error",
      message: err instanceof Error ? err.message : "Unknown Kokoro error"
    } satisfies KokoroWorkerResponse);
  }
};
