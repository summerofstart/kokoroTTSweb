# Kokoro TTS for GitHub Pages

Static browser app for Kokoro TTS. It runs Kokoro through `kokoro-js` and ONNX Runtime Web, so GitHub Pages only needs to host the built files.

## Local development

```bash
pnpm install
pnpm run dev
```

## Build

```bash
pnpm run build
```

The static site is emitted to `dist/`.

## Browser API

The app exposes a reusable browser-side API. It runs in a Web Worker, preloads the model, and returns WAV bytes.

```ts
import { kokoroApi } from "./src/kokoro-api";

await kokoroApi.preload({ device: "auto", dtype: "q4" });

const result = await kokoroApi.synthesize({
  text: "Fast Kokoro synthesis from a browser API.",
  voice: "af_heart",
  speed: 1,
  device: "auto",
  dtype: "q4",
  batchSize: 8
});

const audioUrl = URL.createObjectURL(new Blob([result.wav], { type: result.mimeType }));
```

Use `device: "auto"` for fastest practical startup. It selects WebGPU when available and falls back to WASM. `q4` is the fastest default; WebGPU automatically maps it to `q4f16` for GPU execution.

## Public GitHub Pages API

GitHub Pages cannot run a server-side HTTP API, but the published page can be used as a browser API.

### URL execution

Open this URL to preload the app, synthesize, and play:

```text
https://summerofstart.github.io/kokoroTTSweb/?text=Hello%20from%20Kokoro&voice=af_heart&speed=1&device=auto&dtype=q4&batchSize=8&autoplay=1
```

### iframe postMessage API

```html
<iframe id="kokoro" src="https://summerofstart.github.io/kokoroTTSweb/" hidden></iframe>
<script>
  const frame = document.getElementById("kokoro");

  window.addEventListener("message", (event) => {
    if (event.origin !== "https://summerofstart.github.io") return;
    if (event.data.type !== "kokoro:result") return;

    const audioUrl = URL.createObjectURL(
      new Blob([event.data.result.wav], { type: event.data.result.mimeType })
    );
    new Audio(audioUrl).play();
  });

  frame.addEventListener("load", () => {
    frame.contentWindow.postMessage(
      {
        type: "kokoro:synthesize",
        id: "demo-1",
        options: {
          text: "Kokoro TTS from the GitHub Pages API.",
          voice: "af_heart",
          speed: 1,
          device: "auto",
          dtype: "q4",
          batchSize: 8
        }
      },
      "https://summerofstart.github.io"
    );
  });
</script>
```

### Same-page console API

```js
const result = await window.KokoroTTSWeb.synthesize({
  text: "Run Kokoro from the page console.",
  voice: "af_heart",
  device: "auto",
  dtype: "q4",
  batchSize: 8
});
new Audio(URL.createObjectURL(new Blob([result.wav], { type: result.mimeType }))).play();
```

## GitHub Pages

1. Push this repository to GitHub.
2. In the repository settings, open **Pages**.
3. Use either a GitHub Actions deployment or publish the contents of `dist/`.

For a simple manual deploy from a machine with repository access:

```bash
pnpm install
pnpm run build
pnpm dlx gh-pages -d dist
```

The Vite `base` setting is `./`, so the app works from a project page path like `/your-repo/`.

## Browser notes

The first generation downloads the Kokoro ONNX model into the browser cache. WASM is the safest runtime. WebGPU can be faster on browsers and devices that support it.

## Voice upload

The browser app supports Kokoro-compatible `.bin` voice packs. Uploading WAV/MP3 recordings is not enough to clone a voice; that requires a separate voice-training pipeline. Uploaded `.bin` files are cached locally in the browser and can be selected immediately.

```js
const data = await file.arrayBuffer();
await window.KokoroTTSWeb.importVoice("custom_voice", data);
const result = await window.KokoroTTSWeb.synthesize({
  text: "Using an uploaded Kokoro voice pack.",
  voice: "custom_voice",
  device: "auto",
  dtype: "q4"
});
```
