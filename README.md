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
  dtype: "q4"
});

const audioUrl = URL.createObjectURL(new Blob([result.wav], { type: result.mimeType }));
```

Use `device: "auto"` for fastest practical startup. It selects WebGPU when available and falls back to WASM. `q4` is the fastest default; WebGPU automatically maps it to `q4f16` for GPU execution.

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
