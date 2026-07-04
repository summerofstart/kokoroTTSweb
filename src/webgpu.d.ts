/**
 * WebGPU type declarations for the Kokoro TTS WebGPU Math Engine.
 *
 * TypeScript 5.9.x DOM lib does NOT include WebGPU types.  This file
 * provides everything needed so the project compiles cleanly with
 * `tsc --noEmit` without installing `@webgpu/types`.
 *
 * Only types actually used by webgpu-math.ts and VoiceVectorApp.tsx
 * are declared here.
 */

/* ─── Navigator extension ─────────────────────────────────────────────────── */
interface Navigator {
  readonly gpu?: GPU;
}

interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
}

interface GPURequestAdapterOptions {
  powerPreference?: GPUPowerPreference;
}

type GPUPowerPreference = "low-power" | "high-performance";

/* ─── Adapter ─────────────────────────────────────────────────────────────── */
interface GPUAdapter {
  readonly limits: GPUSupportedLimits;
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUSupportedLimits {
  maxStorageBufferBindingSize: number;
}

interface GPUDeviceDescriptor {
  requiredLimits?: Partial<GPUSupportedLimits>;
  defaultQueue?: GPUQueueDescriptor;
}

interface GPUQueueDescriptor {}

/* ─── Device ──────────────────────────────────────────────────────────────── */
interface GPUDevice {
  readonly limits: GPUSupportedLimits;
  readonly queue: GPUQueue;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor): Promise<GPUComputePipeline>;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
  createPipelineLayout(descriptor?: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  addEventListener(type: "uncapturederror", callback: (evt: GPUUncapturedErrorEvent) => void): void;
}

/* ─── Buffer ──────────────────────────────────────────────────────────────── */
interface GPUBufferDescriptor {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

interface GPUBuffer {
  readonly size: number;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

/* ─── Shader Module ───────────────────────────────────────────────────────── */
interface GPUShaderModuleDescriptor {
  code: string;
  label?: string;
}

interface GPUShaderModule {}

/* ─── Compute Pipeline ────────────────────────────────────────────────────── */
interface GPUComputePipelineDescriptor {
  layout: "auto" | GPUPipelineLayout;
  compute: GPUProgrammableStage;
  label?: string;
}

interface GPUProgrammableStage {
  module: GPUShaderModule;
  entryPoint: string;
}

interface GPUComputePipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

/* ─── Bind Group ──────────────────────────────────────────────────────────── */
interface GPUBindGroup {}

interface GPUBindGroupDescriptor {
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
  label?: string;
}

interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindingResource;
}

type GPUBindingResource = GPUBufferBinding;

interface GPUBufferBinding {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

interface GPUBindGroupLayout {}

interface GPUPipelineLayout {}

interface GPUPipelineLayoutDescriptor {
  bindGroupLayouts?: GPUBindGroupLayout[];
}

/* ─── Command Encoding ────────────────────────────────────────────────────── */
interface GPUCommandEncoderDescriptor {
  label?: string;
}

interface GPUCommandEncoder {
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  finish(): GPUCommandBuffer;
}

interface GPUCommandBuffer {}

interface GPUComputePassDescriptor {
  label?: string;
}

/* ─── Compute Pass ────────────────────────────────────────────────────────── */
interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
  label?: string;
}

/* ─── Queue ───────────────────────────────────────────────────────────────── */
interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  onSubmittedWorkDone(): Promise<undefined>;
  writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: BufferSource, dataOffset?: number, size?: number): void;
}

/* ─── Error ───────────────────────────────────────────────────────────────── */
interface GPUUncapturedErrorEvent extends Event {
  readonly error: GPUError;
}

interface GPUError {
  readonly message: string;
}

/* ─── Usage / Map mode bit flags ──────────────────────────────────────────── */
declare var GPUBufferUsage: {
  readonly MAP_READ: 0x0001;
  readonly MAP_WRITE: 0x0002;
  readonly COPY_SRC: 0x0004;
  readonly COPY_DST: 0x0008;
  readonly INDEX: 0x0010;
  readonly VERTEX: 0x0020;
  readonly UNIFORM: 0x0040;
  readonly STORAGE: 0x0080;
  readonly INDIRECT: 0x0100;
  readonly QUERY_RESOLVE: 0x0200;
};

declare var GPUMapMode: {
  readonly READ: 0x0001;
  readonly WRITE: 0x0002;
};
