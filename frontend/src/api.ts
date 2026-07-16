import type {
  ChatDetail,
  ChatMessage,
  ChatSummary,
  RunningModel,
  VersionInfo,
} from "./types";

/** Strip the `data:image/...;base64,` prefix Ollama doesn't want. */
function toRawBase64(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

function withImages(m: ChatMessage) {
  const images = (m.images ?? []).map(toRawBase64);
  return images.length
    ? { role: m.role, content: m.content, images }
    : { role: m.role, content: m.content };
}

export async function getModels(
  ollamaUrl: string
): Promise<{ vision: string[]; all: string[] }> {
  const res = await fetch(`/api/models?ollama_url=${encodeURIComponent(ollamaUrl)}`);
  if (!res.ok) throw new Error(`models: ${res.status}`);
  return res.json();
}

export async function getVersion(ollamaUrl: string): Promise<VersionInfo> {
  const res = await fetch(
    `/api/ollama/version?ollama_url=${encodeURIComponent(ollamaUrl)}`
  );
  if (!res.ok) throw new Error(`version: ${res.status}`);
  return res.json();
}

export async function getRunning(ollamaUrl: string): Promise<RunningModel[]> {
  const res = await fetch(`/api/ps?ollama_url=${encodeURIComponent(ollamaUrl)}`);
  if (!res.ok) throw new Error(`ps: ${res.status}`);
  return (await res.json()).models ?? [];
}

export async function unloadAll(ollamaUrl: string): Promise<string[]> {
  const res = await fetch(
    `/api/models/unload?ollama_url=${encodeURIComponent(ollamaUrl)}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(`unload: ${res.status}`);
  return (await res.json()).unloaded ?? [];
}

export async function deleteModel(ollamaUrl: string, name: string): Promise<void> {
  const res = await fetch(
    `/api/models/${encodeURIComponent(name)}?ollama_url=${encodeURIComponent(ollamaUrl)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error(`delete: ${res.status}`);
}

/** Generic line-stream reader for ndjson / plain-text streaming endpoints. */
async function readLines(
  res: Response,
  onLine: (line: string) => void
): Promise<void> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      if (line.trim()) onLine(line);
    }
  }
  if (buffer.trim()) onLine(buffer);
}

export interface Usage {
  used: number;
  prompt_tokens: number;
  eval_tokens: number;
  num_ctx: number;
}

interface ChatHandlers {
  onToken: (token: string) => void;
  onUsage?: (usage: Usage) => void;
  onError?: (message: string) => void;
}

/** Stream a chat response (NDJSON events: token / usage / error). */
export async function streamChat(
  ollamaUrl: string,
  model: string,
  messages: ChatMessage[],
  handlers: ChatHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      ollama_url: ollamaUrl,
      messages: messages.map(withImages),
    }),
    signal,
  });
  if (!res.ok) throw new Error(`chat: ${res.status}`);
  await readLines(res, (line) => {
    let ev: { type: string; [k: string]: unknown };
    try {
      ev = JSON.parse(line);
    } catch {
      return;
    }
    if (ev.type === "token") handlers.onToken(ev.text as string);
    else if (ev.type === "usage") handlers.onUsage?.(ev as unknown as Usage);
    else if (ev.type === "error") handlers.onError?.(ev.message as string);
  });
}

export async function pullModel(
  ollamaUrl: string,
  name: string,
  onStatus: (status: string, pct: number | null) => void
): Promise<void> {
  const res = await fetch(
    `/api/models/pull?ollama_url=${encodeURIComponent(ollamaUrl)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, ollama_url: ollamaUrl }),
    }
  );
  if (!res.ok) throw new Error(`pull: ${res.status}`);
  await readLines(res, (line) => {
    try {
      const d = JSON.parse(line);
      const pct =
        d.total && d.completed ? d.completed / d.total : null;
      onStatus(d.status ?? "", pct);
    } catch {
      /* ignore non-json keepalives */
    }
  });
}

// --------------------------------------------------------------------------- #
// Image models. A FLUX.2 model serves every mode; a FLUX.1 one serves exactly one
// (dev creates, Kontext edits) — hence `roles`, not `role`.
// --------------------------------------------------------------------------- #
export type FluxRole = "create" | "edit" | "animate";

export interface FluxModel {
  name: string; // the weight file, which is what /api/generate takes
  label: string;
  roles: FluxRole[];
  bundle: string | null; // catalog id, or null for a user-added model
  family: string;
  size_gb: number;
}

/** An installable model: its weights, text encoder and VAE, downloaded together. */
export interface FluxBundle {
  id: string;
  label: string;
  blurb: string;
  family: string;
  roles: FluxRole[];
  size_gb: number;
  vram_gb: number;
  gated: boolean;
  installed: boolean;
  needed_gb: number; // still to download (a part-finished install counts what it has)
}

export interface FluxInstalling {
  id: string;
  label: string;
  file: string;
  pct: number;
  done: number;
  total: number;
  index: number; // file N…
  count: number; // …of M
}

export interface FluxCatalog {
  runtime_ready: boolean; // engine installed — enough to download a model
  available: boolean; // ...and a model installed, so we can actually generate
  disk_free_gb: number;
  bundles: FluxBundle[];
  hf_token: "saved" | "env" | null;
  /** A download running on the server right now — it outlives the page that started
   *  it, so a reload can find it here rather than assuming nothing is happening. */
  installing: FluxInstalling | null;
}

export interface FluxProgress {
  file: string;
  done: number;
  total: number;
  pct: number;
  index?: number; // file N…
  count?: number; // …of M (a sharded encoder arrives in pieces, then gets stitched)
}

export async function getFluxModels(): Promise<{ available: boolean; models: FluxModel[] }> {
  const res = await fetch("/api/flux/models");
  if (!res.ok) throw new Error(`flux models: ${res.status}`);
  return res.json();
}

export async function getFluxCatalog(): Promise<FluxCatalog> {
  const res = await fetch("/api/flux/catalog");
  if (!res.ok) throw new Error(`flux catalog: ${res.status}`);
  return res.json();
}

/** Install a catalog model. Tens of GB, so progress streams file by file. */
export async function installFluxBundle(
  id: string,
  onStatus: (message: string) => void,
  onProgress: (p: FluxProgress) => void
): Promise<void> {
  const res = await fetch("/api/flux/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `flux install: ${res.status}`);
  }
  let failure: string | null = null;
  await readLines(res, (line) => {
    try {
      const ev = JSON.parse(line);
      if (ev.type === "status") onStatus(ev.message as string);
      else if (ev.type === "progress") onProgress(ev as FluxProgress);
      else if (ev.type === "error") failure = ev.message as string;
    } catch {
      /* ignore keepalives */
    }
  });
  if (failure) throw new Error(failure);
}

export async function deleteFluxBundle(id: string): Promise<void> {
  const res = await fetch(`/api/flux/bundles/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `flux bundle delete: ${res.status}`);
  }
}

// --------------------------------------------------------------------------- #
// Text encoders — the conditioning half of a FLUX.2 model. The bundled one is a
// default, not a fixture: any checkpoint of the same architecture works, and a lighter
// quant is the usual reason to swap ([dev]'s Mistral is 48 GB in bf16).
// --------------------------------------------------------------------------- #
export interface FluxTextEncoder {
  name: string;
  size_gb: number;
  default_for: string[]; // labels of the models that ship with this one
}

export interface FluxTextEncoders {
  encoders: FluxTextEncoder[];
  selected: Record<string, string>; // bundle id -> the encoder it will load
}

export async function getTextEncoders(): Promise<FluxTextEncoders> {
  const res = await fetch("/api/flux/text-encoders");
  if (!res.ok) throw new Error(`text encoders: ${res.status}`);
  return res.json();
}

/** Add one from HuggingFace. These run to 48 GB and a sharded repo is fetched piece by
 *  piece and then stitched, so progress streams the same way an install's does. */
export async function pullTextEncoder(
  repo: string,
  onStatus: (message: string) => void,
  onProgress: (p: FluxProgress) => void
): Promise<void> {
  const res = await fetch("/api/flux/text-encoders/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `text encoder pull: ${res.status}`);
  }
  let failure: string | null = null;
  await readLines(res, (line) => {
    try {
      const ev = JSON.parse(line);
      if (ev.type === "status") onStatus(ev.message as string);
      else if (ev.type === "progress") onProgress(ev as FluxProgress);
      else if (ev.type === "error") failure = ev.message as string;
    } catch {
      /* ignore keepalives */
    }
  });
  if (failure) throw new Error(failure);
}

/** Point a model at a different encoder. An empty name restores its default. */
export async function selectTextEncoder(bundleId: string, name: string): Promise<void> {
  const res = await fetch("/api/flux/text-encoders/select", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bundle_id: bundleId, name }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `select encoder: ${res.status}`);
  }
}

export async function deleteTextEncoder(name: string): Promise<void> {
  const res = await fetch(`/api/flux/text-encoders/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `delete encoder: ${res.status}`);
  }
}

// --------------------------------------------------------------------------- #
// HuggingFace token — only needed for gated repos. It lives on the server; the
// browser can set or clear it but never reads it back.
// --------------------------------------------------------------------------- #
export type HfTokenSource = "saved" | "env" | null;

export async function getHfToken(): Promise<HfTokenSource> {
  const res = await fetch("/api/settings/hf-token");
  if (!res.ok) throw new Error(`hf token: ${res.status}`);
  return (await res.json()).source;
}

/** Validated against HuggingFace before it's saved, so a bad paste fails here. */
export async function setHfToken(token: string): Promise<string> {
  const res = await fetch("/api/settings/hf-token", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  const body = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) throw new Error(body.detail ?? `hf token: ${res.status}`);
  return body.user as string;
}

export async function clearHfToken(): Promise<void> {
  const res = await fetch("/api/settings/hf-token", { method: "DELETE" });
  if (!res.ok) throw new Error(`hf token: ${res.status}`);
}

/** Download an extra FLUX UNet from a HuggingFace repo (owner/name or
 *  owner/name:file.gguf). Streams coarse progress. */
export async function pullFluxModel(
  repo: string,
  onStatus: (message: string) => void,
  onProgress: (p: FluxProgress) => void
): Promise<void> {
  const res = await fetch("/api/flux/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `flux pull: ${res.status}`);
  }
  let failure: string | null = null;
  await readLines(res, (line) => {
    try {
      const ev = JSON.parse(line);
      if (ev.type === "status") onStatus(ev.message as string);
      else if (ev.type === "progress") onProgress(ev as FluxProgress);
      else if (ev.type === "error") failure = ev.message as string;
    } catch {
      /* ignore keepalives */
    }
  });
  if (failure) throw new Error(failure);
}

export async function deleteFluxModel(name: string): Promise<void> {
  const res = await fetch(`/api/flux/models/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `flux delete: ${res.status}`);
  }
}

export type GenMode = "txt2img" | "img2img" | "edit" | "compose" | "animate";

/** Rewrite a prompt with a local vision model that can see the attached images.
 *  Always resolves: if Ollama is unreachable the backend falls back to its static
 *  template and says so via `source`. */
export async function enhancePrompt(params: {
  prompt: string;
  mode: GenMode;
  model: string;
  image_hashes?: string[];
  ollama_url: string;
}): Promise<{ prompt: string; source: "vlm" | "template" }> {
  const res = await fetch("/api/flux/enhance", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `enhance: ${res.status}`);
  }
  return res.json();
}

export interface GenerateParams {
  mode: GenMode;
  flux_model?: string | null; // which FLUX UNet ("" / null = that mode's default)
  prompt: string;
  init_image_hash?: string | null; // img2img / edit: the source image
  ref_image_hashes?: string[]; // compose: reference images to fuse
  steps?: number | null;
  guidance?: number | null;
  strength?: number | null; // img2img: how far to drift from the source
  enhance?: boolean; // wrap create prompts in a photoreal template
  width: number;
  height: number;
  seconds?: number | null; // animate: clip length (capped at 5s)
  seed?: number | null;
  ollama_url: string;
}

/** The final `image` event. `url` and `model_label` are what the backend actually
 *  did — the store's URL carries the real extension, and `model` names the
 *  transformer that ran, which is not necessarily the one that was requested.
 *
 *  `kind` discriminates a still from a video. Both arrive as this one event so the
 *  stream has a single terminal path; only what the client does with `url` differs. */
export interface GeneratedImage {
  kind?: "image" | "video";
  hash: string;
  seed: number;
  width: number;
  height: number;
  url?: string;
  model?: string;
  model_label?: string;
}

interface GenerateHandlers {
  onStatus?: (message: string) => void;
  onProgress?: (step: number, total: number) => void;
  onImage: (r: GeneratedImage) => void;
  onError?: (message: string) => void;
}

/** Stream a local image generation (NDJSON: status / progress / image / error). */
export async function generate(
  params: GenerateParams,
  handlers: GenerateHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `generate: ${res.status}`);
  }
  await readLines(res, (line) => {
    let ev: { type: string; [k: string]: unknown };
    try {
      ev = JSON.parse(line);
    } catch {
      return;
    }
    if (ev.type === "status") handlers.onStatus?.(ev.message as string);
    else if (ev.type === "progress")
      handlers.onProgress?.(ev.step as number, ev.total as number);
    else if (ev.type === "image") handlers.onImage(ev as unknown as GeneratedImage);
    else if (ev.type === "error") handlers.onError?.(ev.message as string);
  });
}

// --------------------------------------------------------------------------- #
// Chat history persistence
// --------------------------------------------------------------------------- #
export async function listChats(): Promise<ChatSummary[]> {
  const res = await fetch("/api/chats");
  if (!res.ok) throw new Error(`chats: ${res.status}`);
  return (await res.json()).chats ?? [];
}

export async function getChat(id: string): Promise<ChatDetail> {
  const res = await fetch(`/api/chats/${id}`);
  if (!res.ok) throw new Error(`chat: ${res.status}`);
  return res.json();
}

export async function putChat(
  id: string,
  body: {
    model: string | null;
    system_prompt: string;
    pinned_hashes: string[];
    system_image_hash: string | null;
  }
): Promise<void> {
  const res = await fetch(`/api/chats/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`putChat: ${res.status}`);
}

export async function appendMessage(
  id: string,
  body: {
    role: string;
    content: string;
    model: string | null;
    image_hashes: string[];
    context_hashes?: string[];
  }
): Promise<void> {
  const res = await fetch(`/api/chats/${id}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`appendMessage: ${res.status}`);
}

/** Upload images (full + thumbnail data-URLs). Returns their content hashes. */
export async function uploadImages(
  items: { full: string; thumb: string | null }[]
): Promise<string[]> {
  if (items.length === 0) return [];
  const res = await fetch("/api/images", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) throw new Error(`uploadImages: ${res.status}`);
  return (await res.json()).hashes ?? [];
}

export async function generateTitle(
  id: string,
  model: string,
  ollamaUrl: string
): Promise<string> {
  const res = await fetch(`/api/chats/${id}/title`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, ollama_url: ollamaUrl }),
  });
  if (!res.ok) throw new Error(`title: ${res.status}`);
  return (await res.json()).title ?? "";
}

export async function deleteChat(id: string): Promise<void> {
  const res = await fetch(`/api/chats/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`deleteChat: ${res.status}`);
}

/** Fetch a stored image URL and convert it to a data-URL for in-memory use,
 *  so the existing send/display code (which expects data-URLs) is unchanged. */
export async function urlToDataUrl(url: string): Promise<string> {
  const res = await fetch(url);
  // Fail loudly on a missing file (e.g. a GC'd image) instead of encoding the
  // 404 body into a bogus data-URL that renders as a broken image.
  if (!res.ok) throw new Error(`image ${res.status}: ${url}`);
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export async function upgradeOllama(
  ollamaUrl: string,
  onLine: (line: string) => void
): Promise<void> {
  const res = await fetch(
    `/api/ollama/upgrade?ollama_url=${encodeURIComponent(ollamaUrl)}`,
    { method: "POST" }
  );
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `upgrade: ${res.status}`);
  }
  await readLines(res, onLine);
}
