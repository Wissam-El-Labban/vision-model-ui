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
// Local image generation (diffusers)
// --------------------------------------------------------------------------- #
export interface SdModel {
  id: string;
  label: string;
  downloaded: boolean;
  turbo: boolean;
  photoreal: boolean;
  size_gb: number;
}

export interface SdInfo {
  available: boolean;
  device: string;
  models: SdModel[];
}

export async function getSdInfo(): Promise<SdInfo> {
  const res = await fetch("/api/generate/models");
  if (!res.ok) throw new Error(`sd models: ${res.status}`);
  return res.json();
}

/** Explicitly download an image model (opt-in). Streams status lines. */
export async function pullSdModel(
  model: string,
  onStatus: (message: string) => void
): Promise<void> {
  const res = await fetch("/api/generate/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!res.ok) throw new Error(`sd pull: ${res.status}`);
  await readLines(res, (line) => {
    try {
      const ev = JSON.parse(line);
      if (ev.type === "status") onStatus(ev.message as string);
      else if (ev.type === "error") throw new Error(ev.message);
    } catch {
      /* ignore keepalives */
    }
  });
}

export interface GenerateParams {
  mode: "txt2img" | "img2img";
  model: string;
  prompt: string;
  negative_prompt: string;
  init_image_hash?: string | null;
  steps?: number | null;
  guidance?: number | null;
  strength?: number | null;
  width: number;
  height: number;
  seed?: number | null;
  ollama_url: string;
}

interface GenerateHandlers {
  onStatus?: (message: string) => void;
  onProgress?: (step: number, total: number) => void;
  onImage: (r: { hash: string; seed: number; width: number; height: number }) => void;
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
    else if (ev.type === "image")
      handlers.onImage(ev as unknown as { hash: string; seed: number; width: number; height: number });
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
