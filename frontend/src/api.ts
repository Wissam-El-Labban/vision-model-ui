import type { ChatMessage, RunningModel, VersionInfo } from "./types";

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

/** Stream chat tokens. Calls onToken for each chunk of assistant text. */
export async function streamChat(
  ollamaUrl: string,
  model: string,
  messages: ChatMessage[],
  onToken: (token: string) => void,
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
  if (!res.ok || !res.body) throw new Error(`chat: ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    onToken(decoder.decode(value, { stream: true }));
  }
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
