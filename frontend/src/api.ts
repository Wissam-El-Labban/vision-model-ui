import type {
  AgentTool,
  AgentToolId,
  ChatDetail,
  ChatMessage,
  ChatSummary,
  ControlKind,
  DocInfo,
  JobInfo,
  JobStatus,
  RunningModel,
  Source,
  TaskInfo,
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

/** One installed model's size and capabilities, as Ollama reports them. */
export interface ModelDetail {
  name: string;
  /** Ollama's own string, e.g. "9.7B" — shown to the user verbatim. */
  parameter_size: string;
  /** That string as a number of billions — what the composer's small-agent-model
   *  warning tests. 0 when Ollama didn't say. */
  params_b: number;
  /** e.g. ["vision", "completion", "tools", "thinking"]. */
  capabilities: string[];
}

/** Installed models grouped by what they can do. `agent` is the subset that can
 *  run agent mode (tools + vision); `details` says why, and how big each is. */
export async function getModels(
  ollamaUrl: string
): Promise<{
  vision: string[];
  all: string[];
  agent: string[];
  details: ModelDetail[];
}> {
  const res = await fetch(`/api/models?ollama_url=${encodeURIComponent(ollamaUrl)}`);
  if (!res.ok) throw new Error(`models: ${res.status}`);
  return res.json();
}

/** The tools the agent can be given, for the composer's Tools menu. */
export async function getAgentTools(): Promise<{
  tools: AgentTool[];
  defaults: AgentToolId[];
}> {
  const res = await fetch("/api/agent/tools");
  if (!res.ok) throw new Error(`agent tools: ${res.status}`);
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

/** Where a loaded-models row is: `loading` (asked for, not in VRAM yet),
 *  `active` (loaded and serving a request), `loaded` (idle), `unloading`. */
export type LoadState = "loading" | "active" | "loaded" | "unloading";

/** A model Ollama has loaded, or is loading. `gpu_pct` under 100 means part of
 *  it runs on the CPU. `expires_at` is when Ollama unloads it (ISO time). Sizes
 *  are null while it's still loading. */
export interface LoadedOllamaModel {
  name: string;
  kind: "vision" | "chat" | "embedding";
  state: LoadState;
  size_gb: number | null;
  vram_gb: number | null;
  gpu_pct: number | null;
  context: number | null;
  expires_at: string | null;
}

/** A model file the image/video engine (ComfyUI) holds. `family`, `kind` and the
 *  bundle `label` are set for transformers only. */
export interface LoadedGenModel {
  file: string;
  role: string;
  label: string;
  family: string | null;
  kind: "image" | "video" | null;
  video: boolean;
  state: LoadState;
}

/** Live memory of this machine, in GiB. `gpu`/`ram` are null when unreadable;
 *  `flux_gb` and `gen_models` are null while the image engine isn't running,
 *  `ollama_models` while Ollama can't be reached. `gen_models.known` is false when
 *  the backend can't tell what the engine holds (it restarted while it ran). */
export interface SystemMemory {
  gpu: { total_gb: number; used_gb: number } | null;
  ram: { total_gb: number; available_gb: number; used_gb: number; swap_gb: number } | null;
  flux_gb: number | null;
  ollama_gb: number | null;
  ollama_models: LoadedOllamaModel[] | null;
  gen_models: {
    status: "starting" | "ready";
    known: boolean;
    models: LoadedGenModel[];
  } | null;
}

export async function getSystemMemory(ollamaUrl: string): Promise<SystemMemory> {
  const res = await fetch(`/api/system/memory?ollama_url=${encodeURIComponent(ollamaUrl)}`);
  if (!res.ok) throw new Error(`memory: ${res.status}`);
  return res.json();
}

/** Whether a chat model plus FLUX models suit this machine. "amber" works but
 *  slower; "red" is likely to crash or crawl; "unknown" means stay quiet. */
export interface SystemFit {
  level: "ok" | "amber" | "red" | "unknown";
  messages: string[];
  /** Whether "Unload vision model before generating" would actually help: the
   *  two fit on the GPU together (so the vision model stays resident) but overrun
   *  system RAM. When they don't fit on the GPU, the backend unloads on its own
   *  and the toggle changes nothing. */
  unload_helps?: boolean;
  /** A rule about the two models *together* tripped (GPU or RAM). */
  pair?: boolean;
}

export async function getFit(
  ollamaUrl: string,
  model: string,
  flux: string[]
): Promise<SystemFit> {
  const q = new URLSearchParams({ ollama_url: ollamaUrl, model });
  for (const f of flux) q.append("flux", f);
  const res = await fetch(`/api/system/fit?${q}`);
  if (!res.ok) throw new Error(`fit: ${res.status}`);
  return res.json();
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

// --------------------------------------------------------------------------- #
// Background tasks: downloads, indexing and the Ollama upgrade run on the backend,
// so a reload loses only the page's view of them, which `attachTask` gets back.
// --------------------------------------------------------------------------- #
export type TaskEvent = { type?: string; [k: string]: unknown };

/** Read a task's NDJSON stream into `onEvent`. Throws the backend's reason for a
 *  refused request, and the task's own error once its stream ends. */
async function followTask(
  res: Response,
  onEvent: (ev: TaskEvent) => void,
  what: string
): Promise<void> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `${what}: ${res.status}`);
  }
  let failure: string | null = null;
  await readLines(res, (line) => {
    let ev: TaskEvent;
    try {
      ev = JSON.parse(line);
    } catch {
      return; // keepalives
    }
    if (ev.type === "job" || ev.type === "end") return;
    if (ev.type === "error") failure = ev.message as string;
    else onEvent(ev);
  });
  if (failure) throw new Error(failure);
}

/** Tasks running on the backend right now. */
export async function listTasks(): Promise<TaskInfo[]> {
  const res = await fetch("/api/tasks");
  if (!res.ok) throw new Error(`tasks: ${res.status}`);
  return (await res.json()).tasks ?? [];
}

/** Follow a task that was started before this page loaded: its events so far,
 *  then live until it ends. Read with the same reader that started it. */
export async function attachTask(
  id: string,
  onEvent: (ev: TaskEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`/api/jobs/${id}/events`, { signal });
  await followTask(res, onEvent, "task");
}

/** An image-engine download's events: status lines and byte progress. */
export function fluxTaskEvents(
  onStatus: (message: string) => void,
  onProgress: (p: FluxProgress) => void
): (ev: TaskEvent) => void {
  return (ev) => {
    if (ev.type === "status") onStatus(ev.message as string);
    else if (ev.type === "progress") onProgress(ev as unknown as FluxProgress);
  };
}

export interface Usage {
  used: number;
  prompt_tokens: number;
  eval_tokens: number;
  num_ctx: number;
}

/** What every turn's stream also says about the backend job running it. */
export interface JobHooks {
  /** First event: the job's id, for Stop and Retry. */
  onJob?: (id: string) => void;
  /** Last event: how the turn ended. */
  onEnd?: (status: JobStatus, error: string | null) => void;
  /** The chat was deleted while the turn ran; nothing more was recorded. */
  onDeleted?: () => void;
}

type JobEvent = { type: string; [k: string]: unknown };

export interface ChatHandlers extends JobHooks {
  onToken: (token: string) => void;
  /** Reasoning tokens, only sent when `think` is on and the model can think. */
  onThinking?: (token: string) => void;
  onUsage?: (usage: Usage) => void;
  onError?: (message: string) => void;
  /** The document excerpts the answer was given, before its first token. */
  onSources?: (sources: Source[]) => void;
  /** Documents were picked but couldn't be searched; the answer goes ahead without. */
  onDocsError?: (message: string) => void;
}

/** How the backend records a chat turn: the chat, what the user typed and
 *  attached, and the numbered images the answer can refer to. */
export interface ChatRecord {
  chat_id: string;
  user_content: string;
  user_image_hashes: string[];
  context_hashes: string[];
  title_model: string | null;
}

/** Stream a chat response (NDJSON events: sources / thinking / token / usage / error).
 *  `docs` are the chat's documents and the text to search them with. With
 *  `record`, the backend writes both turns itself, so a reload loses neither. */
export async function streamChat(
  ollamaUrl: string,
  model: string,
  messages: ChatMessage[],
  handlers: ChatHandlers,
  signal?: AbortSignal,
  think = false,
  docs: { ids: string[]; query: string } = { ids: [], query: "" },
  record?: ChatRecord
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      ollama_url: ollamaUrl,
      messages: messages.map(withImages),
      think,
      doc_ids: docs.ids,
      doc_query: docs.query,
      ...record,
    }),
    signal,
  });
  await followTurn(res, { kind: "chat", on: handlers }, "chat");
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
  await followTask(res, ollamaPullEvents(onStatus), "pull");
}

/** An Ollama pull's events: its own status lines, with bytes as a fraction. */
export function ollamaPullEvents(
  onStatus: (status: string, pct: number | null) => void
): (ev: TaskEvent) => void {
  return (d) => {
    if (d.type) return; // the task's own `done`
    const total = d.total as number | undefined;
    const completed = d.completed as number | undefined;
    onStatus((d.status as string) ?? "", total && completed ? completed / total : null);
  };
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
  /** The text encoder the graph will actually load — the backend's `encoder_for`,
   *  so this can't drift from what runs. FLUX.1 reports its pair as "a + b". */
  encoder: string;
  /** The LoRAs chained onto this transformer, in attach order. Empty when none. */
  loras: FluxLoraPick[];
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
  await followTask(res, fluxTaskEvents(onStatus, onProgress), "flux install");
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
  /** Bundle ids whose architecture this encoder matches — the only models it can
   *  serve. Computed from the checkpoint's embedding shape, so a lighter quant of the
   *  right architecture still fits and a different architecture never does. An
   *  encoder whose architecture can't be read (a GGUF, say) fits everything rather
   *  than nothing, so detection can't hide a working option. */
  fits: string[];
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
  await followTask(res, fluxTaskEvents(onStatus, onProgress), "text encoder pull");
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
// LoRA adapters — an optional low-rank patch over a model's transformer, the
// practical way to change what a model renders without replacing a 64 GB
// checkpoint. Every model starts with none and can be put back to none.
// --------------------------------------------------------------------------- #
export interface FluxLora {
  name: string;
  size_mb: number; // patches, not checkpoints — MB, not GB
  /** Transformer filenames this adapter was trained against — the only ones it can
   *  patch. An adapter for another base binds to almost none of the layers it names
   *  and reads as one that barely works, rather than as a mistake, so it isn't
   *  offered. Empty means nothing installed fits it; an adapter whose base can't be
   *  determined fits everything rather than nothing. */
  fits: string[];
}

/** One attached adapter and its weight. */
export interface FluxLoraPick {
  name: string;
  strength: number;
  /** Whether this is the model's *control* adapter — the one that teaches it to obey
   *  a control map, and so the one the Control tab's strength dial scales. Flagged by
   *  the user rather than guessed from the filename: an adapter's name is a hint and
   *  its job is something only the person who installed it knows. */
  control?: boolean;
}

export interface FluxLoras {
  loras: FluxLora[];
  /** Transformer filename -> its attached picks, in attach order. An empty array
   *  means none. Keyed per transformer, not per bundle: FLUX.1 ships dev and Kontext
   *  together and their adapters don't cross. */
  selected: Record<string, FluxLoraPick[]>;
}

export async function getLoras(): Promise<FluxLoras> {
  const res = await fetch("/api/flux/loras");
  if (!res.ok) throw new Error(`loras: ${res.status}`);
  return res.json();
}

/** Add one from HuggingFace (owner/repo:file, or a repo holding exactly one).
 *  Streams progress like the other pulls, though a LoRA is MBs not GBs. */
/** Where a LoRA is being installed from. HuggingFace wants `owner/repo[:file]`;
 *  CivitAI wants a page URL, a download URL, an AIR, or the id out of any of them. */
export type LoraSource = "huggingface" | "civitai";

export async function pullLora(
  repo: string,
  source: LoraSource,
  onStatus: (message: string) => void,
  onProgress: (p: FluxProgress) => void
): Promise<void> {
  const res = await fetch("/api/flux/loras/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo, source }),
  });
  await followTask(res, fluxTaskEvents(onStatus, onProgress), "lora pull");
}

/** Replace the whole set of LoRAs attached to one transformer. An empty array
 *  detaches everything — the "None" option. */
export async function setLoraPicks(model: string, picks: FluxLoraPick[]): Promise<void> {
  const res = await fetch("/api/flux/loras/select", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, picks }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `select loras: ${res.status}`);
  }
}

/** Remove a LoRA from disk. Any model using it falls back to none. */
export async function deleteLora(name: string): Promise<void> {
  const res = await fetch(`/api/flux/loras/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `delete lora: ${res.status}`);
  }
}

// --------------------------------------------------------------------------- #
// Control preprocessors — the models that turn a source image into a control map.
// Not transformers: they never appear in the model picker and are named by the
// control kind a generation asks for, not chosen per generation.
// --------------------------------------------------------------------------- #
export interface FluxPreprocessor {
  kind: ControlKind;
  /** Install id, or null for one that needs no weights (canny). */
  id: string | null;
  installed: boolean;
  /** True when the control kind is a built-in filter rather than a downloaded model,
   *  so the panel offers no install or delete for it. */
  builtin: boolean;
  label: string;
  note: string;
  /** Download size before install, on-disk size after. 0 for a built-in. */
  size_gb: number;
}

export async function getPreprocessors(): Promise<FluxPreprocessor[]> {
  const res = await fetch("/api/flux/preprocessors");
  if (!res.ok) throw new Error(`preprocessors: ${res.status}`);
  return (await res.json()).preprocessors ?? [];
}

export async function installPreprocessor(
  id: string,
  onStatus: (message: string) => void,
  onProgress: (p: FluxProgress) => void
): Promise<void> {
  const res = await fetch("/api/flux/preprocessors/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  await followTask(res, fluxTaskEvents(onStatus, onProgress), "preprocessor install");
}

export async function deletePreprocessor(id: string): Promise<void> {
  const res = await fetch(`/api/flux/preprocessors/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `delete preprocessor: ${res.status}`);
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

/** The CivitAI API key, stored and reported exactly like the HuggingFace token —
 *  set or cleared here, never read back. Not validated on save: CivitAI has no cheap
 *  whoami, so a bad key surfaces on the next download instead. */
export async function getCivitaiToken(): Promise<HfTokenSource> {
  const res = await fetch("/api/settings/civitai-token");
  if (!res.ok) throw new Error(`civitai token: ${res.status}`);
  return (await res.json()).source;
}

export async function setCivitaiToken(token: string): Promise<HfTokenSource> {
  const res = await fetch("/api/settings/civitai-token", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  const body = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) throw new Error(body.detail ?? `civitai token: ${res.status}`);
  return body.source as HfTokenSource;
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
  await followTask(res, fluxTaskEvents(onStatus, onProgress), "flux pull");
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

export type GenMode = "txt2img" | "img2img" | "edit" | "compose" | "control" | "animate";

/** Rewrite a prompt with a local vision model that can see the attached images.
 *  Always resolves: if Ollama is unreachable the backend falls back to its static
 *  template and says so via `source`. */
export async function enhancePrompt(params: {
  prompt: string;
  mode: GenMode;
  model: string;
  image_hashes?: string[];
  ollama_url: string;
  /** control only: how many figures the studio pose holds, and whether they
   *  touch. The maps aren't sent to the vision model, so without this a
   *  two-person pose is briefed as a one-person scene. */
  subjects?: number;
  contact?: boolean;
  /** Let a thinking-capable model reason first; ignored by models that can't. */
  think?: boolean;
  /** Documents whose details (a character sheet, a style guide) may inform it. */
  doc_ids?: string[];
  /** The chat's "use in prompts" opt-in; off, the backend ignores doc_ids. */
  doc_prompts?: boolean;
}): Promise<{ prompt: string; source: "vlm" | "template"; thinking: string }> {
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
  // The chat to record this generation in. The backend writes both turns itself,
  // from a thread that outlives this page — which is what makes a result survive
  // a reload. The chat must already exist (`putChat`); null skips recording.
  chat_id?: string | null;
  flux_model?: string | null; // which FLUX UNet ("" / null = that mode's default)
  prompt: string;
  // What the user actually typed, when `prompt` is an auto-enhancer's rewrite of
  // it — recorded to chat history in its place so a reload shows what was shown
  // live. Omit when they're the same.
  display_prompt?: string | null;
  init_image_hash?: string | null; // img2img / edit: the source image; control: the structure source
  ref_image_hashes?: string[]; // compose: reference images to fuse
  steps?: number | null;
  guidance?: number | null;
  strength?: number | null; // img2img: how far to drift from the source
  enhance?: boolean; // wrap create prompts in a photoreal template
  width: number;
  height: number;
  seconds?: number | null; // animate: clip length (capped at 5s)
  // control. All optional — a client that predates control sends none of them and
  // gets exactly the behaviour it always did.
  control_kinds?: ControlKind[]; // maps to derive from init_image_hash
  control_map_hashes?: string[]; // maps to use as-is (a re-roll, or one drawn elsewhere)
  structure_lock?: number; // 1 = maps guide only; lower starts from the source's geometry
  control_strength?: number | null; // scales the model's control-adapter LoRA
  canny_low?: number | null;
  canny_high?: number | null;
  seed?: number | null;
  ollama_url: string;
  /** Ollama model to reload in the background once the image is done. */
  warm_model?: string | null;
  /** Always unload the Ollama model before FLUX runs (and don't reload it after). */
  unload_ollama?: boolean;
  /** The model that names the chat after its first exchange; null = leave it. */
  title_model?: string | null;
}

/** A `control` event: one derived control map, emitted before the final image. */
export interface GeneratedControlMap {
  kind: ControlKind;
  hash: string;
  url: string;
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

/** One backend progress snapshot: the whole job's share done, the stage producing
 *  it, and the sampler counters (0 outside sampling).
 *
 *  `live` is false when the update is the backend's own estimate ticking through a
 *  silent model load rather than news from ComfyUI — the difference between "still
 *  working" and "still there", and only the latter clears a stall. */
export interface GenProgressEvent {
  frac: number;
  stage: string;
  live: boolean;
  step: number;
  total: number;
}

export interface GenerateHandlers extends JobHooks {
  onStatus?: (message: string) => void;
  onProgress?: (p: GenProgressEvent) => void;
  onImage: (r: GeneratedImage) => void;
  /** A control map, emitted before the image it conditioned. Zero or more. */
  onControlMap?: (m: GeneratedControlMap) => void;
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
  await followTurn(res, { kind: "generate", on: handlers }, "generate");
}

// --------------------------------------------------------------------------- #
// Agent mode
// --------------------------------------------------------------------------- #
export interface AgentParams {
  /** The conversation, exactly as /api/chat takes it: history with every image
   *  consolidated onto the last message and a manifest numbering them. */
  messages: ChatMessage[];
  /** What the user typed, for chat history — the last message's content carries
   *  the manifest annotation they never wrote. */
  prompt: string;
  /** The manifest as stored-image hashes, same order. Tool argument "image 2"
   *  means `context_hashes[1]`. */
  context_hashes: string[];
  /** Images attached to this message specifically, for the recorded user turn. */
  attachment_hashes: string[];
  model: string;
  enabled_tools: AgentToolId[];
  /** Image tools' model; animate uses `video_model`. null = each role's default. */
  flux_model?: string | null;
  video_model?: string | null;
  steps?: number | null;
  guidance?: number | null;
  width: number;
  height: number;
  seed?: number | null;
  ollama_url: string;
  chat_id?: string | null;
  /** Let a thinking-capable model reason before planning; ignored otherwise. */
  think?: boolean;
  /** Documents to plan and answer from. */
  doc_ids?: string[];
  /** Whether they may shape the prompts the agent writes; off, they only answer. */
  doc_prompts?: boolean;
  /** Always unload the Ollama model before FLUX runs (and don't reload it after). */
  unload_ollama?: boolean;
  /** The model that names the chat after its first exchange; null = leave it. */
  title_model?: string | null;
}

/** One tool the agent decided to call. `index` orders the steps within the turn
 *  and tags every later event, so progress lands on the right step. */
export interface AgentToolCall {
  index: number;
  name: AgentToolId;
  args: Record<string, unknown>;
}

/** A tool call that won't run, with a reason to show. Either the agent's
 *  arguments didn't validate, or the generation itself failed. */
export interface AgentToolError {
  index: number;
  name: AgentToolId;
  message: string;
}

export interface AgentHandlers extends JobHooks {
  /** The agent's own prose, streamed. Empty for a plain "make me X". */
  onToken?: (text: string) => void;
  /** Reasoning tokens, only sent when `think` is on and the model can think. */
  onThinking?: (text: string) => void;
  onUsage?: (u: Usage) => void;
  onStatus?: (message: string, toolIndex?: number) => void;
  onProgress?: (p: GenProgressEvent, toolIndex?: number) => void;
  onToolCall?: (call: AgentToolCall) => void;
  onToolError?: (err: AgentToolError) => void;
  onImage?: (r: GeneratedImage, toolIndex?: number) => void;
  /** The turn finished — including the case where the agent chose to answer in
   *  words and generate nothing. */
  onDone?: () => void;
  /** The turn failed outright (as opposed to one tool failing). */
  onError?: (message: string) => void;
  onSources?: (sources: Source[]) => void;
  onDocsError?: (message: string) => void;
}

/** Stream one agent turn: the model plans, then every tool it called runs.
 *
 * Same NDJSON shape as `generate`, with `token` / `tool_call` / `tool_error` /
 * `done` added and every generation event optionally tagged `tool_index`. */
export async function agentRun(
  params: AgentParams,
  handlers: AgentHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch("/api/agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...params, messages: params.messages.map(withImages) }),
    signal,
  });
  await followTurn(res, { kind: "agent", on: handlers }, "agent");
}

// --------------------------------------------------------------------------- #
// Jobs: every turn runs on the backend, and the page only follows it
// --------------------------------------------------------------------------- #

/** A turn's handlers, tagged with the kind of turn so its events can be read.
 *  Starting a turn, coming back to one and retrying one all read the same
 *  events, so all three go through `dispatch`. */
export type TurnHandlers =
  | { kind: "chat"; on: ChatHandlers }
  | { kind: "generate"; on: GenerateHandlers }
  | { kind: "agent"; on: AgentHandlers };

function genProgress(ev: JobEvent): GenProgressEvent {
  return {
    frac: Math.max(0, Math.min(1, (ev.frac as number) ?? 0)),
    stage: (ev.stage as string) ?? "",
    live: ev.live !== false,
    step: (ev.step as number) ?? 0,
    total: (ev.total as number) ?? 0,
  };
}

function dispatch(t: TurnHandlers, ev: JobEvent): void {
  if (ev.type === "job") return t.on.onJob?.(ev.id as string);
  if (ev.type === "end")
    return t.on.onEnd?.(ev.status as JobStatus, (ev.error as string | null) ?? null);
  if (ev.type === "deleted") return t.on.onDeleted?.();
  if (t.kind === "chat") {
    const h = t.on;
    if (ev.type === "token") h.onToken(ev.text as string);
    else if (ev.type === "thinking") h.onThinking?.(ev.text as string);
    else if (ev.type === "usage") h.onUsage?.(ev as unknown as Usage);
    else if (ev.type === "sources") h.onSources?.(ev.sources as Source[]);
    else if (ev.type === "docs_error") h.onDocsError?.(ev.message as string);
    else if (ev.type === "error") h.onError?.(ev.message as string);
  } else if (t.kind === "generate") {
    const h = t.on;
    if (ev.type === "status") h.onStatus?.(ev.message as string);
    else if (ev.type === "progress") h.onProgress?.(genProgress(ev));
    else if (ev.type === "image") h.onImage(ev as unknown as GeneratedImage);
    else if (ev.type === "control") h.onControlMap?.(ev as unknown as GeneratedControlMap);
    else if (ev.type === "error") h.onError?.(ev.message as string);
  } else {
    const h = t.on;
    const toolIndex = ev.tool_index as number | undefined;
    if (ev.type === "token") h.onToken?.(ev.text as string);
    else if (ev.type === "thinking") h.onThinking?.(ev.text as string);
    else if (ev.type === "usage") h.onUsage?.(ev as unknown as Usage);
    else if (ev.type === "status") h.onStatus?.(ev.message as string, toolIndex);
    else if (ev.type === "progress") h.onProgress?.(genProgress(ev), toolIndex);
    else if (ev.type === "tool_call") h.onToolCall?.(ev as unknown as AgentToolCall);
    else if (ev.type === "tool_error") h.onToolError?.(ev as unknown as AgentToolError);
    else if (ev.type === "image") h.onImage?.(ev as unknown as GeneratedImage, toolIndex);
    else if (ev.type === "done") h.onDone?.();
    else if (ev.type === "sources") h.onSources?.(ev.sources as Source[]);
    else if (ev.type === "docs_error") h.onDocsError?.(ev.message as string);
    else if (ev.type === "error") h.onError?.(ev.message as string);
  }
}

/** Read a turn's NDJSON stream into its handlers. A refusal (a model that isn't
 *  installed, a chat that's already busy) throws with the backend's reason. */
async function followTurn(res: Response, t: TurnHandlers, what: string): Promise<void> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `${what}: ${res.status}`);
  }
  await readLines(res, (line) => {
    let ev: JobEvent;
    try {
      ev = JSON.parse(line);
    } catch {
      return;
    }
    dispatch(t, ev);
  });
}

/** Follow a turn already running (or finished): its events so far, then live. */
export async function attachJob(
  id: string,
  t: TurnHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`/api/jobs/${id}/events`, { signal });
  await followTurn(res, t, "job");
}

/** Run a finished turn again in place of itself, and follow the new run. */
export async function retryJob(
  id: string,
  t: TurnHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`/api/jobs/${id}/retry`, { method: "POST", signal });
  await followTurn(res, t, "retry");
}

/** Stop a turn. Leaving the page doesn't; this does. */
export async function cancelJob(id: string): Promise<void> {
  const res = await fetch(`/api/jobs/${id}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(`cancel: ${res.status}`);
}

/** Turns running right now, in any chat. */
export async function listRunningJobs(): Promise<JobInfo[]> {
  const res = await fetch("/api/jobs");
  if (!res.ok) throw new Error(`jobs: ${res.status}`);
  return (await res.json()).jobs ?? [];
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
    doc_ids?: string[];
    doc_prompts?: boolean;
  }
): Promise<void> {
  const res = await fetch(`/api/chats/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`putChat: ${res.status}`);
}

export async function deleteMessage(chatId: string, messageId: string): Promise<void> {
  const res = await fetch(`/api/chats/${chatId}/messages/${messageId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`deleteMessage: ${res.status}`);
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

/** Run the Ollama installer. `password` is the user's sudo password, sent only
 *  when `VersionInfo.needs_password` says one is needed. */
export async function upgradeOllama(
  ollamaUrl: string,
  password: string | null,
  onEvent: (ev: TaskEvent) => void
): Promise<void> {
  const res = await fetch(
    `/api/ollama/upgrade?ollama_url=${encodeURIComponent(ollamaUrl)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    }
  );
  await followTask(res, onEvent, "upgrade");
}

/** The installer's output. A `replace` line redraws the one before it — the
 *  download's progress bar — and `onUpgraded` names the version now running. */
export function upgradeEvents(
  onLine: (text: string, replace: boolean) => void,
  onUpgraded: (version: string) => void
): (ev: TaskEvent) => void {
  return (ev) => {
    if (ev.type === "line") onLine(ev.text as string, !!ev.replace);
    else if (ev.type === "upgraded") onUpgraded(ev.version as string);
  };
}

// --------------------------------------------------------------------------- #
// Documents (RAG)
// --------------------------------------------------------------------------- #
export interface DocLibrary {
  documents: DocInfo[];
  /** What new documents are embedded with. */
  embed_model: string;
  embed_installed: boolean;
  disk_bytes: number;
}

export async function listDocs(ollamaUrl: string): Promise<DocLibrary> {
  const res = await fetch(`/api/docs?ollama_url=${encodeURIComponent(ollamaUrl)}`);
  if (!res.ok) throw new Error(`docs: ${res.status}`);
  return res.json();
}

/** Indexing progress: reading (and, for scanned pages, OCR), then embedding. */
export interface DocProgress {
  message: string;
  /** 0–1 when the backend can say, else null. */
  frac: number | null;
}

/** Read an indexing stream to the end. Resolves with the finished document;
 *  rejects with the backend's reason if indexing failed. */
async function readIndexStream(
  res: Response,
  onProgress: (p: DocProgress) => void
): Promise<{ doc: DocInfo; duplicate: boolean }> {
  const found: { out: { doc: DocInfo; duplicate: boolean } | null } = { out: null };
  await followTask(res, docIndexEvents(onProgress, (doc, duplicate) => {
    found.out = { doc, duplicate };
  }), "upload");
  if (!found.out) throw new Error("Indexing ended without a result.");
  return found.out;
}

/** A document's indexing events: progress, then the finished document. */
export function docIndexEvents(
  onProgress: (p: DocProgress) => void,
  onDoc: (doc: DocInfo, duplicate: boolean) => void
): (ev: TaskEvent) => void {
  return (ev) => {
    if (ev.type === "status" || ev.type === "progress") {
      const total = (ev.total as number) || 0;
      onProgress({
        message: (ev.message as string) ?? "",
        frac: total ? ((ev.done as number) ?? 0) / total : null,
      });
    } else if (ev.type === "doc") {
      onDoc(ev.doc as DocInfo, !!ev.duplicate);
    }
  };
}

/** Add a PDF or text file to the library and index it. `ocrModel` is the vision
 *  model that reads scanned pages ("" skips them). */
export async function uploadDoc(
  file: File,
  ollamaUrl: string,
  ocrModel: string,
  onProgress: (p: DocProgress) => void
): Promise<{ doc: DocInfo; duplicate: boolean }> {
  const form = new FormData();
  form.append("file", file);
  form.append("ollama_url", ollamaUrl);
  form.append("ocr_model", ocrModel);
  const res = await fetch("/api/docs", { method: "POST", body: form });
  return readIndexStream(res, onProgress);
}

export async function reindexDoc(
  id: string,
  ollamaUrl: string,
  ocrModel: string,
  onProgress: (p: DocProgress) => void
): Promise<DocInfo> {
  const res = await fetch(`/api/docs/${id}/reindex`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ollama_url: ollamaUrl, ocr_model: ocrModel }),
  });
  return (await readIndexStream(res, onProgress)).doc;
}

export async function deleteDoc(id: string): Promise<void> {
  const res = await fetch(`/api/docs/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`deleteDoc: ${res.status}`);
}

export async function setEmbedModel(
  name: string,
  ollamaUrl: string
): Promise<{ embed_model: string; embed_installed: boolean }> {
  const res = await fetch(
    `/api/settings/embed-model?ollama_url=${encodeURIComponent(ollamaUrl)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }
  );
  if (!res.ok) throw new Error(`embed model: ${res.status}`);
  return res.json();
}

/** Whether a dropped file is a document for the library rather than an image to
 *  attach. Mirrors `rag.kind_of` on the backend. */
const DOC_EXTS = new Set(
  (
    "pdf txt md markdown rst csv tsv json jsonl log xml html htm yaml yml ini toml " +
    "cfg conf tex srt vtt py js ts tsx jsx java c h cpp hpp cs go rs rb php sh bash " +
    "sql css scss swift kt lua r"
  ).split(" ")
);
export function isDocFile(f: File): boolean {
  if (f.type.startsWith("image/") || f.type.startsWith("video/")) return false;
  const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
  return f.type === "application/pdf" || f.type.startsWith("text/") || DOC_EXTS.has(ext);
}
export const DOC_ACCEPT = ".pdf,.txt,.md,.markdown,.csv,.json,.html,.htm,.xml,.yaml,.yml,.log,.rst,.tex,text/*,application/pdf";
