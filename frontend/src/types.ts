export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
  /** Data-URL strings (data:image/...;base64,...) for display + sending. */
  images?: string[];
  /** Generated video URLs (`/api/images/<hash>.webm`) — never data-URLs, unlike
   *  `images`. A 5s 720p clip is megabytes, so it stays a URL the <video> element
   *  streams; and everything that consumes `images` (the hash cache's canvas
   *  resize, the pin panel, the vision model) assumes decodable image bytes. */
  videos?: string[];
  /** The ordered images that were in the model's context for this turn (pinned +
   *  in-chat, in manifest order). Data-URLs. Used to resolve the model's
   *  "image N" references to an inline thumbnail. Not displayed as attachments. */
  contextImages?: string[];
  /** Which model this turn was sent to (for the per-chunk model indicator). */
  model?: string;
  /** Verbose-mode prompt enhancer's rewrite of `content` — the text that
   *  actually generated the image, shown underneath the user's own typed
   *  prompt. Live-session only: not persisted, so it's gone on reload. */
  enhancedPrompt?: string;
  /** The control maps a control generation derived and conditioned on. Shown as
   *  small chips beside the result: when a pose comes out wrong, a bad map and an
   *  ignored map look identical from the image alone, and this is what tells them
   *  apart. Data-URLs, so one can be pinned and re-fed on the next roll. */
  controlMaps?: ControlMap[];
  /** What agent mode decided to do this turn: which tools it called, with the
   *  prompts it wrote. Live-session only, like `enhancedPrompt` — the images and
   *  the text survive a reload, the reasoning behind them doesn't. */
  agentSteps?: AgentStep[];
}

/** Which composer mode is active.
 *  "analyze": chat with a vision model about the attached images.
 *  "generate": pick a workflow by hand and drive FLUX directly.
 *  "agent": describe what you want; a tool-capable model picks the workflow,
 *  picks the images and writes the prompt. */
export type ComposerMode = "analyze" | "generate" | "agent";

/** A generation tool the agent can be given. Mirrors `backend/agent.py`'s TOOLS. */
export type AgentToolId =
  | "create_image"
  | "edit_image"
  | "combine_images"
  | "animate_image"
  | "control_image";

/** One entry in the composer's Tools menu, served by `/api/agent/tools` so the
 *  menu can't offer something the backend won't run. */
export interface AgentTool {
  id: AgentToolId;
  label: string;
  icon: string;
  hint: string;
  /** False for tools that exist as a workflow but aren't wired to the agent yet
   *  (control). Shown greyed out rather than hidden, so their absence is
   *  explained rather than mysterious. */
  available: boolean;
  default: boolean;
}

/** One tool call in an agent turn, as the chat renders it. */
export interface AgentStep {
  name: AgentToolId;
  /** What the agent passed — including the prompt it wrote and the image numbers
   *  it picked, which is the whole explanation of why this image looks like it
   *  does. */
  args: Record<string, unknown>;
  state: "running" | "done" | "error";
  /** Why it failed, when `state` is "error". A sentence, ready to show. */
  message?: string;
}

/** One derived control map, as the backend's `control` stream event describes it. */
export interface ControlMap {
  kind: ControlKind;
  /** Data-URL once the client has fetched it, so it can be pinned like any image. */
  url: string;
}

/** Which generation workflow the composer is in. */
export type GenOp = "create" | "edit" | "compose" | "control" | "animate";

/** A control map's type — what structure it carries out of the source image.
 *  "depth" holds the whole scene in 3-D (and so the contact between a subject and
 *  whatever it is sitting or standing on); "canny" holds every edge; "pose" holds
 *  the skeleton and nothing else. Mirrors the backend's `flux_catalog.CONTROL_KINDS`. */
export type ControlKind = "depth" | "canny" | "pose";

/** How the settings-level prompt enhancer runs before a generation.
 *  "off": only the manual ✨ Improve prompt button rewrites, on demand.
 *  "on": rewrites automatically before every generation, silently.
 *  "verbose": rewrites automatically and shows the result in the composer
 *  first, same as the manual button, so it's visible/editable and Undo works. */
export type EnhancerMode = "off" | "on" | "verbose";

/** Progress of the in-flight turn, for the bar under the chat.
 *
 * `frac` is the whole job, not the sampler: the backend prices every node in the
 * ComfyUI graph and reports the share that is finished (see `_Progress` in
 * flux_client.py), which is what makes the bar mean something during the minutes a
 * cold run spends loading weights before step 1 exists. It is null only until the
 * first backend event arrives. `updatedAt` is what separates "slow" from "stuck" —
 * it only moves when the backend actually says something. */
export interface GenProgress {
  /** Last status line from the backend, minus the leading icon. */
  phase: string;
  /** What the graph is doing right now ("Loading the model"), from the backend. */
  stage: string;
  /** Share of the whole job that is done, 0–1. Null before the first event. */
  frac: number | null;
  /** Sampler steps done, and the total for this job. Both 0 outside sampling. */
  step: number;
  total: number;
  /** ms epoch, for elapsed time. */
  startedAt: number;
  /** ms epoch of the last backend event, for the stall detector. */
  updatedAt: number;
}

/** User-tunable generation settings.
 *  No negative prompt is exposed: FLUX.1 samples at cfg=1.0 (the negative branch
 *  has no effect) and FLUX.2 has none at all. Guidance is mode-scaled — ~3.5 for
 *  create (FLUX dev), ~2.5 for edit/compose (Kontext), ~3.5 for animate (Wan,
 *  where it is a real CFG scale). */
export interface GenSettings {
  /** Which FLUX UNet the current mode runs on. "" = that mode's default. */
  fluxModel: string;
  steps: number;
  guidance: number;
  strength: number; // img2img: how far from the source image
  width: number;
  height: number;
  seed: string; // blank = random; kept as string for the input field
  /** control: which maps to derive from the source image. Stackable — depth plus
   *  pose is the pair that holds a hard pose, since depth carries the scene and
   *  pose carries limb identity. Empty means "use the maps I attached as-is". */
  controlKinds: ControlKind[];
  /** control: how far down the schedule to start. 1 = the maps guide and nothing
   *  constrains; lower starts from the source image so its geometry survives. The
   *  dial that decides whether a pose is suggested or held. */
  structureLock: number;
  /** control: with a studio pose, whether the first attachment is the scene the
   *  figures are posed *into* rather than a subject reference. Off, a studio pose
   *  is the whole signal and every attachment describes who is in it; on, the
   *  photo becomes the source image — which is what re-enables the structure
   *  lock, and what lets maps derived from the photo stack with the studio's. */
  studioSource: boolean;
  /** control: scales the model's control-adapter LoRA for this generation. */
  controlStrength: number;
  /** control: Canny edge thresholds. Lower low = more edges kept. */
  cannyLow: number;
  cannyHigh: number;
}

export interface VersionInfo {
  installed: string | null;
  latest: string | null;
  update_available: boolean;
  is_local: boolean;
}

export interface RunningModel {
  name: string;
  size: number;
}

/** One row in the sidebar chat list. `icons` are thumbnail URLs. */
export interface ChatSummary {
  id: string;
  title: string | null;
  model: string;
  updated_at: number;
  icons: string[];
}

/** A message as returned by GET /api/chats/{id} (images are URLs, not data). */
export interface StoredMessage {
  role: Role;
  content: string;
  model: string | null;
  images: string[];
  /** URLs of the images that were in the model's context for this turn. */
  context_images: string[];
}

/** Full chat detail from GET /api/chats/{id}. */
export interface ChatDetail {
  id: string;
  title: string | null;
  model: string | null;
  system_prompt: string;
  system_image: string | null;
  pinned: string[];
  messages: StoredMessage[];
}
