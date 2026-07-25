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
}

/** Which generation workflow the composer is in. */
export type GenOp = "create" | "edit" | "compose" | "animate";

/** Liveness of the in-flight turn, for the progress bar under the chat.
 *
 * The bar exists because a first run goes quiet for a long time: ComfyUI loads
 * tens of GB off disk before it emits step 1, so the status line sits on
 * "editing with flux-2-klein-9b…" with nothing to say whether it is loading or
 * wedged. `total` of 0 means no step has arrived yet and the bar runs
 * indeterminate; `updatedAt` is what separates "slow" from "stuck" — it only
 * moves when the backend actually says something. */
export interface GenProgress {
  /** Last status line from the backend, minus the leading icon. */
  phase: string;
  /** Sampler steps done, and the total for this job. Both 0 before sampling. */
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
  enhance: boolean; // wrap create prompts in a photoreal template
  width: number;
  height: number;
  seed: string; // blank = random; kept as string for the input field
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
