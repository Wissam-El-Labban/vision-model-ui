export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
  /** Data-URL strings (data:image/...;base64,...) for display + sending. */
  images?: string[];
  /** The ordered images that were in the model's context for this turn (pinned +
   *  in-chat, in manifest order). Data-URLs. Used to resolve the model's
   *  "image N" references to an inline thumbnail. Not displayed as attachments. */
  contextImages?: string[];
  /** Which model this turn was sent to (for the per-chunk model indicator). */
  model?: string;
}

/** Which image-generation workflow the composer is in. */
export type GenOp = "create" | "edit" | "compose";

/** User-tunable image-generation settings.
 *  create (txt2img/img2img) uses the SD fields; edit/compose (FLUX Kontext) use
 *  steps + guidance (guidance is Kontext's low ~2.5 scale in those modes). */
export interface GenSettings {
  model: string;
  negativePrompt: string;
  steps: number;
  guidance: number;
  strength: number; // img2img: how far from the source image
  enhance: boolean; // wrap photoreal prompts in a quality template
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
