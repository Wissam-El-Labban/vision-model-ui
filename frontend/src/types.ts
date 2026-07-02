export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
  /** Data-URL strings (data:image/...;base64,...) for display + sending. */
  images?: string[];
  /** Which model this turn was sent to (for the per-chunk model indicator). */
  model?: string;
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
