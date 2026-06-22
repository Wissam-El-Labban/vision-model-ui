export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
  /** Data-URL strings (data:image/...;base64,...) for display + sending. */
  images?: string[];
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
