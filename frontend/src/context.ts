import type { ChatMessage } from "./types";
import type { Usage } from "./api";

// Start trimming once the last turn used this fraction of the window, and trim
// down to this target so there's headroom for the next message + its images.
const TRIM_TRIGGER = 0.8;
const TRIM_TARGET = 0.6;
// Rough per-image token cost (images are resized to <=1280px before sending).
// Used only to decide *how many* old turns to drop; the meter uses exact counts.
const IMG_EST = 1500;

/** Mirror of the backend's context_size_for(). */
export function numCtxFor(imageCount: number): number {
  return Math.min(32768, Math.max(8192, 4096 + 2048 * imageCount));
}

function estTokens(m: ChatMessage): number {
  return Math.ceil((m.content?.length ?? 0) / 4) + (m.images?.length ?? 0) * IMG_EST;
}

/** Drop oldest turns from the history we send when the last measured usage
 *  shows we're near the window limit. The newest message is always kept, and
 *  pinned images (counted via pinnedCount) are accounted for since they ride
 *  on the first surviving user message. Returns the history to send. */
export function trimHistory(
  history: ChatMessage[],
  pinnedCount: number,
  usage: Usage | null
): { sent: ChatMessage[]; trimmed: boolean } {
  if (!usage || usage.used <= TRIM_TRIGGER * usage.num_ctx) {
    return { sent: history, trimmed: false };
  }
  const kept = [...history];
  while (kept.length > 1) {
    const images =
      pinnedCount + kept.reduce((s, m) => s + (m.images?.length ?? 0), 0);
    const budget = TRIM_TARGET * numCtxFor(images);
    const total =
      pinnedCount * IMG_EST + kept.reduce((s, m) => s + estTokens(m), 0);
    if (total <= budget) break;
    kept.shift();
  }
  return { sent: kept, trimmed: kept.length < history.length };
}
