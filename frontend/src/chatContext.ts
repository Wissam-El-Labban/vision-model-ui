import type { ChatMessage } from "./types";
import type { Usage } from "./api";
import { trimHistory } from "./context";

/** Build the message list to send Ollama, with every image the chat has seen.
 *
 * Ollama's /api/chat is stateless AND vision models only attend to images on the
 * CURRENT/last message — images on earlier history messages are ignored. So every
 * image the (trimmed) conversation has seen is consolidated onto the last message:
 * pinned/primary first, then each in-chat attachment in order. Earlier messages go
 * text-only. This keeps the model aware of every photo across follow-ups, not just
 * the one from the latest turn. `trimHistory` already sheds the oldest turns and
 * images under context pressure, so this stays within the window budget (the total
 * image count is unchanged, just relocated and deduped).
 *
 * The chat API hands the model N *unlabeled* images with no anchor for which is
 * which, so it conflates distinct photos when asked to compare them ("this image"
 * vs "the initial one"). We therefore also build a short text manifest, in the same
 * order as the images array, so the model can tell the pinned reference(s) apart
 * from images shared earlier vs. now. Images keep a single global numbering
 * (Image 1..N, in array order) — that's how models actually refer to them and how
 * the UI resolves each "Image N" back to its thumbnail. But we group them under
 * clear section headers so the model doesn't skim past which ones are the
 * persistent pinned references vs. what was actually sent in the chat (even capable
 * models mislabel a pinned image as "shared" when it's just one line in a flat
 * list). Pinned images always come first, so `outImages` = [pinned..., chat...].
 *
 * Shared by analyze mode and agent mode, which need the identical picture of the
 * conversation for different reasons: analyze so the model can talk about the right
 * photo, agent so it can *name* the right photo in a tool argument.
 */
export async function buildOllamaContext(opts: {
  /** Full history including the turn being sent. */
  history: ChatMessage[];
  pinnedImages: string[];
  /** Last measured usage, for trimming. */
  usage: Usage | null;
  /** data-URL -> downscaled copy for the vision model. */
  forOllama: (urls: string[]) => Promise<string[]>;
  /** Number the images even when there's only one.
   *
   *  Analyze annotates only when there are several, because with one image there
   *  is nothing to disambiguate and the note is noise. Agent mode always numbers:
   *  its tool arguments reference the numbers, so "Image 1" has to exist before
   *  the model can ask to edit it. */
  alwaysNumber?: boolean;
}): Promise<{
  /** What goes on the wire: history with images consolidated onto the last turn. */
  merged: ChatMessage[];
  /** The manifest, in order — index i is "Image i+1". Full-size originals, so the
   *  UI can resolve a reference to a thumbnail and the caller can hash them. */
  outImages: string[];
  /** Whether history was dropped to fit the context window. */
  trimmed: boolean;
}> {
  const { history, pinnedImages, usage, forOllama, alwaysNumber = false } = opts;

  // Auto-trim oldest turns from what we SEND (the UI keeps the full history)
  // when the last measured usage shows we're near the window limit.
  const { sent, trimmed } = trimHistory(history, pinnedImages.length, usage);

  const lastIdx = sent.length - 1;
  const seen = new Set<string>(); // dedupe pinned + repeats across history
  const outImages: string[] = [];
  const pinnedLines: string[] = [];
  const chatLines: string[] = [];
  for (const img of pinnedImages) {
    if (seen.has(img)) continue;
    seen.add(img);
    outImages.push(img);
    pinnedLines.push(`  Image ${outImages.length}`);
  }
  sent.forEach((m, i) => {
    for (const img of m.images ?? []) {
      if (seen.has(img)) continue;
      seen.add(img);
      outImages.push(img);
      chatLines.push(
        `  Image ${outImages.length}${i === lastIdx ? " (sent just now)" : " (sent earlier)"}`
      );
    }
  });

  // What actually goes on the wire: downscaled copies. `outImages` stays as the
  // originals, since the UI resolves "Image N" back to those for display.
  const wireImages = await forOllama(outImages);
  const merged = sent.map((m, i) => {
    if (i !== lastIdx) {
      return { role: m.role, content: m.content }; // strip history images (Ollama ignores them)
    }
    if (!outImages.length) return { role: m.role, content: m.content };
    let note = "";
    if (outImages.length > 1 || alwaysNumber) {
      const sections: string[] = [];
      if (pinnedLines.length) {
        sections.push(
          "PINNED REFERENCE IMAGES (kept in view for the whole conversation for " +
            "analysis; NOT part of any single message):\n" +
            pinnedLines.join("\n")
        );
      }
      if (chatLines.length) {
        sections.push("IMAGES SENT IN THE CHAT:\n" + chatLines.join("\n"));
      }
      const heading =
        outImages.length === 1
          ? "There is one image below. It is image 1; refer to it by that number."
          : `The ${outImages.length} images below are numbered 1-${outImages.length} ` +
            "in the order shown; refer to each by its number.";
      note = `[${heading}\n\n${sections.join("\n\n")}]\n\n`;
    }
    return { role: m.role, content: note + m.content, images: wireImages };
  });

  return { merged, outImages, trimmed };
}
