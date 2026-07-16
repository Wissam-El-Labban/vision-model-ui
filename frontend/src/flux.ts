import type { FluxModel, FluxRole, GenMode } from "./api";
import type { GenOp } from "./types";

/** Which set of transformers a workflow draws from. A FLUX.2 model serves both
 *  image roles; on FLUX.1 the sets are disjoint (dev creates, Kontext edits).
 *  `animate` is disjoint from all of them — only a video model can serve it. */
export function roleFor(op: GenOp): FluxRole {
  if (op === "animate") return "animate";
  return op === "create" ? "create" : "edit";
}

/** The backend mode a composer workflow submits as.
 *
 * `create` is the only op that isn't a mode of its own: it splits on whether a
 * source image is attached. Callers must agree on this — the prompt enhancer is
 * briefed per mode, and an edit briefed as a create gets a scene description
 * where it needed an instruction. */
export function modeFor(op: GenOp, images: string[]): GenMode {
  if (op === "animate") return "animate";
  if (op === "edit") return "edit";
  if (op === "compose") return "compose";
  return images.length ? "img2img" : "txt2img";
}

/** The images a workflow conditions on, given everything attached. compose fuses
 *  all of them; edit changes the first and draws subjects from the rest; create
 *  takes a single source (or none, for txt2img); animate takes exactly one — Wan
 *  I2V has a single start frame, and a second attachment would be dropped
 *  silently rather than used. */
export function imagesFor(op: GenOp, images: string[]): string[] {
  return op === "create" || op === "animate" ? images.slice(0, 1) : images;
}

/** The model a mode will actually run on.
 *
 * This is the *only* place that answers that question. It used to be answered
 * twice — once in the composer to decide what to show, once in the backend to
 * decide what to load — off two different orderings, so the UI could display
 * klein while dev did the work. Display and dispatch both call this now, so the
 * two cannot disagree; `list_unets` is sorted in catalog order to match the
 * backend's `_default_for` besides.
 *
 * "" means "no explicit pick", which resolves to the first model that can serve
 * the role. A pick that can't serve the role resolves the same way, mirroring
 * the backend's `_resolve_unet`.
 */
export function resolveFlux(picked: string, models: FluxModel[], role: FluxRole): string {
  const forRole = models.filter((m) => m.roles.includes(role));
  return forRole.find((m) => m.name === picked)?.name ?? forRole[0]?.name ?? "";
}

/** The guidance a mode starts at, which depends on the model that will run it.
 *
 * FLUX.2 uses one value for every job. FLUX.1 is mode-scaled: dev needs ~3.5 to
 * bind a text-only prompt, while Kontext wants ~2.5 — at 3.5 it clings to the
 * reference image and ignores the instruction. Wan's 3.5 is a different quantity
 * that happens to share the number: a real CFG scale over a real negative branch,
 * not FLUX's distilled guidance embedding. Mirrors the backend's
 * `_default_guidance`. Resolves through `resolveFlux` so the guidance follows the
 * model that will actually run, not whichever one happens to sort first. */
export function guidanceFor(op: GenOp, models: FluxModel[], picked = ""): number {
  const role = roleFor(op);
  const name = resolveFlux(picked, models, role);
  const family = models.find((m) => m.name === name)?.family;
  if (family === "wan") return 3.5;
  if (family === "flux2") return 4.0;
  return op === "create" ? 3.5 : 2.5;
}
