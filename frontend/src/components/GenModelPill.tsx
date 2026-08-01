import type { FluxModel } from "../api";
import type { GenOp, GenSettings } from "../types";
import { resolveFlux, roleFor } from "../flux";

/** Drop the extension — every encoder here is .safetensors or .gguf, so it's the
 *  same suffix on all of them and costs pill width for nothing. FLUX.1 reports a
 *  pair joined with " + ", so trim each side. */
function shortEncoder(file: string): string {
  return file
    .split(" + ")
    .map((f) => f.replace(/\.(safetensors|sft|gguf)$/i, ""))
    .join(" + ");
}

interface Props {
  op: GenOp;
  picked: string;
  models: FluxModel[];
  gen: GenSettings;
}

/** What a generate would actually run right now: the transformer the current op
 *  resolves to, and the text encoder bound to it.
 *
 * Resolved through `resolveFlux` — the same function the composer and the backend
 * agree on — so the header can't advertise a model the dispatch wouldn't pick. The
 * encoder comes from the backend's `encoder_for`, which mirrors the graph builder,
 * for the same reason. */
export default function GenModelPill({ op, picked, models, gen }: Props) {
  const name = resolveFlux(picked, models, roleFor(op));
  const model = models.find((m) => m.name === name);
  if (!model) return null;

  const video = model.roles.includes("animate");
  const seedLabel = gen.seed ? `seed ${gen.seed}` : "random seed";
  return (
    <span
      className="gen-pill"
      title={
        `${video ? "Video" : "Image"} model: ${model.label}\n` +
        `Text encoder: ${model.encoder || "—"}\n` +
        `LoRA: ${
          model.loras.length
            ? model.loras.map((l) => `${l.name} @ ${l.strength}`).join(", ")
            : "none"
        }\n` +
        `Steps: ${gen.steps}\n` +
        `Guidance: ${gen.guidance}\n` +
        `Seed: ${gen.seed || "random"}`
      }
    >
      <span className="gen-pill-icon">{video ? "🎬" : "🎨"}</span>
      <span className="gen-pill-model">{model.label}</span>
      {model.encoder && (
        <>
          <span className="gen-pill-sep">·</span>
          <span className="gen-pill-enc">{shortEncoder(model.encoder)}</span>
        </>
      )}
      {/* Only rendered when at least one is attached — "none" is the normal state
          and would be noise on every generate. The weight is part of the identity:
          the same adapter at 0.3 and at 1.2 are different pictures. Past two, the
          pill collapses to a count rather than overflowing — full detail is still
          in the tooltip above. */}
      {model.loras.length > 0 && (
        <span className="gen-pill-lora">
          ⊕{" "}
          {model.loras.length > 2
            ? `${model.loras.length} LoRAs`
            : model.loras
                .map((l) => `${shortEncoder(l.name)}@${l.strength.toFixed(2)}`)
                .join(", ")}
        </span>
      )}
      <span className="gen-pill-sep">·</span>
      <span className="gen-pill-settings">
        {gen.steps} steps · {gen.guidance} cfg · {seedLabel}
      </span>
    </span>
  );
}
