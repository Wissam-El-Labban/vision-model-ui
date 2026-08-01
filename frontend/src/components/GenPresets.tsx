import { useEffect, useState } from "react";
import type { GenPreset } from "../api";

interface Props {
  presets: GenPreset[];
  onApply: (p: GenPreset) => void;
  /** The permanent, non-deletable "Default" row — always first, never in `presets`. */
  onApplyDefault: () => void;
  onSave: (name: string) => void;
  onDelete: (id: string) => void;
  disabled: boolean;
}

/** Saved generation-setting bundles, at the top of the ⚙️ popover: pick one to
 *  instantly restore steps/guidance/size/model and that model's LoRA stack,
 *  instead of re-tuning every control by hand. Structured like the sidebar's
 *  chat list — same two-step-confirm delete, same "type a name, click save" row.
 *
 *  These are pure snapshots: changing a slider after applying one never edits it
 *  back, and nothing here tracks "currently active preset" — so there's no dirty
 *  state to reconcile when the live settings drift from whatever was last applied. */
export default function GenPresets({
  presets,
  onApply,
  onApplyDefault,
  onSave,
  onDelete,
  disabled,
}: Props) {
  const [name, setName] = useState("");
  // Two-step delete guardrail, same as ChatList: first click arms "Delete?", a
  // second click deletes; auto-reverts after a few seconds.
  const [confirmId, setConfirmId] = useState<string | null>(null);
  useEffect(() => {
    if (!confirmId) return;
    const t = setTimeout(() => setConfirmId(null), 5000);
    return () => clearTimeout(t);
  }, [confirmId]);

  function save() {
    const trimmed = name.trim();
    if (!trimmed) return;
    onSave(trimmed);
    setName("");
  }

  return (
    <div className="gen-presets">
      <label className="lbl">Presets</label>
      <ul className="flux-model-list">
        <li>
          <button
            type="button"
            className="flux-model-pick"
            title="Reset to this model's own defaults, with no LoRAs attached"
            onClick={onApplyDefault}
            disabled={disabled}
          >
            Default
          </button>
        </li>
        {presets.map((p) => (
          <li key={p.id}>
            <button
              type="button"
              className="flux-model-pick"
              title="Apply this preset"
              onClick={() => onApply(p)}
              disabled={disabled}
            >
              {p.name}
            </button>
            <button
              className={`chat-del ${confirmId === p.id ? "confirming" : ""}`}
              title={confirmId === p.id ? "Click again to delete" : "Delete preset"}
              onClick={(e) => {
                e.stopPropagation();
                if (confirmId === p.id) {
                  onDelete(p.id);
                  setConfirmId(null);
                } else {
                  setConfirmId(p.id);
                }
              }}
            >
              {confirmId === p.id ? "Delete?" : "✕"}
            </button>
          </li>
        ))}
      </ul>
      <div className="row">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Preset name…"
          onKeyDown={(e) => e.key === "Enter" && save()}
          disabled={disabled}
        />
        <button className="btn small" onClick={save} disabled={disabled || !name.trim()}>
          💾 Save current
        </button>
      </div>
    </div>
  );
}
