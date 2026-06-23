import { useState } from "react";
import { deleteModel, getRunning, pullModel, unloadAll } from "../api";
import type { RunningModel } from "../types";

interface Props {
  ollamaUrl: string;
  allModels: string[];
  onChanged: () => void;
}

export default function ModelManager({ ollamaUrl, allModels, onChanged }: Props) {
  const [open, setOpen] = useState(false);
  const [pullName, setPullName] = useState("");
  const [pullStatus, setPullStatus] = useState<string | null>(null);
  const [pullPct, setPullPct] = useState<number | null>(null);
  const [removeName, setRemoveName] = useState("");
  const [running, setRunning] = useState<RunningModel[] | null>(null);
  const [note, setNote] = useState<string | null>(null);

  async function doPull() {
    if (!pullName.trim()) return;
    setPullStatus("starting…");
    setPullPct(null);
    try {
      await pullModel(ollamaUrl, pullName.trim(), (status, pct) => {
        setPullStatus(status);
        setPullPct(pct);
      });
      setPullStatus("✓ downloaded");
      setPullName("");
      onChanged();
    } catch {
      setPullStatus("✗ download failed");
    }
  }

  async function doRemove() {
    if (!removeName) return;
    try {
      await deleteModel(ollamaUrl, removeName);
      setNote(`Removed ${removeName}`);
      setRemoveName("");
      onChanged();
    } catch {
      setNote("Remove failed");
    }
  }

  async function doUnload() {
    try {
      const names = await unloadAll(ollamaUrl);
      setNote(names.length ? `Unloaded ${names.length} model(s)` : "Nothing loaded");
    } catch {
      setNote("Unload failed");
    }
  }

  async function doShowRunning() {
    try {
      setRunning(await getRunning(ollamaUrl));
    } catch {
      setRunning([]);
    }
  }

  return (
    <div className="section">
      <button className="section-head" onClick={() => setOpen(!open)}>
        🔧 Model Management <span className="chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="section-body">
          <label className="lbl">Download a vision model</label>
          <div className="row">
            <input
              value={pullName}
              onChange={(e) => setPullName(e.target.value)}
              placeholder="e.g. qwen2.5-vl:7b"
            />
            <button className="btn" onClick={doPull}>
              Pull
            </button>
          </div>
          {pullStatus && (
            <div className="pull-progress">
              <div className="muted small">{pullStatus}</div>
              {pullPct !== null && (
                <div className="progress">
                  <div
                    className="progress-bar"
                    style={{ width: `${Math.round(pullPct * 100)}%` }}
                  />
                </div>
              )}
            </div>
          )}

          <label className="lbl">Remove a model</label>
          <div className="row">
            <select
              value={removeName}
              onChange={(e) => setRemoveName(e.target.value)}
            >
              <option value="">Select…</option>
              {allModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <button className="btn danger" onClick={doRemove}>
              Remove
            </button>
          </div>

          <div className="row">
            <button className="btn block" onClick={doUnload}>
              🔄 Unload all
            </button>
            <button className="btn block" onClick={doShowRunning}>
              👀 Running
            </button>
          </div>

          {running && (
            <div className="running">
              {running.length === 0 ? (
                <div className="muted small">No models loaded.</div>
              ) : (
                running.map((m) => (
                  <div key={m.name} className="muted small">
                    • {m.name} ({(m.size / 1024 ** 3).toFixed(2)} GB)
                  </div>
                ))
              )}
            </div>
          )}
          {note && <div className="muted small note">{note}</div>}
        </div>
      )}
    </div>
  );
}
