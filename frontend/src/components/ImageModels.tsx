import { useEffect, useState } from "react";
import {
  clearHfToken,
  deleteFluxBundle,
  deleteFluxModel,
  getFluxCatalog,
  installFluxBundle,
  pullFluxModel,
  setHfToken,
} from "../api";
import type { FluxCatalog, FluxModel } from "../api";

interface Props {
  models: FluxModel[]; // installed transformers, incl. user-added ones
  onChanged: () => void; // re-probe the app's model list after an install/remove
}

/** Install and remove image models.
 *
 * Nothing is downloaded at startup any more, so this is where a fresh machine gets
 * its first model. The downloads are 30-50 GB, which is why this shows a real byte
 * counter rather than a spinner.
 */
export default function ImageModels({ models, onChanged }: Props) {
  const [open, setOpen] = useState(false);
  const [cat, setCat] = useState<FluxCatalog | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // bundle id being installed
  const [status, setStatus] = useState<string | null>(null);
  const [pct, setPct] = useState<number | null>(null);
  const [token, setToken] = useState("");
  const [repo, setRepo] = useState("");

  async function refresh() {
    try {
      setCat(await getFluxCatalog());
    } catch {
      setCat(null);
    }
  }

  useEffect(() => {
    if (open) void refresh();
  }, [open]);

  async function install(id: string) {
    setBusy(id);
    setStatus("starting…");
    setPct(null);
    try {
      await installFluxBundle(
        id,
        (m) => setStatus(m),
        (p) => {
          setStatus(`${p.file} — ${(p.done / 1e9).toFixed(1)}/${(p.total / 1e9).toFixed(1)} GB`);
          setPct(p.pct);
        }
      );
      setStatus("✓ installed");
      setPct(null);
      await refresh();
      onChanged();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
      setPct(null);
    } finally {
      setBusy(null);
    }
  }

  async function remove(id: string, label: string) {
    if (!confirm(`Remove ${label}? You'll have to download it again to use it.`)) return;
    try {
      await deleteFluxBundle(id);
      setStatus(`Removed ${label}`);
      await refresh();
      onChanged();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
    }
  }

  async function saveToken() {
    if (!token.trim()) return;
    try {
      const user = await setHfToken(token.trim());
      setToken("");
      setStatus(`✓ token saved (${user})`);
      await refresh();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
    }
  }

  async function dropToken() {
    await clearHfToken();
    setStatus("Token cleared");
    await refresh();
  }

  async function addRepo() {
    if (!repo.trim()) return;
    setBusy("repo");
    setStatus("starting…");
    try {
      await pullFluxModel(repo.trim(), (m) => setStatus(m));
      setStatus("✓ added");
      setRepo("");
      onChanged();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  async function removeExtra(name: string) {
    try {
      await deleteFluxModel(name);
      onChanged();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
    }
  }

  const extras = models.filter((m) => m.bundle === null);

  return (
    <div className="section">
      <button className="section-head" onClick={() => setOpen(!open)}>
        🖼️ Image Models <span className="chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="section-body">
          {!cat && <div className="muted small">Couldn't reach the server.</div>}

          {cat && !cat.runtime_ready && (
            <div className="muted small">
              The image engine isn't installed. Run <code>./run.sh</code> to set it up.
            </div>
          )}

          {cat?.runtime_ready && (
            <>
              {!cat.available && (
                <div className="muted small note">
                  No image model installed yet — pick one below to enable Create, Edit and
                  Combine.
                </div>
              )}

              <ul className="bundle-list">
                {cat.bundles.map((b) => (
                  <li key={b.id} className={b.installed ? "installed" : ""}>
                    <div className="bundle-head">
                      <span className="bundle-name">{b.label}</span>
                      {b.installed ? (
                        <button
                          className="btn danger small"
                          onClick={() => remove(b.id, b.label)}
                          disabled={busy !== null}
                        >
                          Remove
                        </button>
                      ) : (
                        <button
                          className="btn small"
                          onClick={() => install(b.id)}
                          disabled={busy !== null}
                        >
                          ⬇ {b.needed_gb} GB
                        </button>
                      )}
                    </div>
                    <div className="muted small">{b.blurb}</div>
                    <div className="muted small">
                      {b.roles.join(" + ")} · needs ~{b.vram_gb} GB VRAM
                      {b.gated && " · gated (needs a token)"}
                    </div>
                  </li>
                ))}
              </ul>

              {busy && pct !== null && (
                <div className="progress">
                  <div className="progress-bar" style={{ width: `${pct}%` }} />
                </div>
              )}
              {status && <div className="muted small note">{status}</div>}
              <div className="muted small">{cat.disk_free_gb} GB free on disk.</div>

              <label className="lbl">HuggingFace token</label>
              <div className="muted small">
                {cat.hf_token === "saved" && "A token is saved. "}
                {cat.hf_token === "env" && "Using HF_TOKEN from the environment. "}
                Only needed for gated models — the ones above download without one.
              </div>
              <div className="row">
                <input
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="hf_…"
                />
                <button className="btn" onClick={saveToken}>
                  Save
                </button>
              </div>
              {cat.hf_token === "saved" && (
                <button className="btn block" onClick={dropToken}>
                  Clear saved token
                </button>
              )}

              <label className="lbl">Add a model from any repo</label>
              <div className="row">
                <input
                  value={repo}
                  onChange={(e) => setRepo(e.target.value)}
                  placeholder="owner/model (HuggingFace)"
                />
                <button className="btn" onClick={addRepo} disabled={busy !== null}>
                  ⬇ Add
                </button>
              </div>
              <div className="muted small">
                Extras run on FLUX.1's text encoder, so they need the FLUX.1 model installed.
              </div>
              {extras.map((m) => (
                <div key={m.name} className="row extra-model">
                  <span className="muted small">
                    {m.name} ({m.size_gb} GB)
                  </span>
                  <button
                    className="btn danger small"
                    onClick={() => removeExtra(m.name)}
                    title="Remove"
                  >
                    🗑
                  </button>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
