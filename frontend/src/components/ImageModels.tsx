import { useEffect, useState } from "react";
import {
  clearHfToken,
  deleteFluxBundle,
  deleteFluxModel,
  deleteTextEncoder,
  getFluxCatalog,
  getTextEncoders,
  installFluxBundle,
  pullFluxModel,
  pullTextEncoder,
  selectTextEncoder,
  setHfToken,
} from "../api";
import type { FluxCatalog, FluxModel, FluxTextEncoders } from "../api";

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
  const [tes, setTes] = useState<FluxTextEncoders | null>(null);
  const [teRepo, setTeRepo] = useState("");

  async function refresh() {
    try {
      const c = await getFluxCatalog();
      setCat(c);
      setTes(await getTextEncoders().catch(() => null));
      return c;
    } catch {
      setCat(null);
      return null;
    }
  }

  useEffect(() => {
    if (open) void refresh();
  }, [open]);

  // A download keeps running on the server even if this page reloads mid-install. Adopt
  // it: show its progress, and poll until it finishes (the NDJSON stream that was
  // feeding us died with the old page, so polling is the only way back in).
  const serverInstall = cat?.installing ?? null;
  useEffect(() => {
    if (!open || !serverInstall) return;
    setBusy(serverInstall.id);
    setPct(serverInstall.pct);
    setStatus(
      `[${serverInstall.index}/${serverInstall.count}] ${serverInstall.file} — ` +
        `${(serverInstall.done / 1e9).toFixed(1)}/${(serverInstall.total / 1e9).toFixed(1)} GB`
    );
    const t = setInterval(async () => {
      const c = await refresh();
      if (c && !c.installing) {
        setBusy(null);
        setPct(null);
        setStatus("✓ installed");
        onChanged();
      }
    }, 2000);
    return () => clearInterval(t);
  }, [open, serverInstall?.id, serverInstall?.pct]);

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

  async function addTextEncoder() {
    if (!teRepo.trim()) return;
    setBusy("te");
    setStatus("starting…");
    try {
      await pullTextEncoder(teRepo.trim(), (m) => setStatus(m));
      setStatus("✓ text encoder added");
      setTeRepo("");
      await refresh();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  async function pickTextEncoder(bundleId: string, name: string) {
    try {
      await selectTextEncoder(bundleId, name);
      await refresh();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
    }
  }

  async function removeTextEncoder(name: string) {
    try {
      await deleteTextEncoder(name);
      await refresh();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
    }
  }

  const extras = models.filter((m) => m.bundle === null);
  // Only FLUX.2 models take a swappable encoder; FLUX.1's is wired into its graph.
  const flux2 = (cat?.bundles ?? []).filter((b) => b.family === "flux2" && b.installed);

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
                          {busy === b.id ? "Installing…" : `⬇ ${b.needed_gb} GB`}
                        </button>
                      )}
                    </div>
                    <div className="muted small">{b.blurb}</div>
                    <div className="muted small">
                      {b.roles.join(" + ")} · needs ~{b.vram_gb} GB VRAM
                      {b.gated && " · gated (needs a token)"}
                    </div>
                    {/* Progress lives in the card you clicked — a 50 GB download that
                        reports somewhere else reads as a hang. */}
                    {busy === b.id && (
                      <div className="bundle-progress">
                        <div className="progress">
                          <div className="progress-bar" style={{ width: `${pct ?? 0}%` }} />
                        </div>
                        <div className="muted small">{status ?? "starting…"}</div>
                      </div>
                    )}
                  </li>
                ))}
              </ul>

              {busy !== "repo" && status && !busy && (
                <div className="muted small note">{status}</div>
              )}
              <div className="muted small">{cat.disk_free_gb} GB free on disk.</div>

              <label className="lbl">HuggingFace token</label>
              <div className="muted small">
                {cat.hf_token === "saved" && "✓ A token is saved. "}
                {cat.hf_token === "env" && "✓ Using HF_TOKEN from the environment. "}
                Needed for the gated models — accept the licence on their HuggingFace page
                first, or the download 401s.
              </div>
              <div className="row">
                <input
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="hf_…"
                />
                <button className="btn" onClick={saveToken} disabled={!token.trim()}>
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

              {/* Text encoders. Separate from the model because they're separable: the
                  bundled one is a default, and swapping in a lighter quant of the same
                  architecture is the main way to fit a big model on a small card. */}
              {flux2.length > 0 && (
                <>
                  <label className="lbl">Text encoders</label>
                  {flux2.map((b) => (
                    <div key={b.id} className="row">
                      <span className="muted small te-model">{b.label}</span>
                      <select
                        value={tes?.selected[b.id] ?? ""}
                        onChange={(e) => pickTextEncoder(b.id, e.target.value)}
                        disabled={busy !== null}
                      >
                        {(tes?.encoders ?? []).map((e) => (
                          <option key={e.name} value={e.name}>
                            {e.name} ({e.size_gb} GB)
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                  <div className="muted small">
                    A model loads the encoder it was trained against. Another checkpoint of
                    the same architecture works too — a smaller quant, say. The wrong
                    architecture fails at load rather than generating badly.
                  </div>
                  {(tes?.encoders ?? []).map((e) => (
                    <div key={e.name} className="row extra-model">
                      <span className="muted small">
                        {e.name} ({e.size_gb} GB)
                        {e.default_for.length > 0 && " · default"}
                      </span>
                      <button
                        className="btn danger small"
                        onClick={() => removeTextEncoder(e.name)}
                        title="Remove"
                      >
                        🗑
                      </button>
                    </div>
                  ))}
                  <div className="row">
                    <input
                      value={teRepo}
                      onChange={(e) => setTeRepo(e.target.value)}
                      placeholder="owner/repo (HuggingFace)"
                    />
                    <button className="btn" onClick={addTextEncoder} disabled={busy !== null}>
                      ⬇ Add
                    </button>
                  </div>
                  <div className="muted small">
                    A bare repo works even in the sharded transformers layout — the shards
                    are stitched into the single file ComfyUI loads. Add{" "}
                    <code>:file</code> to name one checkpoint in a repo that holds several.
                    Gated repos use your token.
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
