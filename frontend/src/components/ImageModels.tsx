import { useEffect, useState } from "react";
import {
  clearHfToken,
  deleteFluxBundle,
  deleteFluxModel,
  deleteLora,
  deletePreprocessor,
  deleteTextEncoder,
  getCivitaiToken,
  getFluxCatalog,
  getLoras,
  getPreprocessors,
  getTextEncoders,
  installFluxBundle,
  installPreprocessor,
  pullFluxModel,
  pullLora,
  pullTextEncoder,
  setCivitaiToken,
  setLoraPicks,
  selectTextEncoder,
  setHfToken,
} from "../api";
import type {
  FluxCatalog,
  FluxLoraPick,
  FluxLoras,
  FluxModel,
  FluxPreprocessor,
  FluxTextEncoders,
  HfTokenSource,
  LoraSource,
} from "../api";

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
  // Kept apart from `status`: that renders up beside the bundle list, which on a long
  // panel is off-screen from the encoder form — an error there reads as nothing at all.
  const [teStatus, setTeStatus] = useState<string | null>(null);
  const [loras, setLoras] = useState<FluxLoras | null>(null);
  const [loraRepo, setLoraRepo] = useState("");
  const [loraSource, setLoraSource] = useState<LoraSource>("huggingface");
  const [loraStatus, setLoraStatus] = useState<string | null>(null);
  // The strength a slider is *being dragged to*, before it's saved.
  //
  // A range input fires `onChange` once per notch, and each save is a PUT plus a
  // full model-list refresh — a single drag used to be a dozen or more of them,
  // landing on the server at once. Holding the value here and committing on release
  // makes one drag one request. (The server no longer loses settings when they do
  // overlap, but the cheapest concurrent write is the one never sent.)
  const [draftStrength, setDraftStrength] = useState<Record<string, number>>({});
  const [preps, setPreps] = useState<FluxPreprocessor[]>([]);
  const [prepStatus, setPrepStatus] = useState<string | null>(null);
  // The key itself only ever travels browser -> server. `civitaiSource` is all that
  // comes back — whether one is saved, inherited from the environment, or absent —
  // which is what the field's placeholder reports.
  const [civitaiKey, setCivitaiKey] = useState("");
  const [civitaiSource, setCivitaiSource] = useState<HfTokenSource>(null);

  async function refresh() {
    try {
      const c = await getFluxCatalog();
      setCat(c);
      setTes(await getTextEncoders().catch(() => null));
      setLoras(await getLoras().catch(() => null));
      setPreps(await getPreprocessors().catch(() => []));
      setCivitaiSource(await getCivitaiToken().catch(() => null));
      return c;
    } catch {
      setCat(null);
      return null;
    }
  }

  useEffect(() => {
    // A download that finished while this panel was closed (or the page was
    // reloaded) never ran the polling branch below, so the app's own model list
    // — what the composer's picker actually reads — is still the old one.
    // Reopening the panel is the one moment we know to double check it.
    if (open) void refresh().then(() => onChanged());
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
    setPct(null);
    try {
      await pullFluxModel(
        repo.trim(),
        (m) => setStatus(m),
        (p) => {
          setStatus(`${p.file} — ${(p.done / 1e9).toFixed(1)}/${(p.total / 1e9).toFixed(1)} GB`);
          setPct(p.pct);
        }
      );
      setStatus("✓ added");
      setPct(null);
      setRepo("");
      onChanged();
    } catch (e) {
      setStatus(`✗ ${(e as Error).message}`);
      setPct(null);
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
    setTeStatus("starting…");
    setPct(null);
    try {
      await pullTextEncoder(
        teRepo.trim(),
        (m) => setTeStatus(m),
        (p) => {
          const step = p.count ? `[${p.index}/${p.count}] ` : "";
          setTeStatus(
            `${step}${p.file} — ${(p.done / 1e9).toFixed(1)}/${(p.total / 1e9).toFixed(1)} GB`
          );
          setPct(p.pct);
        }
      );
      setTeStatus("✓ text encoder added");
      setPct(null);
      setTeRepo("");
      await refresh();
    } catch (e) {
      setTeStatus(`✗ ${(e as Error).message}`);
      setPct(null);
    } finally {
      setBusy(null);
    }
  }

  async function pickTextEncoder(bundleId: string, name: string) {
    try {
      await selectTextEncoder(bundleId, name);
      await refresh();
      // The app's model list carries each model's encoder (for the header pill),
      // so a pick changes it — refresh both, not just this panel.
      onChanged();
    } catch (e) {
      setTeStatus(`✗ ${(e as Error).message}`);
    }
  }

  async function removeTextEncoder(name: string) {
    try {
      await deleteTextEncoder(name);
      await refresh();
      // Deleting the selected override drops that model back to its default
      // encoder, which the app's list is likewise reporting.
      onChanged();
    } catch (e) {
      setTeStatus(`✗ ${(e as Error).message}`);
    }
  }

  /** Saved on its own rather than as part of the download, so a key pasted once is
   *  there for every later adapter — and so a wrong one can be corrected without
   *  re-typing the model reference beside it. */
  async function saveCivitaiKey() {
    if (!civitaiKey.trim()) return;
    setBusy("lora");
    setLoraStatus("saving key…");
    try {
      setCivitaiSource(await setCivitaiToken(civitaiKey.trim()));
      setCivitaiKey("");
      setLoraStatus("✓ CivitAI key saved");
    } catch (e) {
      setLoraStatus(`✗ ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  async function addLora() {
    if (!loraRepo.trim()) return;
    setBusy("lora");
    setLoraStatus("starting…");
    setPct(null);
    try {
      await pullLora(
        loraRepo.trim(),
        loraSource,
        (m) => setLoraStatus(m),
        (p) => {
          // MB, not GB like the encoders — a LoRA that reported "0.0/0.2 GB" the
          // whole way down would look stalled.
          setLoraStatus(
            `${p.file} — ${(p.done / 1e6).toFixed(0)}/${(p.total / 1e6).toFixed(0)} MB`
          );
          setPct(p.pct);
        }
      );
      setLoraStatus("✓ LoRA added");
      setPct(null);
      setLoraRepo("");
      await refresh();
    } catch (e) {
      setLoraStatus(`✗ ${(e as Error).message}`);
      setPct(null);
    } finally {
      setBusy(null);
    }
  }

  /** Every mutation to a model's LoRA list sends the whole list — the backend
   *  replaces rather than merges, so a partial update would drop whatever wasn't
   *  named in this call. */
  async function saveModelLoras(model: string, picks: FluxLoraPick[]) {
    try {
      await setLoraPicks(model, picks);
      await refresh();
      onChanged(); // the app's model list carries the picks, for the header pill
    } catch (e) {
      setLoraStatus(`✗ ${(e as Error).message}`);
    }
  }

  // The currently-attached picks for a model come straight off the `models` prop
  // (`FluxModel.loras`), not a separately-fetched copy — that prop is App's single
  // source of truth for a model's LoRAs, kept fresh by anything that changes them.
  // A second, locally fetched copy would only stay in sync with mutations made from
  // this panel itself.
  function currentPicks(model: string): FluxLoraPick[] {
    return models.find((m) => m.name === model)?.loras ?? [];
  }

  /** Attaches as soon as it's picked — same instant-select convention the text
   *  encoder dropdown above already uses. The dropdown always resets to its
   *  placeholder afterward because the just-attached name drops out of `available`
   *  on the next render, so it can't remain selected. */
  function attachLora(model: string, name: string) {
    if (!name) return;
    saveModelLoras(model, [...currentPicks(model), { name, strength: 1.0 }]);
  }

  function reweightLora(model: string, name: string, strength: number) {
    return saveModelLoras(
      model,
      currentPicks(model).map((p) => (p.name === name ? { ...p, strength } : p))
    );
  }

  const strengthKey = (model: string, name: string) => `${model} ${name}`;

  /** Save where a strength slider was let go, if it actually moved.
   *
   * The draft is cleared only once the save has come back, so the slider doesn't
   * flick to the old value for the length of a round-trip and then flick back. */
  async function commitStrength(model: string, name: string) {
    const key = strengthKey(model, name);
    const value = draftStrength[key];
    if (value === undefined) return;
    if (currentPicks(model).find((p) => p.name === name)?.strength !== value) {
      await reweightLora(model, name, value);
    }
    setDraftStrength(({ [key]: _dropped, ...rest }) => rest);
  }

  function detachLora(model: string, name: string) {
    saveModelLoras(
      model,
      currentPicks(model).filter((p) => p.name !== name)
    );
  }

  /** Mark (or unmark) one adapter as this model's control adapter.
   *
   * At most one per model: the Control tab's strength dial is a single number, and
   * two adapters both claiming it would make that number mean whichever the loop
   * reached last. Toggling a second one moves the flag rather than adding it. */
  function setControlAdapter(model: string, name: string, on: boolean) {
    saveModelLoras(
      model,
      currentPicks(model).map((p) => ({ ...p, control: on && p.name === name }))
    );
  }

  async function togglePreprocessor(p: FluxPreprocessor) {
    if (!p.id) return; // built-in: nothing to install or remove
    setBusy(`prep:${p.id}`);
    setPrepStatus(p.installed ? "removing…" : "starting…");
    setPct(null);
    try {
      if (p.installed) {
        await deletePreprocessor(p.id);
        setPrepStatus("✓ removed");
      } else {
        await installPreprocessor(
          p.id,
          (m) => setPrepStatus(m),
          (pr) => {
            setPrepStatus(
              `${pr.file} — ${(pr.done / 1e9).toFixed(1)}/${(pr.total / 1e9).toFixed(1)} GB`
            );
            setPct(pr.pct);
          }
        );
        setPrepStatus("✓ installed");
      }
      setPct(null);
      await refresh();
      onChanged();
    } catch (e) {
      setPrepStatus(`✗ ${(e as Error).message}`);
      setPct(null);
    } finally {
      setBusy(null);
    }
  }

  async function removeLora(name: string) {
    try {
      await deleteLora(name);
      await refresh();
      onChanged(); // deleting it detaches it from whichever model had it
    } catch (e) {
      setLoraStatus(`✗ ${(e as Error).message}`);
    }
  }

  const extras = models.filter((m) => m.bundle === null);
  // One row per *transformer*, not per bundle — FLUX.1 installs dev and Kontext
  // together and each takes its own adapter. Wan is excluded — its graph can't
  // apply one — mirroring the backend's `_takes_lora` (flux_client.py).
  const loraModels = models.filter((m) => m.family !== "wan");
  // FLUX.2 and Qwen each name one encoder per bundle and load it through CLIPLoader,
  // so either can be pointed at another checkpoint of the same architecture. FLUX.1's
  // CLIP-L + T5 pair is wired into its graph as a constant, and Wan's is loaded by its
  // own graph — neither is swappable. Mirrors the backend's
  // `_SWAPPABLE_ENCODER_FAMILIES` (flux_client.py).
  const swappable = (cat?.bundles ?? []).filter(
    (b) => (b.family === "flux2" || b.family === "qwen") && b.installed
  );

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
                  {busy === "repo" ? "Adding…" : "⬇ Add"}
                </button>
              </div>
              {busy === "repo" && (
                <div className="bundle-progress">
                  <div className="progress">
                    <div className="progress-bar" style={{ width: `${pct ?? 0}%` }} />
                  </div>
                  <div className="muted small">{status ?? "starting…"}</div>
                </div>
              )}
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
              {swappable.length > 0 && (
                <>
                  <label className="lbl">Text encoders</label>
                  {swappable.map((b) => (
                    <div key={b.id} className="row">
                      <span className="muted small te-model">{b.label}</span>
                      <select
                        value={tes?.selected[b.id] ?? ""}
                        onChange={(e) => pickTextEncoder(b.id, e.target.value)}
                        disabled={busy !== null}
                      >
                        {/* Only the encoders whose architecture this model can
                            actually load — see `FluxTextEncoder.fits`. Offering the
                            rest made a pick that fails at load look like a normal
                            choice, and the two installed FLUX.2 models take different
                            architectures, so half the list was always wrong. */}
                        {(tes?.encoders ?? [])
                          .filter((e) => e.fits.includes(b.id))
                          .map((e) => (
                            <option key={e.name} value={e.name}>
                              {e.name} ({e.size_gb} GB)
                            </option>
                          ))}
                      </select>
                    </div>
                  ))}
                  <div className="muted small">
                    A model loads the encoder it was trained against, so each list holds
                    only the encoders that fit that model. Another checkpoint of the same
                    architecture works too — a smaller quant, say.
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
                      {busy === "te" ? "Adding…" : "⬇ Add"}
                    </button>
                  </div>
                  {/* Same byte counter the bundles get — these run to 48 GB, and the
                      stitch at the end is its own step, so a spinner would say nothing. */}
                  {busy === "te" && (
                    <div className="bundle-progress">
                      <div className="progress">
                        <div className="progress-bar" style={{ width: `${pct ?? 0}%` }} />
                      </div>
                      <div className="muted small">{teStatus ?? "starting…"}</div>
                    </div>
                  )}
                  {/* Reported here, not up by the bundles: a repo holding several encoders
                      answers with the list to choose from, and that has to land where the
                      person who typed the repo is actually looking. */}
                  {busy !== "te" && teStatus && (
                    <div className="muted small note te-status">{teStatus}</div>
                  )}
                  <div className="muted small">
                    A bare repo works even in the sharded transformers layout — the shards
                    are stitched into the single file ComfyUI loads. Add{" "}
                    <code>:file</code> to name one checkpoint in a repo that holds several.
                    Gated repos use your token.
                  </div>
                </>
              )}

              {/* LoRA adapters. Optional in a way the encoder isn't: no model has a
                  default, so "None" is both where everyone starts and always one
                  selection away. */}
              {loraModels.length > 0 && (
                <>
                  <label className="lbl">LoRA adapters</label>
                  {loraModels.map((m) => {
                    const picks = m.loras;
                    const attached = new Set(picks.map((p) => p.name));
                    // Only adapters trained against *this* transformer, and not
                    // already on it — see `FluxLora.fits`. A klein adapter offered
                    // for [dev] was the same trap the encoder list had: the pick
                    // succeeds, then quietly patches almost nothing.
                    const available = (loras?.loras ?? []).filter(
                      (l) => !attached.has(l.name) && l.fits.includes(m.name)
                    );
                    return (
                      <div key={m.name} className="lora-row">
                        {/* The role leads and never truncates: FLUX.1's two rows
                            come from one bundle, so they share a label that the
                            40%-width column clips to "FLUX.1 dev + K…" — leaving
                            the only distinguishing part off the end. */}
                        <span
                          className="muted small te-model lora-model"
                          title={`${m.label} — ${m.name}`}
                        >
                          <span className="lora-role">{m.roles.join("/")}</span>
                          <span className="lora-label">{m.label}</span>
                        </span>
                        {/* One block per attached adapter — several can stack on the
                            same transformer, each with its own independent weight. */}
                        {picks.map((pick) => (
                          <div key={pick.name} className="lora-pick">
                            <div className="row">
                              <span className="muted small lora-pick-name" title={pick.name}>
                                {pick.name}
                              </span>
                              <button
                                className="btn danger small"
                                onClick={() => detachLora(m.name, pick.name)}
                                disabled={busy !== null}
                                title="Detach"
                              >
                                🗑
                              </button>
                            </div>
                            {/* Most adapters want less than full weight — 1.0 often
                                overcooks the base model's own style away. */}
                            <div className="row lora-strength">
                              <input
                                type="range"
                                min="0"
                                max="1.5"
                                step="0.05"
                                value={
                                  draftStrength[strengthKey(m.name, pick.name)] ?? pick.strength
                                }
                                disabled={busy !== null}
                                onChange={(e) =>
                                  setDraftStrength((d) => ({
                                    ...d,
                                    [strengthKey(m.name, pick.name)]: parseFloat(e.target.value),
                                  }))
                                }
                                // Mouse and touch both end in pointerup; keyup catches
                                // arrow-key stepping, which never fires one. Blur is the
                                // backstop for a pointer released off the control.
                                onPointerUp={() => commitStrength(m.name, pick.name)}
                                onKeyUp={() => commitStrength(m.name, pick.name)}
                                onBlur={() => commitStrength(m.name, pick.name)}
                              />
                              <span className="muted small lora-weight">
                                {(
                                  draftStrength[strengthKey(m.name, pick.name)] ?? pick.strength
                                ).toFixed(2)}
                              </span>
                            </div>
                            {/* Which adapter is the control one can't be read off
                                the file — the name is a hint, not a fact — so it's
                                declared here, and the Control tab's strength dial
                                drives whichever is flagged. */}
                            <label
                              className="lora-control-flag muted small"
                              title="Flag this as the adapter that makes the model follow a control map. The Control tab's strength dial scales it."
                            >
                              <input
                                type="checkbox"
                                checked={!!pick.control}
                                disabled={busy !== null}
                                onChange={(e) =>
                                  setControlAdapter(m.name, pick.name, e.target.checked)
                                }
                              />
                              control adapter
                            </label>
                          </div>
                        ))}
                        {available.length > 0 && (
                          <div className="row">
                            <select
                              value=""
                              onChange={(e) => attachLora(m.name, e.target.value)}
                              disabled={busy !== null}
                            >
                              <option value="">+ Add a LoRA…</option>
                              {available.map((l) => (
                                <option key={l.name} value={l.name}>
                                  {l.name} ({l.size_mb} MB)
                                </option>
                              ))}
                            </select>
                          </div>
                        )}
                      </div>
                    );
                  })}
                  <div className="muted small">
                    A LoRA is a small patch over the transformer — it changes what the
                    model renders without replacing the checkpoint. Several can stack
                    on the same model at once, each at its own weight. Models pick
                    separately, because adapters are trained against one base: a
                    FLUX.2 [dev] LoRA won't bind to klein, and a FLUX.1 dev one won't
                    bind to Kontext. Start around 0.6-0.8. If one of them is a control
                    adapter — trained to make the model follow a pose or depth map —
                    tick <em>control adapter</em> on it, and the Control tab's strength
                    dial will drive that one.
                  </div>
                  {(loras?.loras ?? []).map((l) => (
                    <div key={l.name} className="row extra-model">
                      <span className="muted small">
                        {l.name} ({l.size_mb} MB)
                      </span>
                      <button
                        className="btn danger small"
                        onClick={() => removeLora(l.name)}
                        title="Remove"
                      >
                        🗑
                      </button>
                    </div>
                  ))}
                  <div className="row">
                    <select
                      className="lora-source"
                      value={loraSource}
                      onChange={(e) => setLoraSource(e.target.value as LoraSource)}
                      disabled={busy !== null}
                      title="Where to install this adapter from"
                    >
                      <option value="huggingface">HuggingFace</option>
                      <option value="civitai">CivitAI</option>
                    </select>
                    <input
                      value={loraRepo}
                      onChange={(e) => setLoraRepo(e.target.value)}
                      placeholder={
                        loraSource === "civitai"
                          ? "civitai.com/models/… or the id"
                          : "owner/repo:file.safetensors"
                      }
                      onKeyDown={(e) => e.key === "Enter" && addLora()}
                    />
                    <button className="btn" onClick={addLora} disabled={busy !== null}>
                      {busy === "lora" ? "Adding…" : "⬇ Add"}
                    </button>
                  </div>
                  {busy === "lora" && (
                    <div className="bundle-progress">
                      <div className="progress">
                        <div className="progress-bar" style={{ width: `${pct ?? 0}%` }} />
                      </div>
                      <div className="muted small">{loraStatus ?? "starting…"}</div>
                    </div>
                  )}
                  {busy !== "lora" && loraStatus && (
                    <div className="muted small note te-status">{loraStatus}</div>
                  )}
                  {loraSource === "civitai" && (
                    <div className="row">
                      <input
                        type="password"
                        value={civitaiKey}
                        onChange={(e) => setCivitaiKey(e.target.value)}
                        placeholder={
                          civitaiSource === "saved"
                            ? "API key saved — paste a new one to replace it"
                            : civitaiSource === "env"
                              ? "Using CIVITAI_TOKEN from the environment"
                              : "CivitAI API key (most downloads need one)"
                        }
                        onKeyDown={(e) => e.key === "Enter" && saveCivitaiKey()}
                      />
                      <button
                        className="btn"
                        onClick={saveCivitaiKey}
                        disabled={busy !== null || !civitaiKey.trim()}
                      >
                        Save
                      </button>
                    </div>
                  )}
                  <div className="muted small">
                    {loraSource === "civitai" ? (
                      <>
                        Paste the adapter's page URL — a <code>?modelVersionId=</code> in
                        it picks that exact version, otherwise the newest one is taken.
                        Most CivitAI downloads need an API key (civitai.com → Account
                        settings → API Keys); it's stored server-side and never sent back
                        to the browser.
                      </>
                    ) : (
                      <>
                        <code>owner/repo</code> on its own works when the repo holds
                        exactly one adapter; name the file otherwise. Most FLUX.2
                        adapters are published on CivitAI — switch the source above.
                      </>
                    )}
                  </div>

                  <div className="lbl">Control preprocessors</div>
                  {preps.map((p) => (
                    <div key={p.kind} className="row extra-model prep-row">
                      <span className="muted small prep-name" title={p.note}>
                        {p.label}
                        {p.builtin ? (
                          <span className="prep-badge built-in">built in</span>
                        ) : p.installed ? (
                          <span className="prep-badge">{p.size_gb} GB</span>
                        ) : (
                          <span className="prep-badge muted">{p.size_gb} GB download</span>
                        )}
                      </span>
                      {!p.builtin && (
                        <button
                          className={`btn small ${p.installed ? "danger" : ""}`}
                          onClick={() => togglePreprocessor(p)}
                          disabled={busy !== null}
                          title={p.installed ? "Remove" : "Install"}
                        >
                          {busy === `prep:${p.id}` ? "…" : p.installed ? "🗑" : "⬇"}
                        </button>
                      )}
                    </div>
                  ))}
                  {busy?.startsWith("prep:") && (
                    <div className="bundle-progress">
                      <div className="progress">
                        <div className="progress-bar" style={{ width: `${pct ?? 0}%` }} />
                      </div>
                      <div className="muted small">{prepStatus ?? "starting…"}</div>
                    </div>
                  )}
                  {!busy?.startsWith("prep:") && prepStatus && (
                    <div className="muted small note te-status">{prepStatus}</div>
                  )}
                  <div className="muted small">
                    These turn a reference photo into the control map the{" "}
                    <strong>🕹️ Control</strong> tab conditions on, so you can copy a pose
                    words can't describe. <strong>Depth</strong> is the one to install
                    first: it carries the whole scene, so it can say a subject is
                    <em> sitting on</em> a chair rather than floating near one.{" "}
                    <strong>Pose</strong> adds a skeleton on top, which is what pins
                    which limb is which in a hard pose. Edges needs no download.
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
