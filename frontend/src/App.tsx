import { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import Composer from "./components/Composer";
import ImageBar from "./components/ImageBar";
import ContextMeter from "./components/ContextMeter";
import GenModelPill from "./components/GenModelPill";
import {
  appendMessage,
  createPreset,
  deleteChat,
  deletePreset,
  enhancePrompt,
  generate,
  generateTitle,
  getChat,
  getFluxModels,
  getLoras,
  getModels,
  listChats,
  listPresets,
  putChat,
  setLoraPicks,
  streamChat,
  uploadImages,
  urlToDataUrl,
  type FluxModel,
  type GenPreset,
  type Usage,
} from "./api";
import { fileToDataUrl, resizeDataUrl, rotateDataUrl } from "./fileUtils";
import { guidanceFor, imagesFor, modeFor, resolveFlux, roleFor, stepsFor } from "./flux";
import { trimHistory } from "./context";
import type {
  ChatMessage,
  ChatSummary,
  GenProgress,
  GenSettings,
  GenOp,
  EnhancerMode,
} from "./types";

const DEFAULT_URL = "http://localhost:11434";

/** Extract the sha256 hash from an image/thumb URL like /api/images/<hash>.png.
 *  The store keeps each image in its own format, so match any extension. */
function hashFromUrl(url: string): string {
  return (url.split("/").pop() || "").replace(/\.[a-z0-9]+$/i, "");
}

export default function App() {
  const [ollamaUrl, setOllamaUrl] = useState(
    () => localStorage.getItem("ollamaUrl") || DEFAULT_URL
  );
  const [models, setModels] = useState<{ vision: string[]; all: string[] }>({
    vision: [],
    all: [],
  });
  const [model, setModel] = useState(() => localStorage.getItem("model") || "");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [systemImage, setSystemImage] = useState<string | null>(null);

  // Prompt enhancer settings (sidebar). `enhancerModel` "" = auto-detect, same
  // fallback the manual ✨ button always used. `enhancerMode` gates whether that
  // rewrite also runs automatically before a generation, and whether it's shown.
  const [enhancerModel, setEnhancerModel] = useState(
    () => localStorage.getItem("enhancerModel") || ""
  );
  const [enhancerMode, setEnhancerMode] = useState<EnhancerMode>(
    () => (localStorage.getItem("enhancerMode") as EnhancerMode) || "off"
  );

  // Image generation. `genMode` flips the composer from analyze to generate;
  // `fluxAvailable` reports whether the engine *and* a model are installed;
  // `gen` holds the tunable settings.
  const [genMode, setGenMode] = useState(false);
  // Which generate workflow: create (txt2img/img2img), edit (instruction), or
  // compose (blend multiple reference images).
  const [genOp, setGenOp] = useState<GenOp>("create");
  const [fluxAvailable, setFluxAvailable] = useState(false);
  // Prompt enhancement. `enhanceTemplate` is the independent photoreal-template
  // toggle (sidebar-level, see `PromptEnhancer`). `enhancing` guards the brief
  // async gap while the settings-level auto-enhancer runs before a generation
  // goes out. Verbose mode's rewrite isn't held in state here — it's attached
  // directly to the chat message it produced (`ChatMessage.enhancedPrompt`) by
  // `generateImage`, and rendered under that message, not in the composer.
  const [enhanceTemplate, setEnhanceTemplate] = useState(
    () => localStorage.getItem("enhanceTemplate") !== "false"
  );
  const [enhancing, setEnhancing] = useState(false);
  const [gen, setGen] = useState<GenSettings>({
    fluxModel: "", // "" = let the backend pick this mode's default
    steps: 20,
    guidance: 3.5, // retuned to the installed model — see `guidanceFor`
    strength: 0.6,
    width: 1024, // FLUX is trained at ~1 megapixel
    height: 1024,
    seed: "",
  });
  // Installed image models. Refreshed after an install/removal so the composer's
  // picker stays in sync with the sidebar's Image Models panel.
  const [fluxModels, setFluxModels] = useState<FluxModel[]>([]);
  const guidanceReady = useRef(false);
  const refreshFlux = useCallback(() => {
    getFluxModels()
      .then((r) => {
        setFluxAvailable(r.available);
        setFluxModels(r.models);
        // Guidance defaults depend on which model is installed, which we only learn
        // here. Set it once, on the first list we see, so a value the user has since
        // tuned by hand doesn't get reset by a later install.
        if (!guidanceReady.current && r.models.length) {
          guidanceReady.current = true;
          setGen((g) => ({
            ...g,
            guidance: guidanceFor(genOp, r.models),
            steps: stepsFor(genOp, r.models),
          }));
        }
      })
      .catch(() => {
        /* backend older / no model installed — generation stays hidden */
      });
  }, [genOp]);

  // Switching workflow retunes guidance, and drops the model pick only when that
  // model can't serve the new role — which on FLUX.2 is never, since one model
  // does every job. (On FLUX.1 the two modes draw from disjoint sets, so a create
  // pick genuinely can't edit.) Clearing it unconditionally used to silently swap
  // the model out from under an explicit choice while the picker still showed it.
  // Steps are left alone: they aren't role-dependent, so a hand-tuned value should
  // survive a tab switch.
  const changeOp = useCallback(
    (op: GenOp) => {
      setGenOp(op);
      setGen((g) => {
        const keep = fluxModels.some(
          (m) => m.name === g.fluxModel && m.roles.includes(roleFor(op))
        );
        const fluxModel = keep ? g.fluxModel : "";
        return {
          ...g,
          fluxModel,
          guidance: guidanceFor(op, fluxModels, fluxModel),
          // Steps aren't role-dependent for a given model, so a hand-tuned value
          // survives a tab switch — unless the switch forced the model itself to
          // change (`!keep`), in which case the new model's own default applies.
          steps: keep ? g.steps : stepsFor(op, fluxModels, fluxModel),
        };
      });
    },
    [fluxModels]
  );

  const [pinnedImages, setPinnedImages] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  // Liveness of the in-flight turn. Separate from `streaming` because it carries
  // *when* the last backend event landed, which is the only thing that tells a
  // slow first-run model load apart from a wedged one.
  const [progress, setProgress] = useState<GenProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Persistence: one "current chat" whose id is minted up front; the DB row is
  // created lazily on the first send. `chatExists` gates metadata sync so we
  // never create empty, message-less chats.
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string>(() =>
    crypto.randomUUID()
  );
  const [chatExists, setChatExists] = useState(false);
  // Saved generation-setting bundles (steps/guidance/size/model/LoRAs), named by
  // the user — see `applyPreset`/`saveCurrentAsPreset` below.
  const [presets, setPresets] = useState<GenPreset[]>([]);
  // data-URL -> content hash, so re-sent pinned images aren't re-uploaded.
  const hashCache = useRef<Map<string, string>>(new Map());
  // True while the pinned Images panel is "focused" (clicked). Pasted images are
  // routed there instead of to the message composer while it's armed.
  const pasteToPinnedRef = useRef(false);
  const handleLockChange = useCallback((v: boolean) => {
    pasteToPinnedRef.current = v;
  }, []);

  /** Upload any not-yet-stored images and return their hashes (order-preserved). */
  const ensureHashes = useCallback(async (urls: string[]): Promise<string[]> => {
    return Promise.all(
      urls.map(async (url) => {
        const cached = hashCache.current.get(url);
        if (cached) return cached;
        const thumb = await resizeDataUrl(url, 64);
        const [hash] = await uploadImages([{ full: url, thumb }]);
        hashCache.current.set(url, hash);
        return hash;
      })
    );
  }, []);

  // data-URL -> its downscaled copy for the vision model. Chat sends images from
  // component state, which now holds originals, and a vision model tokenizes by
  // resolution — so a 12 MP photo would blow `context_size_for`'s ceiling. The
  // downscale belongs here, at the point of sending to the consumer that wants it,
  // rather than on the upload that everything else reads from.
  const ollamaCache = useRef<Map<string, string>>(new Map());
  const forOllama = useCallback(async (urls: string[]): Promise<string[]> => {
    return Promise.all(
      urls.map(async (url) => {
        const cached = ollamaCache.current.get(url);
        if (cached) return cached;
        const small = await resizeDataUrl(url, 1280);
        ollamaCache.current.set(url, small);
        return small;
      })
    );
  }, []);

  const refreshChats = useCallback(async () => {
    try {
      setChats(await listChats());
    } catch {
      /* leave the list as-is if the fetch fails */
    }
  }, []);

  const refreshPresets = useCallback(async () => {
    try {
      setPresets(await listPresets());
    } catch {
      /* leave the list as-is if the fetch fails */
    }
  }, []);

  /** Restores everything a preset captured. Settings apply unconditionally; the
   *  LoRA restore is separate and best-effort, so a since-deleted adapter can't
   *  undo the steps/guidance/model that already landed. */
  async function applyPreset(preset: GenPreset) {
    // Set directly rather than through `changeOp`: that helper retunes guidance/
    // steps/fluxModel to *defaults* for the new op, which the preset's own values
    // below would just have to override again.
    setGenOp(preset.gen_op);
    setGen((g) => ({
      ...g,
      fluxModel: preset.flux_model,
      steps: preset.steps,
      guidance: preset.guidance,
      strength: preset.strength,
      width: preset.width,
      height: preset.height,
      // seed is left alone — a preset is a reusable style, not one fixed frame.
    }));
    try {
      const catalog = await getLoras();
      const installed = new Set(catalog.loras.map((l) => l.name));
      const picks = preset.loras.filter((p) => installed.has(p.name));
      await setLoraPicks(preset.flux_model, picks);
      refreshFlux();
    } catch {
      /* the LoRA side of the preset didn't take — the rest of it already did */
    }
  }

  /** The permanent "Default" entry in the presets list — not stored, not
   *  deletable, always available. Unlike a saved preset it's computed live off
   *  `stepsFor`/`guidanceFor`/`resolveFlux` with `""` (the same "no explicit pick"
   *  those already use everywhere else), so it always means whichever model
   *  actually resolves first for this mode rather than a name frozen at save time
   *  — and unlike a saved preset, it also resets the seed and clears LoRAs, since
   *  "default" means the clean state a fresh install starts in. */
  async function applyDefault() {
    setGen((g) => ({
      ...g,
      fluxModel: "",
      steps: stepsFor(genOp, fluxModels, ""),
      guidance: guidanceFor(genOp, fluxModels, ""),
      seed: "",
    }));
    const resolved = resolveFlux("", fluxModels, roleFor(genOp));
    if (!resolved) return;
    try {
      await setLoraPicks(resolved, []);
      refreshFlux();
    } catch {
      /* LoRA clear didn't take — the rest of the reset already applied */
    }
  }

  async function saveCurrentAsPreset(name: string) {
    const role = roleFor(genOp);
    const resolved = resolveFlux(gen.fluxModel, fluxModels, role);
    const activeModel = fluxModels.find((m) => m.name === resolved);
    try {
      const created = await createPreset({
        name,
        gen_op: genOp,
        flux_model: resolved,
        steps: gen.steps,
        guidance: gen.guidance,
        strength: gen.strength,
        width: gen.width,
        height: gen.height,
        loras: activeModel?.loras ?? [],
      });
      setPresets((p) => [...p, created]);
    } catch {
      /* leave the list as-is if the save fails */
    }
  }

  async function removePresetById(id: string) {
    try {
      await deletePreset(id);
      setPresets((p) => p.filter((x) => x.id !== id));
    } catch {
      /* leave the list as-is if the delete fails */
    }
  }

  async function addPinned(files: FileList | File[]) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    // Keep the original bytes. The backend caps and resamples once, with a better
    // filter, and hands each consumer the size it wants (`forOllama` below for the
    // vision model, full-res for FLUX) — so nothing degrades what we store.
    const urls = await Promise.all(list.map((f) => fileToDataUrl(f)));
    setPinnedImages((prev) => [...prev, ...urls]);
  }
  function removePinned(i: number) {
    setPinnedImages((prev) => prev.filter((_, idx) => idx !== i));
  }
  async function rotatePinned(i: number) {
    const rotated = await rotateDataUrl(pinnedImages[i], 90);
    setPinnedImages((prev) => prev.map((img, idx) => (idx === i ? rotated : img)));
  }

  // Persist settings.
  useEffect(() => localStorage.setItem("ollamaUrl", ollamaUrl), [ollamaUrl]);
  useEffect(() => localStorage.setItem("model", model), [model]);
  useEffect(() => localStorage.setItem("enhancerModel", enhancerModel), [enhancerModel]);
  useEffect(() => localStorage.setItem("enhancerMode", enhancerMode), [enhancerMode]);
  useEffect(
    () => localStorage.setItem("enhanceTemplate", String(enhanceTemplate)),
    [enhanceTemplate]
  );

  const refreshModels = useCallback(async () => {
    try {
      const m = await getModels(ollamaUrl);
      setModels(m);
      setModel((cur) =>
        cur && m.vision.includes(cur) ? cur : m.vision[0] ?? cur
      );
      setError(null);
    } catch {
      setError(`Could not reach Ollama at ${ollamaUrl}`);
      setModels({ vision: [], all: [] });
    }
  }, [ollamaUrl]);

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  useEffect(() => {
    refreshChats();
  }, [refreshChats]);

  useEffect(() => {
    refreshPresets();
  }, [refreshPresets]);

  // Probe the image-generation backend once.
  useEffect(() => {
    refreshFlux();
  }, [refreshFlux]);

  // Keep an existing chat's metadata (model / system prompt / pinned + system
  // image) in sync as the user edits it, debounced. Skipped until the chat row
  // exists (created on first send) so we don't spawn empty chats.
  useEffect(() => {
    if (!chatExists) return;
    const t = setTimeout(async () => {
      try {
        const pinned_hashes = await ensureHashes(pinnedImages);
        const system_image_hash = systemImage
          ? (await ensureHashes([systemImage]))[0]
          : null;
        await putChat(currentChatId, {
          model,
          system_prompt: systemPrompt,
          pinned_hashes,
          system_image_hash,
        });
      } catch {
        /* non-fatal */
      }
    }, 500);
    return () => clearTimeout(t);
  }, [
    chatExists,
    currentChatId,
    model,
    systemPrompt,
    pinnedImages,
    systemImage,
    ensureHashes,
  ]);

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    setCurrentChatId(crypto.randomUUID());
    setChatExists(false);
    setMessages([]);
    setPinnedImages([]);
    setSystemPrompt("");
    setSystemImage(null);
    setUsage(null);
    setError(null);
    setComposerText("");
    setComposerImages([]);
  }, []);

  const openChat = useCallback(async (id: string) => {
    try {
      abortRef.current?.abort();
      const d = await getChat(id);

      // Load images back into memory as data-URLs and pre-seed the hash cache
      // so they aren't re-uploaded on the next send. A missing file (e.g. one a
      // past GC removed) resolves to null so we can drop it instead of showing a
      // broken image or re-persisting a dead reference.
      const loadImg = async (url: string): Promise<string | null> => {
        try {
          const data = await urlToDataUrl(url);
          hashCache.current.set(data, hashFromUrl(url));
          return data;
        } catch {
          return null;
        }
      };
      const present = (arr: (string | null)[]) =>
        arr.filter((x): x is string => x !== null);
      // The store holds video under the same `images` list a still uses — it's one
      // content-addressed blob store and the hash doesn't say what it is. The URL's
      // extension does, and it's the only signal here. Split on it so `images`
      // stays "data-URLs of decodable images" on reload, exactly as it is live.
      const isVideoUrl = (url: string) => url.toLowerCase().endsWith(".webm");

      // Dropping missing pinned images here also self-heals the DB: the next
      // send re-persists the pinned set without them.
      const pinned = present(await Promise.all(d.pinned.map(loadImg)));
      const sysImg = d.system_image ? await loadImg(d.system_image) : null;
      const msgs: ChatMessage[] = await Promise.all(
        d.messages.map(async (m) => {
          const videos = m.images.filter(isVideoUrl);
          videos.forEach((url) => hashCache.current.set(url, hashFromUrl(url)));
          const images = present(
            await Promise.all(m.images.filter((u) => !isVideoUrl(u)).map(loadImg))
          );
          // Keep context-image positions stable (missing -> "") so the model's
          // "image N" references still line up; "" simply renders no thumbnail.
          const contextImages = (
            await Promise.all(m.context_images.map(loadImg))
          ).map((x) => x ?? "");
          return {
            role: m.role,
            content: m.content,
            model: m.model ?? undefined,
            images: images.length ? images : undefined,
            videos: videos.length ? videos : undefined,
            contextImages: contextImages.length ? contextImages : undefined,
          };
        })
      );

      setCurrentChatId(d.id);
      setChatExists(true);
      setPinnedImages(pinned);
      setSystemImage(sysImg);
      setSystemPrompt(d.system_prompt || "");
      if (d.model) setModel(d.model);
      setMessages(msgs);
      setUsage(null);
      setError(null);
      setComposerText("");
      setComposerImages([]);
    } catch {
      setError("Could not open that chat.");
    }
  }, []);

  const removeChat = useCallback(
    async (id: string) => {
      try {
        await deleteChat(id);
      } catch {
        /* ignore */
      }
      if (id === currentChatId) newChat();
      refreshChats();
    },
    [currentChatId, newChat, refreshChats]
  );

  const send = useCallback(
    async (text: string, images: string[]) => {
      if (!model) {
        setError("Select a vision model first.");
        return;
      }
      setError(null);

      const chatId = currentChatId;
      const isFirstExchange = messages.length === 0;

      const userMsg: ChatMessage = { role: "user", content: text, images, model };
      const history = [...messages, userMsg];
      setMessages([...history, { role: "assistant", content: "", model }]);
      setStreaming(true);
      // Text has one silent stretch — Ollama loading the model before the first
      // token. The streamed text is its own liveness signal after that, so the
      // bar clears on first token rather than running the whole turn.
      // No fraction: Ollama reports nothing at all while it loads, so the bar shows
      // the phase and the clock rather than inventing a position for itself.
      setProgress({
        phase: `Loading ${model}…`,
        stage: "",
        frac: null,
        step: 0,
        total: 0,
        startedAt: Date.now(),
        updatedAt: Date.now(),
      });

      // Auto-trim oldest turns from what we SEND (the UI keeps the full history)
      // when the last measured usage shows we're near the window limit.
      const { sent } = trimHistory(history, pinnedImages.length, usage);

      // Image-sending policy (Ollama /api/chat is stateless AND only attends to
      // images on the CURRENT/last message — images on earlier history messages
      // are ignored by vision models). So we consolidate every image the
      // (trimmed) conversation has seen onto the last message: pinned/primary
      // first, then each in-chat attachment in order. Earlier messages go
      // text-only. This keeps the model aware of every photo across follow-ups,
      // not just the one from the latest turn. trimHistory (above) already sheds
      // the oldest turns/images under context pressure, so this stays within the
      // window budget (total image count is unchanged, just relocated + deduped).
      //
      // The chat API hands the model N *unlabeled* images with no anchor for
      // which is which, so it conflates distinct photos when asked to compare
      // them ("this image" vs "the initial one"). We therefore also build a short
      // text manifest, in the same order as the images array, so the model can
      // tell the pinned reference(s) apart from images shared earlier vs. now.
      // Images keep a single global numbering (Image 1..N, in array order) —
      // that's how models actually refer to them and how the UI resolves each
      // "Image N" back to its thumbnail. But we group them under clear section
      // headers so the model doesn't skim past which ones are the persistent
      // pinned references vs. what was actually sent in the chat (even capable
      // models mislabel a pinned image as "shared" when it's just one line in a
      // flat list). Pinned images always come first, so contextImages =
      // [pinned..., chat...].
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
      // Record the manifest's ordered image list on the assistant turn so the UI
      // can resolve the model's "image N" references back to a thumbnail.
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, contextImages: outImages };
        }
        return next;
      });
      // What actually goes on the wire: downscaled copies. `outImages` stays as the
      // originals, since the UI resolves "Image N" back to those for display.
      const wireImages = await forOllama(outImages);
      const merged = sent.map((m, i) => {
        if (i !== lastIdx) {
          return { role: m.role, content: m.content }; // strip history images (Ollama ignores them)
        }
        if (!outImages.length) return { role: m.role, content: m.content };
        // Only annotate when there's more than one image (nothing to disambiguate otherwise).
        let note = "";
        if (outImages.length > 1) {
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
          note =
            `[The ${outImages.length} images below are numbered 1-${outImages.length} in the ` +
            `order shown; refer to each by its number.\n\n${sections.join("\n\n")}]\n\n`;
        }
        return { role: m.role, content: note + m.content, images: wireImages };
      });

      // Build the request: optional system message (with persistent image) + history.
      const payload: ChatMessage[] = [];
      if (systemPrompt.trim()) {
        payload.push({
          role: "system",
          content: systemPrompt.trim(),
          images: systemImage ? await forOllama([systemImage]) : undefined,
        });
      }
      payload.push(...merged);

      let assistantText = "";
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        await streamChat(
          ollamaUrl,
          model,
          payload,
          {
            onToken: (token) => {
              assistantText += token;
              setProgress(null);
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = {
                  ...next[next.length - 1],
                  content: next[next.length - 1].content + token,
                };
                return next;
              });
            },
            onUsage: (u) => setUsage(u),
            onError: (msg) =>
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = {
                  ...next[next.length - 1],
                  content:
                    (next[next.length - 1].content || "") + `\n\n⚠️ ${msg}`,
                };
                return next;
              }),
          },
          controller.signal
        );
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          setError(`Could not reach Ollama at ${ollamaUrl}`);
        }
      } finally {
        setStreaming(false);
        setProgress(null);
        abortRef.current = null;

        // Persist this turn (best-effort; a failure here must not break the UI).
        try {
          const pinned_hashes = await ensureHashes(pinnedImages);
          const system_image_hash = systemImage
            ? (await ensureHashes([systemImage]))[0]
            : null;
          await putChat(chatId, {
            model,
            system_prompt: systemPrompt,
            pinned_hashes,
            system_image_hash,
          });
          setChatExists(true);

          const userHashes = await ensureHashes(images);
          await appendMessage(chatId, {
            role: "user",
            content: text,
            model,
            image_hashes: userHashes,
          });
          await appendMessage(chatId, {
            role: "assistant",
            content: assistantText,
            model,
            image_hashes: [],
            context_hashes: await ensureHashes(outImages),
          });

          if (isFirstExchange) {
            await generateTitle(chatId, model, ollamaUrl).catch(() => "");
          }
          await refreshChats();
        } catch (err) {
          console.error("persist failed", err);
        }
      }
    },
    [
      messages,
      model,
      ollamaUrl,
      systemPrompt,
      systemImage,
      pinnedImages,
      usage,
      currentChatId,
      ensureHashes,
      forOllama,
      refreshChats,
    ]
  );

  const generateImage = useCallback(
    // `prompt` is always what the user typed — it's what's shown in the chat
    // turn. `sendPrompt` is what actually goes to the backend: the same text,
    // unless the settings-level auto-enhancer rewrote it, in which case the chat
    // still shows the original while the rewrite does the generating.
    // `displayEnhanced` (Verbose mode only) is that same rewrite, attached to
    // the chat message so it renders underneath the user's prompt.
    async (
      prompt: string,
      op: GenOp,
      images: string[],
      sendPrompt = prompt,
      displayEnhanced: string | null = null
    ) => {
      // Which transformer this run will use. Resolved once, here: it names the
      // placeholder turn, and it's what goes on the wire — so what the composer
      // shows, what the chat says, and what the backend loads are all one value.
      // (The backend echoes back the model it actually used; that wins on arrival.)
      const fluxModel = resolveFlux(gen.fluxModel, fluxModels, roleFor(op));
      const modelId =
        fluxModels.find((m) => m.name === fluxModel)?.label || fluxModel || "FLUX";
      if (!fluxAvailable) {
        setError("FLUX isn't installed on this machine. Run ./run.sh to fetch the weights.");
        return;
      }
      setError(null);
      const chatId = currentChatId;
      const isFirstExchange = messages.length === 0;

      // compose blends every reference image. edit takes the first image as the
      // scene being changed and the rest as subject references. create uses one
      // source image, and infers txt2img vs img2img from its presence.
      // animate takes one source image, like create's img2img, and no references.
      const isCompose = op === "compose";
      const initUrl = isCompose ? null : images[0] ?? null;
      const refUrls = isCompose ? images : op === "edit" ? images.slice(1) : [];
      // Shared with the prompt enhancer, so both brief the model on the same job.
      const mode = modeFor(op, images);
      // Every image the job conditions on, in the order the backend receives them.
      // edit has two kinds — the scene in `initUrl` and the subject references after
      // it — and showing only the first made the references invisible in the turn
      // that used them. compose has no init, so the spread covers it too.
      const conditioning = [...(initUrl ? [initUrl] : []), ...refUrls];
      const shownImages = conditioning.length ? conditioning : undefined;
      const icon = op === "animate" ? "🎬" : "🎨";

      // Show the prompt as a user turn, then an assistant placeholder we fill
      // with progress text and finally the generated image.
      const userMsg: ChatMessage = {
        role: "user",
        content: prompt,
        images: shownImages,
        model: modelId,
        enhancedPrompt: displayEnhanced ?? undefined,
      };
      setMessages((prev) => [
        ...prev,
        userMsg,
        { role: "assistant", content: `${icon} Preparing…`, model: modelId },
      ]);
      setStreaming(true);
      setProgress({
        phase: "Preparing…",
        stage: "",
        frac: null,
        step: 0,
        total: 0,
        startedAt: Date.now(),
        updatedAt: Date.now(),
      });

      const setAssistant = (patch: Partial<ChatMessage>) =>
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = { ...next[next.length - 1], ...patch };
          return next;
        });

      const controller = new AbortController();
      abortRef.current = controller;
      let resultHash: string | null = null;
      let resultDataUrl: string | null = null;
      // Starts as this client's prediction; the backend's echo replaces it.
      let resultLabel = modelId;
      try {
        const initHash = initUrl ? (await ensureHashes([initUrl]))[0] : null;
        const refHashes = refUrls.length ? await ensureHashes(refUrls) : [];

        // Create the chat before sampling, so the backend has a row to record the
        // turns against — it writes them from a thread that outlives this page,
        // which is what lets a reload mid-generation find the result. Best-effort
        // for the same reason it always was: losing the history is a smaller harm
        // than refusing to generate.
        try {
          await putChat(chatId, {
            model: model || modelId,
            system_prompt: systemPrompt,
            pinned_hashes: await ensureHashes(pinnedImages),
            system_image_hash: systemImage
              ? (await ensureHashes([systemImage]))[0]
              : null,
          });
          setChatExists(true);
        } catch (err) {
          console.error("putChat failed; this turn won't be recorded", err);
        }

        await generate(
          {
            mode,
            chat_id: chatId,
            flux_model: fluxModel || null,
            prompt: sendPrompt,
            // Only set when the two diverge, so a reloaded turn shows what the
            // user actually saw live rather than a rewrite they never typed.
            display_prompt: sendPrompt !== prompt ? prompt : null,
            init_image_hash: initHash,
            ref_image_hashes: refHashes,
            steps: gen.steps,
            guidance: gen.guidance,
            strength: gen.strength,
            // The static template is a fallback for an un-enhanced create prompt.
            // Wrapping it around a prompt the settings-level auto-enhancer already
            // rewrote would bury the rewrite's own framing inside a second,
            // blunter one.
            enhance: enhanceTemplate && sendPrompt === prompt,
            width: gen.width,
            height: gen.height,
            seed: gen.seed ? parseInt(gen.seed, 10) : null,
            ollama_url: ollamaUrl,
          },
          {
            onStatus: (m) => {
              setAssistant({ content: `${icon} ${m}` });
              // A status line renames the job; it doesn't rewind it. The fraction
              // covers the whole graph, so it survives every phase change until the
              // job ends.
              setProgress((p) => (p ? { ...p, phase: m, updatedAt: Date.now() } : p));
            },
            onProgress: ({ live, ...p }) => {
              setAssistant({
                content:
                  p.step > 0
                    ? `${icon} ${p.stage}… step ${p.step}/${p.total} (${Math.round(p.frac * 100)}%)`
                    : `${icon} ${p.stage}… ${Math.round(p.frac * 100)}%`,
              });
              // Only a live update means the backend is still there, so only a live
              // update clears the stall clock — an estimate ticking through a silent
              // load looks identical to a wedged one, and must not vouch for it.
              setProgress((cur) =>
                cur
                  ? { ...cur, ...p, updatedAt: live ? Date.now() : cur.updatedAt }
                  : cur
              );
            },
            onImage: async (r) => {
              resultHash = r.hash;
              // The store keeps each result in its own format, so take the URL the
              // backend built rather than assuming an extension here.
              const url = r.url || `/api/images/${r.hash}.png`;
              // The model the backend actually ran, which is the authoritative
              // answer — `modelId` above is only this client's prediction of it.
              if (r.model_label) resultLabel = r.model_label;

              if (r.kind === "video") {
                // Deliberately *not* urlToDataUrl'd. A 5s 720p clip is megabytes,
                // which is a lot to hold base64'd in state for every turn — and
                // `images` is data-URLs by convention, feeding a canvas resize
                // (`ensureHashes`), the pin panel and the vision model, none of
                // which can read a webm. Keeping video in its own field is what
                // stops it reaching them.
                hashCache.current.set(url, r.hash);
                setAssistant({ content: "", videos: [url], model: resultLabel });
              } else {
                // Load the stored image back as a data-URL for display + pin/reuse
                // parity, and seed the hash cache so it isn't re-uploaded.
                try {
                  resultDataUrl = await urlToDataUrl(url);
                  hashCache.current.set(resultDataUrl, r.hash);
                } catch {
                  /* fall back to the URL below */
                }
                setAssistant({
                  content: "",
                  images: [resultDataUrl ?? url],
                  model: resultLabel,
                });
              }
              // Relabel the user turn too, so the pair on screen matches the one the
              // backend recorded — otherwise a fallback shows this client's guess
              // now and the model that really ran on reload.
              setMessages((prev) =>
                prev.map((m, i) =>
                  i === prev.length - 2 && m.role === "user"
                    ? { ...m, model: resultLabel }
                    : m
                )
              );
            },
            onError: (msg) => setAssistant({ content: `⚠️ ${msg}` }),
          },
          controller.signal
        );
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          setAssistant({ content: `⚠️ ${(e as Error).message}` });
        }
      } finally {
        setStreaming(false);
        setProgress(null);
        abortRef.current = null;

        // Both turns are the backend's to write (see `chat_id` above), so what's
        // left here is only the work that needs this page: the title, which runs
        // on the VLM, and the sidebar.
        if (resultHash) {
          try {
            if (isFirstExchange && model) {
              await generateTitle(chatId, model, ollamaUrl).catch(() => "");
            }
            await refreshChats();
          } catch (err) {
            console.error("post-generate refresh failed", err);
          }
        }
      }
    },
    [
      gen,
      enhanceTemplate,
      fluxAvailable,
      fluxModels,
      model,
      ollamaUrl,
      systemPrompt,
      systemImage,
      pinnedImages,
      messages,
      currentChatId,
      ensureHashes,
      refreshChats,
    ]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  // Composer (full-width, bottom) state lives here so the input bar spans the
  // whole width — unobstructed by the left image panel.
  const [composerText, setComposerText] = useState("");
  const [composerImages, setComposerImages] = useState<string[]>([]);
  async function addComposerFiles(files: FileList | File[]) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (list.length === 0) return;
    // Original bytes — these are FLUX's reference images. See `addPinned`.
    const urls = await Promise.all(list.map((f) => fileToDataUrl(f)));
    setComposerImages((prev) => [...prev, ...urls]);
  }
  function removeComposerImage(i: number) {
    setComposerImages((prev) => prev.filter((_, idx) => idx !== i));
  }

  // Paste images (Ctrl+V) anywhere → attach to the current message, or to the
  // pinned panel when the cursor is over it. Text paste into inputs is untouched.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const files = Array.from(e.clipboardData?.items ?? [])
        .filter((it) => it.kind === "file" && it.type.startsWith("image/"))
        .map((it) => it.getAsFile())
        .filter((f): f is File => !!f);
      if (files.length) {
        e.preventDefault();
        if (pasteToPinnedRef.current) addPinned(files);
        else addComposerFiles(files);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);
  async function rotateComposerImage(i: number) {
    const rotated = await rotateDataUrl(composerImages[i], 90);
    setComposerImages((prev) => prev.map((img, idx) => (idx === i ? rotated : img)));
  }
  /** The Ollama model that will do the rewriting: the Settings pick if the user
   *  made one, else the one in use if it can see, else any vision model. ""
   *  means none is installed/picked — skip the call entirely. */
  const effectiveEnhanceModel =
    enhancerModel || (models.vision.includes(model) ? model : models.vision[0] || "");

  /** Settings-level auto-enhance, run right before a generation goes out.
   *  Off: no-op. On/Verbose: rewrites with the picked vision model, reading
   *  whatever images the generation itself will condition on — `sendPrompt` is
   *  what actually goes to the backend either way. `displayEnhanced` is that
   *  same rewrite, but only in Verbose mode: the caller attaches it to the chat
   *  message so it renders underneath the user's own typed prompt, right in the
   *  chat window. On keeps the rewrite invisible — `displayEnhanced` is null.
   *  Fails soft: an unreachable Ollama, or no vision model picked/installed,
   *  just means the typed prompt goes out as-is (`/api/flux/enhance` has the
   *  same fallback contract). */
  async function maybeAutoEnhance(
    prompt: string,
    op: GenOp,
    images: string[]
  ): Promise<{ sendPrompt: string; displayEnhanced: string | null }> {
    if (enhancerMode === "off" || !effectiveEnhanceModel) {
      return { sendPrompt: prompt, displayEnhanced: null };
    }
    const seen = imagesFor(op, images);
    const mode = modeFor(op, seen);
    try {
      const { prompt: rewritten } = await enhancePrompt({
        prompt,
        mode,
        model: effectiveEnhanceModel,
        image_hashes: await ensureHashes(seen),
        ollama_url: ollamaUrl,
      });
      if (rewritten && rewritten !== prompt) {
        return {
          sendPrompt: rewritten,
          displayEnhanced: enhancerMode === "verbose" ? rewritten : null,
        };
      }
    } catch {
      // Swallow — the typed prompt goes out unchanged below.
    }
    return { sendPrompt: prompt, displayEnhanced: null };
  }

  async function submitComposer() {
    const trimmed = composerText.trim();
    if (genMode) {
      if (!trimmed || enhancing) return; // a prompt is required to generate
      // Source images come from the message attachments, else the pinned panel.
      const attached = composerImages.length ? composerImages : pinnedImages;
      let op: GenOp;
      let imgs: string[];
      if (genOp === "animate") {
        // The one image becomes the video's first frame. `imagesFor` caps it at one:
        // Wan I2V has a single start frame, so a second would be dropped in silence.
        if (attached.length === 0) {
          setError("Animate needs a source image to bring to life (attach or pin one).");
          return;
        }
        op = "animate";
        imgs = imagesFor("animate", attached);
      } else if (genOp === "compose") {
        // Blend every available reference image (needs at least one).
        if (attached.length === 0) {
          setError("Combine needs at least one reference image (attach or pin some).");
          return;
        }
        op = "compose";
        imgs = attached;
      } else if (genOp === "edit") {
        // The first image is the one being edited; any others are references the
        // instruction can pull subjects from ("add the man from the second photo").
        if (attached.length === 0) {
          setError("Edit needs a source image to change (attach or pin one).");
          return;
        }
        op = "edit";
        imgs = attached;
      } else {
        // create: txt2img, or img2img from a single source image.
        op = "create";
        imgs = imagesFor("create", attached);
      }
      // Settings-level auto-enhance (if on) runs before the box is cleared
      // below. Verbose's rewrite rides along to `generateImage`, which attaches
      // it to the chat message it's about to create.
      setEnhancing(true);
      const { sendPrompt, displayEnhanced } = await maybeAutoEnhance(trimmed, op, imgs);
      setEnhancing(false);
      generateImage(trimmed, op, imgs, sendPrompt, displayEnhanced);
    } else {
      if (!trimmed && composerImages.length === 0) return;
      send(trimmed, composerImages);
    }
    setComposerText("");
    setComposerImages([]);
  }

  return (
    <div className="app">
      <Sidebar
        ollamaUrl={ollamaUrl}
        setOllamaUrl={setOllamaUrl}
        models={models}
        refreshModels={refreshModels}
        fluxModels={fluxModels}
        refreshFlux={refreshFlux}
        chats={chats}
        currentChatId={currentChatId}
        onNewChat={newChat}
        onOpenChat={openChat}
        onDeleteChat={removeChat}
        enhancerModel={enhancerModel}
        setEnhancerModel={setEnhancerModel}
        enhancerMode={enhancerMode}
        setEnhancerMode={setEnhancerMode}
        enhanceTemplate={enhanceTemplate}
        setEnhanceTemplate={setEnhanceTemplate}
      />
      <main className="main">
        <header className="topbar">
          <h1>👁️ Vision Model Chat</h1>
          <div className="topbar-actions">
            {usage && <ContextMeter used={usage.used} numCtx={usage.num_ctx} />}
            {fluxAvailable && (
              <GenModelPill op={genOp} picked={gen.fluxModel} models={fluxModels} gen={gen} />
            )}
            {model && <span className="model-pill">{model}</span>}
            {messages.length > 0 && (
              <button className="btn ghost" onClick={newChat}>
                ＋ New
              </button>
            )}
          </div>
        </header>
        {error && <div className="banner error">{error}</div>}
        <div className="workspace">
          <div className="work-row">
            <ImageBar
              images={pinnedImages}
              onAdd={addPinned}
              onRemove={removePinned}
              onRotate={rotatePinned}
              onLockChange={handleLockChange}
            />
            <Chat
              messages={messages}
              streaming={streaming}
              progress={progress}
              disabled={!model && !genMode}
              onDropFiles={addComposerFiles}
            />
          </div>
          <Composer
            text={composerText}
            setText={setComposerText}
            images={composerImages}
            onAddFiles={addComposerFiles}
            onRemoveImage={removeComposerImage}
            onRotateImage={rotateComposerImage}
            onSubmit={submitComposer}
            onStop={stop}
            streaming={streaming}
            disabled={!model}
            models={models}
            model={model}
            setModel={setModel}
            systemPrompt={systemPrompt}
            setSystemPrompt={setSystemPrompt}
            systemImage={systemImage}
            setSystemImage={setSystemImage}
            genMode={genMode}
            setGenMode={setGenMode}
            genOp={genOp}
            setGenOp={changeOp}
            fluxAvailable={fluxAvailable}
            fluxModels={fluxModels}
            gen={gen}
            setGen={setGen}
            presets={presets}
            onApplyPreset={applyPreset}
            onApplyDefault={applyDefault}
            onSavePreset={saveCurrentAsPreset}
            onDeletePreset={removePresetById}
            enhancing={enhancing}
            pinnedCount={pinnedImages.length}
            pinnedInit={pinnedImages[0] ?? null}
          />
        </div>
      </main>
    </div>
  );
}
