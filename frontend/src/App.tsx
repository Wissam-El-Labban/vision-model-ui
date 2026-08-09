import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import Composer from "./components/Composer";
import ImageBar from "./components/ImageBar";
import ContextMeter from "./components/ContextMeter";
import GenModelPill from "./components/GenModelPill";
// Lazily loaded: the studio pulls in three.js, which is bigger than the rest of
// the app put together. Nobody who isn't posing a figure should pay for it.
const PoseStudio = lazy(() => import("./components/PoseStudio"));
import {
  agentRun,
  appendMessage,
  deleteChat,
  enhancePrompt,
  generate,
  generateTitle,
  getAgentTools,
  getChat,
  getFluxModels,
  getModels,
  getPreprocessors,
  listChats,
  putChat,
  streamChat,
  uploadImages,
  urlToDataUrl,
  type FluxModel,
  type FluxPreprocessor,
  type Usage,
} from "./api";
import { fileToDataUrl, resizeDataUrl, rotateDataUrl } from "./fileUtils";
import { guidanceFor, imagesFor, modeFor, resolveFlux, roleFor, stepsFor } from "./flux";
import { buildOllamaContext } from "./chatContext";
import type {
  AgentStep,
  AgentTool,
  AgentToolId,
  ChatMessage,
  ChatSummary,
  ComposerMode,
  ControlMap,
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
  const [models, setModels] = useState<{
    vision: string[];
    all: string[];
    agent: string[];
  }>({ vision: [], all: [], agent: [] });
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

  // Image generation. `composerMode` picks analyze / generate / agent;
  // `fluxAvailable` reports whether the engine *and* a model are installed;
  // `gen` holds the tunable settings.
  const [composerMode, setComposerMode] = useState<ComposerMode>(
    () => (localStorage.getItem("composerMode") as ComposerMode) || "analyze"
  );
  const genMode = composerMode === "generate";
  const agentMode = composerMode === "agent";

  // Agent mode. `agentModel` is its own pick rather than the chat model's,
  // because the two lists differ: agent mode needs tools + vision + 7B, and the
  // model someone chats with is often smaller than that. `agentTools` is the
  // Tools menu; its catalog comes from the backend so the menu can't offer
  // something the backend won't run.
  const [agentModel, setAgentModel] = useState(
    () => localStorage.getItem("agentModel") || ""
  );
  const [agentToolCatalog, setAgentToolCatalog] = useState<AgentTool[]>([]);
  const [agentTools, setAgentTools] = useState<AgentToolId[]>(() => {
    const saved = localStorage.getItem("agentTools");
    if (!saved) return [];  // replaced by the backend's defaults once they load
    try {
      return JSON.parse(saved) as AgentToolId[];
    } catch {
      return [];
    }
  });
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
    // control. Depth alone by default: it's the map that carries the scene, so it's
    // the one that answers the pose the user couldn't get with words. The lock
    // starts off — it changes the output a lot, and it should be something the user
    // reaches for once the maps alone haven't landed the pose.
    controlKinds: ["depth"],
    structureLock: 1,
    studioSource: false,
    controlStrength: 1,
    cannyLow: 0.3,
    cannyHigh: 0.4,
  });
  // Installed image models. Refreshed after an install/removal so the composer's
  // picker stays in sync with the sidebar's Image Models panel.
  const [fluxModels, setFluxModels] = useState<FluxModel[]>([]);
  // Which control maps can be built. Refreshed alongside the model list, since
  // installing one is done in the same sidebar panel.
  const [preprocessors, setPreprocessors] = useState<FluxPreprocessor[]>([]);
  // Maps posed in the Pose Studio, held apart from the composer's attachments.
  // They're already control maps, not images to derive one from, and mixing the
  // two lists would make "is this the source or the map?" ambiguous per image.
  const [studioMaps, setStudioMaps] = useState<ControlMap[]>([]);
  // What those maps contain. Held separately because it isn't in the pixels the
  // enhancer is allowed to look at — a control map is the one image its brief
  // forbids describing, so the figure count has to arrive as a number.
  const [studioMeta, setStudioMeta] = useState<{ subjects: number; contact: boolean } | null>(
    null
  );
  const [studioOpen, setStudioOpen] = useState(false);
  const guidanceReady = useRef(false);
  const refreshFlux = useCallback(() => {
    getPreprocessors()
      .then(setPreprocessors)
      .catch(() => {
        /* backend predates control preprocessors — the Control tab offers canny only */
      });
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
  useEffect(() => localStorage.setItem("composerMode", composerMode), [composerMode]);
  // The mode is remembered across reloads, so it can outlive the thing that made
  // it reachable: uninstall the image models, or the only tool-capable model, and
  // the tab that would take you out of the mode is hidden or disabled. Fall back
  // rather than stranding the composer in a mode it can't submit from.
  useEffect(() => {
    if (composerMode === "analyze") return;
    if (!fluxAvailable) setComposerMode("analyze");
    else if (composerMode === "agent" && models.agent.length === 0) {
      setComposerMode("analyze");
    }
  }, [composerMode, fluxAvailable, models.agent]);
  useEffect(() => localStorage.setItem("agentModel", agentModel), [agentModel]);
  useEffect(
    () => localStorage.setItem("agentTools", JSON.stringify(agentTools)),
    [agentTools]
  );

  // The agent's tool catalog, once. Static for the life of the backend, and the
  // defaults only apply to someone who has never opened the Tools menu — an
  // empty saved selection is a real choice (no tools = the agent can only talk).
  useEffect(() => {
    let cancelled = false;
    getAgentTools()
      .then(({ tools, defaults }) => {
        if (cancelled) return;
        setAgentToolCatalog(tools);
        if (localStorage.getItem("agentTools") === null) setAgentTools(defaults);
      })
      .catch(() => {
        /* agent mode stays unavailable; the tab explains why */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      const m = await getModels(ollamaUrl);
      setModels(m);
      setModel((cur) =>
        cur && m.vision.includes(cur) ? cur : m.vision[0] ?? cur
      );
      setAgentModel((cur) =>
        cur && m.agent.includes(cur) ? cur : m.agent[0] ?? ""
      );
      setError(null);
    } catch {
      setError(`Could not reach Ollama at ${ollamaUrl}`);
      setModels({ vision: [], all: [], agent: [] });
    }
  }, [ollamaUrl]);

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  useEffect(() => {
    refreshChats();
  }, [refreshChats]);

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

      // Every image the conversation has seen, consolidated onto the last message
      // and numbered — see `buildOllamaContext` for why both are necessary.
      const { merged, outImages } = await buildOllamaContext({
        history,
        pinnedImages,
        usage,
        forOllama,
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
      // control reads the first image as the structure source and the rest as
      // subject references — the same split edit uses, for the same reason: one
      // image says *how it is arranged*, the others say *what is in it*.
      //
      // With no control type selected, that first image is not a source to derive a
      // map from: it *is* the map. It goes to `control_map_hashes` instead, which is
      // both the re-roll path (pin the map the last run emitted) and how a skeleton
      // posed in Blender or PoseMy.Art gets in without being re-analysed.
      //
      // Studio maps were authored as maps too, so nothing is derived from them
      // either. What they leave open is what an *attachment* then means, and the
      // two answers produce completely different images: a subject reference lends
      // a face and clothing, while a scene is the photograph the figures are posed
      // into and survives the generation. `studioSource` is that choice.
      const isCompose = op === "compose";
      const isControl = op === "control";
      const usingStudio = isControl && studioMaps.length > 0;
      const studioScene = usingStudio && gen.studioSource && images.length > 0;
      const controlAsMap = isControl && !usingStudio && gen.controlKinds.length === 0;
      const initUrl =
        isCompose || controlAsMap || (usingStudio && !studioScene) ? null : images[0] ?? null;
      const mapUrls = usingStudio
        ? studioMaps.map((m) => m.url)
        : controlAsMap && images.length
          ? [images[0]]
          : [];
      const refUrls = isCompose
        ? images
        : usingStudio
          ? // With no scene, nothing is the source and every attachment describes
            // who is in the pose.
            studioScene
            ? images.slice(1)
            : images
          : op === "edit" || isControl
            ? images.slice(1)
            : [];
      // Shared with the prompt enhancer, so both brief the model on the same job.
      const mode = modeFor(op, images);
      // Every image the job conditions on, in the order the backend receives them.
      // edit has two kinds — the scene in `initUrl` and the subject references after
      // it — and showing only the first made the references invisible in the turn
      // that used them. compose has no init, so the spread covers it too.
      const conditioning = [...(initUrl ? [initUrl] : []), ...mapUrls, ...refUrls];
      const shownImages = conditioning.length ? conditioning : undefined;
      const icon = op === "animate" ? "🎬" : op === "control" ? "🕹️" : "🎨";

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
      // Control maps arrive one event at a time, before the image. Accumulated here
      // rather than read back off the message, because `setAssistant` is a state
      // update and the next map can land before it has applied.
      let controlMaps: ControlMap[] = [];
      try {
        const initHash = initUrl ? (await ensureHashes([initUrl]))[0] : null;
        const refHashes = refUrls.length ? await ensureHashes(refUrls) : [];
        const mapHashes = mapUrls.length ? await ensureHashes(mapUrls) : [];

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
            // Only sent for control, so no other mode's request changes shape. The
            // backend defaults every one of these, so omitting them is the same as
            // not knowing about them.
            ...(isControl
              ? {
                  // Studio maps are finished maps. With no scene photo there is
                  // no source to derive more from, and asking would just error;
                  // with one, derived maps stack with the studio's on the
                  // backend, which is how a photo's own depth joins the pose.
                  control_kinds: usingStudio && !studioScene ? [] : gen.controlKinds,
                  control_map_hashes: mapHashes,
                  structure_lock: gen.structureLock,
                  control_strength: gen.controlStrength,
                  canny_low: gen.cannyLow,
                  canny_high: gen.cannyHigh,
                }
              : {}),
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
            onControlMap: async (m) => {
              // Fetched back as a data-URL like a generated image is, so a map can
              // be pinned and re-fed on the next roll without a round trip — that's
              // how you iterate on a prompt while holding one pose fixed.
              let url = m.url;
              try {
                url = await urlToDataUrl(m.url);
                hashCache.current.set(url, m.hash);
              } catch {
                /* fall back to the URL */
              }
              controlMaps = [...controlMaps, { kind: m.kind, url }];
              setAssistant({ controlMaps });
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

  /** One agent turn: the model decides, then whatever it decided on runs.
   *
   * The request half is `send`'s — the same conversation, the same consolidated
   * and numbered images — because the agent has to see what an analyze turn sees
   * before it can say which image it means. The response half is
   * `generateImage`'s: progress, results fetched back as data-URLs, and the turns
   * recorded by the backend thread rather than here, so a reload mid-generation
   * still finds the images.
   *
   * Unlike `generateImage`, status text does not go into the message body: the
   * body is the agent's own words, and overwriting them with "Loading the
   * model…" would throw away the only explanation of what it decided. The
   * progress bar and the step chips carry the state instead.
   */
  const runAgent = useCallback(
    async (text: string, images: string[]) => {
      if (!agentModel) {
        setError("Agent mode needs a tool-capable model. Install one first.");
        return;
      }
      setError(null);
      const chatId = currentChatId;
      const isFirstExchange = messages.length === 0;

      const userMsg: ChatMessage = { role: "user", content: text, images, model: agentModel };
      const history = [...messages, userMsg];
      setMessages([...history, { role: "assistant", content: "", model: agentModel }]);
      setStreaming(true);
      setProgress({
        phase: `Loading ${agentModel}…`,
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

      // `alwaysNumber` because the tool arguments are image numbers: with one
      // image analyze has nothing to disambiguate, but the agent still has to be
      // able to say "image 1".
      const { merged, outImages } = await buildOllamaContext({
        history,
        pinnedImages,
        usage,
        forOllama,
        alwaysNumber: true,
      });
      setAssistant({ contextImages: outImages });

      const payload: ChatMessage[] = [];
      if (systemPrompt.trim()) {
        payload.push({
          role: "system",
          content: systemPrompt.trim(),
          images: systemImage ? await forOllama([systemImage]) : undefined,
        });
      }
      payload.push(...merged);

      // Keyed by the backend's step index, not by arrival order: a call that
      // fails validation is reported before any call runs, so the two orders
      // differ and a chip would otherwise attach its progress to the wrong step.
      const steps = new Map<number, AgentStep>();
      const flush = () =>
        setAssistant({
          agentSteps: [...steps.entries()]
            .sort(([a], [b]) => a - b)
            .map(([, step]) => step),
        });
      const mark = (index: number, patch: Partial<AgentStep>) => {
        const current = steps.get(index);
        if (current) steps.set(index, { ...current, ...patch });
        flush();
      };

      let assistantText = "";
      let produced = 0;
      const resultImages: string[] = [];
      const resultVideos: string[] = [];
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const context_hashes = await ensureHashes(outImages);
        const attachment_hashes = await ensureHashes(images);

        // Before the run, for the same reason generateImage does it: the backend
        // writes both turns from a thread that outlives this page, and it needs a
        // chat row to write them against.
        try {
          await putChat(chatId, {
            model: agentModel,
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

        await agentRun(
          {
            messages: payload,
            prompt: text,
            context_hashes,
            attachment_hashes,
            model: agentModel,
            enabled_tools: agentTools,
            flux_model: gen.fluxModel || null,
            steps: gen.steps,
            guidance: gen.guidance,
            width: gen.width,
            height: gen.height,
            seed: gen.seed ? parseInt(gen.seed, 10) : null,
            ollama_url: ollamaUrl,
            chat_id: chatId,
          },
          {
            onToken: (token) => {
              assistantText += token;
              setAssistant({ content: assistantText });
            },
            onUsage: (u) => setUsage(u),
            onStatus: (m) =>
              setProgress((p) => (p ? { ...p, phase: m, updatedAt: Date.now() } : p)),
            onProgress: ({ live, ...p }) =>
              setProgress((cur) =>
                cur
                  ? { ...cur, ...p, updatedAt: live ? Date.now() : cur.updatedAt }
                  : cur
              ),
            onToolCall: ({ index, name, args }) => {
              steps.set(index, { name, args, state: "running" });
              flush();
            },
            onToolError: ({ index, name, message }) => {
              steps.set(index, {
                ...(steps.get(index) ?? { name, args: {} }),
                state: "error",
                message,
              });
              flush();
            },
            onImage: async (r, toolIndex) => {
              produced += 1;
              const url = r.url || `/api/images/${r.hash}.png`;
              if (r.kind === "video") {
                // Left as a URL, never a data-URL — see the same branch in
                // `generateImage` for why video must not reach `images`.
                hashCache.current.set(url, r.hash);
                resultVideos.push(url);
                setAssistant({ videos: [...resultVideos] });
              } else {
                let dataUrl: string | null = null;
                try {
                  dataUrl = await urlToDataUrl(url);
                  hashCache.current.set(dataUrl, r.hash);
                } catch {
                  /* fall back to the URL */
                }
                resultImages.push(dataUrl ?? url);
                setAssistant({ images: [...resultImages] });
              }
              if (toolIndex !== undefined) mark(toolIndex, { state: "done" });
            },
            onError: (msg) => {
              assistantText = `${assistantText}${assistantText ? "\n\n" : ""}⚠️ ${msg}`;
              setAssistant({ content: assistantText });
            },
          },
          controller.signal
        );
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          assistantText = `${assistantText}${assistantText ? "\n\n" : ""}⚠️ ${
            (e as Error).message
          }`;
          setAssistant({ content: assistantText });
        }
      } finally {
        setStreaming(false);
        setProgress(null);
        abortRef.current = null;

        // The backend recorded both turns (see `chat_id` above), so all that's
        // left is what needs this page: the title and the sidebar. Titling runs
        // on the chat model — a turn that generated nothing still deserves one,
        // since the agent answering in words is a normal outcome.
        try {
          if (isFirstExchange && (produced > 0 || assistantText.trim())) {
            await generateTitle(chatId, model || agentModel, ollamaUrl).catch(() => "");
          }
          await refreshChats();
        } catch (err) {
          console.error("post-agent refresh failed", err);
        }
      }
    },
    [
      agentModel,
      agentTools,
      gen,
      model,
      ollamaUrl,
      systemPrompt,
      systemImage,
      pinnedImages,
      messages,
      usage,
      currentChatId,
      ensureHashes,
      forOllama,
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
        // Only meaningful for a studio pose; the backend ignores it elsewhere.
        ...(op === "control" && studioMaps.length > 0 && studioMeta ? studioMeta : {}),
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
    if (agentMode) {
      // No per-op validation and no auto-enhance: deciding which workflow the
      // request needs, and writing the prompt for it, is the agent's whole job.
      // Attachments are plain attachments — the conversation's images arrive
      // through the same path analyze uses.
      if (!trimmed) return;
      runAgent(trimmed, composerImages);
    } else if (genMode) {
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
      } else if (genOp === "control") {
        // Three ways in: a pose built in the studio, an image to derive maps from,
        // or a finished map attached directly. Only the middle one has a "source".
        if (studioMaps.length === 0) {
          if (attached.length === 0) {
            setError(
              "Control needs a pose. Open the Pose Studio to build one, or attach an image whose pose you want copied."
            );
            return;
          }
          if (gen.controlKinds.length === 0 && gen.structureLock < 1) {
            setError(
              "With no control type selected the attached image is used as a finished control map — there's no source image left to lock onto. Pick a control type, or set the structure lock back to 1."
            );
            return;
          }
        } else if (gen.structureLock < 1 && !(gen.studioSource && attached.length > 0)) {
          setError(
            "A studio pose on its own has no source image to lock onto — the maps are the whole signal. Attach the scene photo and tick “use it as the scene”, or set the structure lock back to 1."
          );
          return;
        }
        op = "control";
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
            {fluxAvailable && !agentMode && (
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
              disabled={composerMode === "analyze" && !model}
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
            agentModel={agentModel}
            setAgentModel={setAgentModel}
            agentToolCatalog={agentToolCatalog}
            agentTools={agentTools}
            setAgentTools={setAgentTools}
            systemPrompt={systemPrompt}
            setSystemPrompt={setSystemPrompt}
            systemImage={systemImage}
            setSystemImage={setSystemImage}
            composerMode={composerMode}
            setComposerMode={setComposerMode}
            genOp={genOp}
            setGenOp={changeOp}
            fluxAvailable={fluxAvailable}
            fluxModels={fluxModels}
            gen={gen}
            setGen={setGen}
            enhancing={enhancing}
            pinnedCount={pinnedImages.length}
            pinnedInit={pinnedImages[0] ?? null}
            preprocessors={preprocessors}
            studioMaps={studioMaps}
            studioMeta={studioMeta}
            onOpenStudio={() => setStudioOpen(true)}
            onClearStudioMaps={() => {
              setStudioMaps([]);
              setStudioMeta(null);
            }}
          />
        </div>
      </main>
      {studioOpen && (
        <Suspense
          fallback={<div className="studio-backdrop"><div className="studio-loading">Loading the Pose Studio…</div></div>}
        >
          <PoseStudio
            onClose={() => setStudioOpen(false)}
            sceneImage={composerImages[0] ?? pinnedImages[0] ?? null}
            onUse={(maps, meta) => {
              setStudioMaps(maps);
              setStudioMeta(meta);
              // Posing is only meaningful in the Control tab, and arriving there
              // is what the user was doing — don't make them find the tab too.
              setComposerMode("generate");
              if (genOp !== "control") changeOp("control");
            }}
          />
        </Suspense>
      )}
    </div>
  );
}
