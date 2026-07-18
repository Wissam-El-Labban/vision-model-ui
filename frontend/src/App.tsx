import { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import Composer from "./components/Composer";
import ImageBar from "./components/ImageBar";
import ContextMeter from "./components/ContextMeter";
import {
  appendMessage,
  deleteChat,
  enhancePrompt,
  generate,
  generateTitle,
  getChat,
  getFluxModels,
  getModels,
  listChats,
  putChat,
  streamChat,
  uploadImages,
  urlToDataUrl,
  type FluxModel,
  type Usage,
} from "./api";
import { fileToDataUrl, resizeDataUrl, rotateDataUrl } from "./fileUtils";
import { guidanceFor, imagesFor, modeFor, resolveFlux, roleFor } from "./flux";
import { trimHistory } from "./context";
import type { ChatMessage, ChatSummary, GenSettings, GenOp } from "./types";

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

  // Image generation. `genMode` flips the composer from analyze to generate;
  // `fluxAvailable` reports whether the engine *and* a model are installed;
  // `gen` holds the tunable settings.
  const [genMode, setGenMode] = useState(false);
  // Which generate workflow: create (txt2img/img2img), edit (instruction), or
  // compose (blend multiple reference images).
  const [genOp, setGenOp] = useState<GenOp>("create");
  const [fluxAvailable, setFluxAvailable] = useState(false);
  // Prompt enhancement. The rewrite replaces the composer text in place, so what
  // the user reads is what gets sent — there is no second copy to drift from it.
  // `originalPrompt` backs Undo; `promptEnhanced` stops the static template being
  // wrapped around an already-rewritten prompt.
  const [enhancing, setEnhancing] = useState(false);
  const [originalPrompt, setOriginalPrompt] = useState<string | null>(null);
  const [promptEnhanced, setPromptEnhanced] = useState(false);
  const [gen, setGen] = useState<GenSettings>({
    fluxModel: "", // "" = let the backend pick this mode's default
    steps: 20,
    guidance: 3.5, // retuned to the installed model — see `guidanceFor`
    strength: 0.6,
    enhance: true,
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
          setGen((g) => ({ ...g, guidance: guidanceFor(genOp, r.models) }));
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
        return { ...g, fluxModel, guidance: guidanceFor(op, fluxModels, fluxModel) };
      });
    },
    [fluxModels]
  );

  const [pinnedImages, setPinnedImages] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
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
    setOriginalPrompt(null);
    setPromptEnhanced(false);
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
      setOriginalPrompt(null);
      setPromptEnhanced(false);
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
    async (prompt: string, op: GenOp, images: string[]) => {
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
      const shownImages = isCompose ? refUrls : initUrl ? [initUrl] : undefined;
      const icon = op === "animate" ? "🎬" : "🎨";

      // Show the prompt as a user turn, then an assistant placeholder we fill
      // with progress text and finally the generated image.
      const userMsg: ChatMessage = {
        role: "user",
        content: prompt,
        images: shownImages,
        model: modelId,
      };
      setMessages((prev) => [
        ...prev,
        userMsg,
        { role: "assistant", content: `${icon} Preparing…`, model: modelId },
      ]);
      setStreaming(true);

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
            prompt,
            init_image_hash: initHash,
            ref_image_hashes: refHashes,
            steps: gen.steps,
            guidance: gen.guidance,
            strength: gen.strength,
            // The static template is a fallback for an un-enhanced create prompt.
            // Wrapping it around a prompt the VLM already rewrote would bury the
            // rewrite's own framing inside a second, blunter one.
            enhance: gen.enhance && !promptEnhanced,
            width: gen.width,
            height: gen.height,
            seed: gen.seed ? parseInt(gen.seed, 10) : null,
            ollama_url: ollamaUrl,
          },
          {
            onStatus: (m) => setAssistant({ content: `${icon} ${m}` }),
            onProgress: (step, total) =>
              setAssistant({ content: `${icon} Generating… step ${step}/${total}` }),
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
      promptEnhanced,
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
  /** The Ollama model that will do the rewriting: the one in use if it can see,
   *  else any vision model. "" means none is installed — skip the call entirely. */
  const enhanceModel = models.vision.includes(model) ? model : models.vision[0] || "";

  async function enhanceComposerPrompt() {
    const trimmed = composerText.trim();
    if (!trimmed || enhancing) return;
    const attached = composerImages.length ? composerImages : pinnedImages;
    // The rewriter is briefed on the same mode, and shown the same images, that
    // the generate will actually run with — both derived by the shared helpers, so
    // an edit can't be briefed as a create.
    const seen = imagesFor(genOp, attached);
    const mode = modeFor(genOp, seen);
    setEnhancing(true);
    setError(null);
    try {
      const { prompt } = await enhancePrompt({
        prompt: trimmed,
        mode,
        model: enhanceModel,
        // No vision model → the backend skips straight to its template.
        image_hashes: enhanceModel ? await ensureHashes(seen) : [],
        ollama_url: ollamaUrl,
      });
      if (prompt && prompt !== trimmed) {
        // Only the first rewrite captures the undo point. Enhancing twice would
        // otherwise record the first rewrite as "the original", and Undo would
        // restore text the user never wrote — losing their prompt for good.
        setOriginalPrompt((prev) => (prev === null ? trimmed : prev));
        setComposerText(prompt);
        setPromptEnhanced(true);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setEnhancing(false);
    }
  }

  function undoEnhance() {
    if (originalPrompt === null) return;
    setComposerText(originalPrompt);
    setOriginalPrompt(null);
    setPromptEnhanced(false);
  }

  function submitComposer() {
    const trimmed = composerText.trim();
    if (genMode) {
      if (!trimmed) return; // a prompt is required to generate
      // Source images come from the message attachments, else the pinned panel.
      const attached = composerImages.length ? composerImages : pinnedImages;
      if (genOp === "animate") {
        // The one image becomes the video's first frame. `imagesFor` caps it at one:
        // Wan I2V has a single start frame, so a second would be dropped in silence.
        if (attached.length === 0) {
          setError("Animate needs a source image to bring to life (attach or pin one).");
          return;
        }
        generateImage(trimmed, "animate", imagesFor("animate", attached));
      } else if (genOp === "compose") {
        // Blend every available reference image (needs at least one).
        if (attached.length === 0) {
          setError("Combine needs at least one reference image (attach or pin some).");
          return;
        }
        generateImage(trimmed, "compose", attached);
      } else if (genOp === "edit") {
        // The first image is the one being edited; any others are references the
        // instruction can pull subjects from ("add the man from the second photo").
        if (attached.length === 0) {
          setError("Edit needs a source image to change (attach or pin one).");
          return;
        }
        generateImage(trimmed, "edit", attached);
      } else {
        // create: txt2img, or img2img from a single source image.
        generateImage(trimmed, "create", imagesFor("create", attached));
      }
    } else {
      if (!trimmed && composerImages.length === 0) return;
      send(trimmed, composerImages);
    }
    setComposerText("");
    setComposerImages([]);
    // The prompt is gone, so its rewrite history goes with it — the next prompt is
    // un-enhanced and the static template is armed again.
    setOriginalPrompt(null);
    setPromptEnhanced(false);
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
      />
      <main className="main">
        <header className="topbar">
          <h1>👁️ Vision Model Chat</h1>
          <div className="topbar-actions">
            {usage && <ContextMeter used={usage.used} numCtx={usage.num_ctx} />}
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
            onEnhance={enhanceComposerPrompt}
            onUndoEnhance={undoEnhance}
            enhancing={enhancing}
            canUndoEnhance={originalPrompt !== null}
            enhanceModel={enhanceModel}
            pinnedCount={pinnedImages.length}
            pinnedInit={pinnedImages[0] ?? null}
          />
        </div>
      </main>
    </div>
  );
}
