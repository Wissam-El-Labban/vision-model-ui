import { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import Composer from "./components/Composer";
import ImageBar from "./components/ImageBar";
import ContextMeter from "./components/ContextMeter";
import {
  appendMessage,
  deleteChat,
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
import { fileToResizedDataUrl, resizeDataUrl, rotateDataUrl } from "./fileUtils";
import { trimHistory } from "./context";
import type { ChatMessage, ChatSummary, GenSettings, GenOp } from "./types";

const DEFAULT_URL = "http://localhost:11434";

/** Extract the sha256 hash from an image/thumb URL like /api/images/<hash>.jpg */
function hashFromUrl(url: string): string {
  return (url.split("/").pop() || "").replace(/\.jpg$/, "");
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

  // Image generation (FLUX). `genMode` flips the composer from analyze to
  // generate; `fluxAvailable` reports whether the sidecar + weights are present;
  // `gen` holds the tunable settings.
  const [genMode, setGenMode] = useState(false);
  // Which generate workflow: create (txt2img/img2img), edit (instruction), or
  // compose (blend multiple reference images).
  const [genOp, setGenOp] = useState<GenOp>("create");
  const [fluxAvailable, setFluxAvailable] = useState(false);
  const [gen, setGen] = useState<GenSettings>({
    fluxModel: "", // "" = the current mode's default FLUX model
    steps: 20,
    guidance: 3.5, // create (FLUX dev); edit/compose retune to 2.5
    strength: 0.6,
    enhance: true,
    width: 1024, // FLUX is trained at ~1 megapixel
    height: 1024,
    seed: "",
  });
  // Installed FLUX UNets (defaults + any the user added). Refreshed after a
  // download/removal so the composer's model picker stays in sync.
  const [fluxModels, setFluxModels] = useState<FluxModel[]>([]);
  const refreshFlux = useCallback(() => {
    getFluxModels()
      .then((r) => {
        setFluxAvailable(r.available);
        setFluxModels(r.models);
      })
      .catch(() => {
        /* backend older / weights missing — generation stays hidden */
      });
  }, []);

  // Switching workflow retunes guidance, which is mode-scaled: create runs FLUX
  // dev (~3.5 to bind a text-only prompt), edit/compose run Kontext (~2.5 to
  // follow an instruction). It also clears `fluxModel`, since the two modes draw
  // from disjoint sets of UNets.
  const changeOp = useCallback((op: GenOp) => {
    setGenOp(op);
    setGen((g) => ({ ...g, fluxModel: "", steps: 20, guidance: op === "create" ? 3.5 : 2.5 }));
  }, []);

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

  const refreshChats = useCallback(async () => {
    try {
      setChats(await listChats());
    } catch {
      /* leave the list as-is if the fetch fails */
    }
  }, []);

  async function addPinned(files: FileList | File[]) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    const urls = await Promise.all(list.map((f) => fileToResizedDataUrl(f)));
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

      // Dropping missing pinned images here also self-heals the DB: the next
      // send re-persists the pinned set without them.
      const pinned = present(await Promise.all(d.pinned.map(loadImg)));
      const sysImg = d.system_image ? await loadImg(d.system_image) : null;
      const msgs: ChatMessage[] = await Promise.all(
        d.messages.map(async (m) => {
          const images = present(await Promise.all(m.images.map(loadImg)));
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
        return { role: m.role, content: note + m.content, images: outImages };
      });

      // Build the request: optional system message (with persistent image) + history.
      const payload: ChatMessage[] = [];
      if (systemPrompt.trim()) {
        payload.push({
          role: "system",
          content: systemPrompt.trim(),
          images: systemImage ? [systemImage] : undefined,
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
      refreshChats,
    ]
  );

  const generateImage = useCallback(
    async (prompt: string, op: GenOp, images: string[]) => {
      // Every mode runs on FLUX: create on FLUX dev, edit/compose on Kontext.
      const isKontext = op === "edit" || op === "compose";
      const modelId = isKontext ? "FLUX Kontext" : "FLUX dev";
      if (!fluxAvailable) {
        setError("FLUX isn't installed on this machine. Run ./run.sh to fetch the weights.");
        return;
      }
      setError(null);
      const chatId = currentChatId;
      const isFirstExchange = messages.length === 0;

      // compose blends every reference image; create/edit use one source image
      // and (for create) infer txt2img vs img2img from its presence.
      const isCompose = op === "compose";
      const initUrl = isCompose ? null : images[0] ?? null;
      const refUrls = isCompose ? images : [];
      const mode =
        op === "edit"
          ? "edit"
          : isCompose
            ? "compose"
            : initUrl
              ? "img2img"
              : "txt2img";
      const shownImages = isCompose ? refUrls : initUrl ? [initUrl] : undefined;

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
        { role: "assistant", content: "🎨 Preparing…", model: modelId },
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
      try {
        const initHash = initUrl ? (await ensureHashes([initUrl]))[0] : null;
        const refHashes = refUrls.length ? await ensureHashes(refUrls) : [];
        await generate(
          {
            mode,
            flux_model: gen.fluxModel || null,
            prompt,
            init_image_hash: initHash,
            ref_image_hashes: refHashes,
            steps: gen.steps,
            guidance: gen.guidance,
            strength: gen.strength,
            enhance: gen.enhance,
            width: gen.width,
            height: gen.height,
            seed: gen.seed ? parseInt(gen.seed, 10) : null,
            ollama_url: ollamaUrl,
          },
          {
            onStatus: (m) => setAssistant({ content: `🎨 ${m}` }),
            onProgress: (step, total) =>
              setAssistant({ content: `🎨 Generating… step ${step}/${total}` }),
            onImage: async (r) => {
              resultHash = r.hash;
              // Load the stored image back as a data-URL for display + pin/reuse
              // parity, and seed the hash cache so it isn't re-uploaded.
              try {
                resultDataUrl = await urlToDataUrl(`/api/images/${r.hash}.jpg`);
                hashCache.current.set(resultDataUrl, r.hash);
              } catch {
                /* fall back to the URL below */
              }
              setAssistant({
                content: "",
                images: [resultDataUrl ?? `/api/images/${r.hash}.jpg`],
              });
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

        // Persist (best-effort). Only if we actually produced an image.
        if (resultHash) {
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
            await appendMessage(chatId, {
              role: "user",
              content: prompt,
              model: modelId,
              image_hashes: isCompose
                ? await ensureHashes(refUrls)
                : initUrl
                  ? await ensureHashes([initUrl])
                  : [],
            });
            await appendMessage(chatId, {
              role: "assistant",
              content: "",
              model: modelId,
              image_hashes: [resultHash],
            });
            if (isFirstExchange && model) {
              await generateTitle(chatId, model, ollamaUrl).catch(() => "");
            }
            await refreshChats();
          } catch (err) {
            console.error("persist failed", err);
          }
        }
      }
    },
    [
      gen,
      fluxAvailable,
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
    const urls = await Promise.all(list.map((f) => fileToResizedDataUrl(f)));
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
  function submitComposer() {
    const trimmed = composerText.trim();
    if (genMode) {
      if (!trimmed) return; // a prompt is required to generate
      // Source images come from the message attachments, else the pinned panel.
      const attached = composerImages.length ? composerImages : pinnedImages;
      if (genOp === "compose") {
        // Blend every available reference image (needs at least one).
        if (attached.length === 0) {
          setError("Combine needs at least one reference image (attach or pin some).");
          return;
        }
        generateImage(trimmed, "compose", attached);
      } else if (genOp === "edit") {
        // Instruction edit works on one source image.
        if (attached.length === 0) {
          setError("Edit needs a source image to change (attach or pin one).");
          return;
        }
        generateImage(trimmed, "edit", attached.slice(0, 1));
      } else {
        // create: txt2img, or img2img from a single source image.
        generateImage(trimmed, "create", attached.slice(0, 1));
      }
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
            pinnedCount={pinnedImages.length}
            pinnedInit={pinnedImages[0] ?? null}
            onFluxModelsChanged={refreshFlux}
          />
        </div>
      </main>
    </div>
  );
}
