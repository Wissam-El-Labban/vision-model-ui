/** Pose Studio — build a pose in 3-D, hand its control maps to a generation.
 *
 * The reason this exists rather than a 2-D stick-figure editor: a flat skeleton
 * cannot say which arm is in front, how far a limb is foreshortened, or that a
 * foot is bearing weight on the seat of a chair instead of floating at the same
 * screen position. Those are precisely the things that break on a hard pose. A
 * posed 3-D figure answers all of them, because the depth map rendered from it
 * carries the answer — and the OpenPose skeleton comes out of the same frame, so
 * the two maps can't disagree about where the body is.
 *
 * It holds several figures for the same reason it holds props: two people in
 * contact is a spatial relationship, and a relationship is exactly what a prompt
 * cannot pin down. "Two dancers, one dipping the other" is a sentence the model
 * will interpret; two mannequins actually touching is a measurement.
 *
 * Interaction model, in one line each:
 *   drag a joint handle  — aim it; the joint above it holds still
 *   drag the hips        — slide that figure across the floor
 *   drag the background  — orbit the camera
 *   wheel                — dolly
 *   props                — blocks you place so the figure has something to rest on
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import {
  HANDLES,
  HANDLE_AIM,
  HANDLE_LABELS,
  IK_CHAINS,
  POLE_HINTS,
  normalizePoseFile,
  type FigurePose,
  type PoseFile,
  type PoseFileV2,
  type PropDef,
} from "../pose/skeleton";
import {
  applyPose,
  buildProp,
  buildRig,
  disposeRig,
  figuresInContact,
  readPose,
  resetRig,
  type Rig,
} from "../pose/mannequin";
import { aimBone, dragPlanePoint, solveTwoBoneIK } from "../pose/ik";
import { makeDepthMaterial, renderDepth, renderPose } from "../pose/render";
import { GROUND_REACH, PRESETS, presetRotations } from "../pose/presets";
import { fileToDataUrl } from "../fileUtils";
import type { ControlKind } from "../types";

interface Frame {
  label: string;
  w: number;
  h: number;
}

const ASPECTS: Frame[] = [
  { label: "Portrait", w: 832, h: 1216 },
  { label: "Square", w: 1024, h: 1024 },
  { label: "Landscape", w: 1216, h: 832 },
];

const STORE_KEY = "poseStudio.saved";
/** How far apart a newly added figure stands from the last one. */
const FIGURE_GAP = 0.7;
const HISTORY_LIMIT = 50;
/** How long two edits of the same control collapse into one undo step. */
const COALESCE_MS = 700;

/** A frame matching a loaded photo, at FLUX's ~1 megapixel and snapped to /64.
 *  Aspect is the part that matters: with the structure lock down, the backend
 *  takes its resolution from the source photo, so maps rendered at a different
 *  aspect land as misaligned reference latents and the pose quietly softens. */
function frameForImage(w: number, h: number): Frame {
  const scale = Math.sqrt((1024 * 1024) / Math.max(w * h, 1));
  const snap = (v: number) => Math.max(64, Math.round((v * scale) / 64) * 64);
  return { label: "Match photo", w: snap(w), h: snap(h) };
}

/** One posable figure: everything that moves together, and the id the handles
 *  and the undo stack refer to it by. */
interface Figure {
  id: string;
  rig: Rig;
}

interface SceneCtx {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  figures: Figure[];
  /** One group for every figure's handles. Deliberately shared: `helpers` is
   *  built once at mount and is what a map render hides, so a per-figure group
   *  added later would not be in it and its handles would render into the depth
   *  map as seventeen floating spheres. */
  handleGroup: THREE.Group;
  handleGeo: THREE.SphereGeometry;
  handleMat: THREE.MeshBasicMaterial;
  handleDimMat: THREE.MeshBasicMaterial;
  propGroup: THREE.Group;
  helpers: THREE.Object3D[];
  depthMaterial: THREE.ShaderMaterial;
  orbit: { theta: number; phi: number; radius: number; target: THREE.Vector3 };
}

/** A whole studio state, small enough to keep fifty of. */
interface StudioSnapshot {
  figures: (FigurePose & { id: string })[];
  props: PropDef[];
  active: string;
  selectedProp: string | null;
}

let figureSeq = 0;
const nextFigureId = () => `f${(figureSeq++).toString(36)}${Date.now().toString(36)}`;

function addFigureToScene(ctx: SceneCtx, id: string, x = 0): Figure {
  const rig = buildRig();
  rig.root.position.x = x;
  ctx.scene.add(rig.root);
  const figure = { id, rig };
  ctx.figures.push(figure);
  for (const bone of HANDLES) {
    const mesh = new THREE.Mesh(ctx.handleGeo, ctx.handleMat);
    mesh.renderOrder = 10;
    mesh.userData.bone = bone;
    mesh.userData.figureId = id;
    ctx.handleGroup.add(mesh);
  }
  return figure;
}

function destroyFigure(ctx: SceneCtx, id: string): void {
  const i = ctx.figures.findIndex((f) => f.id === id);
  if (i < 0) return;
  disposeRig(ctx.figures[i].rig);
  ctx.figures.splice(i, 1);
  for (const mesh of [...ctx.handleGroup.children]) {
    if (mesh.userData.figureId === id) ctx.handleGroup.remove(mesh);
  }
}

interface Props {
  onClose: () => void;
  /** Hands the rendered maps to the composer, with what the scene contains —
   *  the prompt enhancer can't count figures it never sees. */
  onUse: (
    maps: { kind: ControlKind; url: string }[],
    meta: { subjects: number; contact: boolean }
  ) => void;
  /** The composer's first attachment, offered as the scene to pose against. */
  sceneImage?: string | null;
}

export default function PoseStudio({ onClose, onUse, sceneImage }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<SceneCtx | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [figureIds, setFigureIds] = useState<string[]>([]);
  const [active, setActive] = useState("");
  const [rootY, setRootY] = useState(0);
  const [aspect, setAspect] = useState(0);
  const [emit, setEmit] = useState<Record<ControlKind, boolean>>({
    depth: true,
    pose: true,
    canny: false,
  });
  const [props, setProps] = useState<PropDef[]>([]);
  const [selectedProp, setSelectedProp] = useState<string | null>(null);
  const [saved, setSaved] = useState<PoseFile[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
    } catch {
      return [];
    }
  });
  const [saveName, setSaveName] = useState("");
  const [preview, setPreview] = useState<{ kind: ControlKind; url: string }[]>([]);
  const [viewSize, setViewSize] = useState({ w: 1, h: 1 });
  const [backdrop, setBackdrop] = useState<string | null>(sceneImage ?? null);
  const [backdropFrame, setBackdropFrame] = useState<Frame | null>(null);
  const [backdropOpacity, setBackdropOpacity] = useState(0.55);
  const [historyDepth, setHistoryDepth] = useState(0);

  const aspects = useMemo(
    () => (backdropFrame ? [...ASPECTS, backdropFrame] : ASPECTS),
    [backdropFrame]
  );
  const frame = aspects[Math.min(aspect, aspects.length - 1)];

  // Props live in React state (so the panel can edit them) and in the scene (so
  // they render). These refs are what keep a drag, and a history snapshot taken
  // from an event handler, from having to round-trip through a re-render.
  const propsRef = useRef<PropDef[]>(props);
  propsRef.current = props;
  const selectedPropRef = useRef<string | null>(selectedProp);
  selectedPropRef.current = selectedProp;
  const activeRef = useRef(active);
  activeRef.current = active;
  // The animation loop and the resize handler are installed once and must read
  // the *current* output aspect, which is React state.
  const outAspectRef = useRef(frame.w / frame.h);
  outAspectRef.current = frame.w / frame.h;
  const resizeRef = useRef<() => void>(() => {});

  // ------------------------------------------------------------------ history
  const historyRef = useRef<{ snap: StudioSnapshot; tag?: string; at: number }[]>([]);
  const pendingRef = useRef<StudioSnapshot | null>(null);

  const captureSnapshot = useCallback((): StudioSnapshot | null => {
    const ctx = sceneRef.current;
    if (!ctx) return null;
    return {
      figures: ctx.figures.map((f) => ({
        id: f.id,
        rotations: readPose(f.rig),
        rootPosition: f.rig.root.position.toArray() as [number, number, number],
        rootRotationY: f.rig.root.rotation.y,
      })),
      props: propsRef.current.map((p) => ({ ...p })),
      active: activeRef.current,
      selectedProp: selectedPropRef.current,
    };
  }, []);

  const pushSnapshot = useCallback((snap: StudioSnapshot | null, tag?: string) => {
    if (!snap) return;
    const stack = historyRef.current;
    const top = stack[stack.length - 1];
    // A slider fires per pixel. Collapsing consecutive edits of the *same*
    // control into the entry that predates the whole gesture is what makes one
    // Ctrl+Z undo a drag rather than one pixel of it — and unlike a pointer-down
    // hook it also covers holding an arrow key. Discrete actions pass no tag, so
    // two Turn clicks stay two undo steps.
    if (tag && top && top.tag === tag && Date.now() - top.at < COALESCE_MS) {
      top.at = Date.now();
      return;
    }
    stack.push({ snap, tag, at: Date.now() });
    if (stack.length > HISTORY_LIMIT) stack.shift();
    setHistoryDepth(stack.length);
  }, []);

  const pushHistory = useCallback(
    (tag?: string) => pushSnapshot(captureSnapshot(), tag),
    [captureSnapshot, pushSnapshot]
  );

  // A pointer-down on a handle is just as often a click to read its name, and a
  // snapshot pushed for that would make the next Ctrl+Z do nothing visible. So
  // take the snapshot on the way down but only commit it once something moves.
  const armHistory = useCallback(() => {
    pendingRef.current = captureSnapshot();
  }, [captureSnapshot]);

  const commitHistory = useCallback(() => {
    if (!pendingRef.current) return;
    pushSnapshot(pendingRef.current);
    pendingRef.current = null;
  }, [pushSnapshot]);

  const syncRootY = useCallback((id: string) => {
    const fig = sceneRef.current?.figures.find((f) => f.id === id);
    setRootY(fig ? fig.rig.root.position.y : 0);
  }, []);

  const selectFigure = useCallback(
    (id: string) => {
      activeRef.current = id;
      setActive(id);
      syncRootY(id);
    },
    [syncRootY]
  );

  const applySnapshot = useCallback(
    (snap: StudioSnapshot) => {
      const ctx = sceneRef.current;
      if (!ctx) return;
      for (const f of [...ctx.figures]) {
        if (!snap.figures.some((s) => s.id === f.id)) destroyFigure(ctx, f.id);
      }
      for (const s of snap.figures) {
        if (!ctx.figures.some((f) => f.id === s.id)) addFigureToScene(ctx, s.id);
      }
      const order = new Map(snap.figures.map((s, i) => [s.id, i]));
      ctx.figures.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
      for (const s of snap.figures) {
        const fig = ctx.figures.find((f) => f.id === s.id);
        if (!fig) continue;
        resetRig(fig.rig);
        applyPose(fig.rig, s.rotations);
        fig.rig.root.position.set(...s.rootPosition);
        fig.rig.root.rotation.y = s.rootRotationY;
      }
      setFigureIds(ctx.figures.map((f) => f.id));
      setProps(snap.props.map((p) => ({ ...p })));
      setSelectedProp(
        snap.props.some((p) => p.id === snap.selectedProp) ? snap.selectedProp : null
      );
      setSelected(null);
      selectFigure(
        ctx.figures.some((f) => f.id === snap.active) ? snap.active : ctx.figures[0]?.id ?? ""
      );
    },
    [selectFigure]
  );

  const undo = useCallback(() => {
    const stack = historyRef.current;
    const entry = stack.pop();
    setHistoryDepth(stack.length);
    if (entry) applySnapshot(entry.snap);
  }, [applySnapshot]);

  // ------------------------------------------------------------------ scene
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    // No scene.background: the canvas is transparent so a scene photo can show
    // through from behind it. `.studio-viewport` supplies the same colour, and
    // `renderDepth` sets its own black, so no map is affected.
    const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 60);
    const orbit = {
      theta: 0,
      phi: Math.PI / 2.35,
      radius: 3.6,
      target: new THREE.Vector3(0, 0.95, 0),
    };

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x40465a, 2.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(2.5, 4, 3);
    scene.add(key);

    const propGroup = new THREE.Group();
    scene.add(propGroup);

    // Helpers are hidden during a map render — a grid is geometry like anything
    // else, and would come out as a very solid floor in the depth map.
    const grid = new THREE.GridHelper(6, 12, 0x39414f, 0x252b34);
    scene.add(grid);
    const handleGroup = new THREE.Group();
    scene.add(handleGroup);
    const helpers: THREE.Object3D[] = [grid, handleGroup];

    const ctx: SceneCtx = {
      renderer,
      scene,
      camera,
      figures: [],
      handleGroup,
      handleGeo: new THREE.SphereGeometry(0.035, 12, 10),
      handleMat: new THREE.MeshBasicMaterial({
        color: 0xffb454,
        depthTest: false,
        transparent: true,
        opacity: 0.9,
      }),
      // Every figure keeps its handles grabbable — clicking one is how you switch
      // figures — but only the selected figure's read as live.
      handleDimMat: new THREE.MeshBasicMaterial({
        color: 0x8d94a3,
        depthTest: false,
        transparent: true,
        opacity: 0.4,
      }),
      propGroup,
      helpers,
      depthMaterial: makeDepthMaterial(),
      orbit,
    };
    sceneRef.current = ctx;

    const first = addFigureToScene(ctx, nextFigureId(), 0);
    applyPose(first.rig, presetRotations(PRESETS[1])); // start standing, not in a T
    setFigureIds([first.id]);
    activeRef.current = first.id;
    setActive(first.id);

    let raf = 0;
    // The viewport is letterboxed to the output frame rather than filled, so what
    // you see is what renders. It has to be: the camera's FOV is vertical, so a
    // wider output aspect adds width rather than cropping height, and a crop
    // outline drawn over a filled viewport tells you the opposite of the truth
    // for any landscape frame. It also means a scene photo laid behind the canvas
    // lines up with the render by construction.
    const resize = () => {
      const mw = mount.clientWidth;
      const mh = mount.clientHeight;
      if (!mw || !mh) return;
      const a = outAspectRef.current;
      const w = Math.min(mw, mh * a);
      const h = w / a;
      renderer.setSize(w, h);
      camera.aspect = a;
      camera.updateProjectionMatrix();
      setViewSize({ w, h });
    };
    resizeRef.current = resize;
    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    const world = new THREE.Vector3();
    const tick = () => {
      const { theta, phi, radius, target } = orbit;
      camera.position.set(
        target.x + radius * Math.sin(phi) * Math.sin(theta),
        target.y + radius * Math.cos(phi),
        target.z + radius * Math.sin(phi) * Math.cos(theta)
      );
      camera.lookAt(target);
      const rigs = new Map(ctx.figures.map((f) => [f.id, f.rig]));
      for (const f of ctx.figures) f.rig.root.updateWorldMatrix(true, true);
      for (const obj of ctx.handleGroup.children) {
        const mesh = obj as THREE.Mesh;
        const rig = rigs.get(mesh.userData.figureId);
        if (!rig) continue;
        mesh.position.copy(rig.bones[mesh.userData.bone].getWorldPosition(world));
        mesh.material =
          mesh.userData.figureId === activeRef.current ? ctx.handleMat : ctx.handleDimMat;
      }
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      for (const f of ctx.figures) disposeRig(f.rig);
      ctx.handleGeo.dispose();
      ctx.handleMat.dispose();
      ctx.handleDimMat.dispose();
      ctx.depthMaterial.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  // The ResizeObserver only fires when the *viewport* changes; picking a
  // different output frame changes the canvas without touching it.
  useEffect(() => {
    resizeRef.current();
  }, [frame.w, frame.h]);

  // Rebuild the prop meshes whenever the list changes. Cheap — there are a
  // handful of boxes — and it keeps one definition of a prop rather than a scene
  // graph and a state tree that have to be patched in step.
  useEffect(() => {
    const ctx = sceneRef.current;
    if (!ctx) return;
    ctx.propGroup.clear();
    const mat = new THREE.MeshStandardMaterial({ color: 0x8d94a3, roughness: 0.9 });
    const selMat = new THREE.MeshStandardMaterial({ color: 0xffb454, roughness: 0.7 });
    for (const p of props) {
      ctx.propGroup.add(buildProp(p, p.id === selectedProp ? selMat : mat));
    }
  }, [props, selectedProp]);

  // Measure a loaded scene photo so the output frame can match its aspect.
  useEffect(() => {
    if (!backdrop) {
      setBackdropFrame(null);
      setAspect((a) => Math.min(a, ASPECTS.length - 1));
      return;
    }
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (!live) return;
      setBackdropFrame(frameForImage(img.naturalWidth, img.naturalHeight));
      setAspect(ASPECTS.length);
    };
    img.src = backdrop;
    return () => {
      live = false;
    };
  }, [backdrop]);

  // ------------------------------------------------------------------ input
  useEffect(() => {
    const ctx = sceneRef.current;
    const el = ctx?.renderer.domElement;
    if (!ctx || !el) return;

    let mode: "none" | "orbit" | "bone" | "prop" = "none";
    let boneName: string | null = null;
    let dragFigure: string | null = null;
    let propId: string | null = null;
    // Where on the ground the figure was grabbed, relative to its own origin, so
    // it doesn't jump so its hips land under the cursor on the first move.
    let grabOffset = new THREE.Vector3();
    let grabPlaneY = 0;
    let lastX = 0;
    let lastY = 0;
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();

    const toNdc = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      return ndc;
    };

    const rigOf = (id: string | null) =>
      id ? ctx.figures.find((f) => f.id === id)?.rig ?? null : null;

    /** Where the cursor meets the horizontal plane at `y`, or null when the ray
     *  is too near parallel to it for the answer to mean anything. */
    const groundPoint = (e: PointerEvent, y: number): THREE.Vector3 | null => {
      ray.setFromCamera(toNdc(e), ctx.camera);
      if (Math.abs(ray.ray.direction.y) < 0.15) return null;
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -y);
      return ray.ray.intersectPlane(plane, new THREE.Vector3());
    };

    const onDown = (e: PointerEvent) => {
      el.setPointerCapture(e.pointerId);
      lastX = e.clientX;
      lastY = e.clientY;
      ray.setFromCamera(toNdc(e), ctx.camera);

      // Handles first: they're drawn on top and are what the user is aiming at.
      const hit = ray.intersectObjects(ctx.handleGroup.children, false)[0];
      if (hit) {
        mode = "bone";
        boneName = hit.object.userData.bone;
        dragFigure = hit.object.userData.figureId;
        armHistory();
        setSelected(boneName);
        setSelectedProp(null);
        if (dragFigure && dragFigure !== activeRef.current) selectFigure(dragFigure);
        const rig = rigOf(dragFigure);
        if (boneName === "hips" && rig) {
          grabPlaneY = rig.bones.hips.getWorldPosition(new THREE.Vector3()).y;
          const at = groundPoint(e, grabPlaneY);
          grabOffset = at ? at.sub(rig.root.position).setY(0) : new THREE.Vector3();
        }
        return;
      }
      const propHit = ray.intersectObjects(ctx.propGroup.children, false)[0];
      if (propHit) {
        mode = "prop";
        propId = propHit.object.userData.propId;
        armHistory();
        setSelectedProp(propId);
        setSelected(null);
        return;
      }
      mode = "orbit";
    };

    const onMove = (e: PointerEvent) => {
      if (mode === "none") return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;

      if (mode === "orbit") {
        ctx.orbit.theta -= dx * 0.007;
        ctx.orbit.phi = Math.min(Math.max(ctx.orbit.phi - dy * 0.007, 0.12), Math.PI - 0.12);
        return;
      }

      if (mode === "prop" && propId) {
        // Props slide on the ground plane. Lifting one is what the height field
        // in the panel is for — dragging in Y as well would make it impossible to
        // put anything down flat.
        const p = propsRef.current.find((x) => x.id === propId);
        if (!p) return;
        const hit = groundPoint(e, p.position[1]);
        if (hit) {
          commitHistory();
          setProps((cur) =>
            cur.map((x) => (x.id === propId ? { ...x, position: [hit.x, x.position[1], hit.z] } : x))
          );
        }
        return;
      }

      if (mode !== "bone" || !boneName) return;
      const rig = rigOf(dragFigure);
      if (!rig) return;

      if (boneName === "hips") {
        // The hips handle places the figure rather than rotating anything —
        // rotating from the pelvis is what the waist handle already does. It
        // slides on the *floor* rather than across the view, because with two
        // figures the operation that matters is "bring them together", and on a
        // camera-facing plane that takes an orbit first.
        commitHistory();
        const at = groundPoint(e, grabPlaneY);
        if (at) {
          rig.root.position.set(at.x - grabOffset.x, rig.root.position.y, at.z - grabOffset.z);
        } else {
          // Camera nearly level with the floor: the ground ray lands hundreds of
          // metres away. Fall back to the view plane and keep the height.
          const joint = rig.bones.hips.getWorldPosition(new THREE.Vector3());
          const target = dragPlanePoint(toNdc(e), ctx.camera, joint);
          rig.root.position.x += target.x - joint.x;
          rig.root.position.z += target.z - joint.z;
        }
        return;
      }

      // Every other handle sits at a bone's origin, and moving that origin means
      // rotating the bone *above* it: pulling the elbow swings the upper arm.
      // Only that one bone turns, so the joint above the one you grabbed stays
      // exactly where it was — dragging a wrist can no longer move the elbow.
      const aim = HANDLE_AIM[boneName];
      if (!aim) return;
      commitHistory();
      const from = rig.bones[aim.bone].getWorldPosition(new THREE.Vector3());
      const joint = rig.bones[boneName].getWorldPosition(new THREE.Vector3());
      const target = dragPlanePoint(toNdc(e), ctx.camera, joint);
      aimBone(rig, aim.bone, target.clone().sub(from), aim.child);
    };

    const onUp = (e: PointerEvent) => {
      el.releasePointerCapture(e.pointerId);
      mode = "none";
      boneName = null;
      dragFigure = null;
      propId = null;
      pendingRef.current = null;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      ctx.orbit.radius = Math.min(Math.max(ctx.orbit.radius * (1 + Math.sign(e.deltaY) * 0.09), 0.9), 14);
    };

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      el.removeEventListener("wheel", onWheel);
    };
  }, [armHistory, commitHistory, selectFigure]);

  // Ctrl/Cmd+Z on the window, because the modal isn't focusable — but never when
  // the caret is in a text field, where it means the browser's own undo.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== "z" || !(e.ctrlKey || e.metaKey) || e.shiftKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      e.preventDefault();
      undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo]);

  // ------------------------------------------------------------------ actions
  const renderMaps = useCallback((): { kind: ControlKind; url: string }[] => {
    const ctx = sceneRef.current;
    if (!ctx) return [];
    const { w, h } = frame;
    const out: { kind: ControlKind; url: string }[] = [];

    // Handles and the grid are UI, not scenery — a map rendered with them in it
    // would tell the model to draw a floor grid and a fistful of orange spheres.
    const shown = ctx.helpers.map((o) => o.visible);
    ctx.helpers.forEach((o) => (o.visible = false));
    const roots = ctx.figures.map((f) => f.rig.root);
    roots.forEach((r) => r.updateWorldMatrix(true, true));

    if (emit.depth) {
      out.push({
        kind: "depth",
        url: renderDepth(
          ctx.renderer, ctx.scene, ctx.camera, ctx.depthMaterial,
          [...roots, ctx.propGroup], w, h
        ),
      });
    }
    ctx.helpers.forEach((o, i) => (o.visible = shown[i]));

    if (emit.pose) {
      out.push({ kind: "pose", url: renderPose(ctx.figures.map((f) => f.rig), ctx.camera, w, h) });
    }
    return out;
  }, [frame, emit]);

  const refreshPreview = useCallback(() => setPreview(renderMaps()), [renderMaps]);

  function usePose() {
    const ctx = sceneRef.current;
    const maps = renderMaps();
    if (!maps.length || !ctx) return;
    const rigs = ctx.figures.map((f) => f.rig);
    onUse(maps, { subjects: rigs.length, contact: figuresInContact(rigs) });
    onClose();
  }

  const activeFigure = () => sceneRef.current?.figures.find((f) => f.id === active) ?? null;

  function loadPreset(i: number) {
    const ctx = sceneRef.current;
    const fig = activeFigure();
    if (!ctx || !fig) return;
    pushHistory();
    const preset = PRESETS[i];
    // A preset says how a body is folded and how far the hips drop; it does not
    // say where in the room the figure stands. Keeping the placement is what lets
    // you sit the second figure down without it teleporting onto the first.
    const keepX = fig.rig.root.position.x;
    const keepZ = fig.rig.root.position.z;
    const keepY = fig.rig.root.rotation.y;
    const [, presetY, presetZ] = preset.root ?? [0, 0, 0];
    resetRig(fig.rig);
    applyPose(fig.rig, presetRotations(preset));
    fig.rig.root.position.set(keepX, presetY, keepZ);
    fig.rig.root.rotation.y = keepY;
    fig.rig.root.updateWorldMatrix(true, true);

    // Reach the named limbs to the floor. Hand-written angles get the shape of a
    // pose right and the contact wrong every time, and a hand hovering 8 cm above
    // the ground is exactly the tell that makes a generated image look weightless.
    for (const limb of GROUND_REACH[preset.name] ?? []) {
      const handles = limb === "hands" ? (["handR", "handL"] as const) : (["footR", "footL"] as const);
      // Hands land slightly in front of where they hang, which is where they go
      // on a real crouch — planted under the shoulders reads as a collapse.
      const forward = limb === "hands" ? 0.12 : 0;
      for (const h of handles) {
        const chain = IK_CHAINS[h];
        const end = fig.rig.bones[chain.end].getWorldPosition(new THREE.Vector3());
        const root = fig.rig.bones[chain.root].getWorldPosition(new THREE.Vector3());
        const pole = root.clone().add(new THREE.Vector3(...POLE_HINTS[h]));
        solveTwoBoneIK(fig.rig, chain, new THREE.Vector3(end.x, 0.03, end.z + forward), pole);
        fig.rig.root.updateWorldMatrix(true, true);
      }
    }

    // A preset's props are placed relative to the figure it was authored for, so
    // carry that relationship over to wherever this figure is standing. Fresh ids
    // because the ones in `presets.ts` are literals — two figures both sitting
    // would otherwise share a prop id, and every edit would hit both boxes.
    const stamp = Date.now().toString(36);
    const moved = (preset.props ?? []).map((p, n) => ({
      ...p,
      id: `p${stamp}${n}`,
      position: [p.position[0] + keepX, p.position[1], p.position[2] + keepZ - presetZ] as [
        number, number, number,
      ],
    }));
    // With one figure a preset is the whole scene, so it replaces what's there.
    // With several it can only be *this* figure's furniture — clearing the list
    // would take away the other figure's chair.
    setProps((cur) => (ctx.figures.length > 1 ? [...cur, ...moved] : moved));
    setSelectedProp(null);
    setRootY(presetY);
  }

  function savePose() {
    const ctx = sceneRef.current;
    if (!ctx || !saveName.trim()) return;
    const file: PoseFileV2 = {
      version: 2,
      name: saveName.trim(),
      figures: ctx.figures.map((f) => ({
        rotations: readPose(f.rig),
        rootPosition: f.rig.root.position.toArray() as [number, number, number],
        rootRotationY: f.rig.root.rotation.y,
      })),
      props: propsRef.current,
    };
    const next = [...saved.filter((p) => p.name !== file.name), file];
    setSaved(next);
    localStorage.setItem(STORE_KEY, JSON.stringify(next));
    setSaveName("");
  }

  function loadPose(file: PoseFile) {
    const ctx = sceneRef.current;
    if (!ctx) return;
    pushHistory();
    const scene = normalizePoseFile(file);
    while (ctx.figures.length > scene.figures.length) {
      destroyFigure(ctx, ctx.figures[ctx.figures.length - 1].id);
    }
    while (ctx.figures.length < scene.figures.length) {
      addFigureToScene(ctx, nextFigureId());
    }
    scene.figures.forEach((pose, i) => {
      const rig = ctx.figures[i].rig;
      resetRig(rig);
      applyPose(rig, pose.rotations);
      rig.root.position.set(...pose.rootPosition);
      rig.root.rotation.y = pose.rootRotationY;
    });
    setFigureIds(ctx.figures.map((f) => f.id));
    setProps(scene.props.map((p) => ({ ...p })));
    setSelectedProp(null);
    selectFigure(ctx.figures[0]?.id ?? "");
  }

  function deletePose(name: string) {
    const next = saved.filter((p) => p.name !== name);
    setSaved(next);
    localStorage.setItem(STORE_KEY, JSON.stringify(next));
  }

  function addFigure() {
    const ctx = sceneRef.current;
    if (!ctx) return;
    pushHistory();
    const n = ctx.figures.length;
    // Alternate sides so a third figure doesn't land on top of the second.
    const x = (n % 2 === 1 ? 1 : -1) * FIGURE_GAP * Math.ceil(n / 2);
    const fig = addFigureToScene(ctx, nextFigureId(), x);
    applyPose(fig.rig, presetRotations(PRESETS[1]));
    setFigureIds(ctx.figures.map((f) => f.id));
    selectFigure(fig.id);
    setSelected(null);
  }

  function removeFigure(id: string) {
    const ctx = sceneRef.current;
    if (!ctx || ctx.figures.length < 2) return;
    pushHistory();
    destroyFigure(ctx, id);
    setFigureIds(ctx.figures.map((f) => f.id));
    setSelected(null);
    if (activeRef.current === id) selectFigure(ctx.figures[0].id);
  }

  function setFigureLift(y: number) {
    const fig = activeFigure();
    if (!fig) return;
    pushHistory(`figure:${active}:lift`);
    fig.rig.root.position.y = y;
    setRootY(y);
  }

  function addProp(kind: PropDef["kind"]) {
    pushHistory();
    const id = `p${Date.now().toString(36)}`;
    const size: [number, number, number] =
      kind === "box" ? [0.46, 0.46, 0.46] : kind === "cylinder" ? [0.4, 0.5, 0.4] : [0.5, 0.5, 0.5];
    setProps((cur) => [...cur, { id, kind, position: [0, size[1] / 2, -0.05], size, rotationY: 0 }]);
    setSelectedProp(id);
  }

  function patchProp(id: string, patch: Partial<PropDef>) {
    setProps((cur) => cur.map((p) => (p.id === id ? { ...p, ...patch } : p)));
  }

  function turnFigure(deg: number) {
    const fig = activeFigure();
    if (!fig) return;
    pushHistory();
    fig.rig.root.rotation.y += THREE.MathUtils.degToRad(deg);
  }

  function resetFigure() {
    const fig = activeFigure();
    if (!fig) return;
    pushHistory();
    const keepX = fig.rig.root.position.x;
    const keepZ = fig.rig.root.position.z;
    resetRig(fig.rig);
    fig.rig.root.position.set(keepX, 0, keepZ);
    setRootY(0);
  }

  async function loadBackdrop(files: FileList | null) {
    if (!files?.length) return;
    setBackdrop(await fileToDataUrl(files[0]));
  }

  const prop = props.find((p) => p.id === selectedProp) ?? null;

  return (
    <div className="studio-backdrop" onClick={onClose}>
      <div className="studio" onClick={(e) => e.stopPropagation()}>
        <div className="studio-head">
          <strong>🧍 Pose Studio</strong>
          <span className="muted small">
            Drag a joint to aim it · drag the hips to move a figure · drag the background
            to orbit · scroll to zoom
          </span>
          <button className="btn ghost icon" onClick={onClose} title="Close">
            ✕
          </button>
        </div>

        <div className="studio-body">
          <div className="studio-viewport">
            {backdrop && (
              <img
                className="studio-scene-img"
                src={backdrop}
                alt=""
                style={{ width: viewSize.w, height: viewSize.h, opacity: backdropOpacity }}
              />
            )}
            <div className="studio-canvas" ref={mountRef} />
            <div
              className="studio-guide"
              style={{ width: viewSize.w, height: viewSize.h }}
            >
              <span className="studio-guide-tag">{frame.label} frame</span>
            </div>
          </div>

          <div className="studio-panel">
            <div className="lbl">Figures</div>
            <div className="studio-row">
              {figureIds.map((id, i) => (
                <button
                  key={id}
                  className={`control-chip ${id === active ? "on" : ""}`}
                  onClick={() => selectFigure(id)}
                  title={`Select figure ${i + 1}`}
                >
                  {i + 1}
                </button>
              ))}
              <button className="btn small" onClick={addFigure} title="Add another figure">
                + Figure
              </button>
              {figureIds.length > 1 && (
                <button
                  className="btn danger small"
                  onClick={() => removeFigure(active)}
                  title="Remove the selected figure"
                >
                  🗑
                </button>
              )}
            </div>
            <div className="muted small">
              Two figures is how you say <em>they are touching</em>. Drag a figure's hips
              to slide it across the floor — they can overlap, and where they overlap the
              depth map says who is in front.
            </div>
            <label className="studio-field">
              Lift
              <input
                type="range" min={-1.2} max={1} step={0.02}
                value={rootY}
                onChange={(e) => setFigureLift(+e.target.value)}
              />
              <span className="studio-num">{rootY.toFixed(2)}m</span>
            </label>

            <div className="lbl">Start from</div>
            <div className="studio-chips">
              {PRESETS.map((p, i) => (
                <button key={p.name} className="control-chip" title={p.hint} onClick={() => loadPreset(i)}>
                  {p.name}
                </button>
              ))}
            </div>

            <div className="lbl">Selected joint</div>
            <div className="muted small">
              {selected ? (
                <>
                  <strong>{HANDLE_LABELS[selected] ?? selected}</strong>
                  {selected === "hips"
                    ? " — drag to slide this figure across the floor."
                    : " — drag to aim it; everything below follows, and the joint above holds still."}
                </>
              ) : (
                "Click a joint handle in the view."
              )}
            </div>
            <div className="studio-row">
              <button className="btn small" onClick={() => turnFigure(-30)}>↺ Turn</button>
              <button className="btn small" onClick={() => turnFigure(30)}>Turn ↻</button>
              <button className="btn small" onClick={resetFigure}>Reset pose</button>
              <button
                className="btn small"
                onClick={undo}
                disabled={historyDepth === 0}
                title="Undo the last change (Ctrl+Z)"
              >
                ↶ Undo
              </button>
            </div>

            <div className="lbl">Props</div>
            <div className="muted small">
              Put a block where the chair, closet or box is. This is what lets the depth
              map say the figure is <em>resting on</em> something rather than floating —
              the one thing a skeleton can never express.
            </div>
            <div className="studio-row">
              <button className="btn small" onClick={() => addProp("box")}>+ Box</button>
              <button className="btn small" onClick={() => addProp("cylinder")}>+ Cylinder</button>
              <button className="btn small" onClick={() => addProp("sphere")}>+ Ball</button>
            </div>
            {prop && (
              <div className="studio-prop">
                <div className="studio-row">
                  <span className="muted small">{prop.kind}</span>
                  <button
                    className="btn danger small"
                    onClick={() => {
                      pushHistory();
                      setProps((cur) => cur.filter((p) => p.id !== prop.id));
                      setSelectedProp(null);
                    }}
                  >
                    🗑
                  </button>
                </div>
                {(["Width", "Height", "Depth"] as const).map((axis, i) => (
                  <label key={axis} className="studio-field">
                    {axis}
                    <input
                      type="range" min={0.1} max={2.5} step={0.02}
                      value={prop.size[i]}
                      onChange={(e) => {
                        pushHistory(`prop:${prop.id}:size:${i}`);
                        const size = [...prop.size] as [number, number, number];
                        size[i] = +e.target.value;
                        // Keep it sitting on the floor when its height changes,
                        // rather than sinking into it or hovering.
                        const position = [...prop.position] as [number, number, number];
                        if (i === 1) position[1] = size[1] / 2;
                        patchProp(prop.id, { size, position });
                      }}
                    />
                    <span className="studio-num">{prop.size[i].toFixed(2)}m</span>
                  </label>
                ))}
                <label className="studio-field">
                  Lift
                  <input
                    type="range" min={0} max={2} step={0.02}
                    value={prop.position[1]}
                    onChange={(e) => {
                      pushHistory(`prop:${prop.id}:lift`);
                      patchProp(prop.id, {
                        position: [prop.position[0], +e.target.value, prop.position[2]],
                      });
                    }}
                  />
                  <span className="studio-num">{prop.position[1].toFixed(2)}m</span>
                </label>
              </div>
            )}

            <div className="lbl">Scene photo</div>
            <div className="muted small">
              Lay the photo you're generating into behind the figures, to pose them against
              its perspective. It is a guide only — it never reaches the control maps.
            </div>
            <div className="studio-row">
              <label className="btn small studio-file">
                {backdrop ? "Replace…" : "Load image…"}
                <input type="file" accept="image/*" onChange={(e) => loadBackdrop(e.target.files)} />
              </label>
              {backdrop && (
                <button className="btn small" onClick={() => setBackdrop(null)}>Clear</button>
              )}
            </div>
            {backdrop && (
              <label className="studio-field">
                Opacity
                <input
                  type="range" min={0.1} max={1} step={0.05}
                  value={backdropOpacity}
                  onChange={(e) => setBackdropOpacity(+e.target.value)}
                />
                <span className="studio-num">{Math.round(backdropOpacity * 100)}%</span>
              </label>
            )}

            <div className="lbl">Output</div>
            <div className="studio-row">
              {aspects.map((a, i) => (
                <button
                  key={a.label}
                  className={`control-chip ${aspect === i ? "on" : ""}`}
                  onClick={() => setAspect(i)}
                  title={`${a.w}×${a.h}`}
                >
                  {a.label}
                </button>
              ))}
            </div>
            <div className="studio-row">
              {(["depth", "pose"] as ControlKind[]).map((k) => (
                <button
                  key={k}
                  className={`control-chip ${emit[k] ? "on" : ""}`}
                  onClick={() => setEmit((e) => ({ ...e, [k]: !e[k] }))}
                  title={
                    k === "depth"
                      ? "The 3-D shape of the figures and everything they touch. The map that does the work."
                      : "The OpenPose skeletons, in the standard palette. Pins which limb is whose; stack it with depth."
                  }
                >
                  {emit[k] ? "✓ " : ""}{k}
                </button>
              ))}
              <button className="btn small" onClick={refreshPreview}>Preview</button>
            </div>
            {preview.length > 0 && (
              <div className="studio-preview">
                {preview.map((m) => (
                  <figure key={m.kind}>
                    <img src={m.url} alt={`${m.kind} map`} />
                    <figcaption className="muted small">{m.kind}</figcaption>
                  </figure>
                ))}
              </div>
            )}

            <div className="lbl">Saved poses</div>
            <div className="studio-row">
              <input
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                placeholder="Name this pose…"
                onKeyDown={(e) => e.key === "Enter" && savePose()}
              />
              <button className="btn small" onClick={savePose} disabled={!saveName.trim()}>
                Save
              </button>
            </div>
            {saved.map((p) => (
              <div key={p.name} className="row extra-model">
                <button className="btn ghost small studio-load" onClick={() => loadPose(p)}>
                  {p.name}
                </button>
                <button className="btn danger small" onClick={() => deletePose(p.name)}>
                  🗑
                </button>
              </div>
            ))}

            <button className="btn send gen studio-use" onClick={usePose}>
              Use this pose →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
