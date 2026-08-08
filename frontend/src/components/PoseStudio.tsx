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
 * Interaction model, in one line each:
 *   drag a joint handle  — pose it (wrists and ankles solve IK; everything else aims)
 *   drag the background  — orbit the camera
 *   wheel                — dolly
 *   props                — blocks you place so the figure has something to rest on
 */
import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  HANDLES,
  HANDLE_LABELS,
  IK_CHAINS,
  POLE_HINTS,
  type PoseFile,
  type PropDef,
} from "../pose/skeleton";
import {
  applyPose,
  buildProp,
  buildRig,
  readPose,
  resetRig,
  type Rig,
} from "../pose/mannequin";
import { aimBone, dragPlanePoint, solveTwoBoneIK } from "../pose/ik";
import { makeDepthMaterial, renderDepth, renderPose } from "../pose/render";
import { GROUND_REACH, PRESETS, presetRotations } from "../pose/presets";
import type { ControlKind } from "../types";

const ASPECTS = [
  { label: "Portrait", w: 832, h: 1216 },
  { label: "Square", w: 1024, h: 1024 },
  { label: "Landscape", w: 1216, h: 832 },
];

const STORE_KEY = "poseStudio.saved";

interface Props {
  onClose: () => void;
  /** Hands the rendered maps to the composer. */
  onUse: (maps: { kind: ControlKind; url: string }[]) => void;
}

export default function PoseStudio({ onClose, onUse }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    rig: Rig;
    handles: THREE.Mesh[];
    handleGroup: THREE.Group;
    propGroup: THREE.Group;
    helpers: THREE.Object3D[];
    depthMaterial: THREE.ShaderMaterial;
    orbit: { theta: number; phi: number; radius: number; target: THREE.Vector3 };
  } | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
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

  // Props live in React state (so the panel can edit them) and in the scene (so
  // they render). This ref is what keeps a drag from having to round-trip through
  // a re-render for every mouse move.
  const propsRef = useRef<PropDef[]>(props);
  propsRef.current = props;

  // ------------------------------------------------------------------ scene
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14171c);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 60);
    const orbit = { theta: 0, phi: Math.PI / 2.35, radius: 3.6, target: new THREE.Vector3(0, 0.95, 0) };

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x40465a, 2.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(2.5, 4, 3);
    scene.add(key);

    const rig = buildRig();
    scene.add(rig.root);

    const propGroup = new THREE.Group();
    scene.add(propGroup);

    // Helpers are hidden during a map render — a grid is geometry like anything
    // else, and would come out as a very solid floor in the depth map.
    const grid = new THREE.GridHelper(6, 12, 0x39414f, 0x252b34);
    scene.add(grid);
    const helpers: THREE.Object3D[] = [grid];

    const handleGroup = new THREE.Group();
    scene.add(handleGroup);
    const handleGeo = new THREE.SphereGeometry(0.035, 12, 10);
    const handles: THREE.Mesh[] = HANDLES.map((name) => {
      const mesh = new THREE.Mesh(
        handleGeo,
        new THREE.MeshBasicMaterial({ color: 0xffb454, depthTest: false, transparent: true, opacity: 0.9 })
      );
      mesh.renderOrder = 10;
      mesh.userData.bone = name;
      handleGroup.add(mesh);
      return mesh;
    });
    helpers.push(handleGroup);

    const depthMaterial = makeDepthMaterial();
    sceneRef.current = {
      renderer, scene, camera, rig, handles, handleGroup, propGroup, helpers, depthMaterial, orbit,
    };

    applyPose(rig, presetRotations(PRESETS[1])); // start standing, not in a T

    let raf = 0;
    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      setViewSize({ w, h });
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    const tick = () => {
      const { theta, phi, radius, target } = orbit;
      camera.position.set(
        target.x + radius * Math.sin(phi) * Math.sin(theta),
        target.y + radius * Math.cos(phi),
        target.z + radius * Math.sin(phi) * Math.cos(theta)
      );
      camera.lookAt(target);
      rig.root.updateWorldMatrix(true, true);
      handles.forEach((h) => h.position.copy(rig.bones[h.userData.bone].getWorldPosition(new THREE.Vector3())));
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

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

  // ------------------------------------------------------------------ input
  useEffect(() => {
    const ctx = sceneRef.current;
    const el = ctx?.renderer.domElement;
    if (!ctx || !el) return;

    let mode: "none" | "orbit" | "bone" | "prop" = "none";
    let boneName: string | null = null;
    let propId: string | null = null;
    let lastX = 0;
    let lastY = 0;
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();

    const toNdc = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      return ndc;
    };

    const onDown = (e: PointerEvent) => {
      el.setPointerCapture(e.pointerId);
      lastX = e.clientX;
      lastY = e.clientY;
      ray.setFromCamera(toNdc(e), ctx.camera);

      // Handles first: they're drawn on top and are what the user is aiming at.
      const hit = ray.intersectObjects(ctx.handles, false)[0];
      if (hit) {
        mode = "bone";
        boneName = hit.object.userData.bone;
        setSelected(boneName);
        setSelectedProp(null);
        return;
      }
      const propHit = ray.intersectObjects(ctx.propGroup.children, false)[0];
      if (propHit) {
        mode = "prop";
        propId = propHit.object.userData.propId;
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
        const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -p.position[1]);
        ray.setFromCamera(toNdc(e), ctx.camera);
        const hit = ray.ray.intersectPlane(plane, new THREE.Vector3());
        if (hit) {
          setProps((cur) =>
            cur.map((x) => (x.id === propId ? { ...x, position: [hit.x, x.position[1], hit.z] } : x))
          );
        }
        return;
      }

      if (mode !== "bone" || !boneName) return;
      const { rig } = ctx;
      const chain = IK_CHAINS[boneName];
      const joint = rig.bones[boneName].getWorldPosition(new THREE.Vector3());
      const target = dragPlanePoint(toNdc(e), ctx.camera, joint);

      if (boneName === "hips") {
        // The hips handle moves the whole figure rather than rotating anything.
        // Rotating from the pelvis is what the waist handle already does, and
        // *placing* the figure is the operation with no other home — it's how you
        // sit someone down onto a box you've just put on the floor.
        rig.root.position.add(target.clone().sub(joint));
        return;
      }

      if (chain) {
        const hint = POLE_HINTS[boneName] ?? [0, 0, -1];
        // The pole is expressed in the figure's own space, so it keeps meaning
        // "the elbow bends backwards" after the figure has been turned around.
        const pole = rig.bones[chain.root]
          .getWorldPosition(new THREE.Vector3())
          .add(new THREE.Vector3(...hint).applyQuaternion(rig.root.getWorldQuaternion(new THREE.Quaternion())));
        solveTwoBoneIK(rig, chain, target, pole);
        return;
      }

      // Every other handle sits at a bone's origin, and moving that origin means
      // rotating the bone *above* it: pulling the elbow swings the upper arm.
      const parent = rig.bones[boneName].parent;
      if (!parent) return;
      const from = parent.getWorldPosition(new THREE.Vector3());
      aimBone(rig, parentBoneFor(boneName), target.clone().sub(from));
    };

    const onUp = (e: PointerEvent) => {
      el.releasePointerCapture(e.pointerId);
      mode = "none";
      boneName = null;
      propId = null;
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
  }, []);

  // ------------------------------------------------------------------ actions
  const renderMaps = useCallback((): { kind: ControlKind; url: string }[] => {
    const ctx = sceneRef.current;
    if (!ctx) return [];
    const { w, h } = ASPECTS[aspect];
    const out: { kind: ControlKind; url: string }[] = [];

    // Handles and the grid are UI, not scenery — a map rendered with them in it
    // would tell the model to draw a floor grid and seventeen orange spheres.
    const shown = ctx.helpers.map((o) => o.visible);
    ctx.helpers.forEach((o) => (o.visible = false));
    ctx.rig.root.updateWorldMatrix(true, true);

    if (emit.depth) {
      out.push({
        kind: "depth",
        url: renderDepth(
          ctx.renderer, ctx.scene, ctx.camera, ctx.depthMaterial,
          [ctx.rig.root, ctx.propGroup], w, h
        ),
      });
    }
    ctx.helpers.forEach((o, i) => (o.visible = shown[i]));

    if (emit.pose) out.push({ kind: "pose", url: renderPose(ctx.rig, ctx.camera, w, h) });
    return out;
  }, [aspect, emit]);

  const refreshPreview = useCallback(() => setPreview(renderMaps()), [renderMaps]);

  function usePose() {
    const maps = renderMaps();
    if (!maps.length) return;
    onUse(maps);
    onClose();
  }

  function loadPreset(i: number) {
    const ctx = sceneRef.current;
    if (!ctx) return;
    const preset = PRESETS[i];
    resetRig(ctx.rig);
    applyPose(ctx.rig, presetRotations(preset));
    if (preset.root) ctx.rig.root.position.set(...preset.root);
    ctx.rig.root.updateWorldMatrix(true, true);

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
        const end = ctx.rig.bones[chain.end].getWorldPosition(new THREE.Vector3());
        const root = ctx.rig.bones[chain.root].getWorldPosition(new THREE.Vector3());
        const pole = root.clone().add(new THREE.Vector3(...POLE_HINTS[h]));
        solveTwoBoneIK(ctx.rig, chain, new THREE.Vector3(end.x, 0.03, end.z + forward), pole);
        ctx.rig.root.updateWorldMatrix(true, true);
      }
    }
    setProps(preset.props ? preset.props.map((p) => ({ ...p })) : []);
    setSelectedProp(null);
  }

  function savePose() {
    const ctx = sceneRef.current;
    if (!ctx || !saveName.trim()) return;
    const file: PoseFile = {
      version: 1,
      name: saveName.trim(),
      rotations: readPose(ctx.rig),
      rootPosition: ctx.rig.root.position.toArray() as [number, number, number],
      rootRotationY: ctx.rig.root.rotation.y,
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
    resetRig(ctx.rig);
    applyPose(ctx.rig, file.rotations);
    ctx.rig.root.position.set(...file.rootPosition);
    ctx.rig.root.rotation.y = file.rootRotationY;
    setProps(file.props.map((p) => ({ ...p })));
  }

  function deletePose(name: string) {
    const next = saved.filter((p) => p.name !== name);
    setSaved(next);
    localStorage.setItem(STORE_KEY, JSON.stringify(next));
  }

  function addProp(kind: PropDef["kind"]) {
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
    const ctx = sceneRef.current;
    if (ctx) ctx.rig.root.rotation.y += THREE.MathUtils.degToRad(deg);
  }

  const prop = props.find((p) => p.id === selectedProp) ?? null;

  // Where the rendered frame falls inside the viewport. The camera's vertical FOV
  // is fixed, so changing aspect only changes how much is seen horizontally —
  // which means a portrait render off a wide viewport quietly crops the arms off.
  // Showing the crop is cheaper than explaining it.
  const outAspect = ASPECTS[aspect].w / ASPECTS[aspect].h;
  const viewAspect = viewSize.w / Math.max(viewSize.h, 1);
  const guide =
    outAspect <= viewAspect
      ? { width: `${(outAspect / viewAspect) * 100}%`, height: "100%" }
      : { width: "100%", height: `${(viewAspect / outAspect) * 100}%` };

  return (
    <div className="studio-backdrop" onClick={onClose}>
      <div className="studio" onClick={(e) => e.stopPropagation()}>
        <div className="studio-head">
          <strong>🧍 Pose Studio</strong>
          <span className="muted small">
            Drag a joint to pose · drag the background to orbit · scroll to zoom
          </span>
          <button className="btn ghost icon" onClick={onClose} title="Close">
            ✕
          </button>
        </div>

        <div className="studio-body">
          <div className="studio-viewport">
            <div className="studio-canvas" ref={mountRef} />
            <div className="studio-guide" style={guide}>
              <span className="studio-guide-tag">{ASPECTS[aspect].label} frame</span>
            </div>
          </div>

          <div className="studio-panel">
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
                  {IK_CHAINS[selected]
                    ? " — drag it and the limb solves itself."
                    : " — drag to aim it; everything below follows."}
                </>
              ) : (
                "Click a joint handle in the view."
              )}
            </div>
            <div className="studio-row">
              <button className="btn small" onClick={() => turnFigure(-30)}>↺ Turn</button>
              <button className="btn small" onClick={() => turnFigure(30)}>Turn ↻</button>
              <button
                className="btn small"
                onClick={() => sceneRef.current && resetRig(sceneRef.current.rig)}
              >
                Reset pose
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
                    onChange={(e) =>
                      patchProp(prop.id, {
                        position: [prop.position[0], +e.target.value, prop.position[2]],
                      })
                    }
                  />
                  <span className="studio-num">{prop.position[1].toFixed(2)}m</span>
                </label>
              </div>
            )}

            <div className="lbl">Output</div>
            <div className="studio-row">
              {ASPECTS.map((a, i) => (
                <button
                  key={a.label}
                  className={`control-chip ${aspect === i ? "on" : ""}`}
                  onClick={() => setAspect(i)}
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
                      ? "The 3-D shape of the figure and everything it touches. The map that does the work."
                      : "The OpenPose skeleton, in the standard palette. Pins which limb is which; stack it with depth."
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

/** Aiming a handle rotates the bone *above* it — grabbing the elbow and pulling
 *  should swing the upper arm, not spin the forearm about its own origin. The
 *  handle for a bone sits at that bone's origin, so the bone to rotate is its
 *  parent. `hips` is its own parent's child (the root) and is handled by the
 *  caller. */
function parentBoneFor(handle: string): string {
  const map: Record<string, string> = {
    spine: "hips", chest: "spine", neck: "chest", head: "neck",
    upperArmR: "clavicleR", foreArmR: "upperArmR", handR: "foreArmR",
    upperArmL: "clavicleL", foreArmL: "upperArmL", handL: "foreArmL",
    thighR: "hips", shinR: "thighR", footR: "shinR",
    thighL: "hips", shinL: "thighL", footL: "shinL",
  };
  return map[handle] ?? handle;
}
