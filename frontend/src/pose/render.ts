/** Turning the posed scene into the control maps a generation conditions on.
 *
 * Both maps are rendered to match what the corresponding preprocessor produces,
 * because a control adapter was trained on those exact conventions and reads a
 * near-miss as a different signal rather than a slightly worse one:
 *
 *   depth — near is *white*, far is black, background black. Matches
 *           Depth Anything 3's `v2_style` normalisation, which ends on
 *           `norm = 1.0 - norm` with the comment "nearer pixels are brighter".
 *   pose  — OpenPose BODY_18 topology in DWPose's palette, limbs as filled
 *           ellipses, joints as dots, on black. Matches ComfyUI's
 *           `SDPoseDrawKeypoints`.
 *
 * Rendering both from the same frame is the part a photo can't give you: the
 * depth map and the skeleton describe one body in one position, with no
 * estimation error between them.
 */
import * as THREE from "three";
import {
  BODY_LIMBS,
  HEAD_LIMBS,
  POSE_COLORS,
} from "./skeleton";
import { openposeWorld, type Rig } from "./mannequin";

/** Depth shader. Linear view-space distance, normalised over the range the
 *  caller measured — not the camera's near/far, which are set for clipping and
 *  would compress the whole figure into a couple of grey levels. */
const DEPTH_VERT = `
varying float vViewDepth;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const DEPTH_FRAG = `
uniform float uNear;
uniform float uFar;
varying float vViewDepth;
void main() {
  float d = clamp((vViewDepth - uNear) / max(uFar - uNear, 1e-4), 0.0, 1.0);
  // Near = white, matching Depth Anything 3's v2_style output.
  gl_FragColor = vec4(vec3(1.0 - d), 1.0);
}`;

export function makeDepthMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: DEPTH_VERT,
    fragmentShader: DEPTH_FRAG,
    uniforms: { uNear: { value: 1 }, uFar: { value: 6 } },
  });
}

/** The view-space depth range that actually contains the scene.
 *
 * Fitting the range to the content is what gives the map its dynamic range: with
 * the camera's own near/far the figure occupies a sliver of the gradient and the
 * result is a flat grey silhouette — which reads to the model as a cut-out, not
 * as a body with a front and a back. */
export function depthRange(
  objects: THREE.Object3D[],
  camera: THREE.Camera
): { near: number; far: number } {
  // Both matrices are otherwise only refreshed by a render, and this runs from a
  // click handler — between frames, against whatever the last frame left behind.
  // A stale camera matrix quietly shifts the whole depth range; a stale object
  // matrix bounds the figure where it used to be.
  for (const o of objects) o.updateWorldMatrix(true, true);
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();

  const box = new THREE.Box3();
  for (const o of objects) box.expandByObject(o);
  if (box.isEmpty()) return { near: 1, far: 6 };

  const view = new THREE.Matrix4().copy(camera.matrixWorldInverse);
  let near = Infinity;
  let far = -Infinity;
  const corner = new THREE.Vector3();
  for (let i = 0; i < 8; i++) {
    corner.set(
      i & 1 ? box.max.x : box.min.x,
      i & 2 ? box.max.y : box.min.y,
      i & 4 ? box.max.z : box.min.z
    );
    const z = -corner.applyMatrix4(view).z;
    near = Math.min(near, z);
    far = Math.max(far, z);
  }
  // A hair of padding, so the closest surface isn't pinned at pure white and the
  // furthest isn't pinned at pure black against an equally black background.
  const pad = Math.max((far - near) * 0.06, 0.02);
  return { near: Math.max(near - pad, 0.01), far: far + pad };
}

/** Render the depth map and return it as a PNG data-URL.
 *
 * Swaps every material for the depth shader via `scene.overrideMaterial`, so
 * nothing in the scene needs to know this pass exists — including props added
 * after the fact. Helpers and grids must be hidden by the caller; they are
 * geometry too, and a floor grid would render as a very solid floor. */
export function renderDepth(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  depthMaterial: THREE.ShaderMaterial,
  subjects: THREE.Object3D[],
  width: number,
  height: number
): string {
  const { near, far } = depthRange(subjects, camera);
  depthMaterial.uniforms.uNear.value = near;
  depthMaterial.uniforms.uFar.value = far;

  const prevOverride = scene.overrideMaterial;
  const prevBg = scene.background;
  const prevSize = renderer.getSize(new THREE.Vector2());
  const prevRatio = renderer.getPixelRatio();

  scene.overrideMaterial = depthMaterial;
  scene.background = new THREE.Color(0x000000);
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  const prevAspect = camera.aspect;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  renderer.render(scene, camera);
  const url = renderer.domElement.toDataURL("image/png");

  scene.overrideMaterial = prevOverride;
  scene.background = prevBg;
  camera.aspect = prevAspect;
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(prevRatio);
  renderer.setSize(prevSize.x, prevSize.y, false);
  return url;
}

/** Project the 18 keypoints to pixel coordinates, against the camera exactly as
 *  it stands. The caller owns the aspect — see `projectKeypoints`. */
function projectRig(
  rig: Rig,
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number
): { x: number; y: number }[] {
  return openposeWorld(rig).map((p) => {
    const v = p.clone().project(camera);
    return { x: ((v.x + 1) / 2) * width, y: ((1 - v.y) / 2) * height };
  });
}

/** Project the 18 keypoints to pixel coordinates for a `width`x`height` frame. */
export function projectKeypoints(
  rig: Rig,
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number
): { x: number; y: number }[] {
  const prevAspect = camera.aspect;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  const pts = projectRig(rig, camera, width, height);
  camera.aspect = prevAspect;
  camera.updateProjectionMatrix();
  return pts;
}

const rgb = (c: [number, number, number]) => `rgb(${c[0]},${c[1]},${c[2]})`;

/** Draw every figure's OpenPose BODY_18 skeleton onto one frame, and return it as
 *  a PNG data-URL.
 *
 * Limbs are ellipses rather than lines because that is what OpenPose renders and
 * what the pose adapters were trained on: a limb has width, and the width carries
 * scale information a 1-pixel line doesn't.
 *
 * Several people go on one canvas, each in the *same* palette, drawn opaque —
 * which is what ComfyUI's `SDPoseDrawKeypoints` does (it loops the frame's people
 * onto a shared canvas, colours by limb index, and fills without blending). A
 * per-person tint would be a different signal to the adapter, not a clearer one.
 * The only ordering rule is back-to-front, so the nearer figure overdraws the
 * further one the way an occlusion actually looks. */
export function renderPose(
  rigs: Rig[],
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);

  // Scale the stick width with the frame, so a 1216px render doesn't come out
  // looking like a wire diagram of the same pose. Global, not per figure: DWPose
  // draws every person at one width, and a thinner skeleton would read as a
  // smaller person rather than a further one.
  const stick = Math.max(Math.round(Math.min(width, height) / 190), 2);
  const limbs = [...BODY_LIMBS, ...HEAD_LIMBS];

  const prevAspect = camera.aspect;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  const ordered = [...rigs].sort(
    (a, b) =>
      camera.position.distanceToSquared(b.root.getWorldPosition(new THREE.Vector3())) -
      camera.position.distanceToSquared(a.root.getWorldPosition(new THREE.Vector3()))
  );

  for (const rig of ordered) {
    const pts = projectRig(rig, camera, width, height);

    limbs.forEach(([i, j], edge) => {
      const a = pts[i];
      const b = pts[j];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const len = Math.hypot(dx, dy);
      if (len < 1) return;
      ctx.fillStyle = rgb(POSE_COLORS[edge % POSE_COLORS.length]);
      ctx.beginPath();
      ctx.ellipse(
        (a.x + b.x) / 2,
        (a.y + b.y) / 2,
        len / 2,
        stick,
        Math.atan2(dy, dx),
        0,
        Math.PI * 2
      );
      ctx.fill();
    });

    pts.forEach((p, i) => {
      ctx.fillStyle = rgb(POSE_COLORS[i % POSE_COLORS.length]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, stick, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  camera.aspect = prevAspect;
  camera.updateProjectionMatrix();
  return canvas.toDataURL("image/png");
}
