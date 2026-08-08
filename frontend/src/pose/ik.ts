/** Posing maths: aiming a single bone, and solving a two-bone chain.
 *
 * Both work in world space and write back a *local* quaternion, because that is
 * what a hierarchy needs: rotating the upper arm must carry the forearm and hand
 * with it, and it only does that if the rotation is stored on the bone rather
 * than baked into world positions.
 */
import * as THREE from "three";
import { CHILD_OF, OFFSETS, type Rig } from "./mannequin";

const _q = new THREE.Quaternion();
const _pq = new THREE.Quaternion();
const _v = new THREE.Vector3();

/** Rotate `bone` so the segment leaving it points along `worldDir`.
 *
 * Twist is not preserved — a mannequin limb is rotationally symmetric, so there
 * is nothing to preserve, and the shortest-arc rotation is the one that feels
 * like dragging the limb rather than winding it.
 *
 * `child` names which of the bone's children is the one being aimed. It defaults
 * to `CHILD_OF`, which is "whichever child was declared first" — right for every
 * bone with one child, and wrong for the two that have several: aiming `hips`
 * would swing the *spine* when what the user grabbed was a thigh. Callers that
 * know which child they mean pass it. */
export function aimBone(
  rig: Rig,
  bone: string,
  worldDir: THREE.Vector3,
  child: string | undefined = CHILD_OF[bone]
): void {
  if (!child) return;
  const rest = OFFSETS[child];
  if (!rest || rest.lengthSq() < 1e-8) return;

  const obj = rig.bones[bone];
  const parent = obj.parent;
  if (!parent) return;

  // Take the target into the bone's own parent space, so the quaternion we write
  // composes correctly with everything above it in the chain.
  parent.getWorldQuaternion(_pq);
  _v.copy(worldDir).normalize().applyQuaternion(_pq.invert());
  if (_v.lengthSq() < 1e-8) return;
  obj.quaternion.copy(_q.setFromUnitVectors(rest.clone().normalize(), _v));
}

/**
 * Two-bone IK: rotate `root` and `mid` so `end` lands on (or as near as it can
 * reach to) `target`.
 *
 * `poleWorld` disambiguates the plane. Without it the solve is free to bend a
 * knee forwards or an elbow backwards — both satisfy the distance constraint
 * equally, and the maths has no idea one of them is a human being.
 *
 * Rather than deriving a signed rotation about an axis (where the sign is easy to
 * get backwards and hard to notice), this builds an orthonormal basis in the bend
 * plane and reads the upper bone's direction straight off it. Bending *toward*
 * the pole is then true by construction.
 */
export function solveTwoBoneIK(
  rig: Rig,
  chain: { root: string; mid: string; end: string },
  target: THREE.Vector3,
  poleWorld: THREE.Vector3
): void {
  const rootObj = rig.bones[chain.root];
  const midObj = rig.bones[chain.mid];

  const a = rootObj.getWorldPosition(new THREE.Vector3());
  const b = midObj.getWorldPosition(new THREE.Vector3());
  const c = rig.bones[chain.end].getWorldPosition(new THREE.Vector3());

  const l1 = a.distanceTo(b);
  const l2 = b.distanceTo(c);
  if (l1 < 1e-6 || l2 < 1e-6) return;

  const toTarget = new THREE.Vector3().subVectors(target, a);
  let d = toTarget.length();
  if (d < 1e-6) return;
  const dir = toTarget.clone().divideScalar(d);

  // Clamp into the reachable annulus. Just short of full extension, because at
  // exactly l1+l2 the bend plane becomes undefined and the limb snaps flat.
  const dMin = Math.abs(l1 - l2) + 1e-4;
  const dMax = l1 + l2 - 1e-4;
  d = Math.min(Math.max(d, dMin), dMax);

  // The in-plane axis perpendicular to `dir`, on the pole's side.
  const pole = new THREE.Vector3().subVectors(poleWorld, a);
  pole.addScaledVector(dir, -pole.dot(dir));
  if (pole.lengthSq() < 1e-8) {
    // Pole is parallel to the reach direction: pick any perpendicular. Rare, and
    // only reachable by dragging a hand exactly along the elbow's bend axis.
    pole.set(0, 1, 0).addScaledVector(dir, -dir.y);
    if (pole.lengthSq() < 1e-8) pole.set(1, 0, 0).addScaledVector(dir, -dir.x);
  }
  pole.normalize();

  const cosA = Math.min(Math.max((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1), 1);
  const angle = Math.acos(cosA);

  const upperDir = dir
    .clone()
    .multiplyScalar(Math.cos(angle))
    .addScaledVector(pole, Math.sin(angle));

  aimBone(rig, chain.root, upperDir);

  // The mid bone has moved with the root, so its world position is only known
  // after that update. Force it rather than waiting for the next frame's
  // traversal, or the forearm aims at where the elbow used to be.
  rootObj.updateWorldMatrix(true, true);
  const b2 = midObj.getWorldPosition(new THREE.Vector3());
  const lowerDir = new THREE.Vector3().subVectors(target, b2);
  if (lowerDir.lengthSq() > 1e-8) aimBone(rig, chain.mid, lowerDir.normalize());
}

/**
 * Where a screen drag lands in 3-D.
 *
 * A cursor is two numbers and a joint needs three, so the missing one has to come
 * from somewhere. It comes from the plane through the joint's current position
 * facing the camera: drag moves the joint across the view, never toward or away
 * from it. That is the behaviour that makes orbiting the camera and then dragging
 * feel like posing a physical figure — you turn the model to reach the axis you
 * want, then move along it.
 */
export function dragPlanePoint(
  ndc: THREE.Vector2,
  camera: THREE.Camera,
  through: THREE.Vector3,
  out = new THREE.Vector3()
): THREE.Vector3 {
  const normal = camera.getWorldDirection(new THREE.Vector3()).negate();
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, through);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  return ray.ray.intersectPlane(plane, out) ?? out.copy(through);
}
