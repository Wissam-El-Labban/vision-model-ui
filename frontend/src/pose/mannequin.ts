/** Builds the posable figure, and reads keypoints back off it.
 *
 * The body is derived from the rig rather than modelled separately: every bone
 * draws a capsule reaching back to its parent's origin. That means the mesh can't
 * drift out of agreement with the skeleton — a pose map and a depth map rendered
 * from the same frame describe the same body, which is the entire premise of
 * feeding both to the model.
 */
import * as THREE from "three";
import {
  BONES,
  OPENPOSE_BONES,
  type PropDef,
} from "./skeleton";

export interface Rig {
  /** Everything that moves with the figure. */
  root: THREE.Group;
  /** Bone name -> its Object3D. Rotating one rotates everything below it. */
  bones: Record<string, THREE.Object3D>;
  /** The capsules, so a render pass can swap their material. */
  meshes: THREE.Mesh[];
}

/** Rest offsets, kept so a bone can be aimed without re-deriving them. */
export const OFFSETS: Record<string, THREE.Vector3> = Object.fromEntries(
  BONES.map((b) => [b.name, new THREE.Vector3(...b.offset)])
);

const PARENT: Record<string, string | null> = Object.fromEntries(
  BONES.map((b) => [b.name, b.parent])
);

/** The bone whose origin the capsule reaches toward — i.e. this bone's child in
 *  the chain. Used to aim a bone at a point: "point the forearm at my cursor"
 *  means "rotate it so the wrist lands there". */
export const CHILD_OF: Record<string, string | undefined> = (() => {
  const out: Record<string, string | undefined> = {};
  for (const b of BONES) {
    // First child wins. The only bones with several are chest (two clavicles) and
    // hips (two thighs), and neither is aimed — they're rotated as a body part.
    if (b.parent && !out[b.parent]) out[b.parent] = b.name;
  }
  return out;
})();

export function buildRig(): Rig {
  const bones: Record<string, THREE.Object3D> = {};
  const meshes: THREE.Mesh[] = [];
  const material = new THREE.MeshStandardMaterial({
    color: 0xb9bec7,
    roughness: 0.85,
    metalness: 0.0,
  });

  for (const def of BONES) {
    const obj = new THREE.Object3D();
    obj.name = def.name;
    obj.position.set(...def.offset);
    bones[def.name] = obj;
    if (def.parent) bones[def.parent].add(obj);
  }

  // One capsule per bone, spanning from its parent's origin to its own — so it is
  // parented to the *parent* and inherits the parent's rotation, which is what
  // makes a limb swing as one piece.
  for (const def of BONES) {
    if (!def.parent || def.noMesh) continue;
    const offset = new THREE.Vector3(...def.offset);
    const len = offset.length();
    if (len < 1e-4) continue;
    const capsule = new THREE.Mesh(
      new THREE.CapsuleGeometry(def.radius, Math.max(len - def.radius * 0.6, 0.01), 4, 12),
      material
    );
    // Capsules are born along +Y; rotate to lie along the offset, then slide to
    // its midpoint.
    capsule.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      offset.clone().normalize()
    );
    capsule.position.copy(offset).multiplyScalar(0.5);
    bones[def.parent].add(capsule);
    meshes.push(capsule);
  }

  // The torso reads as a stack of thin sausages without this — a depth map of a
  // person has a solid trunk, and the model notices when it doesn't.
  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.145, 0.2, 4, 14),
    material
  );
  torso.scale.set(1.15, 1, 0.72);
  torso.position.set(0, 0.16, 0);
  bones.spine.add(torso);
  meshes.push(torso);

  const pelvis = new THREE.Mesh(new THREE.CapsuleGeometry(0.125, 0.06, 4, 14), material);
  pelvis.scale.set(1.15, 1, 0.78);
  pelvis.position.set(0, -0.01, 0);
  bones.hips.add(pelvis);
  meshes.push(pelvis);

  const root = new THREE.Group();
  root.add(bones.root);
  return { root, bones, meshes };
}

/** Reset every bone to the rest pose the rig was built in. */
export function resetRig(rig: Rig): void {
  for (const def of BONES) {
    rig.bones[def.name].quaternion.identity();
    rig.bones[def.name].position.set(...def.offset);
  }
  rig.root.position.set(0, 0, 0);
  rig.root.rotation.set(0, 0, 0);
}

/** World position of a bone's origin. */
export function boneWorld(rig: Rig, name: string, out = new THREE.Vector3()): THREE.Vector3 {
  return rig.bones[name].getWorldPosition(out);
}

/** The 18 OpenPose keypoints in world space.
 *
 * The face has no bones — a mannequin with modelled eyes would be a different
 * project — so nose, eyes and ears are placed off the head's own orientation.
 * They matter: without them a pose model reads the head as present but facing
 * nowhere, and generated faces end up pointing away from the body. */
export function openposeWorld(rig: Rig): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  for (const bone of OPENPOSE_BONES) {
    pts.push(bone ? boneWorld(rig, bone) : new THREE.Vector3());
  }

  const head = rig.bones.head;
  const headPos = head.getWorldPosition(new THREE.Vector3());
  const q = head.getWorldQuaternion(new THREE.Quaternion());
  const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(q); // face direction
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q); // figure's left

  const face = headPos.clone().addScaledVector(up, 0.09);
  pts[0] = face.clone().addScaledVector(fwd, 0.105); // nose
  pts[14] = face.clone().addScaledVector(fwd, 0.075).addScaledVector(right, -0.032); // R eye
  pts[15] = face.clone().addScaledVector(fwd, 0.075).addScaledVector(right, 0.032); // L eye
  pts[16] = face.clone().addScaledVector(fwd, 0.005).addScaledVector(right, -0.075); // R ear
  pts[17] = face.clone().addScaledVector(fwd, 0.005).addScaledVector(right, 0.075); // L ear
  return pts;
}

/** Serialise the rig's rotations, for save/load. */
export function readPose(rig: Rig): Record<string, [number, number, number, number]> {
  const out: Record<string, [number, number, number, number]> = {};
  for (const def of BONES) {
    const q = rig.bones[def.name].quaternion;
    out[def.name] = [q.x, q.y, q.z, q.w];
  }
  return out;
}

export function applyPose(
  rig: Rig,
  rotations: Record<string, [number, number, number, number]>
): void {
  for (const [name, q] of Object.entries(rotations)) {
    // Skip unknown bones rather than throwing: a pose saved by an older build is
    // still mostly this pose, and losing a wrist beats losing the file.
    if (rig.bones[name]) rig.bones[name].quaternion.set(q[0], q[1], q[2], q[3]);
  }
}

/** Parent bone name, for walking a chain upward. */
export function parentOf(name: string): string | null {
  return PARENT[name] ?? null;
}

// --------------------------------------------------------------------------- #
// Props
// --------------------------------------------------------------------------- #
/** Build the mesh for one prop. Primitives on purpose: this is a blockout, and a
 *  blockout is all a depth map needs to say "the weight is on this". */
export function buildProp(def: PropDef, material: THREE.Material): THREE.Mesh {
  const [w, h, d] = def.size;
  let geo: THREE.BufferGeometry;
  if (def.kind === "box") geo = new THREE.BoxGeometry(w, h, d);
  else if (def.kind === "cylinder") geo = new THREE.CylinderGeometry(w / 2, w / 2, h, 24);
  else geo = new THREE.SphereGeometry(w / 2, 20, 14);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.set(...def.position);
  mesh.rotation.y = def.rotationY;
  mesh.userData.propId = def.id;
  return mesh;
}
