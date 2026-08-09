/** Starting poses.
 *
 * Deliberately a short list of unambiguous ones rather than a yoga library: these
 * are hand-authored as joint rotations, and a pose I can't see while writing it is
 * a pose I can't claim is correct. They exist to save the first thirty seconds of
 * every session — the interesting poses are the ones you build and save.
 *
 * Rotations are Euler XYZ in degrees, converted on apply. Euler is the wrong
 * format to *store* a pose in (see `PoseFile`, which uses quaternions) but the
 * right one to hand-write, because a human can read "left elbow bent 90°".
 */
import * as THREE from "three";
import type { PropDef } from "./skeleton";

export interface Preset {
  name: string;
  hint: string;
  /** bone -> [x, y, z] degrees */
  euler: Record<string, [number, number, number]>;
  /** Where the figure stands. Rotating the legs folds them but doesn't lower the
   *  body — without this a "sitting" preset sits in mid-air above its own chair,
   *  and the depth map shows a floating person next to a box. */
  root?: [number, number, number];
  props?: PropDef[];
}

const box = (
  id: string,
  position: [number, number, number],
  size: [number, number, number]
): PropDef => ({ id, kind: "box", position, size, rotationY: 0 });

export const PRESETS: Preset[] = [
  {
    name: "T-pose",
    hint: "The rest pose. Arms straight out — the neutral start for building anything.",
    euler: {
      upperArmR: [0, 0, 90],
      upperArmL: [0, 0, -90],
    },
  },
  {
    name: "Standing",
    hint: "Relaxed, arms down. Good base for a portrait or a walking pose.",
    euler: {
      upperArmR: [0, 0, 8],
      upperArmL: [0, 0, -8],
      foreArmR: [10, 0, 0],
      foreArmL: [10, 0, 0],
    },
  },
  {
    name: "Arms overhead",
    hint: "Reaching up. The base for a stretch, a jump, or an overhead bind.",
    euler: {
      upperArmR: [0, 0, 168],
      upperArmL: [0, 0, -168],
      chest: [-6, 0, 0],
    },
  },
  {
    name: "Sitting on a box",
    hint: "Seated with the weight on a prop — the case a bare skeleton can't express. Move the box to match a real chair.",
    euler: {
      thighR: [-88, 0, 2],
      thighL: [-88, 0, -2],
      shinR: [85, 0, 0],
      shinL: [85, 0, 0],
      footR: [5, 0, 0],
      footL: [5, 0, 0],
      upperArmR: [0, 0, 12],
      upperArmL: [0, 0, -12],
      foreArmR: [22, 0, 0],
      foreArmL: [22, 0, 0],
    },
    // Hips down onto the 0.46 m seat: the thigh joint sits at 0.90 in the rest
    // pose, and wants to be just above the box's top face.
    root: [0, -0.42, 0.08],
    props: [box("seat", [0, 0.23, -0.06], [0.46, 0.46, 0.46])],
  },
  {
    name: "Kneeling",
    hint: "Knees down, torso upright. Base for a lunge or a floor pose.",
    euler: {
      thighR: [-8, 0, 3],
      thighL: [-8, 0, -3],
      shinR: [138, 0, 0],
      shinL: [138, 0, 0],
      footR: [40, 0, 0],
      footL: [40, 0, 0],
      upperArmR: [0, 0, 10],
      upperArmL: [0, 0, -10],
    },
    // Knees to the floor. They rest at 0.465 with the root at origin.
    root: [0, -0.40, 0],
  },
  {
    name: "Crouch, hands down",
    hint: "Squatting with the palms on the floor — the entry for crow, a handstand kick-up, or a sprint start.",
    euler: {
      // A standing forward fold cannot put the palms down, and it is worth
      // recording why rather than rediscovering it: folding at the hips leaves
      // the shoulders at roughly hip height (~0.95 m) while the arm is only
      // 0.54 m, so the hands stop around shin level however far you fold. Only
      // dropping the hips brings the floor into reach. These angles were solved
      // for, not guessed — at hips 0.34 m the shoulders land at 0.43 m and the
      // hands at 0.06 m, with the knees still clear of the ground at 0.14 m.
      hips: [74, 0, 0],
      spine: [6, 0, 0],
      chest: [4, 0, 0],
      neck: [-40, 0, 0],
      head: [-30, 0, 0],
      thighR: [-140, 0, 5],
      thighL: [-140, 0, -5],
      shinR: [148, 0, 0],
      shinL: [148, 0, 0],
      upperArmR: [0, 0, 4],
      upperArmL: [0, 0, -4],
    },
    root: [0, -0.6, 0],
  },
];

/** Reach the arms straight down to the ground from wherever the shoulders ended
 *  up. Hand-authored Euler angles can't express "until it touches", so the poses
 *  that need contact ask for it by name and the studio solves it on load. */
export const GROUND_REACH: Record<string, ("hands" | "feet")[]> = {
  // Feet first: planting them settles the crouch, and only then is it meaningful
  // to ask where the hands land.
  "Crouch, hands down": ["feet", "hands"],
};

/** Convert a preset's degrees into the quaternion form a pose is stored in. */
export function presetRotations(
  preset: Preset
): Record<string, [number, number, number, number]> {
  const out: Record<string, [number, number, number, number]> = {};
  const e = new THREE.Euler();
  const q = new THREE.Quaternion();
  for (const [bone, deg] of Object.entries(preset.euler)) {
    e.set(
      THREE.MathUtils.degToRad(deg[0]),
      THREE.MathUtils.degToRad(deg[1]),
      THREE.MathUtils.degToRad(deg[2]),
      "XYZ"
    );
    q.setFromEuler(e);
    out[bone] = [q.x, q.y, q.z, q.w];
  }
  return out;
}
