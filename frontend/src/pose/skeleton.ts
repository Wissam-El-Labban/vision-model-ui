/** The mannequin's rig, and how it maps onto OpenPose's 18 keypoints.
 *
 * Why a rig at all, rather than a stick figure the user drags in 2-D: a 2-D
 * skeleton is ambiguous exactly where hard poses break. It cannot say which arm is
 * in front, how much a limb is foreshortened, or that a foot is resting *on* the
 * seat of a chair rather than floating at the same screen position. A posed 3-D
 * figure resolves all three, because the depth map it renders carries the answer.
 *
 * Lengths are metres on a ~1.75 m figure. They are only ever used relative to each
 * other, but keeping them life-sized means a prop sized "0.45 m box" is a chair.
 */

/** A bone: an origin that rotates, positioned relative to its parent's origin. */
export interface BoneDef {
  name: string;
  parent: string | null;
  /** Rest position in the parent's local space. Also the segment the parent's
   *  mesh spans, so the body is derived from the rig rather than modelled twice. */
  offset: [number, number, number];
  /** Radius of the capsule drawn from the parent's origin to this one. The torso
   *  is fat and the wrist is thin, which is most of what makes a depth map read as
   *  a body instead of a wire. */
  radius: number;
  /** Skip the capsule (used for pure reference points like the head's tip). */
  noMesh?: boolean;
}

/** The figure faces +Z (toward the default camera). Its own right is -X, so a
 *  keypoint named `r*` sits on the viewer's left — the same handedness OpenPose
 *  uses, which is what lets the rendered map drop into a real pose pipeline. */
export const BONES: BoneDef[] = [
  { name: "root", parent: null, offset: [0, 0, 0], radius: 0, noMesh: true },
  { name: "hips", parent: "root", offset: [0, 0.95, 0], radius: 0.13, noMesh: true },
  { name: "spine", parent: "hips", offset: [0, 0.13, 0], radius: 0.13 },
  { name: "chest", parent: "spine", offset: [0, 0.17, 0], radius: 0.14 },
  { name: "neck", parent: "chest", offset: [0, 0.19, 0], radius: 0.06 },
  { name: "head", parent: "neck", offset: [0, 0.09, 0], radius: 0.05 },
  { name: "headTop", parent: "head", offset: [0, 0.17, 0], radius: 0.105 },

  // Arms. The clavicle is a separate bone so the shoulder keypoint sits where a
  // shoulder actually is, rather than on the spine.
  { name: "clavicleR", parent: "chest", offset: [-0.05, 0.15, 0], radius: 0.05, noMesh: true },
  { name: "upperArmR", parent: "clavicleR", offset: [-0.13, 0, 0], radius: 0.055 },
  { name: "foreArmR", parent: "upperArmR", offset: [0, -0.28, 0], radius: 0.045 },
  { name: "handR", parent: "foreArmR", offset: [0, -0.26, 0], radius: 0.035 },
  { name: "handEndR", parent: "handR", offset: [0, -0.09, 0], radius: 0.03 },

  { name: "clavicleL", parent: "chest", offset: [0.05, 0.15, 0], radius: 0.05, noMesh: true },
  { name: "upperArmL", parent: "clavicleL", offset: [0.13, 0, 0], radius: 0.055 },
  { name: "foreArmL", parent: "upperArmL", offset: [0, -0.28, 0], radius: 0.045 },
  { name: "handL", parent: "foreArmL", offset: [0, -0.26, 0], radius: 0.035 },
  { name: "handEndL", parent: "handL", offset: [0, -0.09, 0], radius: 0.03 },

  { name: "thighR", parent: "hips", offset: [-0.09, -0.05, 0], radius: 0.075 },
  { name: "shinR", parent: "thighR", offset: [0, -0.44, 0], radius: 0.06 },
  { name: "footR", parent: "shinR", offset: [0, -0.42, 0], radius: 0.045 },
  { name: "toeR", parent: "footR", offset: [0, -0.04, 0.15], radius: 0.04 },

  { name: "thighL", parent: "hips", offset: [0.09, -0.05, 0], radius: 0.075 },
  { name: "shinL", parent: "thighL", offset: [0, -0.44, 0], radius: 0.06 },
  { name: "footL", parent: "shinL", offset: [0, -0.42, 0], radius: 0.045 },
  { name: "toeL", parent: "footL", offset: [0, -0.04, 0.15], radius: 0.04 },
];

/** Bones the user can grab, in the order they're listed in the UI. */
export const HANDLES = [
  "hips", "spine", "chest", "neck", "head",
  "upperArmR", "foreArmR", "handR",
  "upperArmL", "foreArmL", "handL",
  "thighR", "shinR", "footR",
  "thighL", "shinL", "footL",
] as const;

/** The two-bone chains, used to solve a limb onto a contact point.
 *
 * This is *not* the drag path. Dragging a wrist used to run IK, which solves the
 * elbow for you — convenient for "put the hand on the floor", and wrong for
 * everything else, because the shoulder and elbow both move and a pose you had
 * already settled comes apart when you nudge a hand. Drags now aim one bone and
 * leave the joint above it exactly where it was. IK survives here for the one job
 * a user can't do by hand: `GROUND_REACH`, which plants a preset's limbs on the
 * floor at load time. */
export const IK_CHAINS: Record<string, { root: string; mid: string; end: string }> = {
  handR: { root: "upperArmR", mid: "foreArmR", end: "handR" },
  handL: { root: "upperArmL", mid: "foreArmL", end: "handL" },
  footR: { root: "thighR", mid: "shinR", end: "footR" },
  footL: { root: "thighL", mid: "shinL", end: "footL" },
};

/** Which way a chain's middle joint prefers to bend, in the figure's local space.
 *  Without this an IK solve is free to put the knee through the shin — the elbow
 *  bends back, the knee bends forward, and nothing in the maths knows that. */
export const POLE_HINTS: Record<string, [number, number, number]> = {
  handR: [0, -0.3, -1],
  handL: [0, -0.3, -1],
  footR: [0, -0.3, 1],
  footL: [0, -0.3, 1],
};

/** What dragging a handle rotates: the bone above it, aimed so that this handle's
 *  own joint follows the cursor.
 *
 * A handle sits at its bone's origin, so moving that origin means rotating the
 * bone *above* it — grabbing the elbow and pulling swings the upper arm rather
 * than spinning the forearm about its own origin. `child` is spelled out rather
 * than derived because two bones have more than one child: `hips` has the spine
 * and both thighs, and defaulting to the first-declared one made the R/L hip
 * handles swing the entire torso. `hips` itself is absent — it places the figure
 * instead of rotating anything. */
export const HANDLE_AIM: Record<string, { bone: string; child: string }> = {
  spine: { bone: "hips", child: "spine" },
  chest: { bone: "spine", child: "chest" },
  neck: { bone: "chest", child: "neck" },
  head: { bone: "neck", child: "head" },
  upperArmR: { bone: "clavicleR", child: "upperArmR" },
  foreArmR: { bone: "upperArmR", child: "foreArmR" },
  handR: { bone: "foreArmR", child: "handR" },
  upperArmL: { bone: "clavicleL", child: "upperArmL" },
  foreArmL: { bone: "upperArmL", child: "foreArmL" },
  handL: { bone: "foreArmL", child: "handL" },
  thighR: { bone: "hips", child: "thighR" },
  shinR: { bone: "thighR", child: "shinR" },
  footR: { bone: "shinR", child: "footR" },
  thighL: { bone: "hips", child: "thighL" },
  shinL: { bone: "thighL", child: "shinL" },
  footL: { bone: "shinL", child: "footL" },
};

/** Human names for the handles, for the picker and the status line. */
export const HANDLE_LABELS: Record<string, string> = {
  hips: "Hips", spine: "Waist", chest: "Chest", neck: "Neck", head: "Head",
  upperArmR: "R shoulder", foreArmR: "R elbow", handR: "R hand",
  upperArmL: "L shoulder", foreArmL: "L elbow", handL: "L hand",
  thighR: "R hip", shinR: "R knee", footR: "R foot",
  thighL: "L hip", shinL: "L knee", footL: "L foot",
};

// --------------------------------------------------------------------------- #
// OpenPose BODY_18
// --------------------------------------------------------------------------- #
/** Keypoint index -> the rig bone whose *origin* is that keypoint.
 *
 * Ordering is OpenPose's, not ours, and it is not negotiable: the map is only
 * useful because every pose model in the ecosystem reads this exact layout.
 *   0 nose · 1 neck · 2-4 R arm · 5-7 L arm · 8-10 R leg · 11-13 L leg
 *   14/15 eyes · 16/17 ears
 * The face points have no bones — they're derived from the head's orientation in
 * `openposePoints`, which is enough for the model to read where the head is facing. */
export const OPENPOSE_BONES: (string | null)[] = [
  null,        // 0  nose      (derived from head)
  "neck",      // 1  neck
  "upperArmR", // 2  R shoulder
  "foreArmR",  // 3  R elbow
  "handR",     // 4  R wrist
  "upperArmL", // 5  L shoulder
  "foreArmL",  // 6  L elbow
  "handL",     // 7  L wrist
  "thighR",    // 8  R hip
  "shinR",     // 9  R knee
  "footR",     // 10 R ankle
  "thighL",    // 11 L hip
  "shinL",     // 12 L knee
  "footL",     // 13 L ankle
  null,        // 14 R eye
  null,        // 15 L eye
  null,        // 16 R ear
  null,        // 17 L ear
];

/** Limb connections, 0-indexed, in DWPose's order — the colours below are indexed
 *  by position in this list, so reordering it silently recolours every limb. */
export const BODY_LIMBS: [number, number][] = [
  [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
  [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
];

/** Head connections, drawn after the body so the colour indices continue. */
export const HEAD_LIMBS: [number, number][] = [
  [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
];

/** DWPose's palette, copied verbatim from ComfyUI's `KeypointDraw.colors`
 *  (comfy_extras/nodes_sdpose.py). A pose model reads limb identity partly from
 *  hue, so an approximation here is a different signal, not a prettier one. */
export const POSE_COLORS: [number, number, number][] = [
  [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
  [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
  [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],
];

/** One figure's pose: every bone's local rotation as a quaternion, plus where it
 *  stands. Quaternions rather than Euler angles so a reloaded pose can't
 *  gimbal-lock into a different pose than the one saved. */
export interface FigurePose {
  /** bone name -> [x, y, z, w] */
  rotations: Record<string, [number, number, number, number]>;
  rootPosition: [number, number, number];
  rootRotationY: number;
}

/** The original single-figure save format. Still in users' localStorage, so it is
 *  read (and migrated) rather than deleted. */
export interface PoseFileV1 extends FigurePose {
  version: 1;
  name: string;
  props: PropDef[];
}

/** A saved scene: every figure in it, and the props they're arranged around. */
export interface PoseFileV2 {
  version: 2;
  name: string;
  figures: FigurePose[];
  props: PropDef[];
}

export type PoseFile = PoseFileV1 | PoseFileV2;

/** Read any saved pose as the current format. Anything that isn't explicitly v2
 *  is treated as v1 — a record written by a build old enough to predate the
 *  version field is still a single-figure pose. */
export function normalizePoseFile(file: PoseFile): PoseFileV2 {
  if (file.version === 2) return file;
  const { rotations, rootPosition, rootRotationY } = file;
  return {
    version: 2,
    name: file.name,
    figures: [{ rotations, rootPosition, rootRotationY }],
    props: file.props ?? [],
  };
}

/** A block of scenery. Deliberately primitive: the point is not to model a chair,
 *  it is to put a solid where the chair is so the depth map says the figure is
 *  resting on something. */
export interface PropDef {
  id: string;
  kind: "box" | "cylinder" | "sphere";
  position: [number, number, number];
  /** Full extents in metres (diameter for cylinder/sphere). */
  size: [number, number, number];
  rotationY: number;
}
