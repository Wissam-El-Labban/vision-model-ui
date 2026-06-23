/** Read a File into a data-URL string (data:image/...;base64,...). */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/** Normalize a data-URL image to a standard JPEG, downscaling so its longest
 *  side is <= maxDim. Always re-encodes via canvas (even when already small) so
 *  any browser-displayable source becomes a clean JPEG Ollama can decode — this
 *  avoids "Failed to load image" errors from unusual source encodings. Vision
 *  models also tokenize by resolution, so the downscale keeps context cheap. */
export function resizeDataUrl(dataUrl: string, maxDim = 1280): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const longest = Math.max(img.width, img.height);
      const scale = longest > maxDim ? maxDim / longest : 1;
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("no canvas ctx"));
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.9));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

/** Read a File and downscale it in one step. */
export async function fileToResizedDataUrl(file: File, maxDim = 1280): Promise<string> {
  return resizeDataUrl(await fileToDataUrl(file), maxDim);
}

/** Rotate a data-URL image by `deg` (90 increments) via canvas. */
export function rotateDataUrl(dataUrl: string, deg: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const swap = deg % 180 !== 0;
      const canvas = document.createElement("canvas");
      canvas.width = swap ? img.height : img.width;
      canvas.height = swap ? img.width : img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("no canvas ctx"));
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate((deg * Math.PI) / 180);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);
      resolve(canvas.toDataURL("image/jpeg", 0.95));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}
