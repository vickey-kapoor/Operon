// GET /api/blob?url=<blob url>: streams one benchmark screenshot or trace from the
// private Vercel Blob store, so the page never holds the store's token. Only
// bench/ files of the store that BLOB_READ_WRITE_TOKEN belongs to are served.
import { Readable } from "node:stream";
import { get } from "@vercel/blob";
import { sendJson } from "../server/db.js";

// The token's fourth "_" field is the store id, which is also the store's host name.
function storeHost() {
  const storeId = (process.env.BLOB_READ_WRITE_TOKEN || "").split("_")[3];
  return storeId ? `${storeId.toLowerCase()}.private.blob.vercel-storage.com` : null;
}

export function allowedBlobUrl(raw) {
  try {
    const url = new URL(raw);
    return url.protocol === "https:" && url.hostname === storeHost() && url.pathname.startsWith("/bench/") ? url : null;
  } catch {
    return null;
  }
}

export default async function handler(req, res) {
  const url = allowedBlobUrl(new URL(req.url, "http://localhost").searchParams.get("url"));
  if (!url) return sendJson(res, 400, { error: "url must be a bench/ file in the private blob store" });
  let result;
  try {
    result = await get(url.toString(), { access: "private" });
  } catch (err) {
    console.error(err);
    return sendJson(res, 502, { error: "Could not read the blob store" });
  }
  if (!result || result.statusCode !== 200) return sendJson(res, 404, { error: "No such file" });
  res.statusCode = 200;
  res.setHeader("Content-Type", result.blob.contentType || "application/octet-stream");
  // Names carry a random suffix, so a file never changes; "private" keeps it out of shared caches.
  res.setHeader("Cache-Control", "private, max-age=604800, immutable");
  if (url.pathname.endsWith(".zip")) {
    res.setHeader("Content-Disposition", `attachment; filename="${url.pathname.split("/").slice(-2).join("-")}"`);
  }
  Readable.fromWeb(result.stream).pipe(res);
}
