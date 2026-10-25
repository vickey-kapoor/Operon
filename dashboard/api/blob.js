// GET /api/blob?url=<blob url>: streams one benchmark screenshot or trace from the
// private Vercel Blob store, so the page never holds the store's token. Only
// bench/ files of the project's own store are served.
import { Readable } from "node:stream";
import { get } from "@vercel/blob";
import { sendJson } from "../server/db.js";

// The store's host name is its id: BLOB_STORE_ID (set when the store is connected to
// the Vercel project; the client then signs in with the deployment's OIDC token), or
// the fourth "_" field of BLOB_READ_WRITE_TOKEN.
function storeHost() {
  const storeId = (process.env.BLOB_STORE_ID || "").trim().replace(/^store_/, "") ||
    (process.env.BLOB_READ_WRITE_TOKEN || "").split("_")[3];
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
