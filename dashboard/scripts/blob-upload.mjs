// Uploads benchmark screenshots and traces to Vercel Blob for evals/dashboard.py.
// Reads a JSON list of {file, pathname} on stdin and prints the public URLs, in the
// same order, as a JSON list. BLOB_READ_WRITE_TOKEN comes from the environment.
// The store is private: nothing is readable without the token, and the dashboard
// serves the files through its own /api/blob route.
import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { put } from "@vercel/blob";

const TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".zip": "application/zip", ".json": "application/json" };
const PARALLEL = 4;

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

const plan = await readStdin();
const urls = new Array(plan.length);
let next = 0;
async function worker() {
  while (next < plan.length) {
    const i = next++;
    const { file, pathname } = plan[i];
    const blob = await put(pathname, await readFile(file), {
      access: "private",
      addRandomSuffix: true,
      contentType: TYPES[extname(file).toLowerCase()] || "application/octet-stream",
    });
    urls[i] = blob.url;
  }
}
await Promise.all(Array.from({ length: Math.min(PARALLEL, plan.length) }, worker));
process.stdout.write(JSON.stringify(urls));
