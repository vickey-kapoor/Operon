// Local stand-in for Vercel: serves the page and the api/ routes on one port.
// DATABASE_URL=postgres://... npm run dev   (then open http://localhost:3000)
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };
const port = Number(process.env.PORT || 3000);

createServer(async (req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  const route = path.match(/^\/api\/([a-z]+)$/);
  if (route) {
    try {
      const { default: handler } = await import(`./api/${route[1]}.js`);
      return handler(req, res);
    } catch {
      res.statusCode = 404;
      return res.end();
    }
  }
  const file = normalize(join(root, path === "/" ? "index.html" : path));
  if (!file.startsWith(root) || /[\\/](api|server|node_modules)[\\/]/.test(file.slice(root.length - 1))) {
    res.statusCode = 404;
    return res.end();
  }
  try {
    const body = await readFile(file);
    res.setHeader("Content-Type", types[extname(file)] || "application/octet-stream");
    res.end(body);
  } catch {
    res.statusCode = 404;
    res.end();
  }
}).listen(port, () => console.log(`Eval dashboard on http://localhost:${port}`));
