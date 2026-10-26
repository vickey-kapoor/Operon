// Vercel Routing Middleware: runs before every request, static files included.
// Returning nothing lets the request through; see server/auth.js for the check.
import { refuseUnlessAuthorized } from "./server/auth.js";

export default function middleware(request) {
  return refuseUnlessAuthorized(request.headers.get("authorization"), process.env.DASHBOARD_PASSWORD) ?? undefined;
}
