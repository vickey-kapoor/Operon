// Basic auth for the whole dashboard: the page, the API routes and the screenshot
// proxy. The results, screenshots and trace zips are private, so the check fails
// closed: without DASHBOARD_PASSWORD set on the Vercel project nothing is served.
// Any user name is accepted; only the password is checked.

const REALM = 'Basic realm="Operon evals", charset="UTF-8"';

function sameString(a, b) {
  // Constant-time for equal lengths, so the password can't be found a byte at a time.
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function presentedPassword(header) {
  const match = /^Basic\s+([A-Za-z0-9+/=]+)$/i.exec(header || "");
  if (!match) return null;
  let decoded;
  try {
    decoded = new TextDecoder().decode(Uint8Array.from(atob(match[1]), (c) => c.charCodeAt(0)));
  } catch {
    return null;
  }
  const colon = decoded.indexOf(":");
  return colon === -1 ? null : decoded.slice(colon + 1);
}

// Returns the response that refuses the request, or null when it may go through.
export function refuseUnlessAuthorized(authorization, password) {
  if (!password) {
    return new Response("The dashboard is locked: set DASHBOARD_PASSWORD on the Vercel project.", {
      status: 503,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" },
    });
  }
  const presented = presentedPassword(authorization);
  if (presented !== null && sameString(presented, password)) return null;
  return new Response("Sign in to see the Operon eval dashboard.", {
    status: 401,
    headers: { "WWW-Authenticate": REALM, "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" },
  });
}
