import assert from "node:assert/strict";
import { test } from "node:test";
import { refuseUnlessAuthorized } from "../server/auth.js";

const basic = (user, pass) => `Basic ${Buffer.from(`${user}:${pass}`).toString("base64")}`;

test("fails closed when no password is configured", () => {
  assert.equal(refuseUnlessAuthorized(basic("a", "b"), undefined).status, 503);
  assert.equal(refuseUnlessAuthorized(null, "").status, 503);
});

test("asks for credentials when none are sent", () => {
  const res = refuseUnlessAuthorized(null, "s3cret");
  assert.equal(res.status, 401);
  assert.match(res.headers.get("WWW-Authenticate"), /^Basic /);
});

test("refuses a wrong password and malformed headers", () => {
  assert.equal(refuseUnlessAuthorized(basic("vickey", "nope"), "s3cret").status, 401);
  assert.equal(refuseUnlessAuthorized("Bearer s3cret", "s3cret").status, 401);
  assert.equal(refuseUnlessAuthorized("Basic !!!", "s3cret").status, 401);
  assert.equal(refuseUnlessAuthorized(`Basic ${Buffer.from("s3cret").toString("base64")}`, "s3cret").status, 401);
});

test("lets the right password through with any user name", () => {
  assert.equal(refuseUnlessAuthorized(basic("vickey", "s3cret"), "s3cret"), null);
  assert.equal(refuseUnlessAuthorized(basic("", "pa:ss"), "pa:ss"), null);
});
