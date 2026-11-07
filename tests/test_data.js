"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function page() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: "run", open: true, disabled: false, hidden: false, attributes: {},
      addEventListener() {}, pause() {}, load() {}, play: async () => {},
      getAttribute(name) { return this.attributes[name]; },
      removeAttribute(name) { delete this.attributes[name]; },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: {documentElement: {dataset: {basePath: "/lego-robot"}}, getElementById: element},
    fetch: () => new Promise(() => {}),
    setTimeout: callback => callback(), clearTimeout() {}, Date, console, URLSearchParams,
  });
  vm.runInContext(fs.readFileSync(
    process.env.DATA_BROWSER_SOURCE || path.join(__dirname, "../src/so101/dashboard/data.js"),
    "utf8"), context);
  return {context, element, run: code => vm.runInContext(code, context)};
}

const response = (status, body, type = "application/json", redirected = false) => ({
  ok: status >= 200 && status < 300, status, redirected,
  headers: {get: () => type}, json: async () => body,
});

test("HTML gateway and sign-in responses are not parsed as JSON", async () => {
  const p = page();
  p.context.fetch = async () => ({
    ...response(504, null, "text/html"),
    json: () => assert.fail("Must not parse HTML"),
  });
  await assert.rejects(p.run('get("/test")'), /Gateway returned HTTP 504 instead of JSON/);
  p.context.fetch = async () => response(200, null, "text/html", true);
  await assert.rejects(p.run('get("/test")'), /Sign-in may have expired/);
});

test("JSON renderer errors retain their meaningful message", async () => {
  const p = page();
  p.context.fetch = async () => response(500, {error: "Playback rendering exceeded its 180-second limit"});
  await assert.rejects(p.run('get("/test")'), /180-second limit/);
});

test("playback polls pending JSON before loading the exact selected recording", async () => {
  const p = page();
  const routes = [];
  const replies = [
    response(202, {ready: false, status: "preparing"}),
    response(202, {ready: false, status: "waiting"}),
    response(200, {ready: true, status: "ready"}),
  ];
  p.context.fetch = async route => { routes.push(route); return replies.shift(); };
  p.run('detail = {id: "e0b1925f38c0f55b2e70", view: "examples", movie_available: true, frames: [1], playback_description: "Views"};');
  await p.run("playMovie()");
  assert.equal(routes.length, 3);
  assert.ok(routes.every(route => route.includes("item=e0b1925f38c0f55b2e70&movie=prepare")));
  assert.ok(p.element("detail-movie").src.endsWith("item=e0b1925f38c0f55b2e70&movie=1"));
  assert.equal(p.element("play-movie").disabled, false);
});

test("closing the detail stops polling without starting a movie", async () => {
  const p = page();
  let calls = 0;
  p.context.fetch = async () => {
    calls++;
    p.element("detail").open = false;
    return response(202, {ready: false, status: "preparing"});
  };
  p.run('detail = {id: "original", view: "examples", movie_available: true};');
  await p.run("playMovie()");
  assert.equal(calls, 1);
  assert.equal(p.element("detail-movie").src, undefined);
});
