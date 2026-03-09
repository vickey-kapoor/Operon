chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== "EXECUTE_ACTION") return;
  executeAction(msg.action)
    .then((r) => sendResponse({ ok: true, result: r }))
    .catch((e) => sendResponse({ ok: false, error: e.message }));
  return true;
});

async function executeAction(a) {
  if (a.type === "click") {
    const [x, y] = a.coordinate || [0, 0];
    const el = deepElementFromPoint(x, y);
    if (el) {
      el.focus();
      // Full mouse event sequence — works with React, cookie banners, etc.
      for (const t of ["mousedown", "mouseup", "click"]) {
        el.dispatchEvent(new MouseEvent(t, {
          bubbles: true, cancelable: true, view: window, clientX: x, clientY: y,
        }));
      }
    }
    return `clicked (${x},${y})`;
  }

  if (a.type === "type") {
    // If coordinate provided, focus that element first.
    if (a.coordinate) {
      const [x, y] = a.coordinate;
      const el = deepElementFromPoint(x, y);
      el?.focus();
    }
    const el = document.activeElement;
    if (!el || el === document.body) return "type: no focused element";

    if (el.isContentEditable) {
      el.textContent += a.text;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      // Native value setter approach — works with React controlled inputs.
      const nativeSetter =
        Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set ||
        Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
      if (nativeSetter) {
        nativeSetter.call(el, (el.value || "") + a.text);
      } else {
        el.value = (el.value || "") + a.text;
      }
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
    return `typed "${a.text}"`;
  }

  if (a.type === "key") {
    const kmap = { Enter:13, Tab:9, Escape:27, Backspace:8, ArrowDown:40, ArrowUp:38, ArrowLeft:37, ArrowRight:39 };
    const el = document.activeElement || document.body;
    for (const t of ["keydown","keypress","keyup"]) {
      el.dispatchEvent(new KeyboardEvent(t, {
        bubbles: true, cancelable: true, key: a.key, keyCode: kmap[a.key] || 0,
      }));
    }
    // For Enter on forms, also try submitting the closest form.
    if (a.key === "Enter" && el.form) el.form.requestSubmit?.();
    return `key ${a.key}`;
  }

  if (a.type === "scroll") {
    const amt = (a.scroll_amount || 3) * 100;
    const dx = a.scroll_direction === "left" ? -amt : a.scroll_direction === "right" ? amt : 0;
    const dy = a.scroll_direction === "up" ? -amt : a.scroll_direction === "down" ? amt : 0;
    window.scrollBy(dx, dy);
    return `scrolled`;
  }

  if (a.type === "navigate") { window.location.href = a.url; return `navigating`; }
  if (a.type === "wait") { await new Promise(r => setTimeout(r, a.duration_ms || 1000)); return `waited`; }
  return `noop(${a.type})`;
}

// Walk into shadow roots to find the real element at (x, y).
function deepElementFromPoint(x, y) {
  let el = document.elementFromPoint(x, y);
  while (el?.shadowRoot) {
    const inner = el.shadowRoot.elementFromPoint(x, y);
    if (!inner || inner === el) break;
    el = inner;
  }
  return el;
}
