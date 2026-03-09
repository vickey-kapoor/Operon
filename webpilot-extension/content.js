/**
 * WebPilot content script — DOM action executor.
 *
 * Listens for EXECUTE_ACTION messages from the background service worker
 * and performs the requested action in the page context.
 */

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "EXECUTE_ACTION") return false;

  const { action } = msg;
  executeAction(action)
    .then(() => sendResponse({ success: true }))
    .catch((err) => sendResponse({ success: false, error: err.message }));

  return true; // async response
});

async function executeAction(action) {
  const type = action.action;

  if (type === "click") {
    const x = action.x ?? action.target_x;
    const y = action.y ?? action.target_y;
    const el = deepElementFromPoint(x, y);
    if (el) {
      dispatchMouseEvents(el, x, y);
    }

  } else if (type === "type") {
    const x = action.x ?? action.target_x;
    const y = action.y ?? action.target_y;
    const el = deepElementFromPoint(x, y) || document.activeElement;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) {
      setNativeValue(el, action.text || action.value || "");
    }

  } else if (type === "scroll") {
    const direction = action.direction || "down";
    const amount = action.amount || 400;
    window.scrollBy(0, direction === "down" ? amount : -amount);

  } else if (type === "wait") {
    await sleep(action.duration_ms || 1000);

  } else if (type === "key") {
    const el = document.activeElement || document.body;
    el.dispatchEvent(new KeyboardEvent("keydown", { key: action.key, bubbles: true }));
    el.dispatchEvent(new KeyboardEvent("keyup", { key: action.key, bubbles: true }));
  }
}

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

/**
 * elementFromPoint that pierces shadow DOM recursively.
 */
function deepElementFromPoint(x, y) {
  let el = document.elementFromPoint(x, y);
  while (el && el.shadowRoot) {
    const inner = el.shadowRoot.elementFromPoint(x, y);
    if (!inner || inner === el) break;
    el = inner;
  }
  return el;
}

function dispatchMouseEvents(el, x, y) {
  const opts = { bubbles: true, cancelable: true, clientX: x, clientY: y };
  el.dispatchEvent(new MouseEvent("mousedown", opts));
  el.dispatchEvent(new MouseEvent("mouseup", opts));
  el.dispatchEvent(new MouseEvent("click", opts));
}

/**
 * Set value on an input using React's native setter so onChange fires.
 */
function setNativeValue(el, value) {
  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value"
  );
  const nativeTextareaSetter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    "value"
  );
  const setter =
    el.tagName === "TEXTAREA"
      ? nativeTextareaSetter && nativeTextareaSetter.set
      : nativeInputValueSetter && nativeInputValueSetter.set;

  if (setter) {
    setter.call(el, value);
  } else if (el.isContentEditable) {
    el.textContent = value;
  } else {
    el.value = value;
  }

  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
