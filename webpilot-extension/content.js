/**
 * WebPilot content script — DOM action executor.
 *
 * Listens for EXECUTE_ACTION messages from the background service worker
 * and performs the requested action in the page context.
 */

// Re-injection guard: prevent duplicate listeners when content script is re-injected.
if (\!window.__webpilot_content_loaded) {
window.__webpilot_content_loaded = true;

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "WAIT_STABLE") {
    waitForPageStable(msg.timeout || 5000)
      .then(() => sendResponse({ stable: true }))
      .catch(() => sendResponse({ stable: false }));
    return true;
  }

  if (msg.type !== "EXECUTE_ACTION") return false;

  const { action } = msg;
  executeAction(action)
    .then(() => sendResponse({ success: true }))
    .catch((err) => sendResponse({ success: false, error: err.message }));

  return true; // async response
});

} // end re-injection guard

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
    let el = (x \!= null && y \!= null) ? deepElementFromPoint(x, y) : null;
    if (\!el) el = document.activeElement;
    if (el && el \!== document.body) {
      // Focus the element first so keystrokes land in the right place.
      el.focus();
      dispatchMouseEvents(el, x || 0, y || 0);
      if (el.isContentEditable) {
        document.execCommand("selectAll", false, null);
        document.execCommand("insertText", false, action.text || action.value || "");
      } else if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
        setNativeValue(el, action.text || action.value || "");
      }
    }

  } else if (type === "scroll") {
    const direction = action.direction || "down";
    const amount = action.amount || 400;
    if (direction === "down") window.scrollBy(0, amount);
    else if (direction === "up") window.scrollBy(0, -amount);
    else if (direction === "right") window.scrollBy(amount, 0);
    else if (direction === "left") window.scrollBy(-amount, 0);

  } else if (type === "wait") {
    await sleep(action.duration_ms || 1000);

  } else if (type === "key") {
    const el = document.activeElement || document.body;
    const modifiers = {
      key: action.key,
      bubbles: true,
      ctrlKey: \!\!action.ctrlKey,
      shiftKey: \!\!action.shiftKey,
      altKey: \!\!action.altKey,
      metaKey: \!\!action.metaKey,
    };
    el.dispatchEvent(new KeyboardEvent("keydown", modifiers));
    el.dispatchEvent(new KeyboardEvent("keyup", modifiers));
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

/**
 * Wait until the DOM stops mutating for 500ms (quiet period), or until
 * a hard cap (default 5s) is reached — whichever comes first.
 */
function waitForPageStable(hardCapMs = 5000) {
  const QUIET_PERIOD_MS = 500;
  return new Promise((resolve) => {
    let timer = null;
    const hardCap = setTimeout(() => {
      observer.disconnect();
      if (timer) clearTimeout(timer);
      resolve();
    }, hardCapMs);

    const observer = new MutationObserver(() => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        observer.disconnect();
        clearTimeout(hardCap);
        resolve();
      }, QUIET_PERIOD_MS);
    });

    observer.observe(document.body || document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
    });

    // If no mutations happen at all, resolve after the quiet period
    timer = setTimeout(() => {
      observer.disconnect();
      clearTimeout(hardCap);
      resolve();
    }, QUIET_PERIOD_MS);
  });
}
