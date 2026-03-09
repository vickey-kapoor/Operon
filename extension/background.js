chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "CAPTURE_SCREENSHOT") {
    chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
      sendResponse(chrome.runtime.lastError
        ? { error: chrome.runtime.lastError.message }
        : { dataUrl });
    });
    return true;
  }

  if (msg.type === "EXECUTE_ACTION") {
    // Handle navigate directly via tabs API — works on all pages including new tab.
    if (msg.action?.type === "navigate") {
      chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
        if (!tab) { sendResponse({ error: "No active tab" }); return; }
        chrome.tabs.update(tab.id, { url: msg.action.url }, () => {
          sendResponse({ ok: true, result: "navigating" });
        });
      });
      return true;
    }

    // All other actions go to the content script.
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (!tab) { sendResponse({ error: "No active tab" }); return; }
      chrome.tabs.sendMessage(tab.id, msg, (resp) => {
        sendResponse(chrome.runtime.lastError
          ? { error: chrome.runtime.lastError.message }
          : resp);
      });
    });
    return true;
  }
});
