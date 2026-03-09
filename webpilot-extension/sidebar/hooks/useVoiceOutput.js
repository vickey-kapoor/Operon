import { useCallback, useEffect, useRef } from "react";

export function useVoiceOutput() {
  const supported =
    typeof window !== "undefined" && Boolean(window.speechSynthesis);

  const voiceRef = useRef(null);

  // Pick the best soft female voice once voices are loaded.
  useEffect(() => {
    if (!supported) return;

    function pickVoice() {
      const voices = window.speechSynthesis.getVoices();
      if (!voices.length) return;

      // Preference order: Google UK English Female > Microsoft Zira > any female en voice.
      const preferred = [
        "Google UK English Female",
        "Microsoft Zira - English (United States)",
        "Samantha",
        "Karen",
        "Moira",
        "Tessa",
      ];

      let picked = null;
      for (const name of preferred) {
        picked = voices.find((v) => v.name === name);
        if (picked) break;
      }

      // Fallback: any female-sounding en-US / en-GB voice.
      if (!picked) {
        picked = voices.find(
          (v) =>
            (v.lang.startsWith("en")) &&
            /female|woman|girl|zira|samantha|karen|moira/i.test(v.name)
        );
      }

      // Last resort: first English voice.
      if (!picked) {
        picked = voices.find((v) => v.lang.startsWith("en"));
      }

      voiceRef.current = picked || null;
      console.log("[WebPilot] Voice output:", voiceRef.current?.name || "default");
    }

    pickVoice();
    window.speechSynthesis.addEventListener("voiceschanged", pickVoice);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", pickVoice);
  }, [supported]);

  const speak = useCallback(
    (text) => {
      if (!supported || !text) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      if (voiceRef.current) utterance.voice = voiceRef.current;
      utterance.rate = 0.95;   // slightly slower for clarity
      utterance.pitch = 1.1;   // slightly higher = softer / more feminine
      utterance.volume = 1.0;
      console.log(`[WebPilot] Speaking: "${text}" (voice: ${utterance.voice?.name || "default"})`);
      window.speechSynthesis.speak(utterance);
    },
    [supported]
  );

  return { speak, supported };
}
