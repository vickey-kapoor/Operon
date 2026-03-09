import { useCallback, useEffect, useRef, useState } from "react";

const MAX_RESTARTS = 3;

export function useVoiceInput(onTranscript) {
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState(null);
  const recognitionRef = useRef(null);
  const isListeningRef = useRef(false);
  const restartCountRef = useRef(0);

  const SpeechRecognition =
    typeof window !== "undefined" &&
    (window.SpeechRecognition || window.webkitSpeechRecognition);

  const supported = Boolean(SpeechRecognition);

  useEffect(() => {
    if (!SpeechRecognition) {
      console.warn("[WebPilot] SpeechRecognition not available");
      return;
    }

    const r = new SpeechRecognition();
    r.continuous = false;
    r.interimResults = false;
    r.lang = "en-US";
    r.maxAlternatives = 1;

    r.onstart = () => {
      console.log("[WebPilot] Voice: started");
      setError(null);
    };

    r.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      const confidence = (event.results[0][0].confidence * 100).toFixed(0);
      console.log(`[WebPilot] Voice: "${transcript}" (${confidence}%)`);
      onTranscript(transcript);
    };

    r.onerror = (event) => {
      console.error("[WebPilot] Voice error:", event.error);
      isListeningRef.current = false;
      restartCountRef.current = 0;
      setIsListening(false);

      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        setError("mic-denied");
      } else {
        setError(event.error);
      }
    };

    r.onend = () => {
      console.log(`[WebPilot] Voice: ended (restarts: ${restartCountRef.current})`);
      if (isListeningRef.current && restartCountRef.current < MAX_RESTARTS) {
        restartCountRef.current++;
        try { r.start(); } catch {}
      } else {
        isListeningRef.current = false;
        restartCountRef.current = 0;
        setIsListening(false);
      }
    };

    recognitionRef.current = r;
    return () => { r.abort(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const startListening = useCallback(() => {
    if (!recognitionRef.current || isListeningRef.current) return;
    console.log("[WebPilot] Voice: start");
    isListeningRef.current = true;
    restartCountRef.current = 0;
    setIsListening(true);
    setError(null);
    try {
      recognitionRef.current.start();
    } catch (err) {
      console.error("[WebPilot] Voice: start threw:", err.message);
      isListeningRef.current = false;
      setIsListening(false);
    }
  }, []);

  const stopListening = useCallback(() => {
    if (!recognitionRef.current) return;
    console.log("[WebPilot] Voice: stop");
    isListeningRef.current = false;
    restartCountRef.current = 0;
    setIsListening(false);
    try { recognitionRef.current.stop(); } catch {}
  }, []);

  return { isListening, startListening, stopListening, supported, error };
}
