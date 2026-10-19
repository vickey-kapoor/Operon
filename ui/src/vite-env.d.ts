/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Must match one of the API's API_KEYS; leave unset when auth is off. */
  readonly VITE_OPERON_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
