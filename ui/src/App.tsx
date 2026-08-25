import { MainLayout } from "./components/MainLayout";

/** RunState is still exported here because child components import it from "../App". */
export type RunState = {
  runId: string | null;
  intent: string;
  status: string;
};

export default function App() {
  return <MainLayout />;
}
