import React from "react";
import ReactDOM from "react-dom/client";
import { DictationPill } from "./features/dictation/dictation-pill";
import "./app.css";
import "./styles/dictation-pill.css";

ReactDOM.createRoot(document.getElementById("pill-root")!).render(
  <React.StrictMode>
    <DictationPill />
  </React.StrictMode>,
);
