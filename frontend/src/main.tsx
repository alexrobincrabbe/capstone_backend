import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import MemoryBrowser from "./MemoryBrowser";

const isMemoryRoute = window.location.pathname.startsWith("/memories");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {isMemoryRoute ? <MemoryBrowser /> : <App />}
  </React.StrictMode>
);
