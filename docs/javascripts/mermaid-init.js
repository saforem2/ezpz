const mermaidScript = "https://unpkg.com/mermaid@11.6.0/dist/mermaid.min.js";
let mermaidReady;
let mermaidRenderQueue = Promise.resolve();

function ensureMermaid() {
  if (!mermaidReady) {
    mermaidReady = new Promise((resolve, reject) => {
      if (window.mermaid) {
        resolve();
        return;
      }

      const script = document.createElement("script");
      script.src = mermaidScript;
      script.async = true;
      script.onload = resolve;
      script.onerror = () =>
        reject(new Error(`Failed to load ${mermaidScript}`));
      document.head.append(script);
    });
  }
  return mermaidReady;
}

document$.subscribe(({ body }) => {
  const diagrams = [...body.querySelectorAll(".mermaid-source")];
  if (diagrams.length === 0) return;
  const diagramSources = diagrams.map((diagram) => {
    const code = diagram.querySelector("code");
    return code ? code.textContent : diagram.textContent;
  });
  const pagePath = location.pathname;

  mermaidRenderQueue = mermaidRenderQueue
    .catch(() => {})
    .then(() => ensureMermaid())
    .then(() => {
      if (location.pathname !== pagePath) return;

      // The theme can replace the original <pre> nodes while Mermaid loads.
      // Re-query the live DOM and restore the source captured above.
      const currentDiagrams = [...document.querySelectorAll(".mermaid-source")];
      const pending = currentDiagrams
        .map((diagram, index) => ({ diagram, source: diagramSources[index] }))
        .filter(
          ({ diagram }) =>
            diagram.dataset.processed !== "true" && !diagram.querySelector("svg"),
        );
      if (pending.length === 0) return;

      pending.forEach(({ diagram, source }) => {
        if (source) diagram.textContent = source;
      });

      mermaid.initialize({ startOnLoad: false });
      return mermaid.run({ nodes: pending.map(({ diagram }) => diagram) });
    })
    .catch((error) => console.error("Could not render page diagrams:", error));
});
