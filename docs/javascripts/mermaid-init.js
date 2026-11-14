const mermaidScript = "https://unpkg.com/mermaid@11.6.0/dist/mermaid.min.js";
let mermaidReady;

function ensureMermaid() {
  if (!mermaidReady) {
    mermaidReady = new Promise((resolve, reject) => {
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
  const diagrams = [...body.querySelectorAll(".mermaid")];
  if (diagrams.length === 0) return;

  diagrams.forEach((diagram) => {
    const code = diagram.querySelector("code");
    if (code) diagram.textContent = code.textContent;
  });

  ensureMermaid()
    .then(() => {
      mermaid.initialize({ startOnLoad: false });
      return mermaid.run({ nodes: diagrams });
    })
    .catch((error) => console.error("Could not render page diagrams:", error));
});
