const katexCss = "https://unpkg.com/katex@0/dist/katex.min.css";
const katexScript = "https://unpkg.com/katex@0/dist/katex.min.js";
const autoRenderScript =
  "https://unpkg.com/katex@0/dist/contrib/auto-render.min.js";

let katexReady;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.append(script);
  });
}

function loadStylesheet(href) {
  return new Promise((resolve, reject) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    link.onload = resolve;
    link.onerror = () => reject(new Error(`Failed to load ${href}`));
    document.head.append(link);
  });
}

function ensureKatex() {
  if (!katexReady) {
    katexReady = Promise.all([
      loadStylesheet(katexCss),
      loadScript(katexScript),
    ]).then(() => loadScript(autoRenderScript));
  }
  return katexReady;
}

document$.subscribe(({ body }) => {
  if (!body.querySelector(".arithmatex")) return;

  ensureKatex()
    .then(() => {
      renderMathInElement(body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
      });
    })
    .catch((error) => console.error("Could not render page math:", error));
});
