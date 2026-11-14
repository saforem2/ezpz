document$.subscribe(({ body }) => {
  for (const link of body.querySelectorAll('a[id^="__codelineno-"]')) {
    const parts = link.id.split("-");
    if (parts.length !== 3 || parts[0] !== "__codelineno") continue;
    const block = Number(parts[1]);
    const line = Number(parts[2]);
    if (!Number.isInteger(block) || !Number.isInteger(line)) continue;
    if (link.hasAttribute("aria-label")) continue;

    link.setAttribute(
      "aria-label",
      "Link to code block " + (block + 1) + ", line " + line,
    );
  }
});
