document$.subscribe(({ body }) => {
  for (const link of body.querySelectorAll('a[id^="__codelineno-"]')) {
    const match = link.id.match(/^__codelineno-(\\d+)-(\\d+)$/);
    if (!match || link.hasAttribute("aria-label")) continue;
    const [, block, line] = match;
    link.setAttribute("aria-label", "Link to code block " + (Number(block) + 1) + ", line " + line);
  }
});
