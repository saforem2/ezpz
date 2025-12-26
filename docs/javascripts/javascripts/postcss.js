const postcss = require("postcss");
const postcssOKLabFunction = require("@csstools/postcss-oklab-function");

postcss([postcssOKLabFunction({ preserve: true })]).process(
  "../../_site/css/*.css" /*, processOptions */,
);
