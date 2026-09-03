// Syntax-check the <script> block of a generated report before shipping it.
const fs = require("fs");
const vm = require("vm");
const f = process.argv[2];
const h = fs.readFileSync(f, "utf8");
const i = h.lastIndexOf("<script>"), j = h.lastIndexOf("</script>");
if (i < 0 || j < 0) { console.error("no <script> block"); process.exit(2); }
const js = h.slice(i + 8, j);
try {
  new vm.Script(js, { filename: f });
  console.log(`OK  ${f}  (${js.length} chars of JS parse cleanly)`);
} catch (e) {
  console.error(`FAIL ${f}\n  ${e.message}`);
  const m = /:(\d+)$/.exec(e.stack.split("\n")[0]) || [];
  process.exit(1);
}
