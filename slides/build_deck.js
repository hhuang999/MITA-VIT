// Conference-talk deck (16:9, white content slides, dark title and closing slides) for the course paper.
// Numbers come from slides/data.json (slides/slide_data.py) and figures from paper/figures (scripts/paper_assets.py).
// Usage: NODE_PATH=<dir with pptxgenjs> node slides/build_deck.js <path to the pptx skill's apply_theme.js>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const OUT = path.join(__dirname, "mita_probe_talk.pptx");
const { applyTheme } = require(process.argv[2]);

// Palette follows the paper figures: blue = top-k probe, ink = dense attention, orange = CLS, gold = GAE.
const THEME = {
  name: "MiTA Probe Talk",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2430", lt1: "FFFFFF", dk2: "1B2B48", lt2: "EEF3FA",
    accent1: "2A78D6", accent2: "EB6834", accent3: "3A3F4B", accent4: "E0A100",
    accent5: "8A8F99", accent6: "1BAF7A", hlink: "2A78D6", folHlink: "6D5BD0",
  },
};
const HEX = THEME.colors;
const SHORT = "What Do Vision Transformers Read?";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = SHORT;
pres.author = "Author Name";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "Title",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.3, w: 11.7, h: 2.3, fontSize: 36, bold: true,
      color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.75, w: 11.7, h: 0.7, fontSize: 22,
      color: C.background2, align: "left", valign: "top" }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.8, y: 5.6, w: 11.7, h: 1.0, fontSize: 16,
      color: C.background2, align: "left", valign: "top" }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30, bold: true,
      color: C.text2, align: "left", valign: "middle" }, text: "" } },
    { text: { text: SHORT, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

pres.defineSlideMaster({
  title: "Closing",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30, bold: true,
      color: C.background1, align: "left", valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

// ---------- helpers ----------
function content(section, title) {
  const slide = pres.addSlide({ masterName: "Content", sectionTitle: section });
  slide.addText(title, { placeholder: "title" });
  return slide;
}

function card(slide, { x, y, w, h, name }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name,
  });
}

function badge(slide, { x, y, label, color }) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.55, h: 0.55, fill: { color }, line: { color }, objectName: `badge ${label}` });
  slide.addText(label, { x, y, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.background1, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}

function stat(slide, { x, y, w, value, label, color }) {
  slide.addText(value, { x, y, w, h: 0.9, fontSize: 44, bold: true, color, margin: 0, isTextBox: true, valign: "bottom" });
  slide.addText(label, { x, y: y + 0.95, w, h: 0.85, fontSize: 14, color: C.text1, margin: 0, isTextBox: true,
    valign: "top" });
}

function bullets(slide, items, box) {
  slide.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 },
  })), { ...box, fontSize: 16, color: C.text1, valign: "top", isTextBox: true });
}

function chartStyle(extra) {
  return {
    catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    valGridLine: { color: "E3E7EE", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.dk1,
    dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: HEX.dk1,
    ...extra,
  };
}

// ---------- 1. Title ----------
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "Title", sectionTitle: "Opening" });
s.addText("What Do Vision Transformers Read?\nExecutable Attention Reading Sets\nunder Centre-Bias Control",
  { placeholder: "title", fontSize: 36 });
s.addText("Author Name  ·  HKUST(GZ)", { placeholder: "subtitle" });
s.addText("DSAA 2012 Deep Learning · Course project · Fall 2026", { placeholder: "meta" });
s.addNotes("Hi, I'm [name]. This talk asks a simple question: when a Vision Transformer looks at an image, which parts " +
  "does it actually read? I'll show a way to read this from an unmodified model, a way to check that the reading is " +
  "real, and what it tells us about three ViTs.");

// ---------- 2. Problem ----------
pres.addSection({ title: "Motivation" });
s = content("Motivation", "Attention maps are easy to draw and hard to trust");
card(s, { x: 0.6, y: 1.5, w: 5.9, h: 4.9, name: "card picture" });
card(s, { x: 6.83, y: 1.5, w: 5.9, h: 4.9, name: "card centre" });
badge(s, { x: 0.95, y: 1.85, label: "1", color: C.accent1 });
badge(s, { x: 7.18, y: 1.85, label: "2", color: C.accent2 });
s.addText("A map is a picture, not a computation", { x: 1.7, y: 1.8, w: 4.6, h: 0.65, fontSize: 20, bold: true,
  color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
s.addText("Photos put objects in the middle", { x: 7.93, y: 1.8, w: 4.6, h: 0.65, fontSize: 20, bold: true,
  color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
bullets(s, ["Nothing guarantees the model relies on the highlighted tokens",
  "“Attention is not explanation” debate (Jain & Wallace 2019; Wiegreffe & Pinter 2019)",
  "We want a reading we can run and check"], { x: 0.95, y: 2.75, w: 5.3, h: 3.4 });
stat(s, { x: 7.18, y: 2.65, w: 5.3, value: "0.41",
  label: "Spearman of a fixed centred Gaussian with ClickMe human maps, higher than our top-k probe on every model",
  color: C.accent2 });
stat(s, { x: 7.18, y: 4.5, w: 5.3, value: "0.10–0.14",
  label: "deletion faithfulness (SRG) of the same centre map on ImageNet-S; it beats Rollout on DeiT-B", color: C.accent2 });
s.addNotes("Attention maps are the default view into ViTs, but they have two problems. First, a map is just a picture. " +
  "Nothing guarantees the model uses the highlighted tokens. Second, photos put objects in the middle, so a map that " +
  "always lights up the centre already looks good. A fixed Gaussian in the centre correlates 0.41 with human maps, and " +
  "even scores 0.10 to 0.14 on a deletion test. So every test in this work subtracts the centre.");

// ---------- 3. Fast weights ----------
pres.addSection({ title: "Method" });
s = content("Method", "Attention is a fast-weight MLP: every token is a neuron");
s.addText([
  { text: "Attn(q", options: {} }, { text: "i", options: { subscript: true } }, { text: ") = Σ", options: {} },
  { text: "j", options: { subscript: true } }, { text: " softmax(q", options: {} }, { text: "i", options: { subscript: true } },
  { text: "·k", options: {} }, { text: "j", options: { subscript: true } }, { text: " / √d) · v", options: {} },
  { text: "j", options: { subscript: true } },
], { x: 0.6, y: 1.5, w: 12.1, h: 0.8, fontSize: 26, color: C.text2, align: "center", isTextBox: true });
const head = { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 16, align: "center" };
const cell = { color: HEX.dk1, fontSize: 16, align: "center", fill: { color: HEX.lt2 } };
s.addTable([
  [{ text: "Two-layer MLP", options: head }, { text: "One attention head", options: head }],
  [{ text: "input x", options: cell }, { text: "query qᵢ", options: cell }],
  [{ text: "hidden neuron j: first-layer weights", options: cell }, { text: "key kⱼ of token j", options: cell }],
  [{ text: "activation", options: cell }, { text: "softmax", options: cell }],
  [{ text: "hidden neuron j: second-layer weights", options: cell }, { text: "value vⱼ of token j", options: cell }],
], { x: 1.4, y: 2.6, w: 6.4, colW: [3.4, 3.0], rowH: 0.5, border: { type: "solid", color: "FFFFFF", pt: 2 },
  objectName: "MLP vs attention table" });
card(s, { x: 8.4, y: 2.6, w: 4.3, h: 2.5, name: "card neuron" });
s.addText([
  { text: "Neuron j  =  image patch j", options: { bold: true, fontSize: 20, color: HEX.accent1, breakLine: true } },
  { text: "So “which neurons does this region use?” is a set of patches we can draw on the image.",
    options: { fontSize: 16, color: HEX.dk1 } },
], { x: 8.7, y: 2.8, w: 3.8, h: 2.1, valign: "top", isTextBox: true });
s.addText("Schlag et al. 2021; MiTA (Wen et al. 2026). The weights come from the image itself, hence “fast weights”.",
  { x: 1.4, y: 5.55, w: 11.3, h: 0.5, fontSize: 12, color: C.accent5, italic: true, isTextBox: true, margin: 0 });
s.addNotes("One attention head computes exactly what a two-layer MLP computes: keys are the first-layer weights, " +
  "softmax is the activation, values are the second-layer weights. So every token is a hidden neuron, and that neuron " +
  "sits at a fixed image patch. This view is not ours; it comes from fast-weight work and from MiTA. What we add is " +
  "using it to read and test a pretrained model.");

// ---------- 4. Reading sets ----------
s = content("Method", "Reading sets: read top-k from an unmodified ViT, then execute");
const steps = [
  ["1", "Split the 14×14 patches into 16 regions", "4 × 4 grid, per layer and head"],
  ["2", "Score every key with the region's mean query", "one ordinary forward pass, no training"],
  ["3", "Keep the top k = 16 keys", "every query in the region reads only this set; CLS gets its own"],
  ["4", "Execute and compare", "restrict layers to the sets; does the top-1 prediction survive?"],
];
steps.forEach(([n, title, sub], i) => {
  const x = 0.6 + i * 3.1;
  card(s, { x, y: 1.5, w: 2.85, h: 2.85, name: `card step ${n}` });
  badge(s, { x: x + 0.3, y: 1.75, label: n, color: i === 3 ? C.accent2 : C.accent1 });
  s.addText(title, { x: x + 0.3, y: 2.45, w: 2.3, h: 1.05, fontSize: 17, bold: true, color: C.text2, margin: 0,
    valign: "top", isTextBox: true });
  s.addText(sub, { x: x + 0.3, y: 3.55, w: 2.3, h: 0.7, fontSize: 13, color: C.text1, margin: 0, valign: "top",
    isTextBox: true });
});
s.addText("Three readouts from the same forward pass (layers 9–12):", { x: 0.6, y: 4.55, w: 12.1, h: 0.45,
  fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
[["Top-k probe", "attention received inside the sets", C.accent1],
  ["Dense attention", "attention received from all tokens (token-pruning score)", C.accent3],
  ["CLS attention", "attention of the CLS query only", C.accent2]].forEach(([name, text, color], i) => {
  const x = 0.6 + i * 4.15;
  s.addShape(pres.shapes.OVAL, { x, y: 5.25, w: 0.3, h: 0.3, fill: { color }, line: { color }, objectName: `dot ${name}` });
  s.addText([{ text: `${name}: `, options: { bold: true } }, { text }], { x: x + 0.45, y: 5.1, w: 3.55, h: 0.9,
    fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("Here is the probe. For each layer and head, we split the patch grid into 16 regions, score all keys with " +
  "the region's mean query, and keep the top 16. Every query in the region may read only that set. Because this is a " +
  "restriction of the model's own computation, we can run it and see whether the prediction survives. From the same " +
  "forward pass we get three readouts: attention inside the sets, plain attention received from all tokens, and " +
  "CLS attention.");

// ---------- 5. Execution ----------
s = content("Method", "The reading sets are what the model computes with");
s.addImage({ path: path.join(ROOT, "paper/figures/execution.png"), x: 0.6, y: 1.35, w: 12.1, h: 12.1 * 585 / 2025,
  altText: "Probe maps per layer, execution k sweep and single-layer agreement", objectName: "execution figure" });
stat(s, { x: 0.6, y: 5.05, w: 3.8, value: "72% vs 9%", label: "DeiT-S predictions kept when all 12 layers read only their sets vs random sets",
  color: C.accent1 });
stat(s, { x: 4.75, y: 5.05, w: 3.8, value: "0.1%", label: "kept with no attention at all: the 72% is not the FFNs alone",
  color: C.accent1 });
stat(s, { x: 8.9, y: 5.05, w: 3.8, value: "83%", label: "for the larger DeiT-B (random: 30%)", color: C.accent1 });
s.addNotes("Does the model really compute with these sets? The test is the whole network. Restricting all twelve " +
  "layers at once keeps 72 percent of DeiT-S predictions, against 9 percent for random sets of the same size and 0.1 " +
  "percent with no attention at all; the larger DeiT-B keeps 83 percent. One layer at a time is not a test: deleting " +
  "a single layer's attention outright already keeps 93 to 97 percent, so panel c only shows which layers are harder " +
  "to replace. So we can treat attention reading as a measured quantity.");

// ---------- 6. Protocol ----------
s = content("Method", "Every test controls for centre bias");
const tests = [
  ["Same-ring enrichment", "Compare only patches at the same distance from the centre; a centre prior scores 0", C.accent1],
  ["Image-specific human alignment", "ρ with this image's ClickMe map minus ρ with another image's map", C.accent1],
  ["Centre-matched removal", "Remove object tokens vs. as many tokens at matched distances from the centre", C.accent2],
  ["Part removal", "Remove 4 head or 4 body patches; how much does the prediction drop?", C.accent2],
];
tests.forEach(([title, text, color], i) => {
  const x = 0.6 + (i % 2) * 6.15;
  const y = 1.45 + Math.floor(i / 2) * 1.85;
  card(s, { x, y, w: 5.95, h: 1.65, name: `card ${title}` });
  badge(s, { x: x + 0.3, y: y + 0.3, label: String(i + 1), color });
  s.addText(title, { x: x + 1.05, y: y + 0.22, w: 4.7, h: 0.5, fontSize: 18, bold: true, color: C.text2, margin: 0,
    isTextBox: true, valign: "middle" });
  s.addText(text, { x: x + 1.05, y: y + 0.75, w: 4.7, h: 0.8, fontSize: 14, color: C.text1, margin: 0,
    isTextBox: true, valign: "top" });
});
s.addText([
  { text: "Models  ", options: { bold: true, color: HEX.dk2 } },
  { text: "DeiT-S, DeiT-B (supervised) · DINO ViT-S (self-supervised)", options: { breakLine: true } },
  { text: "Data  ", options: { bold: true, color: HEX.dk2 } },
  { text: "ImageNet-S · Hard ImageNet · Oxford Pet · ClickMe · PartImageNet", options: { breakLine: true } },
  { text: "Statistics  ", options: { bold: true, color: HEX.dk2 } },
  { text: "paired bootstrap over images · all predictions registered before the runs" },
], { x: 0.6, y: 5.3, w: 12.1, h: 1.3, fontSize: 15, color: C.text1, valign: "top", isTextBox: true, margin: 0,
  paraSpaceAfter: 4 });
s.addNotes("To take the centre out, we use four tests. Same-ring enrichment compares only patches equally far from " +
  "the centre. Human alignment subtracts the score against another image's human map. For the model side, we remove " +
  "object tokens and compare with tokens at matched distances, and we remove head or body patches. We test three " +
  "ViTs on five datasets, with bootstrap intervals, and we wrote every prediction down before running.");

// ---------- 7. Comparison ----------
pres.addSection({ title: "Results" });
s = content("Results", "As saliency maps, attention readouts trail gradient methods");
const methodNames = ["Top-k probe", "Dense attention", "CLS attention", "GAE (gradients)", "Centre (reference)"];
const keys = ["probe_mass", "attention_mass", "cls_attention", "gae", "centre"];
s.addChart(pres.charts.BAR, keys.map((k, i) => ({ name: methodNames[i], labels: DATA.models, values: DATA.srg[k] })),
  chartStyle({ x: 0.6, y: 1.4, w: 7.6, h: 5.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.accent1, HEX.accent3, HEX.accent2, HEX.accent4, "C5CAD3"], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 0.5,
    showTitle: true, title: "Deletion faithfulness (SRG), ImageNet-S", titleFontFace: "+mj-lt", titleFontSize: 14,
    titleColor: HEX.dk2, objectName: "SRG chart" }));
stat(s, { x: 8.7, y: 1.6, w: 4.0, value: "−0.10 to −0.14", label: "the best attention readout vs GAE, per model",
  color: C.accent2 });
bullets(s, ["Dense attention is the best attention map on DeiT-S and DeiT-B; CLS attention on DINO",
  "Same ranking on Hard ImageNet, Pet and an ImageNet subset",
  "Probe, CLS attention and Rollout pass the weight-randomisation sanity check; Integrated Gradients fails"],
  { x: 8.7, y: 3.55, w: 4.0, h: 3.1 });
s.addNotes("First, the honest part. Used as a saliency map, every attention readout is clearly less faithful than " +
  "GAE, which uses gradients of the predicted class. The best attention readout is 0.10 to 0.14 SRG behind. Among " +
  "attention readouts, dense attention wins on the supervised models and CLS attention on DINO, whose classifier reads " +
  "the CLS token. The attention readouts do pass the sanity check, while Integrated Gradients does not.");

// ---------- 8. Top-k vs dense ----------
s = content("Results", "Top-k selection does not improve the map");
s.addChart(pres.charts.BAR, [
  { name: "Top-k probe", labels: DATA.models, values: DATA.head.probe_mass },
  { name: "Dense attention", labels: DATA.models, values: DATA.head.attention_mass },
  { name: "GAE (gradients)", labels: DATA.models, values: DATA.head.gae },
], chartStyle({ x: 6.4, y: 1.4, w: 6.3, h: 5.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: [HEX.accent1, HEX.accent3, HEX.accent4], showValue: true, dataLabelPosition: "outEnd",
  dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 2.2, showTitle: true,
  title: "Animal head enrichment (1 = no preference)", titleFontFace: "+mj-lt", titleFontSize: 14, titleColor: HEX.dk2,
  objectName: "head enrichment chart" }));
stat(s, { x: 0.6, y: 1.5, w: 5.4, value: "ρ = 0.90", label: "median per-image correlation between the probe map and dense attention",
  color: C.accent1 });
bullets(s, ["Dense attention is as good or better on faithfulness (10/10 pairs), object focus and human alignment",
  "The probe's stronger head preference comes from truncation",
  "It does not name the part the model relies on more often: 46.7% vs 46.4% (chance 40%)",
  "So: draw the map with dense attention, verify it with the top-k sets"], { x: 0.6, y: 3.45, w: 5.4, h: 3.2 });
s.addNotes("We also asked whether the top-k step itself helps. It doesn't. The probe map and plain attention received " +
  "correlate at 0.90 per image, and plain attention is equal or slightly better on every map-quality test. The probe " +
  "favours animal heads much more, but that is the truncation sharpening the map; it does not pick the part the " +
  "model relies on any more often. So we draw maps with dense attention and use the top-k sets to verify them.");

// ---------- 9. Beyond the centre ----------
s = content("Results", "Beyond the centre: models read objects and rely on heads");
const findings = [
  ["> 0", "same-ring enrichment for every readout and model: objects win even among patches equally far from the centre", C.accent1],
  ["63–79%", "of the object effect survives when removed tokens are matched for distance to the centre (DeiT-S, DINO)", C.accent1],
  ["2–3.4×", "larger drop when 4 head patches are removed than 4 body patches (quadrupeds and bipeds)", C.accent2],
];
findings.forEach(([value, label, color], i) => {
  const x = 0.6 + i * 4.15;
  card(s, { x, y: 1.5, w: 3.9, h: 3.0, name: `card finding ${i + 1}` });
  stat(s, { x: x + 0.3, y: 1.75, w: 3.3, value, label, color });
});
card(s, { x: 0.6, y: 4.85, w: 12.1, h: 1.35, name: "card shift" });
badge(s, { x: 0.9, y: 5.25, label: "→", color: C.accent1 });
s.addText([{ text: "Objects moved off-centre: ", options: { bold: true, color: HEX.dk2 } },
  { text: "when the crop window shifts 48 px towards a corner, the probe's object mass drops less than a centre prior's on all three models" }],
  { x: 1.65, y: 4.95, w: 10.8, h: 1.15, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "middle" });
s.addNotes("Now what the models read. Every readout prefers object patches even among patches at the same distance " +
  "from the centre. When we shift the object to a corner, the reading follows it. On the model side, removing object " +
  "tokens hurts much more than removing tokens at matched distances: 63 to 79 percent of the effect survives the " +
  "matching. And the models rely on animal heads: removing four head patches hurts two to three times more than " +
  "removing four body patches.");

// ---------- 10. Layers ----------
s = content("Results", "Layer by layer: reading is most human-like in mid-to-late layers");
s.addImage({ path: path.join(__dirname, "layers_three.png"), x: 0.6, y: 1.35, w: 8.3, h: 8.3 * 810 / 1509,
  altText: "Human alignment, head enrichment and high-norm reading per layer", objectName: "layers figure" });
bullets(s, ["Peaks: DeiT-S layer 9, DINO layer 11, DeiT-B layers 6–10",
  "Supervised last layer drifts off the object (dense attention, layer 12 vs best of 8–11: −0.09 DeiT-S, −0.12 DeiT-B)",
  "DINO barely drops (−0.03)",
  "Same curves for all three readouts (r = 0.77–0.99): a property of the model, not of the probe"],
  { x: 9.2, y: 1.45, w: 3.5, h: 5.2 });
s.addNotes("Reading layer by layer, human alignment and head preference rise and peak in mid-to-late layers. In the " +
  "two supervised models, the last layer drops sharply and moves off the object. DINO barely drops. Importantly, the " +
  "curves are the same for all three readouts, so this is a property of the models' attention, not an artefact of " +
  "our probe.");

// ---------- 11. High-norm ----------
s = content("Results", "High-norm tokens absorb reading, but do not explain the drift");
const layers = Array.from({ length: 12 }, (_, i) => String(i + 1));
s.addChart(pres.charts.LINE, [
  { name: "DeiT-S", labels: layers, values: DATA.high_share.deit },
  { name: "DINO ViT-S", labels: layers, values: DATA.high_share.dino },
  { name: "DeiT-B", labels: layers, values: DATA.high_share.deit_b },
], chartStyle({ x: 0.6, y: 1.4, w: 7.6, h: 5.2, chartColors: [HEX.accent1, HEX.accent6, HEX.dk2], lineSize: 2.5,
  lineDataSymbol: "circle", lineDataSymbolSize: 6, valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisLabelFormatCode: "0%",
  showTitle: true, title: "Share of the probe's reading on high-norm patches, per layer", titleFontFace: "+mj-lt",
  titleFontSize: 14, titleColor: HEX.dk2, showCatAxisTitle: true, catAxisTitle: "Layer", catAxisTitleFontSize: 12,
  catAxisTitleColor: HEX.accent5, objectName: "high-norm chart" }));
stat(s, { x: 8.7, y: 1.5, w: 4.0, value: "3% → 61%", label: "in DeiT-B layer 11, 3% of patches receive 61% of the probe's reading",
  color: C.text2 });
bullets(s, ["DINO has no high-norm patches at all, as Darcet et al. (2024) report",
  "But the share falls at layer 12, exactly where the drift happens",
  "Why the last layer leaves the object remains open"], { x: 8.7, y: 3.45, w: 4.0, h: 3.1 });
s.addNotes("Could high-norm artefact tokens explain the drift? Supervised ViTs do send a lot of their mid-to-late " +
  "reading to a handful of high-norm patches; in DeiT-B layer 11, three percent of patches get 61 percent of the " +
  "reading. DINO has none, as Darcet et al. reported. But at layer 12 the share goes down, not up, so these tokens " +
  "do not explain why the last layer leaves the object. That question remains open.");

// ---------- 12. Takeaways ----------
pres.addSection({ title: "Closing" });
s = pres.addSlide({ masterName: "Closing", sectionTitle: "Closing" });
s.addText("Takeaways", { placeholder: "title" });
const takeaways = [
  "Top-k reading sets make attention checkable: run the model through them and the prediction survives",
  "Under centre control, ViTs read objects and animal heads, and rely on them",
  "Reading is most human-like in mid-to-late layers; supervised ViTs drift at the last layer, DINO does not",
  "Honest limits: attention maps trail GAE, and top-k does not beat dense attention as a map",
];
takeaways.forEach((t, i) => {
  const y = 1.45 + i * 1.15;
  badge(s, { x: 0.8, y: y + 0.1, label: String(i + 1), color: i === 3 ? C.accent2 : C.accent1 });
  s.addText(t, { x: 1.65, y, w: 10.9, h: 0.8, fontSize: 20, color: C.background1, margin: 0, valign: "middle",
    isTextBox: true });
});
s.addText("Registered before the runs: of 29 directional predictions, 19 held, 7 held partly, 3 failed.   Questions?",
  { x: 0.8, y: 6.2, w: 11.8, h: 0.5, fontSize: 15, italic: true, color: C.background2, margin: 0, isTextBox: true });
s.addNotes("To sum up. Top-k reading sets make attention checkable. Under centre control, ViTs read objects and animal " +
  "heads and rely on them. Reading is most human-like in mid-to-late layers, and supervised models drift at the last " +
  "layer while DINO does not. And the limits: attention maps trail gradient methods, and top-k selection does not beat " +
  "plain attention as a map. We registered every prediction in advance; most held, and we report the ones that failed. " +
  "Thank you, I'm happy to take questions.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
