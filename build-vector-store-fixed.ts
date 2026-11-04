// Load a built-in helper that reads and writes files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// Load both chunk-cutting strategies
import { splitIntoChunks, splitIntoFixedChunks, Chunk } from "./chunker";

// The folder with our sample documents
const DOCS_FOLDER = "docs";

// The embedding model, and how many numbers it gives per text
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
const EXPECTED_NUMBERS = 384;

// The three versions of the fixed-size strategy we build and compare.
// Each one has a name, a piece size, an overlap, and the file where its store is saved.
const variants = [
  {
    name: "A. Baseline (200 characters, overlap 40)",
    size: 200,
    overlap: 40,
    file: "vector-store-fixed.json",
  },
  {
    name: "B. Larger pieces (300 characters, overlap 40)",
    size: 300,
    overlap: 40,
    file: "vector-store-fixed-300.json",
  },
  {
    name: "C. No overlap (200 characters, overlap 0)",
    size: 200,
    overlap: 0,
    file: "vector-store-fixed-no-overlap.json",
  },
];

// COLORS: these are standard terminal color codes.
// Colors are used only when the output goes to a real terminal screen.
// If the output is saved to a file, the text stays plain.
const USE_COLOR = process.stdout.isTTY === true && !process.env.NO_COLOR;
const YELLOW = "33";
const GREEN = "32";
const RED = "31";
const CYAN = "1;36";
const MAGENTA = "1;35";

// Wraps a piece of text in a color (or leaves it plain if colors are off)
function paint(colorCode: string, text: string) {
  if (!USE_COLOR) {
    return text;
  }
  return "\u001b[" + colorCode + "m" + text + "\u001b[0m";
}

// Turns true/false into a green PASS or a red FAIL
function verdict(ok: boolean) {
  return ok ? paint(GREEN, "PASS") : paint(RED, "FAIL");
}

// A piece of text "ends cleanly" when its last character closes a sentence
function endsCleanly(text: string) {
  const last = text.charAt(text.length - 1);
  return last === "." || last === "?" || last === "!";
}

// Counts how many chunks stop in the middle of a sentence
function countMidSentence(chunks: Chunk[]) {
  let count = 0;
  for (const chunk of chunks) {
    if (!endsCleanly(chunk.text)) {
      count++;
    }
  }
  return count;
}

// What we remember about each finished build, for the comparison at the end
type BuildResult = {
  name: string;
  file: string;
  chunkCount: number;
  midSentence: number;
  sizeKb: number;
  checksPassed: number;
};

async function main() {
  console.log(paint(CYAN, "BUILD VECTOR STORES (fixed-size chunks, three versions)"));
  console.log("Goal: build one store for each version of the fixed-size strategy, check each store,");
  console.log("and compare them. The first store (vector-store.json) is not changed.");
  console.log("");

  // Read every document once
  const fileNames = fs
    .readdirSync(DOCS_FOLDER)
    .filter((name) => name.endsWith(".md"))
    .sort();
  const texts: { [name: string]: string } = {};
  for (const fileName of fileNames) {
    texts[fileName] = fs.readFileSync(DOCS_FOLDER + "/" + fileName, "utf8");
  }

  // The section strategy, only for the comparison at the end
  const sectionChunks: Chunk[] = [];
  for (const fileName of fileNames) {
    sectionChunks.push(...splitIntoChunks(fileName, texts[fileName]));
  }

  // Load the model once, and use it for all three builds
  console.log("Loading the embedding model (the first run may download it)...");
  const embedder = await pipeline("feature-extraction", MODEL_NAME);
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // The results of the three builds go here
  const results: BuildResult[] = [];

  for (const variant of variants) {
    console.log(paint(CYAN, "BUILDING " + variant.name));

    // Step 1: cut every document with this version's settings
    const allChunks: Chunk[] = [];
    let filesWithChunks = 0;
    for (const fileName of fileNames) {
      const chunks = splitIntoFixedChunks(fileName, texts[fileName], variant.size, variant.overlap);
      if (chunks.length === 0) {
        console.log(paint(YELLOW, "  WARNING: " + fileName + " gave no chunks, so it is NOT in the store."));
      } else {
        filesWithChunks++;
      }
      allChunks.push(...chunks);
    }
    console.log("  Chunks to store: " + allChunks.length + " (from " + filesWithChunks + " of " + fileNames.length + " documents)");

    // Step 2: turn every chunk into numbers. Only the text is embedded, because these chunks have no real title.
    const textsToEmbed = allChunks.map((chunk) => chunk.text);
    const output = await embedder(textsToEmbed, { pooling: "mean", normalize: true });
    const vectors: number[][] = output.tolist();

    // Step 3: build the store. Each entry is the chunk plus its numbers (rounded to 6 decimals).
    const entries = allChunks.map((chunk, i) => ({
      source: chunk.source,
      section: chunk.section,
      text: chunk.text,
      vector: vectors[i].map((n) => Number(n.toFixed(6))),
    }));
    const store = {
      model: MODEL_NAME,
      strategy: "fixed-size",
      chunkSize: variant.size,
      overlap: variant.overlap,
      numbersPerVector: EXPECTED_NUMBERS,
      builtAt: new Date().toLocaleString(),
      entries: entries,
    };

    // Step 4: save the store (each run replaces the old file)
    fs.writeFileSync(variant.file, JSON.stringify(store));
    const sizeKb = Math.round(fs.statSync(variant.file).size / 1024);
    console.log(paint(GREEN, "  Saved " + entries.length + " entries to " + variant.file + " (" + sizeKb + " KB)."));

    // Step 5: the four QA checks, one line each
    let checksPassed = 0;

    // Check 1: one vector for every chunk
    const check1 = entries.length === allChunks.length;
    if (check1) {
      checksPassed++;
    }
    console.log(paint(MAGENTA, "  Check 1: every chunk has exactly one vector") + " | expected " + allChunks.length + ", actual " + entries.length + " | " + verdict(check1));

    // Check 2: every vector has the right length
    let wrongLength = 0;
    for (const entry of entries) {
      if (entry.vector.length !== EXPECTED_NUMBERS) {
        wrongLength++;
      }
    }
    const check2 = wrongLength === 0;
    if (check2) {
      checksPassed++;
    }
    console.log(paint(MAGENTA, "  Check 2: every vector has " + EXPECTED_NUMBERS + " numbers") + " | expected 0 wrong, actual " + wrongLength + " wrong | " + verdict(check2));

    // Check 3: no broken numbers (NaN means "not a number")
    let brokenNumbers = 0;
    for (const entry of entries) {
      for (const n of entry.vector) {
        if (Number.isNaN(n)) {
          brokenNumbers++;
        }
      }
    }
    const check3 = brokenNumbers === 0;
    if (check3) {
      checksPassed++;
    }
    console.log(paint(MAGENTA, "  Check 3: no broken numbers inside the vectors") + " | expected 0, actual " + brokenNumbers + " | " + verdict(check3));

    // Check 4: every document in the docs folder made it into the store
    const check4 = filesWithChunks === fileNames.length;
    if (check4) {
      checksPassed++;
    }
    console.log(paint(MAGENTA, "  Check 4: every document is in the store") + " | expected " + fileNames.length + " of " + fileNames.length + ", actual " + filesWithChunks + " of " + fileNames.length + " | " + verdict(check4));
    console.log("");

    // Remember this build for the comparison
    results.push({
      name: variant.name,
      file: variant.file,
      chunkCount: allChunks.length,
      midSentence: countMidSentence(allChunks),
      sizeKb: sizeKb,
      checksPassed: checksPassed,
    });
  }

  // THE COMPARISON, written as plain sentences
  console.log(paint(CYAN, "COMPARISON OF THE THREE VERSIONS"));
  console.log("  For reference, cutting by section: " + sectionChunks.length + " chunks, " + countMidSentence(sectionChunks) + " stop in the middle of a sentence, " + (fileNames.length - new Set(sectionChunks.map((c) => c.source)).size) + " document(s) cannot be searched.");
  console.log("");
  for (const r of results) {
    const lineColor = r.checksPassed === 4 ? GREEN : RED;
    console.log(paint(lineColor, "  " + r.name));
    console.log("    " + r.chunkCount + " chunks, " + r.midSentence + " of them stop in the middle of a sentence, file size " + r.sizeKb + " KB");
    console.log("    " + r.checksPassed + " of 4 checks passed. Saved in " + r.file);
  }
  console.log("");

  // What the differences mean, using the numbers we just measured
  const baseline = results[0];
  const larger = results[1];
  const noOverlap = results[2];
  console.log(paint(CYAN, "WHAT THE DIFFERENCES MEAN"));
  console.log("  - Larger pieces (B) made " + larger.chunkCount + " chunks instead of " + baseline.chunkCount + ". " + larger.midSentence + " stop mid-sentence, against " + baseline.midSentence + " for the baseline.");
  console.log("    Bigger pieces keep more of each answer together, but each piece then covers more topics at once.");
  console.log("  - Removing the overlap (C) made " + noOverlap.chunkCount + " chunks instead of " + baseline.chunkCount + ". " + noOverlap.midSentence + " stop mid-sentence, against " + baseline.midSentence + " for the baseline.");
  console.log("    Without overlap, a sentence that falls on a cut is split with no copy of it in the next piece.");
  console.log("  - None of this says which version finds answers better. That needs the same questions run on each store.");
}

// Run the function
main();