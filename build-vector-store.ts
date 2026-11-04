// Load a built-in helper that reads and writes files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// Load the shared chunk-cutting code
import { splitIntoChunks, Chunk } from "./chunker";

// The folder with our sample documents, and the file where the store is saved
const DOCS_FOLDER = "docs";
const STORE_FILE = "vector-store.json";

// The embedding model, and how many numbers it gives per text
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
const EXPECTED_NUMBERS = 384;

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

async function main() {
  console.log(paint(CYAN, "BUILD VECTOR STORE"));
  console.log("Goal: cut every document into chunks, turn each chunk into numbers, and save them in " + STORE_FILE + ".");
  console.log("");

  // Step 1: read every .md file and cut it into chunks
  const fileNames = fs
    .readdirSync(DOCS_FOLDER)
    .filter((name) => name.endsWith(".md"))
    .sort();

  const allChunks: Chunk[] = [];
  for (const fileName of fileNames) {
    const fullText = fs.readFileSync(DOCS_FOLDER + "/" + fileName, "utf8");
    const chunks = splitIntoChunks(fileName, fullText);

    // A file with no ## headings gives no chunks, so warn clearly in yellow
    if (chunks.length === 0) {
      console.log(paint(YELLOW, "WARNING: " + fileName + " has no section headings, so it is NOT in the store."));
    } else {
      console.log(paint(GREEN, fileName + ": " + chunks.length + " chunks"));
    }
    allChunks.push(...chunks);
  }
  console.log("Total chunks to store: " + allChunks.length);
  console.log("");

  // Step 2: load the model
  console.log("Loading the embedding model (the first run may download it)...");
  const embedder = await pipeline("feature-extraction", MODEL_NAME);
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // Step 3: turn every chunk into numbers.
  // We embed the section title together with the text, so the title helps the search.
  const textsToEmbed = allChunks.map((chunk) => chunk.section + ". " + chunk.text);
  const output = await embedder(textsToEmbed, { pooling: "mean", normalize: true });
  const vectors: number[][] = output.tolist();

  // Step 4: build the store. Each entry is the chunk plus its numbers.
  // The numbers are rounded to 6 decimals, which keeps the file small and loses nothing that matters.
  const entries = allChunks.map((chunk, i) => ({
    source: chunk.source,
    section: chunk.section,
    text: chunk.text,
    vector: vectors[i].map((n) => Number(n.toFixed(6))),
  }));

  const store = {
    model: MODEL_NAME,
    numbersPerVector: EXPECTED_NUMBERS,
    builtAt: new Date().toLocaleString(),
    entries: entries,
  };

  // Step 5: save the store (each run replaces the old file)
  fs.writeFileSync(STORE_FILE, JSON.stringify(store));
  const sizeKb = Math.round(fs.statSync(STORE_FILE).size / 1024);
  console.log(paint(GREEN, "Saved " + entries.length + " entries to " + STORE_FILE + " (" + sizeKb + " KB).") + "\n");

  // Step 6: QA checks on the store we just built
  console.log(paint(CYAN, "QA CHECKS"));
  let checksPassed = 0;

  // Check 1: one vector for every chunk
  const check1 = entries.length === allChunks.length;
  if (check1) {
    checksPassed++;
  }
  console.log(paint(MAGENTA, "Check 1: every chunk has exactly one vector"));
  console.log("  Expected: " + allChunks.length + " vectors");
  console.log("  Actual:   " + entries.length + " vectors");
  console.log("  Verdict:  " + verdict(check1));

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
  console.log(paint(MAGENTA, "Check 2: every vector has " + EXPECTED_NUMBERS + " numbers"));
  console.log("  Expected: 0 vectors with the wrong length");
  console.log("  Actual:   " + wrongLength + " vectors with the wrong length");
  console.log("  Verdict:  " + verdict(check2));

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
  console.log(paint(MAGENTA, "Check 3: no broken numbers inside the vectors"));
  console.log("  Expected: 0 broken numbers");
  console.log("  Actual:   " + brokenNumbers + " broken numbers");
  console.log("  Verdict:  " + verdict(check3));

  // The summary: green if every check passed, red if any failed
  console.log("");
  console.log(paint(CYAN, "SUMMARY"));
  const summaryColor = checksPassed === 3 ? GREEN : RED;
  console.log(paint(summaryColor, "  " + checksPassed + " of 3 checks passed."));
}

// Run the function
main();