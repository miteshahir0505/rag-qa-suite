// Load a built-in helper that reads files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// The file where the store was saved, and the model that must be used to search it
const STORE_FILE = "vector-store.json";
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

// How many best matches we show
const TOP_RESULTS = 3;

// A score below this gets a yellow warning ("weak match").
// This is a starting guess for this check, not an official value.
const WEAK_SCORE = 0.3;

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

// Compares two lists of numbers and gives one score (0 = nothing in common, 1 = same meaning).
// The model already scales each list to the same size, so multiplying and adding is enough.
function similarity(a: number[], b: number[]) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total = total + a[i] * b[i];
  }
  return total;
}

async function main() {
  // The question is whatever you type after the file name
  const question = process.argv.slice(2).join(" ").trim();

  console.log(paint(CYAN, "SEARCH THE VECTOR STORE"));

  // Stop with a clear message if no question was given
  if (question === "") {
    console.log(paint(YELLOW, "No question given. Type your question in quotes after the file name, for example:"));
    console.log('  npx tsx search-store.ts "your question here"');
    return;
  }

  // Stop with a clear message if the store has not been built yet
  if (!fs.existsSync(STORE_FILE)) {
    console.log(paint(RED, STORE_FILE + " was not found. Run: npx tsx build-vector-store.ts"));
    return;
  }

  // Read the store
  const store = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));

  // The question must be turned into numbers by the SAME model that built the store
  if (store.model !== MODEL_NAME) {
    console.log(paint(RED, "The store was built with " + store.model + " but this search uses " + MODEL_NAME + ". The scores would mean nothing."));
    return;
  }

  console.log("Question: " + question);
  console.log("Store:    " + store.entries.length + " chunks, built at " + store.builtAt);
  console.log("");

  // Load the model and turn the question into numbers
  const embedder = await pipeline("feature-extraction", MODEL_NAME);
  const output = await embedder([question], { pooling: "mean", normalize: true });
  const questionVector: number[] = output.tolist()[0];

  // Score every chunk against the question
  const scored = store.entries.map((entry: any) => ({
    source: entry.source,
    section: entry.section,
    text: entry.text,
    score: similarity(questionVector, entry.vector),
  }));

  // Put the best score first
  scored.sort((a: any, b: any) => b.score - a.score);

  // Show the best matches
  console.log(paint(CYAN, "TOP " + TOP_RESULTS + " MATCHES"));
  for (let i = 0; i < TOP_RESULTS && i < scored.length; i++) {
    const result = scored[i];

    // A weak score is shown in yellow, a normal one in green
    const scoreColor = result.score < WEAK_SCORE ? YELLOW : GREEN;

    console.log(paint(MAGENTA, "Match " + (i + 1)));
    console.log("  From file: " + result.source);
    console.log("  Section:   " + result.section);
    console.log("  Score:     " + paint(scoreColor, result.score.toFixed(2)));

    // Show only the first 200 characters, so the screen stays readable
    let shortText = result.text;
    if (shortText.length > 200) {
      shortText = shortText.slice(0, 200) + "...";
    }
    console.log("  Text:      " + shortText);
    console.log("");
  }

  // A plain-words note about the best score
  if (scored[0].score < WEAK_SCORE) {
    console.log(paint(YELLOW, "Note: even the best match scored below " + WEAK_SCORE + ". The documents may not cover this question."));
  }
}

// Run the function
main();