// Load a built-in helper that reads files and folders
import * as fs from "fs";

// Load both chunk-cutting strategies from chunker.ts
import { splitIntoChunks, splitIntoFixedChunks, Chunk } from "./chunker";

// The folder that holds our sample documents
const DOCS_FOLDER = "docs";

// The settings of the fixed-size strategy.
// These are starting guesses, not official values.
const CHUNK_SIZE = 200;
const OVERLAP = 40;

// COLORS: these are standard terminal color codes.
// Colors are used only when the output goes to a real terminal screen.
// If the output is saved to a file, the text stays plain.
const USE_COLOR = process.stdout.isTTY === true && !process.env.NO_COLOR;
const YELLOW = "33";
const GREEN = "32";
const CYAN = "1;36";
const MAGENTA = "1;35";

// Wraps a piece of text in a color (or leaves it plain if colors are off)
function paint(colorCode: string, text: string) {
  if (!USE_COLOR) {
    return text;
  }
  return "\u001b[" + colorCode + "m" + text + "\u001b[0m";
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

function main() {
  // Find every .md file in the docs folder, in alphabetical order
  const fileNames = fs
    .readdirSync(DOCS_FOLDER)
    .filter((name) => name.endsWith(".md"))
    .sort();

  // Explain the rules first, so anyone reading the output knows what is being counted
  console.log(paint(CYAN, "CHUNKING CHECK (fixed-size pieces)"));
  console.log("Question being tested: what happens when we cut the documents into pieces of equal size,");
  console.log("instead of one piece per section?");
  console.log("");
  console.log(paint(CYAN, "HOW CHUNKS ARE CUT HERE"));
  console.log("  - A chunk is one piece of a document. Search finds and returns whole chunks.");
  console.log("  - The headings are ignored. The text is cut into pieces of about " + CHUNK_SIZE + " characters.");
  console.log("  - Each piece repeats the last " + OVERLAP + " characters of the piece before it, so nothing falls exactly on a cut.");
  console.log("  - A piece never starts or ends in the middle of a word, but it can stop in the middle of a sentence.");
  console.log("  - A piece has no section title, so it is called \"part 1\", \"part 2\" and so on.");
  console.log("  - Chunks that stop in the middle of a sentence are marked in yellow.");
  console.log("");

  // All fixed-size chunks and all section chunks, per file
  const allFixed: Chunk[] = [];
  const allSection: Chunk[] = [];
  const fixedPerFile: { [name: string]: number } = {};
  const sectionPerFile: { [name: string]: number } = {};

  for (const fileName of fileNames) {
    // Read the file and cut it with both strategies
    const fullText = fs.readFileSync(DOCS_FOLDER + "/" + fileName, "utf8");
    const fixedChunks = splitIntoFixedChunks(fileName, fullText, CHUNK_SIZE, OVERLAP);
    const sectionChunks = splitIntoChunks(fileName, fullText);
    fixedPerFile[fileName] = fixedChunks.length;
    sectionPerFile[fileName] = sectionChunks.length;
    allSection.push(...sectionChunks);

    // Show each fixed-size chunk in plain words
    for (const chunk of fixedChunks) {
      allFixed.push(chunk);
      console.log(paint(MAGENTA, "Chunk " + allFixed.length));
      console.log("  From file:  " + chunk.source);
      console.log("  Piece:      " + chunk.section);
      console.log("  Length:     " + chunk.text.length + " characters");
      console.log("  Text:       " + chunk.text);
      if (!endsCleanly(chunk.text)) {
        console.log(paint(YELLOW, "  Note:       this chunk stops in the middle of a sentence."));
      }
      console.log("");
    }
  }

  // The summary: one line per file, comparing the two strategies
  console.log(paint(CYAN, "SUMMARY: HOW MANY CHUNKS EACH STRATEGY MAKES"));
  for (const fileName of fileNames) {
    const line = "  " + fileName + ": " + sectionPerFile[fileName] + " chunks by section, " + fixedPerFile[fileName] + " chunks of fixed size";
    if (sectionPerFile[fileName] === 0 && fixedPerFile[fileName] > 0) {
      console.log(paint(GREEN, line + "  (the fixed-size strategy can read this file; the section strategy cannot)"));
    } else {
      console.log(line);
    }
  }
  console.log("  Total: " + allSection.length + " chunks by section, " + allFixed.length + " chunks of fixed size");
  console.log("");

  // How clean are the cuts?
  const midSection = countMidSentence(allSection);
  const midFixed = countMidSentence(allFixed);
  console.log(paint(CYAN, "SUMMARY: HOW CLEAN ARE THE CUTS?"));
  console.log("  By section: " + midSection + " of " + allSection.length + " chunks stop in the middle of a sentence.");
  const fixedColor = midFixed === 0 ? GREEN : YELLOW;
  console.log(paint(fixedColor, "  Fixed size: " + midFixed + " of " + allFixed.length + " chunks stop in the middle of a sentence."));
  if (midFixed > 0) {
    console.log("  A chunk that stops mid-sentence can lose the very detail a customer asks about,");
    console.log("  for example a number or a \"not\" that falls just after the cut.");
  }
  console.log("");

  // The size of the chunks
  if (allFixed.length > 0 && allSection.length > 0) {
    let shortestFixed = allFixed[0];
    let longestFixed = allFixed[0];
    for (const chunk of allFixed) {
      if (chunk.text.length < shortestFixed.text.length) {
        shortestFixed = chunk;
      }
      if (chunk.text.length > longestFixed.text.length) {
        longestFixed = chunk;
      }
    }
    let shortestSection = allSection[0];
    let longestSection = allSection[0];
    for (const chunk of allSection) {
      if (chunk.text.length < shortestSection.text.length) {
        shortestSection = chunk;
      }
      if (chunk.text.length > longestSection.text.length) {
        longestSection = chunk;
      }
    }
    console.log(paint(CYAN, "SUMMARY: CHUNK SIZES (in characters)"));
    console.log("  By section: from " + shortestSection.text.length + " to " + longestSection.text.length);
    console.log("  Fixed size: from " + shortestFixed.text.length + " to " + longestFixed.text.length);
  }
}

// Run the function
main();