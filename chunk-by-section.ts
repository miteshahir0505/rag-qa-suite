// Load a built-in helper that reads files and folders
import * as fs from "fs";

// Load the shared chunk-cutting code from chunker.ts
import { splitIntoChunks, Chunk } from "./chunker";

// The folder that holds our sample documents
const DOCS_FOLDER = "docs";

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

function main() {
  // Find every .md file in the docs folder, in alphabetical order
  const fileNames = fs
    .readdirSync(DOCS_FOLDER)
    .filter((name) => name.endsWith(".md"))
    .sort();

  // Explain the rules first, so anyone reading the output knows what is being counted
  console.log("CHUNKING CHECK (one chunk per section)");
  console.log("Question being tested: are the documents cut into sensible pieces?");
  console.log("");
  console.log(paint(CYAN, "HOW CHUNKS ARE COUNTED"));
  console.log("  - A chunk is one piece of a document. Search finds and returns whole chunks.");
  console.log("  - Each line that starts with ## is a section heading. One section = one chunk.");
  console.log("  - A chunk holds the heading's text and everything under it, up to the next ## heading.");
  console.log("  - The document title (a line that starts with a single #) is not part of any chunk.");
  console.log("  - A file with no ## headings gives 0 chunks. It is NOT COUNTED and cannot be searched.");
  console.log("");

  // All chunks from all files, and a count per file
  const allChunks: Chunk[] = [];
  const countPerFile: { [name: string]: number } = {};

  for (const fileName of fileNames) {
    // Read the file and cut it into chunks
    const fullText = fs.readFileSync(DOCS_FOLDER + "/" + fileName, "utf8");
    const chunks = splitIntoChunks(fileName, fullText);
    countPerFile[fileName] = chunks.length;

    // If the file gave no chunks, warn right away in yellow (the rules above explain why)
    if (chunks.length === 0) {
      console.log(paint(YELLOW, "WARNING: " + fileName + " has no section headings, so it is NOT COUNTED."));
      console.log(paint(YELLOW, "  To fix it: add a line that starts with ## above each part of its text."));
      console.log("");
    }

    // Show each chunk in plain words. The "Chunk N" line is magenta so each chunk is easy to spot.
    for (const chunk of chunks) {
      allChunks.push(chunk);
      console.log(paint(MAGENTA, "Chunk " + allChunks.length));
      console.log("  From file:  " + chunk.source);
      console.log("  Section:    " + chunk.section);
      console.log("  Length:     " + chunk.text.length + " characters");
      console.log("  Text:       " + chunk.text);
      console.log("");
    }
  }

  // Count the files that gave at least one chunk
  let countedFiles = 0;
  for (const fileName of fileNames) {
    if (countPerFile[fileName] > 0) {
      countedFiles++;
    }
  }

  // The summary: one short line per file. Counted files are green, a skipped file is yellow.
  console.log(paint(CYAN, "SUMMARY"));
  for (const fileName of fileNames) {
    if (countPerFile[fileName] === 0) {
      console.log(paint(YELLOW, "  " + fileName + ": NOT COUNTED (no section headings)"));
    } else {
      console.log(paint(GREEN, "  " + fileName + ": " + countPerFile[fileName] + " chunks"));
    }
  }
  console.log(paint(GREEN, "  Total: " + allChunks.length + " chunks from " + countedFiles + " of " + fileNames.length + " files"));

  // Find the shortest and the longest chunk (only if there is at least one chunk)
  if (allChunks.length > 0) {
    let shortest = allChunks[0];
    let longest = allChunks[0];
    for (const chunk of allChunks) {
      if (chunk.text.length < shortest.text.length) {
        shortest = chunk;
      }
      if (chunk.text.length > longest.text.length) {
        longest = chunk;
      }
    }
    console.log("  Shortest chunk: " + shortest.source + " / " + shortest.section + " (" + shortest.text.length + " characters)");
    console.log("  Longest chunk:  " + longest.source + " / " + longest.section + " (" + longest.text.length + " characters)");
  }
}

// Run the function
main();