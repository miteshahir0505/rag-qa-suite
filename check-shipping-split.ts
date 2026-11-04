// Load a built-in helper that reads files
import * as fs from "fs";

// Load the shared fixed-size cutting code
import { splitIntoFixedChunks } from "./chunker";

// The document we are checking
const FILE_NAME = "shipping-notes.md";
const FILE_PATH = "docs/" + FILE_NAME;

// The three versions we built (same settings as build-vector-store-fixed.ts)
const versions = [
  { name: "A. Baseline (200 characters, overlap 40)", size: 200, overlap: 40 },
  { name: "B. Larger pieces (300 characters, overlap 40)", size: 300, overlap: 40 },
  { name: "C. No overlap (200 characters, overlap 0)", size: 200, overlap: 0 },
];

// COLORS: used only when the output goes to a real terminal screen
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

// Makes all spaces and line breaks into single spaces, so texts can be compared fairly
function tidy(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function main() {
  console.log(paint(CYAN, "SHIPPING NOTES: IS ANY KEY FACT SPLIT BETWEEN TWO PIECES?"));
  console.log("Question being tested: when the document is cut into fixed-size pieces,");
  console.log("does every sentence that holds a number still appear whole in at least one piece?");
  console.log("A sentence that is split loses part of its meaning, for example a deadline without its unit.");
  console.log("");

  // Read the document and tidy it
  const fullText = fs.readFileSync(FILE_PATH, "utf8");
  const cleanText = tidy(fullText);

  // Cut the text into sentences, then keep only the ones that contain a digit
  const sentences = cleanText.split(/(?<=[.!?])\s+/);
  const factSentences = sentences.filter((s) => /\d/.test(s));

  console.log("Document: " + FILE_NAME + " (" + cleanText.length + " characters)");
  console.log("Sentences that contain a number: " + factSentences.length);
  console.log("");

  // If there are no such sentences, the check has nothing to test
  if (factSentences.length === 0) {
    console.log(paint(YELLOW, "No sentence with a number was found, so this check has nothing to test."));
    return;
  }

  // A short summary line per version, filled in as we go
  const summary: string[] = [];

  for (const version of versions) {
    // Cut the document the way this version does, and tidy each piece
    const chunks = splitIntoFixedChunks(FILE_NAME, fullText, version.size, version.overlap);
    const pieces = chunks.map((chunk) => tidy(chunk.text));

    console.log(paint(MAGENTA, version.name));
    console.log("  The document gives " + pieces.length + " pieces.");

    let wholeCount = 0;

    for (let i = 0; i < factSentences.length; i++) {
      const sentence = factSentences[i];

      // Find every piece that holds this whole sentence
      const holders: number[] = [];
      for (let p = 0; p < pieces.length; p++) {
        if (pieces[p].includes(sentence)) {
          holders.push(p + 1);
        }
      }

      const ok = holders.length > 0;
      if (ok) {
        wholeCount++;
      }

      // Show the check like a line in a QA report
      console.log("  Fact " + (i + 1) + ": " + sentence);
      console.log("    Requirement: this sentence must appear whole in at least one piece.");
      console.log("    Expected:    whole in 1 or more pieces");
      if (ok) {
        console.log("    Actual:      whole in piece " + holders.join(" and "));
      } else {
        console.log("    Actual:      split across pieces, no piece holds all of it");
      }
      console.log("    Verdict:     " + verdict(ok));
    }

    console.log("");
    summary.push(
      version.name + ": " + wholeCount + " of " + factSentences.length + " facts kept whole"
    );
  }

  // The summary: green when every fact is whole, red when any fact is split
  console.log(paint(CYAN, "SUMMARY"));
  for (let i = 0; i < summary.length; i++) {
    const allWhole = summary[i].includes(
      factSentences.length + " of " + factSentences.length
    );
    console.log(paint(allWhole ? GREEN : RED, "  " + summary[i]));
  }
  console.log("");
  console.log("A FAIL here is a finding about the cutting strategy, not a coding mistake.");
  console.log("A tester would record it and check whether a question about that fact also fails in the next step.");
}

// Run the function
main();