// One piece of a document: where it came from, its section title, and its text
export type Chunk = { source: string; section: string; text: string };

// STRATEGY 1: cuts one document into chunks, one chunk for each "## " section.
// It returns a list of chunks. Each chunk says which file it came from, its section title, and its text.
export function splitIntoChunks(fileName: string, fullText: string) {
  // Split the document into lines (works for Windows and other line endings)
  const lines = fullText.split(/\r?\n/);

  // The chunks we have found so far
  const chunks: Chunk[] = [];

  // The section we are reading right now, and its lines
  let currentSection = "";
  let currentLines: string[] = [];

  // Go through the document one line at a time
  for (const line of lines) {
    if (line.startsWith("## ")) {
      // A new section starts here. First, save the section we were reading (if any).
      if (currentSection !== "") {
        chunks.push({
          source: fileName,
          section: currentSection,
          text: currentLines.join(" ").trim(),
        });
      }
      // Start the new section: its title is the heading without the "## "
      currentSection = line.slice(3).trim();
      currentLines = [];
    } else if (currentSection !== "" && line.trim() !== "") {
      // A normal line inside a section: keep it (empty lines are skipped)
      currentLines.push(line.trim());
    }
    // Lines before the first "## " (the document title) are not part of any chunk
  }

  // Save the last section, because no heading comes after it
  if (currentSection !== "") {
    chunks.push({
      source: fileName,
      section: currentSection,
      text: currentLines.join(" ").trim(),
    });
  }

  return chunks;
}

// STRATEGY 2: cuts one document into pieces of about the same size, ignoring headings.
// size    = the most characters one piece may have
// overlap = how many characters one piece shares with the next one
// The pieces never start or end in the middle of a word, but they may stop in the middle of a sentence.
export function splitIntoFixedChunks(fileName: string, fullText: string, size: number, overlap: number) {
  // Turn the whole document into one line of text (every run of spaces or line breaks becomes one space)
  const text = fullText.replace(/\s+/g, " ").trim();

  // The pieces we have found so far
  const chunks: Chunk[] = [];

  // Where the current piece starts
  let start = 0;

  // An empty document gives no pieces
  while (start < text.length) {
    // The piece should end "size" characters after it starts
    let end = start + size;

    // If that is not the end of the document, move the end back to the last space, so no word is cut in half
    if (end < text.length) {
      const lastSpace = text.lastIndexOf(" ", end);
      if (lastSpace > start) {
        end = lastSpace;
      }
    }

    // Save this piece. It has no section title, so we call it "part 1", "part 2" and so on.
    chunks.push({
      source: fileName,
      section: "part " + (chunks.length + 1),
      text: text.slice(start, end).trim(),
    });

    // Stop when we reached the end of the document
    if (end >= text.length) {
      break;
    }

    // The next piece starts a little before this one ended (the overlap), but always moves forward
    start = Math.max(end - overlap, start + 1);

    // Do not start in the middle of a word: move on to the start of the next word
    const nextSpace = text.indexOf(" ", start);
    if (nextSpace !== -1 && nextSpace < end) {
      start = nextSpace + 1;
    }
  }

  return chunks;
}