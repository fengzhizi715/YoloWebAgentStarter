import type { ReactNode } from "react";

// A deliberately small, text-only Markdown subset for model reports. Never
// interpret raw HTML, embedded images or model-provided URLs as active content.
function inline(text: string): ReactNode[] {
  return text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g).map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    return part;
  });
}

export function ReportText({ content }: { content: string }) {
  const lines = content.replace(/\r\n?/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  const startsBlock = (line: string) => /^(?:#{1,6}\s|```|\s*[-*+]\s|\s*\d+\.\s|>\s)/.test(line);
  for (let i = 0; i < lines.length;) {
    const key = i;
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      if (i < lines.length) i++;
      blocks.push(<pre key={key}><code>{code.join("\n")}</code></pre>);
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    if (heading) { blocks.push(<h3 key={key}>{inline(heading[1])}</h3>); i++; continue; }
    const list = line.match(/^\s*([-*+]|\d+\.)\s+(.+)$/);
    if (list) {
      const ordered = /\d/.test(list[1]);
      const pattern = ordered ? /^\s*\d+\.\s+(.+)$/ : /^\s*[-*+]\s+(.+)$/;
      const items: ReactNode[] = [];
      while (i < lines.length) {
        const item = lines[i].match(pattern);
        if (!item) break;
        items.push(<li key={i++}>{inline(item[1])}</li>);
      }
      blocks.push(ordered ? <ol key={key} start={parseInt(list[1], 10)}>{items}</ol> : <ul key={key}>{items}</ul>);
      continue;
    }
    if (line.startsWith("> ")) { blocks.push(<blockquote key={key}>{inline(line.slice(2))}</blockquote>); i++; continue; }
    if (/^\s*(?:---+|\*\*\*+)\s*$/.test(line)) { blocks.push(<hr key={key} />); i++; continue; }
    const paragraph = [lines[i++]];
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) paragraph.push(lines[i++]);
    blocks.push(<p key={key}>{inline(paragraph.join("\n"))}</p>);
  }
  return <div className="agent-report-text">{blocks}</div>;
}
