import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ReportText } from "./ReportText";

describe("Agent report typography", () => {
  it("renders headings, emphasis, code and both kinds of list", () => {
    const html = renderToStaticMarkup(<ReportText content={'### 结论\n\n**需要复核** `ds_a`\n\n- 事实一\n- 事实二\n\n2. 下一步\n3. 人工确认'} />);
    expect(html).toContain("<h3>结论</h3>");
    expect(html).toContain("<strong>需要复核</strong>");
    expect(html).toContain("<code>ds_a</code>");
    expect(html).toContain("<ul><li>事实一</li><li>事实二</li></ul>");
    expect(html).toContain('<ol start="2"><li>下一步</li><li>人工确认</li></ol>');
  });
  it("treats HTML, image embeds and untrusted URLs as inert text", () => {
    const html = renderToStaticMarkup(<ReportText content={'<script>alert(1)</script>\n<img src=x onerror=alert(1)>\n[link](javascript:alert(1))\n![image](https://example.test/tracker)'} />);
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<img");
    expect(html).not.toContain("<a ");
    expect(html).toContain("&lt;script&gt;");
  });
  it("preserves fenced code and incomplete fences without interpreting markup", () => {
    const html = renderToStaticMarkup(<ReportText content={'```html\n<strong>literal</strong>\n**literal**'} />);
    expect(html).toContain("<pre><code>&lt;strong&gt;literal&lt;/strong&gt;\n**literal**</code></pre>");
  });
});
