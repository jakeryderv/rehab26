"""Build the APA-formatted docx/pdf for the Part 1 report from the markdown draft."""
import re, subprocess, sys, zipfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "REHAB26_Capstone_Part1_Draft.md"
OUT = ROOT / "dist"; OUT.mkdir(exist_ok=True)
DOCX = OUT / "VanSlyke_Capstone_Part1.docx"

md = SRC.read_text()
# Title block: first heading + following non-blank lines -> YAML metadata
lines = md.splitlines()
title = lines[0].lstrip("# ").strip()
i = 1
while lines[i].strip() == "": i += 1
meta = []
while lines[i].strip(): meta.append(lines[i].strip()); i += 1
body = "\n".join(lines[i:])
# References -> Bibliography style (hanging indent)
body = re.sub(r"(## References\n\n)(.*)$", lambda m: m.group(1) + '::: {custom-style="Bibliography"}\n' + m.group(2).strip() + "\n:::\n", body, flags=re.S)
# Figure/table labels, titles and notes -> no first-line indent
body = re.sub(r"^(\*\*(?:Figure|Table) \d+\*\*|\*[A-Z][^*\n]*\*|\*Note\.\* .*)$", r'::: {custom-style="FigureCaption"}\n\1\n:::', body, flags=re.M)
yaml = "---\ntitle: \"%s\"\nauthor:\n%s\n---\n" % (title, "\n".join(f"  - \"{m}\"" for m in meta))
tmp = OUT / "_report.md"; tmp.write_text(yaml + body)

subprocess.run(["pandoc", str(tmp), "-f", "markdown-implicit_figures", "-o", str(DOCX), "--resource-path", str(ROOT)], check=True)

# Patch styles.xml for APA 7: Times New Roman 12, double spacing, 1" margins, centered bold headings
FONT = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/>'
def patch_style(xml, sid, ppr=None, rpr=None):
    m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % re.escape(sid), xml, flags=re.S)
    if not m: print("missing style", sid); return xml
    s = m.group(0)
    s = re.sub(r"<w:pPr>.*?</w:pPr>", "", s, flags=re.S); s = re.sub(r"<w:rPr>.*?</w:rPr>", "", s, flags=re.S)
    ins = (f"<w:pPr>{ppr}</w:pPr>" if ppr is not None else "") + (f"<w:rPr>{rpr}</w:rPr>" if rpr is not None else "")
    s = s.replace("</w:style>", ins + "</w:style>")
    return xml.replace(m.group(0), s)

DOUBLE = '<w:spacing w:line="480" w:lineRule="auto" w:before="0" w:after="0"/>'
BASE_R = FONT + '<w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/>'
z = zipfile.ZipFile(DOCX); files = {n: z.read(n) for n in z.namelist()}; z.close()
st = files["word/styles.xml"].decode()
st = re.sub(r"<w:docDefaults>.*?</w:docDefaults>",
            f'<w:docDefaults><w:rPrDefault><w:rPr>{BASE_R}<w:lang w:val="en-US"/></w:rPr></w:rPrDefault>'
            f'<w:pPrDefault><w:pPr>{DOUBLE}</w:pPr></w:pPrDefault></w:docDefaults>', st, flags=re.S)
st = patch_style(st, "Normal", DOUBLE, BASE_R)
for sid in ("BodyText", "FirstParagraph"):
    st = patch_style(st, sid, DOUBLE + '<w:ind w:firstLine="720"/>', BASE_R)
st = patch_style(st, "Title", DOUBLE + '<w:jc w:val="center"/>', BASE_R + "<w:b/>")
for sid in ("Author", "Date"):
    st = patch_style(st, sid, DOUBLE + '<w:jc w:val="center"/>', BASE_R)
st = patch_style(st, "Heading1", DOUBLE + '<w:keepNext/><w:jc w:val="center"/>', BASE_R + "<w:b/>")
st = patch_style(st, "Heading2", DOUBLE + '<w:keepNext/><w:jc w:val="center"/>', BASE_R + "<w:b/>")   # APA level 1
st = patch_style(st, "Heading3", DOUBLE + '<w:keepNext/>', BASE_R + "<w:b/>")                          # APA level 2
st = patch_style(st, "Bibliography", DOUBLE + '<w:ind w:left="720" w:hanging="720"/>', BASE_R)
st = patch_style(st, "Compact", '<w:spacing w:line="240" w:lineRule="auto" w:before="0" w:after="0"/><w:ind w:firstLine="0"/>', FONT + '<w:sz w:val="20"/><w:szCs w:val="20"/>')
st = patch_style(st, "VerbatimChar", None, '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="20"/><w:szCs w:val="20"/>')
if 'w:styleId="FigureCaption"' not in st:
    st = st.replace("</w:styles>", '<w:style w:type="paragraph" w:styleId="FigureCaption"><w:name w:val="FigureCaption"/><w:basedOn w:val="Normal"/></w:style></w:styles>')
st = patch_style(st, "FigureCaption", DOUBLE + '<w:keepNext/><w:ind w:firstLine="0"/>', BASE_R)
files["word/styles.xml"] = st.encode()
doc = files["word/document.xml"].decode()
doc = re.sub(r'<w:pgMar [^/]*/>', '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>', doc)
# page break after the title block
last_author = [m for m in re.finditer(r'<w:p>\s*<w:pPr>\s*<w:pStyle w:val="Author"\s*/>.*?</w:p>', doc, flags=re.S)][-1]
doc = doc[:last_author.end()] + '<w:p><w:r><w:br w:type="page"/></w:r></w:p>' + doc[last_author.end():]
# page-number header
HDR = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
       '<w:p><w:pPr><w:jc w:val="right"/><w:spacing w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
       '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
       '<w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:hdr>')
files["word/header1.xml"] = HDR.encode()
rels = files["word/_rels/document.xml.rels"].decode()
rels = rels.replace("</Relationships>", '<Relationship Id="rIdHdr1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/></Relationships>')
files["word/_rels/document.xml.rels"] = rels.encode()
ct = files["[Content_Types].xml"].decode()
ct = ct.replace("</Types>", '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/></Types>')
files["[Content_Types].xml"] = ct.encode()
doc = re.sub(r"<w:sectPr(.*?)>", lambda m: '<w:sectPr%s><w:headerReference w:type="default" r:id="rIdHdr1"/>' % m.group(1) if 'headerReference' not in m.group(0) else m.group(0), doc, count=1, flags=re.S)
files["word/document.xml"] = doc.encode()
with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zo:
    for n, b in files.items(): zo.writestr(n, b)
tmp.unlink()
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(DOCX)], check=True, capture_output=True)
print("built", DOCX, DOCX.with_suffix(".pdf"))
