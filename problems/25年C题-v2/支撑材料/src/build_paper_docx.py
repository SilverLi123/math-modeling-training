"""Build the common manuscript with the installed math-modeling DOCX helpers."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(skill_root):
    sys.path.insert(0, str(skill_root/'tools/docx/scripts'))
    import paper_format as pf
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    source = ROOT/'paper/manuscript.md'
    doc = pf.new_document(contest='cumcm',
                          template_path=skill_root/'references/roles/论文手/references/论文模板.docx')
    # Electronic format starts at the abstract, no identity/numbering pages.
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Cm(2.5)
        section.left_margin = section.right_margin = Cm(2.5)
        footer = section.footer.paragraphs[0]
        footer.alignment = 1
        field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE')
        footer._p.append(field)
    lines = source.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line:
            continue
        if line == '<PAGEBREAK>':
            pf.page_break(doc)
        elif line == '$$':
            equation = []
            while i < len(lines) and lines[i].strip() != '$$':
                equation.append(lines[i]); i += 1
            if i == len(lines):
                raise ValueError('Unclosed equation')
            i += 1
            pf.equation(doc, ' '.join(equation))
        elif line.startswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip()); i += 1
            parsed = [[c.strip() for c in r.strip('|').split('|')] for r in rows]
            parsed = [r for r in parsed if not all(re.fullmatch(r'[:\- ]+', c) for c in r)]
            table = pf.three_line_table(doc, parsed)
            # Tables contain dense numeric evidence; 10 pt is still legible on
            # A4 and prevents a two-row table tail from occupying a whole page.
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.line_spacing = 1.0
                        for run in paragraph.runs:
                            run.font.size = Pt(10)
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
            if not match:
                raise ValueError(f'Invalid image: {line}')
            path = ROOT/'paper'/match[2]
            pf.image(doc, path, width_cm=15.2)
            pf.figure_caption(doc, match[1])
        elif line.startswith('# '):
            pf.title(doc, line[2:])
        elif line == '## 摘要':
            pf.abstract_title(doc)
        elif line.startswith('## '):
            heading = pf.heading1(doc, line[3:])
            # Start a major heading at the next available position.  Word's
            # page-break-before setting can otherwise introduce a blank page
            # when the prior table exactly fills a page.
            heading.paragraph_format.page_break_before = False
        elif line.startswith('### '):
            pf.heading2(doc, line[4:])
        elif line.startswith('关键词：'):
            pf.keywords(doc, line[4:])
        else:
            # The official template leaves no mandatory fixed line spacing.
            # Keep a readable compact layout so a figure or a section heading
            # does not get stranded on a nearly blank page.
            paragraph = pf.body(doc, line)
            paragraph.paragraph_format.line_spacing = 1.10
    doc.core_properties.author = ''
    doc.core_properties.last_modified_by = ''
    output = pf.save_document(doc, ROOT, filename='完整论文.docx', overwrite=True)
    report = {'manuscript_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'docx_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'body_structure_issues': pf.validate_paper_structure(doc, require_rendered_pages=False)}
    (ROOT/'paper/build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--skill-root', type=Path, required=True)
    main(parser.parse_args().skill_root)
