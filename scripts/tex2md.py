"""Convert ``paper/main.tex`` into readable Markdown.

Usage:
    python scripts/tex2md.py
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parent.parent
TEX = ROOT / 'paper' / 'main.tex'
OUT = ROOT / 'paper' / 'paper_readable.md'

MACRO_MAP = {
    r'\btheta': r'\boldsymbol{\theta}',
    r'\blambda': r'\boldsymbol{\lambda}',
    r'\Loss': r'\mathcal{L}',
    r'\Net': r'\mathcal{N}',
    r'\SSR': r'\mathcal{S}',
    r'\bV': r'\mathbf{V}',
    r'\bx': r'\mathbf{x}',
    r'\by': r'\mathbf{y}',
    r'\bz': r'\mathbf{z}',
    r'\relu': r'\mathrm{ReLU}',
    r'\R': r'\mathbb{R}',
}


def extract_body(src: str) -> str:
    start_tag = r'\begin{document}'
    end_tag = r'\end{document}'
    start = src.find(start_tag)
    end = src.find(end_tag)
    if start < 0 or end < 0 or end <= start:
        raise ValueError('Could not locate \\begin{document} ... \\end{document}.')
    return src[start + len(start_tag):end]


def replace_macros(text: str) -> str:
    for key in sorted(MACRO_MAP, key=len, reverse=True):
        text = text.replace(key, MACRO_MAP[key])
    return text


def replace_block_env(text: str, env: str, placeholder: str) -> str:
    pattern = rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}'
    return re.sub(pattern, placeholder, text, flags=re.DOTALL)


def convert_math_env(text: str, env: str) -> str:
    pattern = rf'\\begin\{{{env}\*?\}}(.*?)\\end\{{{env}\*?\}}'

    def repl(match: re.Match) -> str:
        block = re.sub(r'\\label\{[^}]+\}', '', match.group(1)).strip()
        return f'\n$$\n{block}\n$$\n'

    return re.sub(pattern, repl, text, flags=re.DOTALL)


def convert_theorem_like(text: str) -> str:
    def begin_repl(match: re.Match) -> str:
        env = match.group(1)
        title = match.group(2)
        label = env.capitalize()
        if title:
            return f'\n**{label} ({title.strip()}).**\n'
        return f'\n**{label}.**\n'

    text = re.sub(
        r'\\begin\{(definition|theorem|proposition|remark)\}(?:\[([^\]]+)\])?',
        begin_repl,
        text,
    )
    text = re.sub(r'\\end\{(definition|theorem|proposition|remark)\}', '\n', text)
    return text


def convert_basic_commands(text: str) -> str:
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    text = re.sub(r'\\maketitle\b', '', text)
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    text = re.sub(r'\\ref\{[^}]+\}', '[ref]', text)
    text = re.sub(r'\\cite\{[^}]+\}', '[cite]', text)

    text = re.sub(r'\\begin\{abstract\}', '\n## Abstract\n', text)
    text = re.sub(r'\\end\{abstract\}', '\n', text)
    text = re.sub(r'\\begin\{IEEEkeywords\}', '\n**Keywords:** ', text)
    text = re.sub(r'\\end\{IEEEkeywords\}', '\n', text)

    text = re.sub(r'\\section\*?\{([^}]+)\}', r'\n# \1\n', text)
    text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\n## \1\n', text)
    text = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'\n### \1\n', text)

    text = re.sub(r'\\begin\{(?:itemize|enumerate)\}', '', text)
    text = re.sub(r'\\end\{(?:itemize|enumerate)\}', '', text)
    text = re.sub(r'\\item\s+', '\n- ', text)

    text = re.sub(r'\\textbf\{([^{}]*)\}', r'**\1**', text)
    text = re.sub(r'\\textit\{([^{}]*)\}', r'*\1*', text)
    text = re.sub(r'\\emph\{([^{}]*)\}', r'*\1*', text)
    text = re.sub(r'\\texttt\{([^{}]*)\}', r'`\1`', text)
    text = re.sub(r'\\url\{([^{}]+)\}', r'\1', text)

    text = re.sub(r'\\(noindent|centering|small|footnotesize|normalsize|newline)\b', '', text)
    text = text.replace('~', ' ')
    return text


def finalize(text: str) -> str:
    text = re.sub(r'\n[ \t]+\n', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n'


def main() -> None:
    src = TEX.read_text(encoding='utf-8')
    body = extract_body(src)
    body = replace_macros(body)

    body = replace_block_env(body, 'figure', '\n> [Figure omitted; see the figures directory.]\n')
    body = replace_block_env(body, 'table', '\n> [Table omitted in readable markdown.]\n')
    body = replace_block_env(body, 'algorithm', '\n> [Algorithm omitted in readable markdown.]\n')

    for env in ('equation', 'align', 'gather', 'multline'):
        body = convert_math_env(body, env)
    body = re.sub(r'\\\[(.*?)\\\]', lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", body, flags=re.DOTALL)

    body = convert_theorem_like(body)
    body = convert_basic_commands(body)
    body = finalize(body)

    title = re.search(r'\\title\{([^}]*)\}', src)
    md_title = title.group(1).strip() if title else 'Static Security Region Analysis'
    header = (
        f'# {md_title}\n\n'
        '> Readable Markdown generated from `paper/main.tex`.\n'
        '> Mathematical expressions are preserved in LaTeX form (`$...$`, `$$...$$`).\n\n'
    )

    OUT.write_text(header + body, encoding='utf-8')
    print(f'Written: {OUT}')
    print(f'Lines: {body.count(chr(10))}')


if __name__ == '__main__':
    main()
