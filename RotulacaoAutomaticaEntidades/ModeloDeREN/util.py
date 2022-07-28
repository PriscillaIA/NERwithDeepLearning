from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import align_tokens
from nltk.tokenize.destructive import MacIntyreContractions
import re
import jsonlines
class TreebankWordTokenizer(TokenizerI):
    r"""
    The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.
    This is the method that is invoked by ``word_tokenize()``.  It assumes that the
    text has already been segmented into sentences, e.g. using ``sent_tokenize()``.

    This tokenizer performs the following steps:

    - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
    - treat most punctuation characters as separate tokens
    - split off commas and single quotes, when followed by whitespace
    - separate periods that appear at the end of line

        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
        >>> TreebankWordTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
        >>> s = "They'll save and invest more."
        >>> TreebankWordTokenizer().tokenize(s)
        ['They', "'ll", 'save', 'and', 'invest', 'more', '.']
        >>> s = "hi, my name can't hello,"
        >>> TreebankWordTokenizer().tokenize(s)
        ['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']
    """

    # starting quotes
    STARTING_QUOTES = [
        (re.compile(r"^\""), r"``"),
        (re.compile(r"(``)"), r" \1 "),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
    ]

    # punctuation
    PUNCTUATION = [
        (re.compile(r"([:,])([^\d])"), r" \1 \2"),
        (re.compile(r"([:,])$"), r" \1 "),
        (re.compile(r"\.\.\."), r" ... "),
        (re.compile(r"[;@#&-/]"), r" \g<0> "),
        (re.compile(r"(\d{2}) / (\d{2}) / (\d{2,4})"), r"\1/\2/\3"),
        (
            re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
            r"\1 \2\3 ",
        ),  # Handles the final period.
        (re.compile(r"[?!]"), r" \g<0> "),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")

    # Optionally: Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = [
        (re.compile(r"\("), "-LRB-"),
        (re.compile(r"\)"), "-RRB-"),
        (re.compile(r"\["), "-LSB-"),
        (re.compile(r"\]"), "-RSB-"),
        (re.compile(r"\{"), "-LCB-"),
        (re.compile(r"\}"), "-RCB-"),
    ]

    DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r'"'), " '' "),
        (re.compile(r"(\S)(\'\')"), r"\1 \2 "),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text, convert_parentheses=False, return_str=False):
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)
        
        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handles parentheses.
        regexp, substitution = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)
        # Optionally convert parentheses
        if convert_parentheses:
            for regexp, substitution in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)

        # Handles double dash.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r" \1 \2 ", text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r" \1 \2 ", text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self._contractions.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        return text if return_str else text.split()

    def span_tokenize(self, text):
        r"""
        Uses the post-hoc nltk.tokens.align_tokens to return the offset spans.

            >>> from nltk.tokenize import TreebankWordTokenizer
            >>> s = '''Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).'''
            >>> expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23),
            ... (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38),
            ... (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59),
            ... (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]
            >>> list(TreebankWordTokenizer().span_tokenize(s)) == expected
            True
            >>> expected = ['Good', 'muffins', 'cost', '$', '3.88', 'in',
            ... 'New', '(', 'York', ')', '.', 'Please', '(', 'buy', ')',
            ... 'me', 'two', 'of', 'them.', '(', 'Thanks', ')', '.']
            >>> [s[start:end] for start, end in TreebankWordTokenizer().span_tokenize(s)] == expected
            True

            Additional example
            >>> from nltk.tokenize import TreebankWordTokenizer
            >>> s = '''I said, "I'd like to buy some ''good muffins" which cost $3.88\n each in New (York)."'''
            >>> expected = [(0, 1), (2, 6), (6, 7), (8, 9), (9, 10), (10, 12),
            ... (13, 17), (18, 20), (21, 24), (25, 29), (30, 32), (32, 36),
            ... (37, 44), (44, 45), (46, 51), (52, 56), (57, 58), (58, 62),
            ... (64, 68), (69, 71), (72, 75), (76, 77), (77, 81), (81, 82),
            ... (82, 83), (83, 84)]
            >>> list(TreebankWordTokenizer().span_tokenize(s)) == expected
            True
            >>> expected = ['I', 'said', ',', '"', 'I', "'d", 'like', 'to',
            ... 'buy', 'some', "''", "good", 'muffins', '"', 'which', 'cost',
            ... '$', '3.88', 'each', 'in', 'New', '(', 'York', ')', '.', '"']
            >>> [s[start:end] for start, end in TreebankWordTokenizer().span_tokenize(s)] == expected
            True

        """
        raw_tokens = self.tokenize(text)

        # Convert converted quotes back to original double quotes
        # Do this only if original text contains double quote(s) or double
        # single-quotes (because '' might be transformed to `` if it is
        # treated as starting quotes).
        if ('"' in text) or ("''" in text):
            # Find double quotes and converted quotes
            matched = [m.group() for m in re.finditer(r"``|'{2}|\"", text)]

            # Replace converted quotes back to double quotes
            tokens = [
                matched.pop(0) if tok in ['"', "``", "''"] else tok
                for tok in raw_tokens
            ]
        else:
            tokens = raw_tokens

        for tok in align_tokens(tokens, text):
            yield tok

## Metodo para tokenizar tudo (Todas as sentença do aruivo Dados_Utilizados_Para_RotulacaoManual.txt) 
def tokenizeAn(lista):
    resultado = []
    for item in lista:
        sentenca = item['data']
        labels = item['label']
        wordSpansList = list(TreebankWordTokenizer().span_tokenize(sentenca))
        lista_aux = []
        for wordSpan in wordSpansList:
            ## Adicionando a setença original, e o index da setença do arquivo original .txt
            aux = (wordSpan[0], wordSpan[1], '')
            for label in labels:
                if wordSpan[0] >= label[0] and wordSpan[1] <= label[1]:
                    aux = (label[0], label[1], label[2])
                    break
            if aux not in lista_aux:
                lista_aux.append(aux)
        resultado.append((sentenca, {"entities" : lista_aux}))
    
    ## Metodo retorna resultado (sentanças tokenizada, juntamente com a rotulação da palavra)
    return resultado