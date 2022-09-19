import re
from nltk.corpus import wordnet
R_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'I\'m', 'I am'),
    (r'Let\'s', 'Let us'),
    (r'(\w+)\'d', '\g<1> would'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
]


class REReplacer(object):
    def __init__(self, patterns=R_patterns):
        self.patterns = [(re.compile(regex), repl)
                         for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s
