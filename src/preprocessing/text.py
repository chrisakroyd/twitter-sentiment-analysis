import re
import unicodedata
from wordsegment import load, segment

load()

FLAGS = re.MULTILINE | re.DOTALL
whitespace_chars = {' ', '\t', '\n', '\r', '\u200b', '\u200c', '\u200d', '\ufeff', '\u200e'}
dash_chars = {'-'}
articles = re.compile(r'\b(a|an|the)\b')
apostrophe = re.compile(r"('')")
# Should filter out wiki style references e.g. [3], [123], [citation needed]
double_apostrophe = re.compile(r"('')")
apostrophe_like = re.compile(r'(`)')
multi_spaces = re.compile(r'\s{2,}')
elipsiss = re.compile(r'([.]{2,})')
emoji_regex = re.compile(r'[\uD83C-\uDBFF\uDC00-\uDFFF]+')

space_before = re.compile(r'([:$\\])')
double_punct = re.compile(r'(\w+[.,\/#!$%~\'\"^&\*;:{}=\-_`~()\[\]])([.,\/#!$%~\'\"^&\*;:{}=\-_`~()\[\]]\w+)')

space_before_paren = re.compile(r'(\w+|[^\w\s])(\((\w+))')
space_after_paren = re.compile(r'(\))(\w+)')

hashtag_regex = re.compile("#\S+")
hashtag_splitter_regex = re.compile(r'((?<=[a-z])[A-Z]|[A-Z](?=[a-z]))')
word_split = re.compile(r'[/\-_\\/]')


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    hashtag_body = ' '.join(segment(hashtag_body))

    hashtag_body = hashtag_splitter_regex.sub(r' \1', hashtag_body)
    result = " ".join([" -hashtag- "] + hashtag_body.split(r"(?=[A-Z])") + [" -/hashtag- "])
    return result


def annotate_hashtags(text):
    text = re.sub(r'#', ' #', text)
    text = hashtag_regex.sub(hashtag, text)
    return text


def replace_smileys(text):
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    loleyes = r"[8:=;]"
    eyes = r"[8:=;Xx]"
    nose = r"['`^\-0Oo]?"

    text = re_sub(r"\B{}{}[)D]+|[(D]+{}{}".format(eyes, nose, nose, eyes), ' -smile- ')
    text = re_sub(r"\B{}{}[pPbB]+".format(loleyes, nose), ' -lolface- ')
    text = re_sub(r"\B{}{}[(?]+|\)+{}{}".format(eyes, nose, nose, eyes), ' -sadface- ')
    text = re_sub(r"\B{}{}[\/\\|l]".format(eyes, nose), ' -neutralface- ')
    text = re_sub(r"\B{}{}[*]".format(eyes, nose), ' -kisses- ')

    # Replace kisses e.g. xx, xoxoxo
    text = re.sub(r'(\b([Xx][Oo]){1,}\b)|(\b[Xx]{2,}\b)|(\b[Xx]$)', ' -kisses- ', text)

    return text


def normalize(text):
    """
        Normalizes unicode whitespace, dashes and invalid characters. As this does not modify the length or position
        of words it is considered non-destructive.
        Args:
            text: String text to be cleaned.
        Returns:
            Cleaned string.
    """
    text = text.strip()
    text = list(text)
    # Normalize spaces + remove invalid characters.
    out_text = []

    for char in text:
        if is_invalid(char):
            continue

        if is_whitespace(char):
            out_text.append(' ')
        elif is_dash(char):
            out_text.append(' - ')
        elif is_math_symbol(char):
            out_text.append(' {} '.format(char))
        else:
            out_text.append(char)

    return ''.join(out_text)


def clean(text):
    """ Cleans the given text by removing wikipedia noise ([citation needed], [1], etc.) recurring punctuation and
        multiple spaces. As this may significantly modify the string, any answer pointers will need to be updated
        before used for training.

        Args:
            text: String text to be cleaned.
        Returns:
            Cleaned string.
    """
    text = word_split.sub(' ', text)
    text = space_before.sub(r' \1', text)
    text = elipsiss.sub(r' \1 ', text)
    text = apostrophe_like.sub(r' \1', text)
    text = space_before_paren.sub(r'\1 \2', text)
    text = space_after_paren.sub(r'\1 \2', text)
    text = emoji_regex.sub('', text)
    text = normalize(text)
    text = annotate_hashtags(text)
    text = replace_smileys(text)
    text = double_punct.sub(r'\1 \2', text)
    text = multi_spaces.sub(' ', text)
    text = text.strip()
    return text


def is_whitespace(char):
    """ Checks if the unicode character is a form of whitespace """
    cat = unicodedata.category(char)
    if char in whitespace_chars or cat == 'Zs':
        return True
    return False


def is_dash(char):
    """ Checks if the unicode character is a dash character """
    cat = unicodedata.category(char)
    return char in dash_chars or cat == 'Pd'


def is_math_symbol(char):
    """ Checks if the unicode character is a math symbol """
    cat = unicodedata.category(char)
    return cat == 'Sm'


def is_invalid(char):
    char_int = ord(char)
    return char_int == 0 or char_int == 0xfffd
