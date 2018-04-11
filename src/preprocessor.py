import re
from ftfy import fix_text
from unidecode import unidecode

VALID_TAGS = {'<ALLCAPS>', '<IP>', '<URL>', '<EMAIL>', '<USER>', '<DATE>', '<TIME>', '<NUMBER>', '<CURRENCY>',
              '<REPEAT>', '<ELONG>', '<EMPTY>', '<OOV>'}

FLAGS = re.MULTILINE | re.DOTALL

REGEX_LOOKUP = {
    "DATE": "(?:(?:(?:(?:(?<!:)\\b\\'?\\d{1,4},? ?)?\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b))|(?:(?:(?<!:)\\b\\'?\\d{1,4},? ?)\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b)?))|(?:\\b(?<!\\d\\.)(?:(?:(?:[0123]?[0-9][\\.\\-\\/])?[0123]?[0-9][\\.\\-\\/][12][0-9]{3})|(?:[0123]?[0-9][\\.\\-\\/][0123]?[0-9][\\.\\-\\/][12]?[0-9]{2,3}))(?!\\.\\d)\\b))",
    "EMAIL": "(?:^|(?<=[^\\w@.)]))(?:[\\w+-](?:\\.(?!\\.))?)*?[\\w+-]@(?:\\w-?)*?\\w+(?:\\.(?:[a-z]{2,})){1,3}(?:$|(?=\\b))",
    "EMOJI": "[\uD83C-\uDBFF\uDC00-\uDFFF]+",
    "IP": "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    "TIME": "(?:(?:\d+)?\.?\d+(?:AM|PM|am|pm|a\.m\.|p\.m\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.))?)",
    "MONEY": "(?:[$€£¢]\d+(?:[\.,']\d+)?(?:[MmKkBb](?:n|(?:il(?:lion)?))?)?)|(?:\d+(?:[\.,']\d+)?[$€£¢])",
    "URL": "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
    "NUMBERS": "[-+]?[.\d]*[\d]+[:,.\d]*"
}

# Preprocessing Regex's
url_regex = re.compile(REGEX_LOOKUP['URL'])
ip_regex = re.compile(REGEX_LOOKUP['IP'])
date_regex = re.compile(REGEX_LOOKUP['DATE'])
emoji_regex = re.compile(REGEX_LOOKUP['EMOJI'])
email_regex = re.compile(REGEX_LOOKUP['EMAIL'])
time_regex = re.compile(REGEX_LOOKUP['TIME'])
money_regex = re.compile(REGEX_LOOKUP['MONEY'])
numbers_regex = re.compile(REGEX_LOOKUP['NUMBERS'])
control_chars = re.compile('[\n\t\r\v\f\0]')

parenthesis_regex = re.compile('([\[\]()])')

hearts_regex = re.compile(r'<3')
users_regex = re.compile("@\w+")
tokenize_punct = re.compile(r'([.,?!"]{1})')
repeated_punct = re.compile('([!?.]){2,}')
elongated_words = re.compile(r"\b(\S*?)(.)\2{2,}\b")
word_split = re.compile(r'[/\-_\\]')
all_caps_regex = re.compile(r'([A-Z]){2,}')
hashtag_regex = re.compile("#\S+")

mentions_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_.]))@([A-Za-z_]+[A-Za-z0-9_]+)')


def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> "


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        test_regex = re.compile(r'((?<=[a-z])[A-Z]|[A-Z](?=[a-z]))')
        hashtag_body = test_regex.sub(r' \1', hashtag_body)
        result = " ".join(["<hashtag>"] + hashtag_body.split(r"(?=[A-Z])") + ["</hashtag>"])
    return result


class TextPreProcessor:
    def __init__(self, embedding_profile=None):
        self.embedding_profile = embedding_profile

    def preprocess(self, string):
        string = self.clean(string)
        return string

    def unpack_contractions(self, text):
        """
        Replace *English* contractions in ``text`` str with their unshortened forms.
        N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
        so are left as-is.
        ---------
        Important Note: The function is taken from textacy (https://github.com/chartbeat-labs/textacy).
        """
        text = re.sub(
            r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
            r"\1\2 not", text)
        text = re.sub(
            r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
            r"\1\2 will", text)
        text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are",
                      text)
        text = re.sub(
            r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",
            r"\1\2 have", text)
        # non-standard
        text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
        text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
        text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
        text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
        return text

    def unpack_placements(self, text):
        text = re.sub(r'1[sS][tT]', 'first', text)
        text = re.sub(r'2[nN][dD]', 'second', text)
        text = re.sub(r'3[rR][dD]', 'third', text)
        text = re.sub(r'4[tT][hH]', 'fourth', text)
        text = re.sub(r'5[tT][hH]', 'fifth', text)
        text = re.sub(r'6[tT][hH]', 'sixth', text)
        text = re.sub(r'7[tT][hH]', 'seventh', text)
        text = re.sub(r'8[tT][hH]', 'eigth', text)
        text = re.sub(r'9[tT][hH]', 'ninth', text)
        text = re.sub(r'10[tT][hH]', 'tenth', text)

        return text

    def replace_smileys(self, text):
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)

        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")

        return text

    def clean(self, text):
        # Fix unicode characters
        text = fix_text(text)
        text = unidecode(text)

        # Replace newline characters
        text = control_chars.sub(' ', text)
        # Replace ips
        text = ip_regex.sub(' <ip> ', text)
        # Replace URLs
        text = url_regex.sub(' <url> ', text)
        # Replace Emails
        text = email_regex.sub(' <email> ', text)
        # Replace User Names
        text = users_regex.sub(' <user> ', text)
        # Replace Dates/Time
        text = date_regex.sub(' <date> ', text)
        text = time_regex.sub(' <time> ', text)
        # Replace money symbols
        text = money_regex.sub(' <currency> ', text)

        text = self.unpack_placements(text)
        text = self.replace_smileys(text)
        text = word_split.sub(' ', text)

        text = hashtag_regex.sub(hashtag, text)

        text = hearts_regex.sub(' <heart> ', text)
        # Replace Numbers
        text = numbers_regex.sub(' <number> ', text)

        text = parenthesis_regex.sub(r' \1 ', text)

        # Remove multi spaces
        text = re.sub('\s+', ' ', text)

        text = repeated_punct.sub(r' \1 <repeat> ', text)

        text = tokenize_punct.sub(r' \1 ', text)

        text = all_caps_regex.sub(allcaps, text)
        # text = elongated_words.sub(r"\1", text)
        text = elongated_words.sub(r"\1\2 <elong> ", text)

        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')

        # Remove a load of unicode emoji characters
        text = emoji_regex.sub('', text)
        text = self.unpack_contractions(text)

        # Remove multi spaces
        text = re.sub('\s+', ' ', text)
        # Remove ending space if any
        if len(text) > 1:
            text = re.sub('\s+$', '', text)

        # If this string is a single space replace with an <empty> tag.
        if text == ' ':
            text = '<EMPTY>'

        return text.strip().lower()
