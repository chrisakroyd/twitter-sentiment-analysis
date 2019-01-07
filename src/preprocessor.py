import re
from ftfy import fix_text
from unidecode import unidecode
from wordsegment import load, segment

load()

FLAGS = re.MULTILINE | re.DOTALL

REGEX_LOOKUP = {
    "DATE": "(?:(?:(?:(?:(?<!:)\\b\\'?\\d{1,4},? ?)?\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b))|(?:(?:(?<!:)\\b\\'?\\d{1,4},? ?)\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b)?))|(?:\\b(?<!\\d\\.)(?:(?:(?:[0123]?[0-9][\\.\\-\\/])?[0123]?[0-9][\\.\\-\\/][12][0-9]{3})|(?:[0123]?[0-9][\\.\\-\\/][0123]?[0-9][\\.\\-\\/][12]?[0-9]{2,3}))(?!\\.\\d)\\b))",
    "EMAIL": "(?:^|(?<=[^\\w@.)]))(?:[\\w+-](?:\\.(?!\\.))?)*?[\\w+-]@(?:\\w-?)*?\\w+(?:\\.(?:[a-z]{2,})){1,3}(?:$|(?=\\b))",
    "EMOJI": "[\uD83C-\uDBFF\uDC00-\uDFFF]+",
    "IP": "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    "TIME": "(?:(?:\d+)?\.?\d+(?:AM|PM|am|pm|a\.m\.|p\.m\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.))?)",
    "MONEY": "(?:[$€£¢]\d+(?:[\.,']\d+)?(?:[MmKkBb](?:n|(?:il(?:lion)?))?)?)|(?:\d+(?:[\.,']\d+)?[$€£¢])",
    "URL": "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
    "NUMBERS": "[-+]?([\d]+[.\d]*)",
    "PERCENT": "[-+]?([\d]+[.\d]*)\s{0,1}%",
    "SCORES": "\d+\s{0,1}-\s{0,1}\d+"
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
percent_regex = re.compile(REGEX_LOOKUP['PERCENT'])
scores_regex = re.compile(REGEX_LOOKUP['SCORES'])

control_chars = re.compile('[\n\t\r\v\f\0]')
parenthesis_regex = re.compile('([\[\]()])')
hearts_regex = re.compile(r'(♥)|(<3{1,})')
users_regex = re.compile("@\w+")
tokenize_punct = re.compile(r'([.,?!"]{1})')
repeated_punct = re.compile('([!?.]){2,}')
elongated_words = re.compile(r"\b(\S*?)(.)\2{2,}\b")

word_split = re.compile(r'[/\-_\\/]')
all_caps_regex = re.compile(r'([A-Z]){2,}')

hashtag_regex = re.compile("#\S+")
mentions_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_.]))@([A-Za-z_]+[A-Za-z0-9_]+)')

hashtag_splitter_regex = re.compile(r'((?<=[a-z])[A-Z]|[A-Z](?=[a-z]))')
# seperates things like the'
seperate_apostrophes = re.compile("(([A-zA-Z]){3,})(')")
# seperates things like 'the
seperate_errant_apostrophes = re.compile("(')(([A-zA-Z]){3,})")
arrows = re.compile("[<>]")


def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> "


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    is_uppercase = hashtag_body.isupper()
    hashtag_body = ' '.join(segment(hashtag_body))

    if is_uppercase:
        result = " <hashtag> {} <allcaps> </hashtag>".format(hashtag_body)
    else:
        hashtag_body = hashtag_splitter_regex.sub(r' \1', hashtag_body)
        result = " ".join([" <hashtag>"] + hashtag_body.split(r"(?=[A-Z])") + ["</hashtag> "])
    return result


class TextPreProcessor:
    def __init__(self, embedding_profile=None):
        self.embedding_profile = embedding_profile

    def preprocess(self, string):
        string = self.clean(string)
        # Add annotations e.g. 11:55pm -> <time>
        string = self.annotate(string)

        return string

    def clean(self, text):
        # Fix unicode characters
        text = fix_text(text)
        text = unidecode(text)
        # Replace newline and other control characters
        text = control_chars.sub(' ', text)
        text = arrows.sub(' ', text)
        return text

    def unpack_contractions(self, text):
        """
        Replace *English* contractions in ``text`` str with their unshortened forms.
        N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
        so are left as-is.
        ---------
        Important Note: The function is taken from textacy (https://github.com/chartbeat-labs/textacy).
        """
        text = re.sub(
            r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]are||[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii][Ss]|[Mm]ay|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]as|[Ww]ere|[Ww]ould|[Nn]eed)[Nn]'[Tt]",
            r"\1\2 not", text)
        text = re.sub(
            r"(\b)([Hh]e|[Hh]ow|[Ii]|[Ss]he|[Tt]hat|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
            r"\1\2 will", text)
        text = re.sub(r"(\b)([Tt]hat|[Tt]hey|[Hh]ow|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are",
                      text)
        text = re.sub(
            r"(\b)([Ii]|[Cc]ould|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou|[Mm]ight|[Mm]ust|[Mm]ay)'ve",
            r"\1\2 have", text)
        # non-standard
        text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
        text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
        text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
        text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
        text = re.sub(r"(\b)([Cc])'[Mm]on", r"\1\2ome on", text)
        text = re.sub(r"(\b)([Ww])/", " with ", text)
        text = re.sub(r"(\b)([Ww])/([Oo])", " without ", text)
        text = re.sub(r"(\b)([Hh])rs", "hours", text)
        text = re.sub(r"(\b)([Mm])ins", "minutes", text)
        text = re.sub(r"(\b)([Pp])ls", "please", text)
        text = re.sub(r"(\b)([Ss])ry", "sorry", text)
        text = re.sub(r"(\b)([Tt])ht", "that", text)
        text = re.sub(r"(\b)([Ii])nt'l", "international", text)
        text = re.sub(r"(\b)([Uu])(\b)", " you ", text)
        text = re.sub(r"\b([\/])\b", " or ", text)
        # Unambigous shortened day names.
        text = re.sub(
            r"(\b)([Mm]on|[Tt]ues|[Tt]hurs|[Ff]ri)(\b)",
            r"\1\2day", text)

        # Add a space around 's, 'd, 'll and n't due to enable word vectors to be used.
        text = re.sub(r"\b(\w+)('s|'d|'ll)", r"\1 \2", text)
        text = re.sub(r"\b(\w+)(n't)", r"\1 n't", text)

        # Expand out common text symbols
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')

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

        loleyes = r"[8:=;]"
        eyes = r"[8:=;Xx]"
        nose = r"['`^\-0Oo]?"

        text = re_sub(r"\B{}{}[)D]+|[(D]+{}{}".format(eyes, nose, nose, eyes), ' <smile> ')
        text = re_sub(r"\B{}{}[pPbB]+".format(loleyes, nose), ' <lolface> ')
        text = re_sub(r"\B{}{}[(?]+|\)+{}{}".format(eyes, nose, nose, eyes), ' <sadface> ')
        text = re_sub(r"\B{}{}[\/\\|l]".format(eyes, nose), ' <neutralface> ')
        text = re_sub(r"\B{}{}[*]".format(eyes, nose), ' <kisses> ')

        # Replace kisses e.g. xx, xoxoxo
        text = re.sub(r'(\b([Xx][Oo]){1,}\b)|(\b[Xx]{2,}\b)|(\b[Xx]$)', ' <kisses> ', text)

        return text

    def annotate_basic_attributes(self, text):
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
        # Replace abstract number values with an annotation
        text = money_regex.sub(' <currency> ', text)
        text = percent_regex.sub(' <percent> ', text)
        # Replaces simple sport scores etc.
        text = scores_regex.sub(' <score> ', text)
        return text

    def annotate_hashtags(self, text):
        text = re.sub(r'#', ' #', text)
        text = hashtag_regex.sub(hashtag, text)

        return text

    def annotate_text_features(self, text):
        # Add in repeat annotation
        text = repeated_punct.sub(r' \1 <repeat> ', text)

        text = tokenize_punct.sub(r' \1 ', text)
        # Add in annotations for all-caps and elongated words
        text = all_caps_regex.sub(allcaps, text)
        text = elongated_words.sub(r"\1\2 <elong> ", text)
        return text

    def annotate(self, text):
        # Annotate basic variables e.g. IP, URL's, Dates, times, scores etc.
        text = self.annotate_basic_attributes(text)
        # Unpack can't -> can not, c'mon -> come on
        text = self.unpack_contractions(text)
        # 1st, 2nd, 3rd  -> First, Second, Third. Performed post-date annotation so we don't remove any useful details
        text = self.unpack_placements(text)
        # Replace various different smileys with a common annotation.
        text = self.replace_smileys(text)

        # Split words
        text = word_split.sub(' ', text)

        # Add spaces around parenthesis.
        text = parenthesis_regex.sub(r' \1 ', text)

        # Add in wraparound annotations which surround text
        # Introduce a space before hash symbols to prevent issues where hastags are next to each other
        # e.g. #test#chicken
        text = self.annotate_hashtags(text)

        # Replace <3 (less than three) with the more meaningful <heart> annotation
        text = hearts_regex.sub(' <heart> ', text)
        # Replace Numbers
        text = numbers_regex.sub(' <number> ', text)
        # Remove multi spaces - Prevents errant <repeat> signals appearing in text
        text = re.sub('\s+', ' ', text)
        # Annotate text features e.g. elongated words, repeated punctuation.
        text = self.annotate_text_features(text)

        text = seperate_apostrophes.sub(r' \1 \2 ', text)
        text = seperate_errant_apostrophes.sub(r' \1 \2 ', text)
        # Remove a load of unicode emoji characters
        text = emoji_regex.sub('', text)

        text = text.replace('*', '')

        # Remove multi spaces
        text = re.sub('\s+', ' ', text)
        # Remove ending space if any
        if len(text) > 1:
            text = re.sub('\s+$', '', text)

        # If this string is a single space replace with an <empty> tag.
        if text == ' ':
            text = '<empty>'

        return text.strip().lower()
