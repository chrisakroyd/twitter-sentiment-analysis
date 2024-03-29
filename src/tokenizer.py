import string
from collections import Counter
import spacy
import re

default_punct = set(string.punctuation)


REGEX_LOOKUP = {
    "EMAIL": "(?:^|(?<=[^\\w@.)]))(?:[\\w+-](?:\\.(?!\\.))?)*?[\\w+-]@(?:\\w-?)*?\\w+(?:\\.(?:[a-z]{2,})){1,3}(?:$|(?=\\b))",
    "URL": "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
    "NUMBERS": "[-+]?([\d]+[.\d]*)",
}

url_regex = re.compile(REGEX_LOOKUP['URL'])
email_regex = re.compile(REGEX_LOOKUP['EMAIL'])
numbers_regex = re.compile(REGEX_LOOKUP['NUMBERS'])


class Tokenizer(object):
    def __init__(self, lower=False, filters=default_punct, max_words=25000, max_chars=2500, min_word_occurrence=-1,
                 min_char_occurrence=-1, vocab=None, word_index=None, char_index=None, oov_token='<oov>',
                 trainable_words=None):
        """
            Constructs a Tokenizer Object, this is a wrapper around Spacy which tracks word/character usage as well
            as handling trainable words, unknown word tokens and filters.

            Args:
                lower: Whether to lower case all strings.
                filters: Set, list or string of any punctuation we should ignore.
                max_words: Maximum number of words to include in the word_index.
                max_chars: Maximum number of chars to include in the char_index.
                min_word_occurrence: How many times a word must occur to be included in the word_index.
                min_char_occurrence: How many times a char must occur to be included in the char_index.
                vocab: A list of strings that are considered our vocab e.g. list of words with embeddings.
                word_index: A dict of word: index mappings.
                char_index: A dict of char: index mappings.
                oov_token: Token to replace out of vocabulary words/chars with.
                trainable_words: A list of words which we want to have trainable embeddings.
        """
        self.word_counter = Counter()
        self.char_counter = Counter()
        self.tag_counter = Counter()
        self.vocab = set(vocab if vocab else [])
        self.trainable_words = set(trainable_words if trainable_words else [])
        self.vocab = self.vocab | self.trainable_words

        self.word_index = word_index if word_index else {}
        self.char_index = char_index if char_index else {}
        self.max_words = max_words
        self.max_chars = max_chars
        self.oov_token = oov_token
        self.lower = lower
        self.min_word_occurrence = min_word_occurrence
        self.min_char_occurrence = min_char_occurrence
        # Flag for if we have just run the fit_text or if we have a set vocab.
        self.just_fit = False
        self.given_vocab = vocab is not None

        self.nlp = spacy.load('en_core_web_sm', disable=['parser'])
        self.tag_index = {key: i for i, key in enumerate(self.nlp.tokenizer.vocab.morphology.tag_map.keys())}

        if not isinstance(filters, set) and filters is not None:
            self.filters = set(filters)
        else:
            self.filters = set()

        self.init()

    def tokenize(self, text, error_correct=True):
        """ Splits a text or list of text into its constituent words.
            Args:
                text: string or list of string of untokenized text.
                error_correct: If we have a vocab, attempts to correct minor errors against it. e.g. Token is steve
                               but vocab only has Steve -> We replace steve with Steve. If in the vocab we do nothing.
            returns:
                Two lists, a list of original tokens and a list of corrected tokens.
        """
        if self.lower:
            text = text.lower()

        original_tokens = []
        modified_tokens = []
        pos_tags = []

        for token in self.nlp(text):
            text = token.text
            tag = token.tag_
            token_corrected = False

            if self.given_vocab and error_correct and text not in self.vocab:
                # Replace numbers, urls and emails with a token that is optionally trainable.
                if numbers_regex.match(text):
                    text = '-number-'
                    token_corrected = True
                elif url_regex.match(text):
                    text = '-url-'
                    token_corrected = True
                elif email_regex.match(text):
                    text = '-email-'
                    token_corrected = True
                else:
                    # We generate a short list of candidate words and replace the
                    # original text if any of them are in the vocab.
                    for word in (text.lower(), text.capitalize(), text.lower().capitalize(), text.upper(), token.lemma_):
                        if word in self.vocab:
                            text = word
                            token_corrected = True
                            break

            if text not in self.filters and len(text) > 0:
                if token_corrected:
                    original_tokens.append(token.text)
                    modified_tokens.append(text)
                else:
                    original_tokens.append(text)
                    modified_tokens.append(text)

            pos_tags.append(tag)

        assert len(original_tokens) == len(modified_tokens)

        return original_tokens, modified_tokens, pos_tags

    def fit_on_texts(self, texts, error_correct=True):
        """ Counts word/character occurrence.
            Args:
                texts: string or list of string of untokenized text.
                error_correct: If we have a vocab, attempts to correct minor errors against it. e.g. Token is steve
                               but vocab only has Steve -> We replace steve with Steve. If in the vocab we do nothing.
            returns:
                A list of tuples containing tokens and modified tokens.
        """
        tokenized = []
        if not isinstance(texts, list):
            texts = [texts]

        for text in texts:
            tokens, modified_tokens, pos_tags = self.tokenize(text, error_correct)

            for token, tag in zip(modified_tokens, pos_tags):
                self.word_counter[token] += 1
                self.tag_counter[tag] += 1

                for char in list(token):
                    self.char_counter[char] += 1

            tokenized.append((tokens, modified_tokens, pos_tags, ))

        self.just_fit = True
        return tokenized

    def update_indexes(self):
        """ Creates word, character indexes and handles trainable words.

            To facilitate trainable embeddings only for a subset of word and OOV tokens we need a special way of
            handling the word and char index creation. We create the word indexes as normal, but instead of assigning
            trainable word ids based on how often it occurs they are always assigned to the highest Id's. For details
            on why refer to docstrings in src/models/embedding_layer.
        """
        print('Total Unique Words: %d' % len(self.word_counter))
        sorted_words = self.word_counter.most_common(self.max_words)
        sorted_chars = self.char_counter.most_common(self.max_chars)

        word_index = [word for (word, count) in sorted_words if count > self.min_word_occurrence and
                      word in self.vocab and word not in self.filters and word not in self.trainable_words]

        print('Words in vocab: %d' % len(word_index))

        char_index = [char for (char, count) in sorted_chars if count > self.min_char_occurrence and
                      char not in self.filters]

        # Put all the Ids in a continues range from 1 to len(vocab), this is necessary to keep indices in sync.
        word_index = {word: i for i, word in enumerate(word_index, start=1)}
        char_index = {char: i for i, char in enumerate(char_index, start=1)}

        # Add any trainable words to the end of the index (Refer to src/models/embedding_layer for a full reason why)
        vocab_size = len(word_index)
        for i, word in enumerate(self.trainable_words, start=1):
            assert (vocab_size + i) not in word_index.values()
            word_index[word] = vocab_size + i

        # Add OOV token to the word + char index (So we always have an OOV token in the vocab).
        if self.oov_token not in word_index:
            assert len(word_index) + 1 not in word_index.values()
            word_index[self.oov_token] = len(word_index) + 1  # Add OOV as the last character

        if self.oov_token not in char_index:
            assert len(char_index) + 1 not in char_index.values()
            char_index[self.oov_token] = len(char_index) + 1  # Add OOV as the last character

        self.word_index = word_index
        self.char_index = char_index

    def update_vocab(self):
        """ If we don't have a vocab sets the vocab to all the words within the word index. """
        if not self.given_vocab:
            self.vocab = set([word for word, _ in self.word_counter.items()])

    def set_vocab(self, vocab):
        """ Sets the vocab and prevents it being automatically changed.
            Args:
                vocab: List of strings for words that are in the vocab (e.g. words with embeddings).
        """
        self.vocab = set(vocab)
        self.given_vocab = True
        self.just_fit = True

    def init(self):
        """ Initialises the vocab, word and char indices if they have not been set or need updating. """
        # Word indexes haven't been initialised or need updating.
        if (len(self.word_index) == 0 or len(self.char_index)) == 0 or self.just_fit:
            self.update_vocab()
            self.update_indexes()
            self.just_fit = False
