import regex as re

# Convert Vietnamese text to Telex-style input when IME (Unikey) is off

vowel_table = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
               ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
               ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
               ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
               ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
               ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
               ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
               ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
               ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
               ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
               ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
               ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
tone_suffixes = ['', 'f', 's', 'r', 'x', 'j']

vowel_to_ids = {}

for i in range(len(vowel_table)):
    for j in range(len(vowel_table[i]) - 1):
        vowel_to_ids[vowel_table[i][j]] = (i, j)


def vietnamese_word_to_telex_type(word):
    tone_idx = 0
    new_word = ''
    for char in word:
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x == -1:
            new_word += char
            continue
        if y != 0:
            tone_idx = y
        new_word += vowel_table[x][-1]
    new_word += tone_suffixes[tone_idx]
    return new_word


def vietnamese_sentence_to_telex_type(sentence):
    """
    Convert a Vietnamese sentence with diacritics to Telex typing style.
    :param sentence:
    :return:
    """
    words = sentence.split()
    for index, word in enumerate(words):
        words[index] = vietnamese_word_to_telex_type(word)
    return ' '.join(words)

unicode_chars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsign_chars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def load_char_map():
    # Map all Vietnamese diacritic variants to their canonical Unicode counterparts
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


CHAR_MAP = load_char_map()


def convert_unicode(txt):
    # Normalize various encodings to standard Unicode codepoints
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: CHAR_MAP[x.group()], txt)

# Normalize to legacy tone placement: use "òa", "úy" instead of "oà", "uý"
def normalize_vietnamese_word_tone_legacy(word):
    if not is_valid_vietnamese_word(word):
        return word

    chars = list(word)
    tone_idx = 0
    vowel_pos = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check "qu"
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check "gi"
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            tone_idx = y
            chars[index] = vowel_table[x][0]
        if not qu_or_gi or index != 1:
            vowel_pos.append(index)
    if len(vowel_pos) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = vowel_to_ids.get(chars[1])
                chars[1] = vowel_table[x][tone_idx]
            else:
                x, y = vowel_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowel_table[x][tone_idx]
                else:
                    chars[1] = vowel_table[5][tone_idx] if chars[1] == 'i' else vowel_table[9][tone_idx]
            return ''.join(chars)
        return word

    for index in vowel_pos:
        x, y = vowel_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowel_table[x][tone_idx]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return ''.join(chars)

    if len(vowel_pos) == 2:
        if vowel_pos[-1] == len(chars) - 1:
            x, y = vowel_to_ids[chars[vowel_pos[0]]]
            chars[vowel_pos[0]] = vowel_table[x][tone_idx]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = vowel_to_ids[chars[vowel_pos[1]]]
            chars[vowel_pos[1]] = vowel_table[x][tone_idx]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = vowel_to_ids[chars[vowel_pos[1]]]
        chars[vowel_pos[1]] = vowel_table[x][tone_idx]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return ''.join(chars)


def is_valid_vietnamese_word(word):
    # Valid if vowels appear in consecutive positions (basic heuristic)
    chars = list(word)
    vowel_last = -1
    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x != -1:
            if vowel_last == -1:
                vowel_last = index
            else:
                if index - vowel_last != 1:
                    return False
                vowel_last = index
    return True


def normalize_vietnamese_sentence_tone_legacy(sentence):
    """
        Normalize tone placement across a Vietnamese sentence (legacy style).
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        if len(cw) == 3:
            cw[1] = normalize_vietnamese_word_tone_legacy(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)


import enchant
en_dict = enchant.Dict("en_US")

def remove_dup_characters(sentence):
  # Collapse duplicated letters in non-English words (e.g., "goooood" -> "good")
  sentence = str(sentence)
  words = []
  for word in sentence.strip().split():
    if en_dict.check(str(word)):
      words.append(word)
      continue
    words.append(re.sub(r'([A-Z])\1+', lambda m: m.group(1), word, flags = re.IGNORECASE))
  return ' '.join(words)

def remove_dup_special_characters(sentence):
    # Collapse duplicated punctuation in non-English words (e.g., "what!!!" -> "what!")
    sentence = str(sentence)
    words = []
    for word in sentence.strip().split():
        if en_dict.check(str(word)):
            words.append(word)
            continue
        cleaned_word = re.sub(r'([^\w\s])\1+', r'\1', word)
        words.append(cleaned_word)
    return ' '.join(words)

def prep_text(document):

    document = str(document)

    # lowercase
    document = document.lower()
    # remove extra whitespaces
    document = re.sub(r'\s+', ' ', document).strip()

    # Unicode normalization for Vietnamese characters
    document = convert_unicode(document)

    # remove duplicated chars in non-English words
    document = remove_dup_characters(document)
    # normalize Vietnamese tone placement (legacy style)
    document = normalize_vietnamese_sentence_tone_legacy(document)
    return document

def prep_batch(xs):
    return [prep_text(x) for x in xs]