---
layout: blog
author: cohlem
---

### Unicode

- Character encoding standard
- aims to incorporate all the available digital characters
- Each character in Unicode has a unique 4 to 6-digit hexadecimal number. For Example, the letter 'A' has the code 0041, represented as U+0041.
- compatible with ASCII
- first 128 characters in Unicode directly correspond to the characters represented in the 7-bit ASCII table

### Unicode Transformation Format (UTF-8)

- uses 1-4 bytes to represent each character
- can encode all the unicode code points
- backward compatible with ASCII

```
Example:
(1 byte) The character 'A' (U+0041) is encoded as `01000001` (0x41 in hexadecimal).
(2 byte) The character '¢' (U+00A2) is encoded as `11000010 10100010` (0xC2 0xA2 in hexadecimal).
(3 byte) The character '€' (U+20AC) is encoded as `11100010 10000010 10101100` (0xE2 0x82 0xAC in hexadecimal).
(4 byte) The character '𠜎' (U+2070E) is encoded as `11110000 10100000 10011100 10001110`(0xF0 0xA0 0x9C 0x8E in hexadecimal).

```

let's understand some difference between unicode and utf-8
character '€' has

unicode code point (hex): U+20AC
unicode code point (decimal): 8364

So there is a single number (decimal) that represents characters in unicode

but, the same character in utf-8 is represented as

- **Binary**: `11100010 10000010 10101100`
- **Hexadecimal:** `0xE2 0x82 0xAC`
- **Decimal:** `226, 130, 172`

Why ? utf-8 is a standard that stores characters in 1-4 bytes as described above.

similarly in python, we can get it's hex values by `'€'.encode('utf-8')` and converting it to list gives us it's list of decimal values and doing `ord(`'€') gives us it's unicode code point in decimal`

### Build vocabulary

We build our vocabulary by gathering chunks of bytes that appear together most of the times,

Suppose the data to be encoded is

`aaabdaaabac`

the byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, such as "Z". Now there is the following data and replacement table:

```
ZabdZabac
Z=aa
```

Then the process is repeated with byte pair "ab", replacing it with "Y":

```
ZYdZYac
Y=ab
Z=aa
```

again, the pair ZY occurs twice so our data and replacement table becomes.

```
XdXac
X=ZY
Y=ab
Z=aa
```

We now write code to implement this same functionality.

The function below constructs a dictionary that keeps track of frequency of bytes that come together.

```python
def get_stats(text):
    freq = dict()
    for t1,t2 in zip(text, text[1:]):
        freq[(t1,t2)] = freq.get((t1,t2),0) + 1
    return freq
```

This merge function is now used to merge the two bytes into one. similar to Z=aa in the example above.

```python
def merge(pair, ids, idx):
    n_ids = []
    i=0
    while i < len(ids):

        if i<len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            n_ids.append(idx)
            i+=2
        else:
            n_ids.append(ids[i])
            i+=1
    return n_ids
```

We do this iteratively for `total_merges` times,

```python
max_vocab = 356
total_merges = max_vocab - 256
merge_dict = {}

for i in range(total_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256+i
    ids = merge(pair, text_utf8, idx)
    merge_dict[pair] = idx

```

### Encode

lets encode our text into our tokens using our merge_dict which keeps track of all the possible combination of characters.

```python
def encode(text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merge_dict.get(p, float("inf")))
        print(pair)
        if pair not in merges:
            break  # nothing else can be merged
        idx = merge_dict[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


```

here we need to be careful about how we encode the tokens, i.e for instance the new 256 index in our merge_dict should be encoded first because later index for ex: 352 could be combination of 256 and 108. So we have to maintain this order.

to do that we first get all the token combination available to us in our dataset using stats = get_stats(tokens), and this line of code `pair = min(stats, key=lambda p: merges.get(p, float("inf")))` finds the pair with the lowest key (lets say 101,32 -> 256) and then merges those 101,32 tokens to be 256 and the process is continued until there are no pairs that can be combined using our merge_dict mapping.

### Decode

```python
vocab = {idx : bytes([idx]) for idx in range(256)}

for (p1,p2),idx in merge_dict.items():
    vocab[idx] = vocab[p1] + vocab[p2]

```

The code above maps index to its corresponding byte in utf-8, and the loop combines the byte information corresponding to their indexs, for instance lets say 256 is combo of 101,32, the bytes of 101 and 32 will be combined. lets say 352 is now a combo of 256 and 32, their byte information will be combined, which will be easier to decode the information in the code below.

```python
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode('utf-8', errors = "replace")
```

the first line in the function above maps their index to byte information, and then those utf-8 bytes will be decoded to their corresponding characters in utf-8.

This is the building block of tokenizer, everything that comes next is a more complex and efficient version of the tokenizer.

### Openai's BPE tokenizer

Let's first understand the problem our current implementation of tokenizer has.

It will tokenize the whole sequence. i.e lets say we have a sentence

I've chosen a bit funky sentence here, just for the purpose of explanation.

`you are 52 years old, u are good too, you've achieved so much`

if you look closely, this sequence of characters `u are` appear twice and will have their own mapping in the vocabulary.

so for token `u are` it will have one index in our vocabulary, we can see it combines characters from two separate words, which we would like to minimize, and we would like to separate out these types of tokens `'ve` because they generally go along with other words as well. For this purpose, we process our initial text through this regex.

```python
re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

as you can see these tokens `s|'t|'re|'ve|'m|'ll|'d|` are separated from text, and we separate out words and numbers as well

```python
import regex as re

samplere = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

samplere.findall(" you've done soo much 1245234 ")
```

The sentence is processed in this way.

```
[' you', "'ve", ' done', ' soo', ' much', ' 1245234', ' ']
```

and then we train our tokenizer on each elements that we get from this.

### Complete code for tokenizer

```python
class Tokenizer:
    def __init__(self):
        self.pattern = None
        self.merges = dict()
        self.vocab = None

    def get_stats(self,ids, freq=None):
        freq = dict() if freq is None else freq
        for t1,t2 in zip(ids, ids[1:]):

            freq[(t1,t2)] = freq.get((t1,t2),0) + 1
        return freq


    def merge(self, ids, pair, idx):
        n_ids = []
        i=0
        while i < len(ids):

            if i<len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                n_ids.append(idx)
                i+=2
            else:
                n_ids.append(ids[i])
                i+=1
        return n_ids


```

```python
GPT4Pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class SampleTokenizer(Tokenizer):
    def __init__(self,max_vocab, pattern=None):
        super().__init__()
        self.pattern = re.compile(GPT4Pattern if pattern is None else pattern)
        self.max_vocab = max_vocab
        self.total_merges = max_vocab - 256

    def train(self, x):
        """This objective of this function is to build the merges dictionary mapping and build vocab"""
        chunks = self.pattern.findall(x)
        ids = [list(ch.encode('utf-8')) for ch in chunks]


        for p in range(self.total_merges):
            freq = dict()
            idx = 256 + p

            for item in ids:
                self.get_stats(item, freq)

            pair = max(freq, key=freq.get)
            ids = [self.merge(i,pair, idx) for i in ids]

            self.merges[pair] = idx
            self._build_vocab()

    def _build_vocab(self):
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p1,p2),idx in self.merges.items():
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]


    def encode(self, x):
        # given a string, return list of integers (the tokens)
        tokens = list(x.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self,ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode('utf-8', errors = "replace")


st = SampleTokenizer(356)
st.train(text)
a = st.encode("hey how are you doing 124")
st.decode(a)
# outputs: 'hey how are you doing 124'

```
