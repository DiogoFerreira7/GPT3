from tqdm import tqdm

class Tokeniser:
    
    def __init__(self, vocabulary_size=50256):
        with open("Pre Train Datasets/small_shakespeare.txt", "r") as text_file:
            self.tokens = text_file.read().encode("utf-8")
        # self.tokens = list("this is simply a test of unicode encoding in python, the longer the string the better we can test our bit pair algorithm - notice the amount of 'the' used. GPT Generated Sentece for testing - The quick brown fox jumps over the lazy dog! こんにちは世界! Добро пожаловать! ¡Hola, mundo! 你好，世界! Bonjour le monde! 안녕하세요 세계! Γειά σου Κόσμε! שלום עולם! नमस्ते दुनिया! 🌍🚀💻✨ नमस्ते, मेरा नाम है GPT-4. This is a long sentence designed to test a tokenizer's capabilities. 一只敏捷的棕色狐狸跳过了懒狗! हरिओम, यहां हम विभिन्न भाषाओं और लिपियों का प्रयोग कर रहे हैं। Lorem ipsum dolor sit amet, consectetur adipiscing elit. カタカナとひらがなも使います。 Моя цель — проверить токенизацию. ¿Puedes entender este texto? 😊✨👾🎉 Python is great for scripting! எங்கள் விஞ்ஞானிகள் நியூயார்க்கில் உள்ளனர். الطقس جميل اليوم! Будем рады видеть вас снова. ここに多くの異なる文字があります。 Это предложение становится длиннее и длиннее. 我们正在测试各种字符。 Δοκιμάζουμε διαφορετικούς χαρακτήρες. הקפיצה המהירה של השועל החום מעל הכלב העצלן! Всем привет! 🌟🌐📚👩‍💻🧑‍🚀🎨 βελτιώνουμε συνεχώς το μοντέλο μας. ¿Qué tal tu día? မင်္ဂလာပါ။ हमने बहुत सारी भाषाएँ शामिल की हैं। ताजमहल भारत में है। 🚗🚀📱💡💬🌈🙌 Этот текст продолжает расти. Qu'est-ce que vous en pensez? 今日はどうですか? Aloha ʻāina! फिर मिलेंगे। 🏖️🏔️🗽🕌🏯 🚴‍♂️🏊‍♀️⛷️🏋️‍♀️🤹‍♂️".encode("utf-8"))
        self.initialise()

        # Hyperparameters
        # The longer we run it for the larger the vocabulary but the shorter our encoding is so there are always tradeoffs
        # GPT4 uses 100k tokens
        self.vocabulary_size = vocabulary_size
        self.num_merges = self.vocabulary_size - 256

    def initialise(self):
        self._special_tokens = {'<|endoftext|>': 50257}
        self.vocabulary = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

    def byte_pair_algorithm(self):
        # We copy the original list here so that we don't destroy it
        chars = list(self.tokens)

        for merge_count in tqdm(range(self.num_merges)):
            pairs = self.get_pairs(chars)
            max_pair = self.get_top_pair(pairs)

            # We can choose a new token that is 256 values higher than the current one - since we are using utf 8 it will be 256 + any value will always be out of the range
            new_index = 256 + merge_count
            # We need to merge as we might get merges that turn into more merges as they might be highly likely combinations of merges
            chars = self.merge(chars, max_pair, new_index)
            self.merges[max_pair] = new_index

        # Adding to the original 256 token dictionary
        # Then all the merges we had we take the new index and put it back into the two values merged 
        for (p0, p1), idx in self.merges.items():
            self.vocabulary[idx] = self.vocabulary[p0] + self.vocabulary[p1]

    def get_top_pair(self, pairs):
        return max(pairs, key=pairs.get)
    
    def get_pairs(self, data):
        counts = {}
        # Here we can use zip to iterate over consecutive elements
        for pair in zip(data, data[1:]):
            # We get the value, by default if it does not exist it will be 0 and we will add 1
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, text, pair, new_index):
        char_pointer = 0
        while char_pointer < len(text) - 1:
            # The only time we dont want to do it is when we are at the last position
            # If we match the text to our pair
            if text[char_pointer] == pair[0] and text[char_pointer+1] == pair[1]:
                text[char_pointer:char_pointer+2] = [new_index]
            char_pointer += 1

        return text 

    def decode(self, ids):
        tokens = b"".join(self.vocabulary[idx] for idx in ids)
        text = tokens.decode("utf-8")
        return text

    def encode(self, text):
        # Get a list of integers from the utf encoding merge based on our trained merges vocabulary
        tokens = list(text.encode("utf-8"))

        # Now we have to implement the byte pair algorithm - since we are doing pairings we have to make sure that the lenght of tokens is at least 2
        while len(tokens) >= 2:
            pairs = self.get_pairs(tokens)
            # Finding the lowest possible merge based on our vocabulary that exists in the tokens pairs 
            # we need to start from the lowest index as higher indexes may be combinations of lower pair indexes
            pair = min(pairs, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                # We know that nothing else can be merged if everything is infinity
                break
            new_index = self.merges[pair]
            # Keep merging the pairs using the index token that we have defined in our vocabulary
            tokens = self.merge(tokens, pair, new_index)
        return tokens

    def save(self, file_prefix):
        model_file = file_prefix + ".tokeniser"
        with open(model_file, 'w') as tokeniser_file:
            # Output the merges dictionary so we can load it later after training
            for token_one, token_two in self.merges:
                tokeniser_file.write(f"{token_one} {token_two}\n")

    def load(self, model_file):
        assert model_file.endswith(".tokeniser")
        merges = {}

        new_index = 256
        with open(model_file, 'r', encoding="utf-8") as tokeniser_file:
            # Since we know we save them to the file in order they can be read in order which the token value known without being saved
            for line in tokeniser_file:
                token_one, token_two = map(int, line.split())
                merges[(token_one, token_two)] = new_index
                new_index += 1
    
        self.merges = merges
        self.vocabulary = self.create_vocabulary()

    def create_vocabulary(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self._special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

if __name__ == "__main__":
    # If you wish to train the tokeniser uncomment this code
    # The byte pair algorithm will be finished and the vocabulary file will be saved
    # tokeniser = Tokeniser()
    # tokeniser.byte_pair_algorithm()
    # tokeniser.save("shakespeare_tokeniser")

    # Here we then test whether loading a new tokeniser with that vocabulary file works
    second_tokeniser = Tokeniser()
    second_tokeniser.load("Tokeniser Vocabulary/shakespeare_tokeniser.tokeniser")
    encoding = second_tokeniser.encode("Hello, this is a test to make sure that the tokeniser is working correctly! <|endoftext|>")
    # print(encoding)
    # print(second_tokeniser.decode(encoding))