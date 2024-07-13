# TODO figure out how to train the tokeniser
    # Should probably find a better dataset - the larger thee vocabulary the better
    # TODO separate the dataloader from the other class as and use it to train the tokeniser

# TODO Create tokeniser class that can be imported and can load and save the values trained for it
    # This can be good practice for loading and storing values
    # If need be do the loading and storing for the model first and then tokeniser

class Tokeniser:
    # TODO rename everything and organise it all to make sense, comment everything too

    def __init__(self):
        self.tokens = list("this is simply a test of unicode encoding in python, the longer the string the better we can test our bit pair algorithm - notice the amount of 'the' used. GPT Generated Sentece for testing - The quick brown fox jumps over the lazy dog! こんにちは世界! Добро пожаловать! ¡Hola, mundo! 你好，世界! Bonjour le monde! 안녕하세요 세계! Γειά σου Κόσμε! שלום עולם! नमस्ते दुनिया! 🌍🚀💻✨ नमस्ते, मेरा नाम है GPT-4. This is a long sentence designed to test a tokenizer's capabilities. 一只敏捷的棕色狐狸跳过了懒狗! हरिओम, यहां हम विभिन्न भाषाओं और लिपियों का प्रयोग कर रहे हैं। Lorem ipsum dolor sit amet, consectetur adipiscing elit. カタカナとひらがなも使います。 Моя цель — проверить токенизацию. ¿Puedes entender este texto? 😊✨👾🎉 Python is great for scripting! எங்கள் விஞ்ஞானிகள் நியூயார்க்கில் உள்ளனர். الطقس جميل اليوم! Будем рады видеть вас снова. ここに多くの異なる文字があります。 Это предложение становится длиннее и длиннее. 我们正在测试各种字符。 Δοκιμάζουμε διαφορετικούς χαρακτήρες. הקפיצה המהירה של השועל החום מעל הכלב העצלן! Всем привет! 🌟🌐📚👩‍💻🧑‍🚀🎨 βελτιώνουμε συνεχώς το μοντέλο μας. ¿Qué tal tu día? မင်္ဂလာပါ။ हमने बहुत सारी भाषाएँ शामिल की हैं। ताजमहल भारत में है। 🚗🚀📱💡💬🌈🙌 Этот текст продолжает расти. Qu'est-ce que vous en pensez? 今日はどうですか? Aloha ʻāina! फिर मिलेंगे। 🏖️🏔️🗽🕌🏯 🚴‍♂️🏊‍♀️⛷️🏋️‍♀️🤹‍♂️".encode("utf-8"))


    def byte_pair_algorithm(self):
        # This byte pair algorithm has a hyper parameter for tokenisation where we can actually dictate how many steps we want to run it for
        # The longer we run it for the larger the vocabulary but the shorter our encoding is so there are always tradeoffs
        # GPT4 uses 100k tokens

        # Hyperparameters
        vocabulary_size = 276
        # TODO figure out why this 256 is here 
        num_merges = vocabulary_size - 256

        # We copy the original list here so that we don't destroy it
        ids = list(self.tokens)

        # Keep track of our merges
        merges = {}
        for i in range(num_merges):
            # Can optimise this function later by making it so that it will get the top one and replace all the ones that match based on the same number of pair values instead of just one
            pairs = self.get_pairs(ids)
            pair = max(pairs, key = pairs.get)
            # We can choose a new token that is 256 values higher than the current one - since we are using utf 8 it will be 256 + any value will always be out of the range
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx

        return ids
        
    def get_sorted_pairs(self, data):
        counts = {}
        # Here we can use zip to iterate over consecutive elements
        for pair in zip(data, data[1:]):
            # We get the value, by default if it does not exist it will be 0 and we will add 1
            counts[pair] = counts.get(pair, 0) + 1

        return sorted(((v, k) for k, v in counts.items()), reverse=True)
    
    def get_top_pair(self, pairs):
        return max(pairs, key=pairs.get)
    
    def merge(self, ids, pair, idx):
        # replace all occurrences of pairs with that id with the new index token
        newids = []

        i = 0
        while i < len(ids):
            # The only time we dont want to do it is when we are at the last position
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids 
    
