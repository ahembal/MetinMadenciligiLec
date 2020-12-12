import math, random

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    padded_tokens = []
    padded_tokens = "~"*(n) + text + "."
    if n == 1:
      lst = [((padded_tokens[i-1]), token) for i, token in enumerate(padded_tokens) if i >= n-1]    # >>> ngrams(1, 'abc') => [('~', 'a'), ('a', 'b'), ('b', 'c')]
      return  lst[1:]
    else:
      lst = [(("".join(padded_tokens[i-n:i])), token) for i, token in enumerate(padded_tokens) if i >= n-1] # >>> ngrams(2, 'abc') => [('~~', 'a'), ('~a', 'b'), ('ab', 'c')]
      return  lst[1:]

def create_ngram_model(model_class, path, n=2):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model
    
class NgramModel():

    def __init__(self, n):
        self.n = n
        self.context_dic = {}
        self.context_count_dic = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.context_count_dic.keys()

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for context, token in ngrams(self.n, text):
            # keep count
            if context in self.context_count_dic:
                self.context_count_dic[context] += 1
            else:
                self.context_count_dic[context] = 1

            # insert data
            if context in self.context_dic:
                token_dic = self.context_dic.get(context)
                if token in token_dic:
                    token_dic[token] += 1
                else:
                    token_dic[token] = 1
            else:
                self.context_dic[context] = {token: 1}
        return

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context in self.context_dic:
            token_dic = self.context_dic[context]
            if char in token_dic:
                return float(token_dic[char]) / self.context_count_dic[context]
            else:
                return 0.001
        else:
            return 0.001

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()

        if context in self.context_dic:
            denominator = self.context_count_dic[context]
            token_dic = self.context_dic[context]
            sorted_keys = sorted(token_dic.keys())

            for i, token in enumerate(sorted_keys):
                minus_i_sum = sum([token_dic[k] for k in sorted_keys[:i]])
                if float(minus_i_sum)/denominator <= r < float(minus_i_sum + token_dic[sorted_keys[i]])/denominator:
                    return token

        else:
            return None

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        if self.n != 1:
            context = "".join(["~"] * (self.n))
            generated = []

            for __ in range(length):
                token = self.random_token(context)
                generated.append(token)

                if token != ".":
                    context = context[1:] + (token)
                else:
                    context = "".join(["~"] * (self.n))

            return "".join(generated)
        else:
            return "".join(self.random_char(random.choice(list(self.get_vocab()))) for _ in range(length))

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        product = 0

        for context, token in ngrams(self.n, text):
          if self.prob(context,token) != 0:
            product += math.log(self.prob(context, token))
        return (1/math.exp(product)) ** (float(1)/(len(text)+1))
    
   
