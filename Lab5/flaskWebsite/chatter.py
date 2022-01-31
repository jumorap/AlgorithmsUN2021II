from transformers import BlenderbotForConditionalGeneration, BlenderbotSmallTokenizer, BlenderbotTokenizer


# https://huggingface.co/facebook
# facebook/blenderbot_small-90M
# facebook/blenderbot-90M
# ^ For these models use tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
# facebook/blenderbot-400M-distill
# facebook/blenderbot-1B-distill
# facebook/blenderbot-3B
# ^ For these models use tokenizer = BlenderbotTokenizer.from_pretrained(mname)
mname = 'facebook/blenderbot-400M-distill'
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)


def say_something(text):
    inputs = tokenizer([text], return_tensors='pt')
    reply_ids = model.generate(**inputs)
    data = [tokenizer.decode(g,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True) for g in reply_ids][0]
    return data


if __name__ == '__main__':
    print("Up and running!")

