from emosent import get_emoji_sentiment_rank, EMOJI_SENTIMENT_DICT

def get_emoji_metric(inp: str, metric: str, delete_emojis: bool):
    value = 0
    emojis_found = set()
    for char in inp:
        if char in EMOJI_SENTIMENT_DICT:
            new_value = get_emoji_sentiment_rank(char)[metric]
            value += new_value
            print(char)
            print(new_value)
            emojis_found.add(char)
    if delete_emojis:
        for emoji in emojis_found:
            inp = inp.replace(emoji, "")

    return (inp, value)

print(get_emoji_metric("Hola 😀😊😎", "positive", False))
