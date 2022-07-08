import torch
from transformers import AutoModel, AutoTokenizer

bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

## Input Japanese Text
line = "吾輩は猫である。"

inputs = tokenizer(line, return_tensors="pt")
print(inputs)

print(tokenizer.decode(inputs["input_ids"][0]))

outputs = bertjapanese(**inputs)
print(outputs)

long = "　世界はゾンビで溢れかえっていた。そんなバカげた漫画や小説が、昔はたくさんあったらしい。\n　――今となっては、ただの現実でしかないけれど。\n　ゾンビとなった者、取り残された者が、唯一意志を交わし合える手段――「ヘミシンク・ノイズ」。\n\n「――ヘルマン！　ヘルマンだ！　ヘルマン・ボーテ！　やっと見つけた」\n\n　その声に、僕は振り向いたんだ。\n\n創られた二人きりの世界で出会ったのは、かつての幼馴染だった。\n\n「にごたん」企画作品。カクヨム、読み書きしよう！　にも投稿しております。\nお題は【ヘミシンク音】【二人だけで通じ合える秘密の合言葉】【諦めたくなんてない】【かぼちゃの馬車】\n\n「にごたん」ルール\n・【　】内の単語を3つ以上使って二時間半で即興短編を書く。\n・他小説サイトに投稿した場合、参加者は評価をしない\n\n　気になる方は、ツイッター内＃にごたん、で検索です。\n"

inputs = tokenizer(long, return_tensors="pt")
print(inputs)

print(tokenizer.decode(inputs["input_ids"][0]))

#outputs = bertjapanese(**inputs)
#print(outputs)