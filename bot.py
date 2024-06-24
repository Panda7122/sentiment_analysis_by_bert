import discord
import json
import loadBert
import re
from discord import app_commands
from discord.ext import commands
#client 是我們與 Discord 連結的橋樑，intents 是我們要求的權限
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

TARGET_CHANNEL_ID =00000000 #changeme

@client.event
async def on_ready():
    print(f"目前登入身份 --> {client.user}")
   
@client.event
async def on_message(message: discord.Message) -> None:  # This event is called when a message is sent
    if message.author == client.user:
        return
    # 檢查消息是否來自指定的頻道
    if message.channel.id != TARGET_CHANNEL_ID:
        return
    
    channel = client.get_channel(TARGET_CHANNEL_ID)
    #emotion = loadBert.predict_emotion(message.content)
    #await message.channel.send(f'> {message.content}\n`{emotion}`')  # This is required to process commands
    emotion_predictions = loadBert.predict_emotion(message.content)
    
    # 構建情緒與表情符號的對應關係
    emotion_to_emoji = {
        'joy': ':grinning:',
        'neutral': ':neutral_face:',
        'sadness': ':cry:',
        'anger': ':rage:',
        'fear': ':fearful:',
        'love': ':smiling_face_with_3_hearts:',
        'surprise': ':open_mouth:'
    }

    # 發送原始消息和預測的情緒分佈
    prediction_text = '\n'.join([f'{emotion}: {prob:.2%}' for emotion, prob in emotion_predictions[0]])
    await message.channel.send(f'> {message.content}\n`{prediction_text}`')

    # 發送每個預測情緒對應的表情符號
    for emotion, prob in emotion_predictions[0]:
        if emotion in emotion_to_emoji:
            await message.channel.send(emotion_to_emoji[emotion])

with open("./token.json", "r", encoding='utf-8') as f:#透過讀檔的方式讀取token這樣比較安全，不會容易外洩token。
    data = json.load(f)
client.run(data["data"])
