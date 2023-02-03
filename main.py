import io
import os
import discord
from discord.ext import commands

from inference import generate


intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')


@bot.command()
async def doggydolle(ctx, prompt: str):
    await ctx.reply(f"Just a second, I'm preparing an image for '{prompt}'")

    image = generate(prompt)[0]
    with io.BytesIO() as b_image:
        image.save(b_image, 'PNG')
        b_image.seek(0)
        await ctx.reply(file=discord.File(fp=b_image, filename=f'{prompt}.png'))
    

bot.run(os.getenv("BOT_TOKEN"))
