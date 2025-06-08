API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySUQiOiI2ODQ0NGNjYmMyODAzYzlkYmI0ZTQ3MTIiLCJ0eXBlIjoiYXV0aCIsImlhdCI6MTc0OTMwNzAwMH0.4xuTmA2p0onfvGHo8XlwV4vIjlcE7REMVre0luU-2yU'

from openai import OpenAI

client = OpenAI(
  base_url="https://ai.liara.ir/api/v1/68444daf64f28c83a27063e1",
  api_key=API_KEY,
)

completion = client.chat.completions.create(
  model="openai/gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": 'Do you know what gossiping is?'
    }
  ]
)

print(completion.choices[0].message.content)