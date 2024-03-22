import openai


class gpt:
    def __init__(self, commands) -> None:
        # optional; defaults to `os.environ['OPENAI_API_KEY']`
        openai.api_key = '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'
        self.commands = commands
        self.commonRequest = "These were the commands you can only use as a response. Can you interpret the speech for Baxter robot that I am going to pass in next sentence into these commands."

    def makeRequest(self, request):
        completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": self.commands + self.commonRequest + request,
            },
        ],
        )
        return completion.choices[0].message.content

