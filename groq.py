from typing import Optional
from litellm import completion


class Chatbot:
    def __init__(
        self,
        model_name="groq/llama-3.1-70b-versatile",
        api_base=None,
        system_prompt: Optional[str] = None,
        user_profile: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.system_prompt = system_prompt

        if system_prompt is not None:
            self.conversation_history = [{"role": "system", "content": system_prompt}]
        else:
            self.conversation_history = []

    def send_message(self, message):
        self.conversation_history.append({"role": "user", "content": message})
        response = completion(
            model=self.model_name,
            messages=self.conversation_history,
            api_base=self.api_base,
            temperature=0.7,
            stream=False,
        )
        bot_response = response["choices"][0]["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        return bot_response

    def predict(self, prompt):
        message = []
        message.append({"role": "user", "content": prompt})
        response = completion(
            model=self.model_name,
            messages=message,
            api_base=self.api_base,
            temperature=0.7,
            stream=False,
        )
        bot_response = response["choices"][0]["message"]["content"]
        # print(bot_response)
        return bot_response

    def get_history(self, k=0):
        return self.conversation_history[-k:]

    def get_user_history(self, k=0):
        even_indexed_history = [
            self.conversation_history[i]
            for i in range(len(self.conversation_history))
            if i % 2 == 0
        ]
        return even_indexed_history
