from openai import OpenAI
import json
api_key =  ''

class BaseAgent:
    def __init__(self, system_prompt="", use_history=False, temp=0, top_p=1):
        self.use_history = use_history
        self.client = OpenAI(api_key=api_key)
        self.system = system_prompt
        self.temp = temp
        self.top_p = top_p
        self.input_tokens_count = 0
        self.output_tokens_count = 0

    
    def __call__(self, message, parse=False):
        #self.messages.append({"role": "user", "content": message})
        result = self.generate(message, parse)
        #self.messages.append({"role": "assistant", "content": result})
        if parse:
            try:
                result = self.parse_json(result)
            except:
                result = result
            
        return result
        
    
    
    def generate(self, message, json_format):
        #if self.use_history:
        #    input_messages = self.messages
        #else:
        input_messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": message}
        ]
        try:
            if json_format:
                response = self.client.chat.completions.create(
                    model="gpt-5-mini", # gpt-4
                    messages=input_messages,

                    top_p=self.top_p,
                    response_format={"type": "json_object"}
                    )
    #                 temperature=self.temp,
            else:
                response = self.client.chat.completions.create(
                    model="gpt-5-mini", # gpt-4
                    messages=input_messages,
                    top_p=self.top_p,
                    )
            #                 temperature=self.temp,
            self.update_tokens_count(response)
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"
    
    
    def parse_json(self, response):
        return json.loads(response)

    
    def update_tokens_count(self, response):
        self.input_tokens_count += response.usage.prompt_tokens
        self.output_tokens_count += response.usage.completion_tokens
    
    
    def show_usage(self):
        print(f"Total input tokens used: {self.input_tokens_count}\nTotal output tokens used: {self.output_tokens_count}")
        



