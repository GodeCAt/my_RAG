import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
from transformers.generation.utils import GenerationConfig
from peft import PeftModel
from colorama import Fore,Style
from threading import Thread

from .my_logger import logger

class Baichuan2:
    def __init__(self,model_base):
        self.model_base = model_base
        self.model,self.tokenizer=self.init_baichuan2()

    def init_baichuan2(self):
        model_base=self.model_base
        model=AutoModelForCausalLM.from_pretrained(
            model_base['model_path_or_id'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.generation_config=GenerationConfig.from_pretrained(
            model_base['model_path_or_id'],
        )
        tokenizer=AutoTokenizer.from_pretrained(
            model_base['model_path_or_id'],
            use_fast=True,
            trust_remote_code=True
        )
        if "lora_weights_path" in model_base:
            logger.info("正在加载Lora权重")
            model=PeftModel.from_pretrained(model,model_base['lora_weights_path'])
        return model,tokenizer

    def chat(self,prompt,messages=[],stream=False):
        model=self.model
        tokenizer=self.tokenizer
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            responses=[]
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    responses.append(response[position:])
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                messages.append({"role": "assistant", "content": response})
            except KeyboardInterrupt:
                pass
        else:
            responses = model.chat(tokenizer, messages)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            messages.append({"role": "assistant", "content": responses})

        return responses,messages

    def conversation(self,messages=[],stream=True):
        model = self.model
        tokenizer=self.tokenizer
        while True:
            prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
            if prompt.strip() == "exit":
                break
            print(Fore.CYAN + Style.BRIGHT + "\nBaichuan 2：" + Style.NORMAL, end='')
            if prompt.strip() == "stream":
                stream = not stream
                print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
                continue
            messages.append({"role": "user", "content": prompt})
            if stream:
                position = 0
                try:
                    for response in model.chat(tokenizer, messages, stream=True):
                        print(response[position:], end='', flush=True)
                        position = len(response)
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                except KeyboardInterrupt:
                    pass
                print()
            else:
                response = model.chat(tokenizer, messages)
                print(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            messages.append({"role": "assistant", "content": response})
        print(Style.RESET_ALL)

class Qwen1_5:
    def __init__(self,model_base):
        self.model_base = model_base
        self.model,self.tokenizer=self.init_qwen15()

    def init_qwen15(self):
        model_base=self.model_base
        model=AutoModelForCausalLM.from_pretrained(
            model_base['model_path_or_id'],
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer=AutoTokenizer.from_pretrained(model_base['model_path_or_id'])
        if "lora_weights_path" in model_base:
            logger.info("正在加载Lora权重")
            model=PeftModel.from_pretrained(model,model_base['lora_weights_path'])
        return model,tokenizer

    def chat(self,prompt,messages=[],stream=False):
        model=self.model
        tokenizer=self.tokenizer
        device="cuda"
        if not messages:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        if not stream:
            # Directly use generate() and tokenizer.decode() to get the output.
            # Use `max_new_tokens` to control the maximum output length.
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            messages.append({"role": "assistant", "content": responses})
        else:
            streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
            generation_kwargs=dict(model_inputs,streamer=streamer,max_new_tokens=512)
            thread=Thread(target=model.generate,kwargs=generation_kwargs)
            thread.start()
            responses=[]
            response=""
            for new_text in streamer:
                response+=new_text
                responses.append(new_text)
            messages.append({"role": "assistant", "content": response})

        return responses,messages

    def conversation(self,messages=[],stream=True):
        model = self.model
        tokenizer=self.tokenizer
        while True:
            device = "cuda"
            if not messages:
                messages.append({"role": "system", "content": "You are a helpful assistant."})
            prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
            if prompt.strip() == "exit":
                break
            print(Fore.CYAN + Style.BRIGHT + "\nQwen 1.5：" + Style.NORMAL, end='')
            if prompt.strip() == "stream":
                stream = not stream
                print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
                continue
            print(messages)
            messages.append({"role": "user", "content": prompt})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            if stream:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                responses = []
                response = ""
                for new_text in streamer:
                    print(new_text,end='', flush=True)
                    response += new_text
                    responses.append(new_text)
                messages.append({"role": "assistant", "content": response})
                print()
            else:
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(response)
            messages.append({"role": "assistant", "content": responses})
        print(Style.RESET_ALL)