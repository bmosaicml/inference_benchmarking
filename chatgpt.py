import os
import openai
import time
from dotenv import dotenv_values
config = dotenv_values(".env")

openai.api_key = config['OPENAI']






ITERS = 5
with open('openai.tsv', 'w') as f:
    f.write(f"model_name\tprompt_len\toutput_len\tcall_latency\tsecs_per_tok\n")
    for num1 in [1, 16, 32, 64, 128, 256]:
        for num2 in [1, 16, 32, 64, 128, 256]:
            prompt = f"Here are the first {num1} even numbers: " + ', '.join([str(2 * i) for i in range(1, num1+1)]) + f"\nGive the first {num2} odd numbers:"
            for engine in ["text-babbage-001", "text-davinci-003", "text-curie-001",  "davinci-instruct-beta"]:
                print(f"{engine}")
                avg_latency = 0.0
                avg_latency_per_token = 0.0
                avg_prompt_len = 0.0
                avg_output_len = 0.0
                for i in range(ITERS):
                    try:
                        try:
                            start = time.time()
                            response = openai.ChatCompletion.create(
                                engine=engine,
                                prompt=prompt,
                                max_tokens=100,  
                                temperature=0.25,
                                top_p=1,
                                frequency_penalty=0.5,
                                presence_penalty=0
                            )
                            end = time.time()
                        except:
                            start = time.time()
                            response = openai.Completion.create(
                                engine=engine,
                                prompt=prompt,
                                max_tokens=100,  
                                temperature=0.25,
                                top_p=1,
                                frequency_penalty=0.5,
                                presence_penalty=0
                            )
                            end = time.time()
                    except openai.error.RateLimitError:
                        time.sleep(10)

                    prompt_len =  response.usage.prompt_tokens
                    output_len =  response.usage.completion_tokens

                    avg_latency += end-start
                    avg_latency_per_token += (end-start)/output_len
                    avg_prompt_len += prompt_len
                    avg_output_len += output_len

                f.write(f"{engine}\t{avg_prompt_len/ITERS}\t{avg_output_len/ITERS}\t{avg_latency/ITERS}\t{avg_latency_per_token/ITERS}\n")
                f.flush()
