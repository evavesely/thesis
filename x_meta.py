import openai
openai.api_key = "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"  # register with OpenAI to get one
import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

error_test = pd.read_csv("error_test.csv")
error_test['gpt_label'] = ['def'] * len(error_test)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

for index, row in error_test.iterrows(): 
            if pd.notna(row['text']):

                system_prompt = '''Assign a gender label to the speaker of this dialogue using their character and movie name 
                and/or contextual clues. Your options are 'm' if they are male, 'f' if they are female, and '?' if you can't tell. Keep in mind that
                each instance of dialogue is only spoken by one person. Please only include the single letter label in your response.'''

                messages = [ {"role": "system", "content": system_prompt} ]

                try:
                    user_content = row['meta.character_name'] + ' from ' + row['meta.movie_name'] + ':' + row['text']
                except TypeError as e:
                    user_content = 'dialogue: ' + row['text']
 
                messages.append({"role": "user", "content": user_content})

                try:
                    #result = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=messages)
                    result = completion_with_backoff(model="gpt-3.5-turbo-1106", messages=messages)

                    if result['object'] == 'chat.completion':
                        model_response = result.choices[-1].message.content
                        error_test.at[index, 'gpt_label'] = model_response

                    else:
                        error_test.at[index, 'gpt_label'] = 'err'
                        print("Unexpected API response format.")
                    
                    if index % 1000 == 0:
                            print(f"Processed {index} rows out of {len(error_test)}.")
                            print(result.choices[-1].message.content)

                    messages.append(result.choices[-1].message)

                except Exception as e:
                    #print(row['text'])
                    #print(row['speaker'])
                    print("Error occurred:", str(e))
                    error_test.at[index, 'gpt_label'] = 'err'
error_test.to_csv('labeled_meta.csv')




