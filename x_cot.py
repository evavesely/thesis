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
error_test['gpt_reason'] = ['def'] * len(error_test)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

for index, row in error_test.iterrows(): 
        if pd.notna(row['text']):

            system_prompt1 = '''Think step by step and explain your gender label for the speaker of this dialogue using their character name 
            and/or contextual clues. Your options are 'm' if they are male, 'f' if they are female, and '?' if you can't tell. Keep in mind that
            each instance of dialogue is only spoken by one person.'''

            system_prompt2 = '''Therefore, based on your explanation, what is your final gender label for the speaker? Only include
            a 'm', 'f', or '?' in your response. Your explamation: {}'''

            messages = [ {"role": "system", "content": system_prompt1 } ]

            try:
                user_content = row['meta.character_name'] + ' from ' + row['meta.movie_name'] + ':' + row['text']
            except TypeError as e:
                user_content = row['text']

            messages.append({"role": "user", "content": user_content})

            try:
                explanation_result = completion_with_backoff(model="gpt-3.5-turbo-1106", messages=messages)
                model_explanation = explanation_result.choices[-1].message.content

                if index % 1000 == 0:
                        print(f"Processed {index} rows out of {len(error_test)}.")
                        print(model_explanation)
                
                if explanation_result['object'] == 'chat.completion':
                    error_test.at[index, 'gpt_reason'] = model_explanation
                else:
                    error_test.at[index, 'gpt_reason'] = "err"
                    print("Unexpected API response format.")

                messages.append(explanation_result.choices[-1].message)
                messages.append({"role": "system", "content": system_prompt2.format(model_explanation)})
                result = completion_with_backoff(model="gpt-3.5-turbo-1106", messages=messages)

                if result['object'] == 'chat.completion':
                    model_response = result.choices[-1].message.content
                    error_test.at[index, 'gpt_label'] = model_response
                else:
                    error_test.at[index, 'gpt_label'] = "err"
                    print("Unexpected API response format.")
                
                if index % 1000 == 0:
                        print(model_response)

                messages.append(result.choices[-1].message)

            except Exception as e:
                error_test.at[index, 'gpt_reason'] = "err"
                error_test.at[index, 'gpt_label'] = "err"
                print("Error occurred:", str(e))

error_test.to_csv('labeled_cot.csv')
