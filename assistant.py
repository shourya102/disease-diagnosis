import re

from openai import OpenAI

OPENAI_API_KEY = ''
client = OpenAI(api_key=OPENAI_API_KEY)
ASSISTANT_ID = 'asst_bfUp5J4DxQUl6g7Ozdexd43O'


def response(messages):
    thread = client.beta.threads.create(messages=messages)
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    )
    while run.status != 'completed':
        pass
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        return messages
    else:
        return 'error'


def filter_response(messages):
    messages_l = response(messages)
    res1 = {}
    message1 = messages_l.data[0]
    if message1.role == 'assistant':
        con = message1.content[0].text.value
        message, data = extract_data_and_message(con)
        if data:
            res1['data'] = data.split(',')
        else:
            res1['data'] = []
        res1['message'] = message
    return res1


def extract_data_and_message(text):
    message = re.sub(r'\[DATA\](.*?)\[\/?DATA\]', '', text, flags=re.DOTALL).strip()
    data_match = re.search(r'\[DATA\](.*?)\[\/?DATA\]', text, re.DOTALL)
    data = data_match.group(1).strip() if data_match else None
    return message, data


if __name__ == '__main__':
    res = [{'content': "I feel cough and itchiness", 'role': 'user'}]
    print(filter_response(res))
