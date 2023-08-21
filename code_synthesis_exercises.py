import random

with open('words.txt','r') as f:
    words_set = f.readlines()
    
words_set = [word.strip() for word in words_set]

def synthesize_exercises(words_set, num_words = 5):
    selected_words = random.sample(words_set,num_words)
    random.shuffle(selected_words)
    word_str = ''
    for word in selected_words:
        word_str += '* '+word + '\n'
    prompt = f'''You are an experienced Python programmer. \
Your task is to create practice exercises for Python learners. \
These exercises should be presented in the form of functions. \
You will provide the function names and include detailed information within comments,\
explaining the function's purpose, input-output formats, \
and more. The complete implementation should be provided within the function body. \
The format for your output should strictly follow this structure:

```python
def function_name(parameter: parameter_type) -> return_type:
    """
    Description of the function and scenario.
    
    Parameters:
    parameter_description
    
    Returns:
    return_value_description
    """
    # Function body
```

To ensure a variety of exercise questions, you need to construct an exercise question using words from the following random word list. Your function name should incorporate at least two or more words from this list (you can also add other words to the function name). The choice is entirely yours, as long as the selected words can create a meaningful function name for the exercise question. The random word list you have is as follows:

{word_str}
Please note that the exercise questions you generate are intended for Python learners. \
Therefore, it's crucial to clearly describe the function's purpose in the comment section, \
so that learners can understand the function's intent and grasp your provided solution. \
In your solution, try to avoid relying on excessive external libraries. \
If you need necessary libraries, please import them at the beginning of the function.\
The exercises you provide should not be overly simple; on the contrary, they should focus on algorithms, which help enhance students' thinking and coding abilities.\
Now, please provide the exercise and finish it! No additional explanations or examples are required, and there is no need to comment on or discuss this instruction! Just generate the one function as exercise! Remember to follow the format provided.
    '''
    return prompt


if __name__ == '__main__':
    
    print(synthesize_exercises(words_set,num_words=5))