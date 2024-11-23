from openai import OpenAI
import os
import json
import openai

class ResultEvaluation:
    def __init__(self, api_key=os.environ.get('GPT_API_KEY'), verbose = False):
        self.client = OpenAI(api_key=api_key)
        self.verbose = verbose

    def generate_prompt(self, entry, task = "rewrite"):
        """
        生成文本响应
        :param prompt: 用户输入的提示
        :param max_tokens: 最大 tokens 数
        :param temperature: 生成的随机性
        :return: 生成的文本
        """
        standard = """1. 风格的忠实度
            定义：评估改写的文本是否成功模仿了目标作者的特定写作风格。
            细分规则：
                语言使用：目标作者是否有特定的语言风格，如使用古典词汇、俚语或特定的语言修辞技巧。
                句式结构：目标作者倾向使用哪种类型的句子结构，如简洁明了的短句，复杂并列或从句结构。
                用词习惯：目标作者是否偏好某些特定的词语或短语，如文学色彩浓厚的描述性用词或具体的术语使用。
            评分指标：
                几乎无法识别目标作者的风格（1分）
                部分体现目标作者风格，但存在明显偏差（2-3分）
                较好地模仿了目标作者的风格，偶有不符（4-5分）
                非常贴近目标作者的风格，细节处理到位（6-7分）
                几乎完美地复制了目标作者的风格（8-10分）
                
            2. 内容的适应性
            定义：检查改写的文本在保持原文主题和情感的同时，如何适应目标作者的表达风格。
            细分规则：
                主题保持：改写后的文本是否保持了原文的核心主题和主要信息。
                情感表达：改写后的文本是否保持或加强了原文的情感色彩，适应目标作者的情感表达习惯。
                表达方式：改写文本是否采用了目标作者可能偏好的表达方式，如隐喻、比喻或直白陈述。
            评分指标：
                原意和情感严重失真（1分）
                保留了部分原意和情感，但改动较多（2-3分）
                大部分原意和情感得以保留，小部分改动（4-5分）
                原意和情感基本保留，改写合乎目标作者风格（6-7分）
                完整保留原意和情感，改写精准符合目标作者风格（8-10分）
            3. 语言的流畅性
            定义：评估改写文本的语言是否自然流畅，易于理解，没有语法或语义错误。
            细分规则：
                语法正确性：文本中的语法是否正确，没有错误。
                语义清晰：文本表达的意思是否清楚，没有歧义或模糊的地方。
                读者理解：文本是否易于读者理解，符合语言习惯。
            评分指标：
                多处语法错误，难以理解（1分）
                语言表达生硬，偶尔语法错误（2-3分）
                基本无语法错误，部分表达不够流畅（4-5分）
                语言流畅自然，极少数小错误（6-7分）
                语言完全正确和流畅，非常自然（8-10分）"""
        prompt = f"对于任务\"{entry['task']}\"，以下是两个输出结果。\n第一个输出结果：\"{entry['output1']}\"\n第二个输出结果：\"{entry['output2']}\"\n"
        prompt = prompt + "满分30分，请根据以下评分标准分别为两个结果打分并简短评价。\n" + standard
        return prompt
       
    def evaluate_results(self, prompt, max_tokens=1500, temperature=0.2):
        """
        获得对两个结果的评价
        :param llm_instance: LLM类的实例
        :param problem_statement: 输入的用户问题
        :return: 评价结果
        """
        # 定义函数的 JSON 架构
        functions = [
            {
                "name": "evaluate_outputs",
                "description": "对两个输出结果进行评分和简短评价。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score1": {
                            "type": "integer",
                            "description": "第一个输出的评分，0到10之间。",
                            "minimum": 0,
                            "maximum": 10
                        },
                        "review1": {
                            "type": "string",
                            "description": "对第一个输出的简短评价。"
                        },
                        "score2": {
                            "type": "integer",
                            "description": "第二个输出的评分，0到10之间。",
                            "minimum": 0,
                            "maximum": 10
                        },
                        "review2": {
                            "type": "string",
                            "description": "对第二个输出的简短评价。"
                        }
                    },
                    "required": ["score1", "review1", "score2", "review2"]
                }
            }
        ]
        # 发送请求到 OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            functions=functions,
            function_call={"name": "evaluate_outputs"}
        )
    
        # 解析函数返回的结果
        function_response = response.choices[0].message.function_call.arguments
        result = json.loads(function_response)
    
        
        # 输出结果
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

entries = []

with open('sample_list.json', 'r') as json_file:
    entries = json.load(json_file)
    
for entry in entries:
    llm = ResultEvaluation()
    prompt = llm.generate_prompt(entry)
    print("prompt: ", prompt)
    result = llm.evaluate_results(prompt)
    print("result: ", result)

    
