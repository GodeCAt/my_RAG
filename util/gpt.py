from openai import OpenAI
import httpx
import time
import re

base_role="You are a helpful assistant."
topic_extractor_role="You are a topic extraction assistant."
topic_extractor_prompt="""抽取一个问答对的话题，以下是几个例子：
问：1000A有哪些特征特性？
答：属籼型三系不育系，南昌春播播始历期90-95天，夏播85-90天，海南三亚播始历期70-75天。主茎叶片数15.5叶左右。该不育系株型适中，叶色浓绿，剑叶挺直，茎秆粗壮，株高68.0厘米，单株成穗12穗，穗长23.7厘米，平均每穗颖花数161.0粒，包颈粒率20.9%，千粒重22.5克。种子饱满，稃尖紫色，柱头紫色。柱头外露率51.6%，双边外露率26.3%。花粉镜检花粉败育彻底，典败99.22%，圆败0.78%，染败0%，花粉败育率100%；套袋自交不育度为100%。出糙率81.3％，精米率74.8％，整精米率50.7％，粒长6.6mm，粒型长宽比3.3，垩白粒率46％，垩白度5.3％，直链淀粉含量12.3％，胶稠度74mm，碱消值4.0级。2018年井冈山稻瘟病田间自然诱发鉴定：穗颈瘟为9级，穗颈瘟平均损失率为43.1%，稻瘟病抗性综合指数6.5。"
话题：作物品种信息-1000A-特征特性

问：属籼型三系不育系，南昌春播播始历期90-95天，夏播85-90天，海南三亚播始历期70-75天。主茎叶片数15.5叶左右。该不育系株型适中，叶色浓绿，剑叶挺直，茎秆粗壮，株高68.0厘米，单株成穗12穗，穗长23.7厘米，平均每穗颖花数161.0粒，包颈粒率20.9%，千粒重22.5克。种子饱满，稃尖紫色，柱头紫色。柱头外露率51.6%，双边外露率26.3%。花粉镜检花粉败育彻底，典败99.22%，圆败0.78%，染败0%，花粉败育率100%；套袋自交不育度为100%。出糙率81.3％，精米率74.8％，整精米率50.7％，粒长6.6mm，粒型长宽比3.3，垩白粒率46％，垩白度5.3％，直链淀粉含量12.3％，胶稠度74mm，碱消值4.0级。2018年井冈山稻瘟病田间自然诱发鉴定：穗颈瘟为9级，穗颈瘟平均损失率为43.1%，稻瘟病抗性综合指数6.5。回答满足要求的水稻品种？
答：1000A
话题：作物品种推荐-籼型三系不育系-春播播始历期90-95天-夏播85-90天-海南三亚播始历期70-75天-主茎叶片数15.5叶左右-该不育系株型适中，叶色浓绿，剑叶挺直，茎秆粗壮-株高68.0厘米-单株成穗12穗-穗长23.7厘米-平均每穗颖花数161.0粒-包颈粒率20.9%-千粒重22.5克-柱头外露率51.6%-双边外露率26.3%-典败99.22%-圆败0.78%-染败0%-花粉败育率100%-套袋自交不育度为100%-出糙率81.3％-精米率74.8％-整精米率50.7％-粒长6.6mm-粒型长宽比3.3-垩白粒率46％-垩白度5.3％-直链淀粉含量12.3％-胶稠度74mm-碱消值4.0级--穗颈瘟为9级-穗颈瘟平均损失率43.1%-稻瘟病抗性综合指数6.5-1000A

问：该品种属早籼型光温敏两系不育系，播始历期56-73天，株高60-65厘米，株型适中，叶缘、叶鞘、稃尖、柱头均无色，主茎叶片数12叶。分蘖力中等，单株有效穗9.2-10.33个,穗长22厘米，穗总粒数76.4-98粒，千粒重27-28克。不育株率和不育度均为100%，无花粉型或少量典败，不育起点温度低于23.5℃。柱头总外露率79%，其中双边外露率29%；穗包颖粒率21.4%。异交结实率56.9%，亲和性好。抗性：叶瘟8级，穗瘟9级，高感稻瘟病；白叶枯病7级。米质：糙米率79.7%，精米率72.2%，整精米率53.2%，粒长6.8毫米，长宽比2.9，垩白粒率2%，垩白度0.1%，透明度2级，碱消值4.1级，胶稠度84毫米，直链淀粉含量12.2%。繁殖要点：繁殖要求育性敏感期平均温度在19-22℃。制种育性敏感期应安排在7月上旬-8月中旬。制种时每亩喷施15-16克[九二0]，及时防治病虫害。一般每亩繁殖产量150公斤左右。回答满足要求的水稻品种？
答：157S
话题：作物品种推荐-早籼型光温敏两系不育系-播期56-73天-株高60-65厘米-主茎叶片数12叶-单株有效穗9.2-10.33个-穗长22厘米-穗总粒数76.4-98粒-千粒重27-28克-不育株率和不育度均为100%-不育起点温度低于23.5℃-柱头总外露率79%-穗包颖粒率21.4%-异交结实率56.9%-叶瘟8级-穗瘟9级-高感稻瘟病-白叶枯病7级-糙米率79.7%-精米率72.2%-整精米率53.2%-粒长6.8毫米-长宽比2.9-垩白粒率2%-垩白度0.1%-透明度2级-碱消值4.1级-胶稠度84毫米-直链淀粉含量12.2%--平均温度在19-22℃-每亩喷施15-16克-每亩产量150公斤-157S

问：10优18产量表现如何？
答：2007年参加京、津、唐粳稻组品种区域试验，平均亩产679.2千克，比对照津原45增产15.2%（极显著）；2008年续试平均亩产617.4千克，比对照津原45增产8.4%（极显著）；两年区域试验平均亩产648.3千克，比对照津原45增产11.8%，增产点比例75%。2008年生产试验平均亩产611.9千克，比对照津原45增产11.5%。
话题：作物品种信息-10优18-产量表现-平均亩产679.2千克-对照津原45增产15.2%-平均亩产617.4千克-对照津原45增产8.4%-平均亩产648.3千克-津原45增产11.8%-亩产611.9千克

问：10优18适宜范围/适宜推广区域？
答：适宜在北京市、天津市、河北省冀东及中北部的一季春稻区种植。
话题：作物品种推荐-10优18-适宜范围-北京市、天津市、河北省冀东及中北部的一季春稻区

问：适宜在北京市、天津市、河北省冀东及中北部的一季春稻区种植。水稻品种10优18有哪些栽培技术要点？
答：1.育秧：适时播种，播种前做好晒种与消毒，防治干尖线虫病和恶苗病。秧田用种量约为常规品种的1/2，培育带蘖壮秧。2.移栽：秧龄35天左右移栽，行株距为30厘米×13.3厘米，每穴栽双株。3.肥水管理：氮、磷、钾、锌肥配合使用，注意干湿交替，确保每亩有效穗数在18万左右。4.病虫防治：注意防治稻曲病，其它病虫草害同一般常规稻。
话题：种植流程化管理-10优18-栽培技术要点-育秧：秧田用种量约为常规品种的1/2-移栽：秧龄35天左右移栽，行株距为30厘米×13.3厘米，每穴栽双株-每亩有效穗数在18万左右-注意防治稻曲病

问：适宜在江西省、湖南省、湖北省、安徽省、浙江省双季稻区的稻瘟病轻发区作晚稻种植，稻瘟病重发区不宜种植。水稻品种10香优郁香有哪些栽培技术要点？
答：在长江中下游作双季晚稻种植，一般6月上旬至6月下旬播种， 秧田播种量每亩9.0千克， 大田亩用种量1.0-1.3千克。水育秧移栽叶龄叶左右，秧龄控制在25天以内。栽插株行距20.0厘米×22.0厘米，每亩插足基本苗6万以上。科学施肥，重施底肥，早施分蘖肥，忌后期偏施氮肥。其他田间管理、栽培和收获措施均按该类型品种常规方法实施。注意及时防治稻瘟病、白叶枯病、褐飞虱等病虫害。
话题：种植流程化管理-10香优郁香-栽培要点-6月上旬至6月下旬播种-双季晚稻种植-秧田每亩9.0千克-大田种量1.0-1.3千克-秧龄25天-株行距20.0厘米×22.0厘米-每亩插足基本苗6万-早施分蘖肥-忌后期偏施氮肥-防治稻瘟病、白叶枯病、褐飞虱

问：{query}
答：{answer}
话题："""

Variety_disease_extractor_prompt="""提取以下一段话中的作物品种名称或者病害名称，若有多个，以英文逗号','分隔，以下是几个例子：
文本：6优160有哪些特征特性？
提取结果：6优160
文本：Y两优8866适宜范围\/适宜推广区域？
提取结果：Y两优8866
文本：白眉野草螟的分布及危害？
提取结果：白眉野草螟
文本：斑鞘豆叶甲卵形态特征有哪些？
提取结果：斑鞘豆叶甲
文本：{插入位置}
提取结果："""

# 文本：该品种属感温型三系杂交水稻，桂中、桂北早稻种植，全生育期127天左右，比对照中优838迟熟2天；晚稻种植，全生育期111天左右，与对照中优838相当。主要农艺性状（平均值）表现：每亩有效穗数17.5万，株高101.8厘米，穗长23.4厘米，每穗总粒数153.8粒，结实率84.4%，千粒重24.1克。米质主要指标：糙米率81.7%，整精米率69.6%，长宽比2.9，垩白米率16%，垩白度2.6%，胶稠度78毫米，直链淀粉含量14.8%；抗性：苗叶瘟7级，穗瘟9级，穗瘟损失指数80.5%，稻瘟病抗性指数8.5；白叶枯病Ⅳ型7级，Ⅴ型9级。回答满足要求的水稻品种？
# 提取结果：感温型三系杂交水稻,桂中,桂北,127天,111天,17.5万,101.8厘米,23.4厘米,153.8粒,结实率84.4%,千粒重24.1克,糙米率81.7%,整精米率69.6%,长宽比2.9,垩白米率16%,垩白度2.6%,胶稠度78毫米,淀粉含量14.8%,苗叶瘟7级,穗瘟9级,80.5%,8.5,Ⅳ型7级,Ⅴ型9级

# 文本：适宜在辽宁省≥10℃活动积温2800℃以上的中晚熟春玉米类型区种植。回答满足要求的玉米品种？
# 提取结果：辽宁省≥10℃活动积温2800℃以上,中晚熟春玉米类型区

# 文本：椭圆形，有光泽，长梨形，有小柄，与叶面垂直，长宽约为0.21mm×0.096mm 。 卵柄通过产卵 器插入叶表面裂缝中。卵初产时淡黄绿色，孵化前颜色加深，至深褐色。可能得了什么疾病？
# 提取结果：椭圆形,有光泽,长梨形,有小柄,与叶面垂直,长宽约为0.21mm×0.096mm,淡黄绿色,深褐色

# 文本：体长约6.98mm,  宽约2mm。体背面淡黄色，向后至臀板逐渐加深成淡褐色，腹面色浅；头  骨前半部色渐深呈黑色，前胸背板前缘色略深。胸部具3对足，足淡褐色。腹部多横褶，背面尤其明显； 体表具成列刚毛；化蛹前体变粗而稍弯曲(彩图3-42-5)。可能得了什么疾病？
# 提取结果：体长约6.98mm,宽约2mm,体背面淡黄色,淡褐色,腹面色浅,渐深呈黑色,前胸背板前缘色略深,3对足,足淡褐色,腹部多横褶,成列刚毛,化蛹前体变粗
fuzzysearch_extractor_prompt="""提取以下一段话中可能作为数据库模糊搜索的字符串，若有多个，以英文逗号','分隔，以下是几个例子：
文本：该品种属感温型三系杂交水稻，桂中、桂北早稻种植，全生育期127天左右，比对照中优838迟熟2天；晚稻种植，全生育期111天左右，与对照中优838相当。主要农艺性状（平均值）表现：每亩有效穗数17.5万，株高101.8厘米，穗长23.4厘米，每穗总粒数153.8粒，结实率84.4%，千粒重24.1克。米质主要指标：糙米率81.7%，整精米率69.6%，长宽比2.9，垩白米率16%，垩白度2.6%，胶稠度78毫米，直链淀粉含量14.8%；抗性：苗叶瘟7级，穗瘟9级，穗瘟损失指数80.5%，稻瘟病抗性指数8.5；白叶枯病Ⅳ型7级，Ⅴ型9级。回答满足要求的水稻品种？
提取结果：感温型三系杂交水稻,桂中,桂北,127天,111天,17.5万,101.8厘米,23.4厘米,153.8粒,结实率84.4%,千粒重24.1克,糙米率81.7%,整精米率69.6%,长宽比2.9,垩白米率16%,垩白度2.6%,胶稠度78毫米,淀粉含量14.8%,苗叶瘟7级,穗瘟9级,80.5%,8.5,Ⅳ型7级,Ⅴ型9级

文本：适宜在辽宁省≥10℃活动积温2800℃以上的中晚熟春玉米类型区种植。回答满足要求的玉米品种？
提取结果：辽宁省≥10℃活动积温2800℃以上,中晚熟春玉米类型区

文本：椭圆形，有光泽，长梨形，有小柄，与叶面垂直，长宽约为0.21mm×0.096mm 。 卵柄通过产卵 器插入叶表面裂缝中。卵初产时淡黄绿色，孵化前颜色加深，至深褐色。可能得了什么疾病？
提取结果：椭圆形,有光泽,长梨形,有小柄,与叶面垂直,长宽约为0.21mm×0.096mm,淡黄绿色,深褐色

文本：体长约6.98mm,  宽约2mm。体背面淡黄色，向后至臀板逐渐加深成淡褐色，腹面色浅；头  骨前半部色渐深呈黑色，前胸背板前缘色略深。胸部具3对足，足淡褐色。腹部多横褶，背面尤其明显； 体表具成列刚毛；化蛹前体变粗而稍弯曲(彩图3-42-5)。可能得了什么疾病？
提取结果：体长约6.98mm,宽约2mm,体背面淡黄色,淡褐色,腹面色浅,渐深呈黑色,前胸背板前缘色略深,3对足,足淡褐色,腹部多横褶,成列刚毛,化蛹前体变粗

文本：{插入位置}
提取结果："""

class GPT:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url="",
            api_key="",
            http_client=httpx.Client(
                base_url="",
                follow_redirects=True,
            ),
        )

    def gpt_chat(self, content, role=base_role, model="gpt-35-turbo", max_attempts=3):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
        )
        return completion.choices[0].message.content

    def topic_extract(self, query, answer, prompt=topic_extractor_prompt, role=topic_extractor_role,
                      model="gpt-35-turbo", max_attempts=3):
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{answer}", answer)
        for i in range(max_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": prompt}
                    ], stream=True
                )
                response = ""
                for chunk in completion:
                    if chunk.choices is not None and chunk.choices[0].delta.content is not None:
                        response += chunk.choices[0].delta.content
                if response:
                    return response
            except Exception as e:
                print(e)
                print(f"抽取话题：正在进行第{i + 2}次尝试")

        print("话题抽取失败，已返回空字符串")
        return ""

    def judge_answer(self, query, context: list):
        #     combined_context=""
        #     for idx in range(1,len(context)+1):
        #         combined_context+=f"{idx}. {context[idx-1]}\n\n"

        #     prompt=f"""判断以下的每个上下文是否分别能够回答下面的问题，若上下文包含回答问题所需信息则输出其序号，序号以英文逗号","隔开，没有则返回空字符串
        # 上下文：{combined_context}
        # 问题：{query}
        # 回答："""
        prompt = f"""判断以下的上下文是否能够回答问题，若上下文包含回答问题所需信息则输出'【是】',否则输出'【否】'
    上下文：{context}
    问题：{query}
    回答："""
        # print(prompt)
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        # {"role": "system", "content": "You are a document usefulness judgment assistant, and your output are numbers separated by English commas."},
                        {"role": "system",
                         "content": "You are a document usefulness judgment assistant, and your output is '【是】' or '【否】'"},
                        {"role": "user", "content": prompt}
                    ], stream=True
                )
                break
            except Exception as e:
                print(e)
                time.sleep(3)

        response = ""
        # for chunk in completion:
        #     if not chunk.choices:
        #         return self.judge_answer(query,context)
        #     if chunk.choices[0].delta.content is not None:
        #         response+=chunk.choices[0].delta.content
        # # print(response)
        # ids=response.split(',')
        # ids=[int(i) for i in ids if i.isdigit()]
        # ids=[i for i in ids if i>=1 and i<=len(context)]
        # return ids
        try:
            for chunk in completion:
                if chunk.choices is not None and chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
        except Exception as e:
            print("judge_answer函数中上下文为：", context)
            print("judge_answer函数中问题为：", query)
            print("发生异常：", str(e))
            print(chunk)
        if "【是】" in response:
            return 1
        else:
            return 0

    def final_judge(self, valid_ans, query):
        if not valid_ans:
            return None
        prompt = f"""给出问题以及一些检索上下文及回答，上下文是检索事实，选择其中能够最好回答问题的回答序号，最佳回答序号只有一个。
    问题：{query}
    """
        for index, ans in enumerate(valid_ans):
            prompt += f""""上下文{index + 1}：{ans['knowledge']}\n回答{index + 1}：{ans['answer']}\n\n"""
        prompt += "最佳答案序号为："
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system",
                         "content": "You are an assistant who determines the best answer, and your output is Arabic numerals."},
                        {"role": "user", "content": prompt}
                    ], stream=True
                )
                break
            except Exception as e:
                print(e)
                time.sleep(3)

        response = ""
        for chunk in completion:
            if not chunk.choices:
                return self.final_judge(valid_ans, query)
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content

        # print(prompt)
        # print(response)
        if len(re.findall(r'\d+', response)) == 0:
            return None
        return int(re.findall(r'\d+', response)[0]) - 1

    def rag_judge(self, query, ori_res, rag_res):
        prompt = f"""能够更加具体地回答问题：“{query}”的回答是？
    1：{ori_res}
    2：{rag_res}
    你只需要回答1或2
    """
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system",
                         "content": "You are an assistant who determines the best answer, and your output is Arabic numerals."},
                        {"role": "user", "content": prompt}
                    ], stream=True
                )
                break
            except Exception as e:
                print(e)
                time.sleep(3)

        response = ""
        for chunk in completion:
            if not chunk.choices:
                # return self.final_judge(valid_ans, query)
                pass
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content

        # print(prompt)
        # print(response)
        if len(re.findall(r'\d+', response)) == 0:
            return self.rag_judge(query, ori_res, rag_res)
        return int(re.findall(r'\d+', response)[0]) - 1

    def generate_subclaim(self, answer):
        # print(f"问题：{query}")
        # print(f"回答：{answer}")
        if len(answer) <= 20:
            return f"[\"{answer}\"]"
        prompt = f"""阅读段落，从段落中提取3个声明,并将声明以列表形式输出。

    段落：全生育期141天，比对照吉玉粳早2天。株高92厘米，株型紧凑，平均分蘖5个。紧穗，穗长15厘米，穗粒数125，结实率88.5%。籽粒椭圆型、无芒，颖壳及颖尖黄色，千粒重23.5g。抗性鉴定，中抗苗瘟病，高抗叶瘟病，中抗穗瘟病。米质主要指标：出糙率83.4%，整精米率72.1%，垩白粒率8.5%，垩白度1.9%，粗蛋白7.76%，长\/宽1.7，直链淀粉18.65%，胶稠度70.0毫米。食味评价76分。
    声明：["全生育期141天，比对照吉玉粳早2天。株高92厘米，株型紧凑，平均分蘖5个。紧穗，穗长15厘米，穗粒数125，结实率88.5%。籽粒椭圆型、无芒，颖壳及颖尖黄色，千粒重23.5g。","病害抗性有：中抗苗瘟病，高抗叶瘟病，中抗穗瘟病。","米质指标：出糙率83.4%，整精米率72.1%，垩白粒率8.5%，垩白度1.9%，粗蛋白7.76%，长\/宽1.7，直链淀粉18.65%，胶稠度70.0毫米。食味评价76分。]

    段落：2015年参加水稻品种区域试验，平均亩产670.4千克，比组均值增产6%；2016年参加水稻品种区域试验，平均亩产594.4千克，比组均值增产4.4%；2017年参加水稻品种生产试验，平均亩产619.6千克，比对照增产12.1%。
    声明：["在2015年参加水稻品种区域试验，平均亩产670.4千克，比组均值增产6%。","在2016年参加水稻品种区域试验，平均亩产594.4千克，比组均值增产4.4%。","在2017年参加水稻品种生产试验，平均亩产619.6千克，比对照增产12.1%。]

    段落：4月上旬播种，播种密度适中均匀，播种量机播每盘折干籽115克左右。催芽播种育壮苗，5月中下旬插秧；株距13厘米-15厘米，行距30厘米-33厘米，每穴5-6苗。公顷施纯氮120-130千克，纯磷50-60千克，纯钾80-90千克，按底肥、蘖肥、穗肥5:3:2比例，分3次施肥，并配合适量的硅、锌、钙、镁、硼、铁、锰、钼、铜等中微量元素。综合营养、稳健生长，杆状穗大，产量高，有效掌握[底肥足、蘖肥早、穗肥巧]的原则。注意事项：田间水层管理采取分蘖期浅，孕穗期深，籽粒灌浆期浅的灌溉方法，全生育期注意防治各种病、虫、草害。
    声明：["在4月上旬播种，播种密度适中均匀，播种量机播每盘折干籽115克左右。催芽播种育壮苗，5月中下旬插秧；株距13厘米-15厘米，行距30厘米-33厘米，每穴5-6苗。","公顷施纯氮120-130千克，纯磷50-60千克，纯钾80-90千克，按底肥、蘖肥、穗肥5:3:2比例，分3次施肥，并配合适量的硅、锌、钙、镁、硼、铁、锰、钼、铜等中微量元素。综合营养、稳健生长，杆状穗大，产量高，有效掌握[底肥足、蘖肥早、穗肥巧]的原则。","栽培注意事项：田间水层管理采取分蘖期浅，孕穗期深，籽粒灌浆期浅的灌溉方法，全生育期注意防治各种病、虫、草害。]

    段落：4月上旬播种，播种密度适中均匀，播种量机播每盘折干籽115克左右。催芽播种育壮苗，5月中下旬插秧；株距13厘米-15厘米，行距30厘米-33厘米，每穴5-6苗。公顷施纯氮120-130千克，纯磷50-60千克，纯钾80-90千克，按底肥、蘖肥、穗肥5:3:2比例，分3次施肥，并配合适量的硅、锌、钙、镁、硼、铁、锰、钼、铜等中微量元素。综合营养、稳健生长，杆状穗大，产量高，有效掌握[底肥足、蘖肥早、穗肥巧]的原则。注意事项：田间水层管理采取分蘖期浅，孕穗期深，籽粒灌浆期浅的灌溉方法，全生育期注意防治各种病、虫、草害。
    声明：["4月上旬播种，播种密度适中均匀，播种量机播每盘折干籽115克左右。催芽播种育壮苗，5月中下旬插秧；株距13厘米-15厘米，行距30厘米-33厘米，每穴5-6苗。","公顷施纯氮120-130千克，纯磷50-60千克，纯钾80-90千克，按底肥、蘖肥、穗肥5:3:2比例，分3次施肥，并配合适量的硅、锌、钙、镁、硼、铁、锰、钼、铜等中微量元素。综合营养、稳健生长，杆状穗大，产量高，有效掌握[底肥足、蘖肥早、穗肥巧]的原则。","注意事项：田间水层管理采取分蘖期浅，孕穗期深，籽粒灌浆期浅的灌溉方法，全生育期注意防治各种病、虫、草害。"]

    段落：{answer}
    声明："""
        # print(prompt)
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helper that outputs a list."},
                        {"role": "user", "content": prompt}
                    ], stream=True
                )
                break
            except Exception as e:
                print("An error occurred:", e)

        response = ""
        for chunk in completion:
            if chunk.choices is not None and chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content

        return response

    def list_generator(self, prompt):
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helper that outputs a list."},
                        {"role": "user", "content": prompt}
                    ], stream=True
                )
                break
            except Exception as e:
                print("An error occurred:", e)

        response = ""
        for chunk in completion:
            if chunk.choices is not None and chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content

        return response

    def compute_claim_recall(self, text, sub_claim):
        cl_list = []
        sys_role = "You are an text entailment assistant, and your output is 'ENTAILMENT', 'CONTRADICTION' or 'NEUTRAL'."

        prompt = f"""推理前提与假设的蕴含关系，若前提能推出假设，则输出ENTAILMENT，若前提与假设矛盾，则输出CONTRADICTION，若前提不能推出假设，也没有矛盾，则输出NEUTRAL。

    前提：1、适时播种：根据当地种植习惯与特优63同期播种，秧龄30天左右。2、合理密植：一般早稻插植规格23厘米×13厘米，中稻26厘米×16厘米，每蔸插2粒谷苗，保证亩基本苗7～9万。3、肥水管理：以基肥和前期追肥为主，一般每亩需施纯氮10～12公斤，氮、磷、钾比例为1∶0.5∶0.8；前期浅灌，够苗晒田，后期干湿交替。4、注意白叶枯病等病虫害的防治。
    假设：合理密植：一般早稻插植规格23厘米×13厘米，中稻26厘米×16厘米，每蔸插2粒谷苗，保证亩基本苗7～9万。
    输出：ENTAILMENT

    前提：在2005年生产试验平均亩产459.2公斤，比对照特优63增产1.3%.
    假设：在2005年生产试验平均亩产478公斤，比对照特优63增产4%.
    输出：CONTRADICTION

    前提：湘品22
    假设：省区试结果：每亩有效穗18.6万穗，每穗总粒数123.2粒，结实率81.7%，千粒重29.2克。抗性：稻瘟病抗性综合指数7.8，白叶枯病抗性7级，高感稻瘟病，感白叶枯病；耐低温能力较强。
    输出：NEUTRAL

    前提：{text}
    假设：{sub_claim}
    输出："""
        res = self.gpt_chat(prompt, role=sys_role)
        # print(text)
        # print(sub_claim)
        # print(res)
        # print()
        if res == 'ENTAILMENT':
            cl_list.append(1)
        elif res == 'CONTRADICTION' or res == 'NEUTRAL':
            cl_list.append(0)

        return sum(cl_list) / len(cl_list) if len(cl_list) else 0

    def RAG_process(self, context, query):
        template = f"""你是问答任务的助手。
        使用以下检索到的上下文来回答问题。
        保持答案具体。
        上下文：{context}
        问题：{query}
        回答："""
        res = self.gpt_chat(template)
        ans = {}
        ans['knowledge'] = context
        ans['answer'] = res
        return ans

    # BB
    def self_revise(self, answer):
        template = f"""请检查下面这个语句中是否存在某些字词的无意义重复。如果存在，请删除那些无意义的重复部分并返回删除后的语句；如果不存在，则直接返回原始语句。
        【例1】
        原始语句：农村土地制度改革为农民带来了以下好处：1. 保障了农民的土地权益，使农民有了更多的话语权，提高了农民的参与度；2. 提高了农民的收入水平，通过土地流转、土地入股、土地信托等方式，实现了农民收入的多元化；3. 保障了农民的土地权益，使农民有了更多的话语权，提高了农民的参与度；4. 保障了农民的土地权益，使农民有了更多的话语权，提高了农民的参与度；
        回答：农村土地制度改革为农民带来了以下好处：1. 保障了农民的土地权益，使农民有了更多的话语权，提高了农民的参与度；2. 提高了农民的收入水平，通过土地流转、土地入股、土地信托等方式，实现了农民收入的多元化。
        【例1结束】
        【例2】
        原始语句：教育技术可以通过提供远程教育、个性化学习、教育资源共享等方式，帮助提升边远地区学生的学习成果和教育机会。
        回答：教育技术可以通过提供远程教育、个性化学习、教育资源共享等方式，帮助提升边远地区学生的学习成果和教育机会。
        【例2结束】
        【输入】
        原始语句：{answer}
        回答："""
        ans = self.gpt_chat(template)
        return ans

    def chat(self,query, answer, prompt=topic_extractor_prompt, role=topic_extractor_role,
                      model="gpt-35-turbo", max_attempts=3):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content,[]