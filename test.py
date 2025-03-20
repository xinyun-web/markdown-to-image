import requests,json
import utils
import os 
import time
import torch
import logging
import subprocess
import aiohttp
import asyncio
import zipfile
from io import BytesIO
import copy
import shutil
from datetime import datetime

file_path = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=os.path.join(file_path, 'error_log.txt'), level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

take_num = 30
proc_task_num = 30
isinit=True

# web_url = "http://127.0.0.1:8000/kolors_generate_image2image"  # FastAPI 服务地址
web_url = "http://127.0.0.1:8000/app2/kolors_generate_image2image"  # FastAPI 服务地址

def write_file(file_path,content):
    with open(file_path, 'a', encoding='utf-8') as file:
        # 写入一行数据
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {content}\n") 

# 定义一个带时间戳的打印函数
def print_with_timestamp(message):
    # 获取当前日期和时间，格式化为 "年-月-日 时:分:秒"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 打印带时间戳的消息
    print(f"{current_time}: {message}")

async def fetch_image(session,params, retries=3):
    id = params.get('id')
    sParams = copy.deepcopy(params)  # 创建深拷贝
    global proc_task_num 
    for i in range(retries):
        try:
            timeout = aiohttp.ClientTimeout(total=6000)  # 设置超时时间为6000秒,100分钟。
            async with session.post(web_url,params=sParams, timeout=timeout) as response:
                response_text = await response.text()
                if response.status == 200: 
                    json_data = await response.json()  # 解析返回的JSON数据
                    proc_task_num -= 1
                    new_message = None
                    if 'message' in json_data:
                        new_message = json_data['message']
                    if json_data.get('status') == 'success':
                        print_with_timestamp(f"{id}图像生成成功")
                        write_file('datalog.txt',f"ok - id:{id},{new_message}")
                        break
                    else:
                        print_with_timestamp(f"id:{id}生成或上传失败")
                        logging.error(f"id:{id},生成或上传失败:{str(json_data)}")
                        write_file('datalog.txt',f"error - id:{id},{new_message}")
                        utils.update_status(ai_id=id, memo=f"生成或上传失败!")
                        proc_task_num -= 1
                        break
                else:
                    await asyncio.sleep(2)  
                    if i>=retries-1:
                        proc_task_num -= 1
                        print_with_timestamp(f"id:{id}生成或上传失败,响应状态：{response.status}")
                        logging.error(f"id:{id},生成或上传失败,响应状态：{response.status},响应内容: {response_text}")
                        write_file('datalog.txt',f"error - id:{id},f'响应状态：{response.status}'")
                        utils.update_status(ai_id=id, memo=f"生成或上传失败!")
        except asyncio.TimeoutError:
            await asyncio.sleep(2)   # 指数退避
            if i>=retries-1:
                print_with_timestamp(f'{id}请求超时')
                proc_task_num -= 1
                logging.error(f"id:{id},请求超时")
                write_file('datalog.txt',f"error - id:{id},请求超时!")
                utils.update_status(ai_id=id, memo=f"请求超时!")
        except Exception as e:
            await asyncio.sleep(2) 
            if i>=retries-1:
                error_message = str(e)
                if len(error_message) > 100:
                    error_message = error_message[:100] + "..."  # 截断错误信息
                print_with_timestamp(f'{id}网络请求出错：{error_message}')
                proc_task_num -= 1
                logging.error(f"id:{id},网络请求出错：{error_message}")
                write_file('datalog.txt',f"error - id:{id},网络请求出错!")
                utils.update_status(ai_id=id, memo=f"网络请求出错!")
    # 运行完一个，补充一下数据
    await supply_data() 
    
async def get_image(params_list):
    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for params in params_list:
                tasks.append(asyncio.create_task(fetch_image(session, params)))
            print_with_timestamp('请求开始.......')
            await asyncio.gather(*tasks)
    except Exception as e:
        print_with_timestamp(f'批量获取图片出错：{e}')
        logging.error(f"批量获取图片出错：{e}")

async def supply_data():
    global proc_task_num
    try:
        pageSize = take_num-proc_task_num
        print_with_timestamp(f'补充数据{pageSize}条')
        if pageSize>0:
            result= utils.fetch_info(pageSize)
            if result is False:
                return 
            # 如果数据有变化，处理数据并更新
            if result and 'data' in result:
                result_len = len(result['data'])
                write_file('datanumber.txt',f"请求条数{pageSize},获取数据长度:{result_len}")
                proc_task_num += result_len
                await process_data(result['data'])
    except Exception as e:
        print_with_timestamp(f'补充数据出错：{e}')
        logging.error(f'补充数据出错：{e}')

# 当值大于指定的数时，返回指定的数,  小于 0 时，返回 0
def set_zero_if_greater_than(num, threshold):
    if num > threshold:
        return threshold
    elif num <= 0:
        return 0
    else:
        return num
    
# 处理从服务器上拿取的数据并在 SD 中生成图片,最后处理生成后的图片
async def process_data(result):
    try:
        params_list = []
        for res in result:
            try:
                if not res: 
                    continue
                with open('detils.txt', 'a', encoding='utf-8') as file:
                # 写入一行数据
                    file.write(str(res) + "\n")
                id = res["id"]
                data={
                        "id":res.get("id", 123),
                        "prompt":res.get("prompt", None),
                        "ratio":res.get("fbl", "9:16"),
                        "image":res.get("image", None),
                        "num_image":res.get("num_image", 2),
                        "weight":res.get("weight", 1.5), # 越小越像参考图
                    }

            except Exception as e:
                print_with_timestamp(f"秒爆通用爆款(图生图)获取数据信息过程出错: {e}")
                logging.error(f"秒爆通用爆款(图生图)获取数据信息过程出错: {e}")
                continue
            try:
                num_image = data["num_image"]
                pos_prompt = data['prompt']
                weights = set_zero_if_greater_than(float(data['weight']), 2)
                weight = float(2 - weights)
                
                # 宽高
                width, hight = utils.get_w_h(data['ratio'])
                if width % 8 != 0:
                    width = width - width % 8
                if hight % 8 != 0:
                    hight = hight - hight % 8

                params={
                    'prompt': pos_prompt,
                    'ip_img_path': data['image'],  # 将 bool 转为 str
                    'height': int(hight),
                    'width': int(width),
                    'guidance_scale': 5.0,
                    'num_inference_steps': 50,
                    'num_images_per_prompt': num_image,
                    'id':id,
                    'weight':weight,
                    'negative_prompt':''
                }

                params_list.append(params)
            except Exception as e:
                print_with_timestamp(f"{id}秒爆通用爆款(图生图)运行过程出错: {e}")
                logging.error(f"{id}秒爆通用爆款(图生图)运行过程出错: {e}")
                error_message = str(e)
                if len(error_message) > 100:
                    error_message = error_message[:100] + "..."  # 截断错误信息
                write_file('datalog.txt',f"error - id:{id},运行过程出错!")
                utils.update_status(id, memo=error_message)
                continue
        print_with_timestamp(f"params_list:{len(params_list)}")
        #获取图片
        if not params_list:
            return
        await get_image(params_list)
         # 设置监视间隔
        await asyncio.sleep(2)
    except Exception as e:
        logging.error(f"秒爆通用爆款(图生图)运行出错: {e}")

# 不停的从服务器拿去数据
async def monitor_data_changes():
    try:
        global proc_task_num
        global isinit
        while True:
            print_with_timestamp("从服务器拿取数据")
            pageSize = take_num
            if not isinit:
                pageSize = take_num-proc_task_num
                print_with_timestamp(f'take_num:{take_num},proc_task_num:{proc_task_num},需补充数据{pageSize}条')
            if pageSize <= 0:
                await asyncio.sleep(5)
                continue
            result= utils.fetch_info(pageSize)
            # print_with_timestamp("服务器拿取的数据：", result)
            if result is False:
                # 设置监视间隔 
                time.sleep(5)
                continue
            # 如果数据有变化，处理数据并更新
            if result and 'data' in result:
                result_len = len(result['data'])
                print_with_timestamp(f"result 长度:{result_len}")
                write_file('datanumber.txt',f"请求条数{pageSize},获取数据长度init:{result_len}")
                if isinit:
                    proc_task_num = result_len
                else:
                    proc_task_num += result_len
                isinit=False
                # 创建任务并调度
                task = asyncio.create_task(process_data(result['data']))
                # 可选：添加任务完成后的回调
                task.add_done_callback(lambda t: print_with_timestamp("任务完成"))
            await asyncio.sleep(5)
    except Exception as e:
        print_with_timestamp(f'任务运行出错：{e}')
        logging.error(f"任务运行出错：{e}")
        await asyncio.sleep(10)

async def main():
    await monitor_data_changes()

if __name__ == "__main__":
    # 启动监视
    asyncio.run(main())
