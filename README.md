https://github.com/vllm-project/vllm-ascend/issues/17110
<img width="1204" height="575" alt="image" src="https://github.com/user-attachments/assets/00e871e9-e263-417f-b819-4effb21dc928" />
ms-e5a1708a-d4c4-4e70-8039-970f4ce486ae
Eco-Tech/gemma-4-26B-A4B-it-w4a4-mxfp4
不是 SSL，也不是线程池报错；是 ModelScope 上传大 blob 时的 HTTP ReadTimeout，并发可能让它更容易触发，但根因是请求 10 秒没读到响应。
