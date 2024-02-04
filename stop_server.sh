
# 停止所有相关服务进程
docker-compose -f docker-compose-all.yaml stop

pkill -f rerank_server.py
pkill -f ocr_server.py
pkill -f sanic_api.py

echo "所有服务已暂停。"
