
# 定义日志文件
log_file="all_services.log"

# 清空之前的日志内容
echo "" > $log_file
# 启动docker服务
docker-compose -f docker-compose-all.yaml up -d
# 启动rerank服务
nohup python3 -u ./qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py >> $log_file 2>&1 &
echo "The rerank service is ready! (1/3)" | tee -a $log_file
echo "rerank服务已就绪! (1/3)" | tee -a $log_file

# 启动OCR服务
CUDA_VISIBLE_DEVICES=0 nohup python3 -u ./qanything_kernel/dependent_server/ocr_serve/ocr_server.py >> $log_file 2>&1 &
echo "The ocr service is ready! (2/3)" | tee -a $log_file
echo "OCR服务已就绪! (2/3)" | tee -a $log_file

# 启动qanything后端服务
nohup python3 -u ./qanything_kernel/qanything_server/sanic_api.py >> $log_file 2>&1 &
echo "The qanything backend service is ready! (3/3)" | tee -a $log_file
echo "qanything后端服务已就绪! (3/3)" | tee -a $log_file
