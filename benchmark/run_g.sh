# python -u graph_pred.py --device 0 --model gcn > gpred_gcn.log &
python -u graph_pred.py --device 1 --model gin > gpred_gin.log &
python -u test.py --device 2 --model gin > gpred_gin_vn.log &
# python -u test.py --device 3 --model gcn > gpred_gcn_vn.log &