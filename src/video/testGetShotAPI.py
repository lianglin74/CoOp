from video.getShotMoments import getShotAPI

videoFileName="/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_selected_training/5102222619_5004703696_92.mp4"
predict_file="/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_selected_training/5102222619_5004703696_92.tsv"

pred_results = getShotAPI(videoFileName, predict_file)
print(pred_results)