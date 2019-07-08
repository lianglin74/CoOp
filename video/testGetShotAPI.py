from video.getShotMoments import getShotAPI

videoFileName="/home/mtcadmin/test/1551538896210_sc99_01_q1.mp4"
predict_file="/home/mtcadmin/test/1551538896210_sc99_01_q1.tsv"

pred_results = getShotAPI(videoFileName, predict_file)
print pred_results